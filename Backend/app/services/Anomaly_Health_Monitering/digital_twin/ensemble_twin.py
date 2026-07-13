"""
4-Model Ensemble Digital Twin for CA-EDT-AHMA.

CA-EDT-AHMA:
Context-Aware Ensemble Digital Twin for Explainable Health Monitoring
and Anomaly Reasoning.

Purpose:
Combines four digital twin regressors for raw measured X_s sensors only:

1. Random Forest Regressor
2. XGBoost Regressor
3. LightGBM Regressor
4. MLP Digital Twin Regressor implemented using TensorFlow/Keras

Research-correct behavior:
- Uses dev split only to calculate ensemble weights.
- Uses inverse dev RMSE weighting.
- Uses raw X_s sensors only.
- Does not use Y_dev/Y_test.
- Does not train or fit using test split.
- Test split is inference/evaluation only.

Memory-safety:
- Does not load full prediction CSV files into RAM.
- Reads RF/XGB/LGBM/MLP prediction CSV files in aligned chunks.
- Writes ensemble predictions chunk by chunk to a temporary CSV.
- Replaces final ensemble CSV only after successful completion.

Final ensemble formula:
ensemble_predicted_Xs =
    rf_weight   * rf_predicted_Xs
  + xgb_weight  * xgb_predicted_Xs
  + lgbm_weight * lgbm_predicted_Xs
  + mlp_weight  * tf_predicted_Xs
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List
import gc
import os
import sys

import numpy as np
import pandas as pd


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    if BACKEND_ROOT not in sys.path:
        sys.path.append(BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_json,
    read_json_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_raw_xs_columns


logger = get_logger(__name__)


class EnsembleDigitalTwin:
    """
    Memory-safe 4-model ensemble combiner.

    Ensemble members:
    - Random Forest
    - XGBoost
    - LightGBM
    - MLP Digital Twin implemented using TensorFlow/Keras
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize ensemble digital twin.

        Args:
            chunk_size: Number of rows processed per CSV chunk.
        """
        Config.create_directories()

        self.chunk_size = int(getattr(Config, "ENSEMBLE_CHUNK_SIZE", chunk_size))

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        self.mlp_predictions_csv: Path = getattr(
            Config,
            "MLP_TWIN_PREDICTIONS_CSV",
            getattr(Config, "TF_PREDICTIONS_CSV", Config.OUTPUT_DIR / "mlp_twin_predictions.csv"),
        )

        self.ensemble_metadata_path: Path = getattr(
            Config,
            "ENSEMBLE_METADATA_PATH",
            Config.DIGITAL_TWIN_MODEL_DIR / "ensemble_metadata.json",
        )

        self.ensemble_members = ["rf", "xgb", "lgbm", "mlp"]

        print("[PROGRESS] EnsembleDigitalTwin initialized successfully")
        print(f"[PROGRESS] Ensemble chunk size: {self.chunk_size}")
        print("[PROGRESS] Ensemble members: RF + XGBoost + LightGBM + MLP")
        print(f"[PROGRESS] RF predictions: {Config.RF_PREDICTIONS_CSV}")
        print(f"[PROGRESS] XGB predictions: {Config.XGB_PREDICTIONS_CSV}")
        print(f"[PROGRESS] LGBM predictions: {Config.LGBM_PREDICTIONS_CSV}")
        print(f"[PROGRESS] MLP predictions: {self.mlp_predictions_csv}")
        print(f"[PROGRESS] Ensemble output: {Config.ENSEMBLE_PREDICTIONS_CSV}")

    # ==================================================================================
    # Compatibility helper
    # ==================================================================================

    def _safe_merge_by_occurrence(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        merge_columns: List[str],
    ) -> pd.DataFrame:
        """
        Merge while avoiding many-to-many expansion when merge keys are duplicated.

        This method is kept for compatibility. The main ensemble flow uses
        streaming chunk alignment instead of full-memory merges.

        Args:
            left_df: Left DataFrame.
            right_df: Right DataFrame.
            merge_columns: Merge key columns.

        Returns:
            Safely merged DataFrame.
        """
        print("[PROGRESS] Entering EnsembleDigitalTwin._safe_merge_by_occurrence")
        print(f"[PROGRESS] Left DataFrame shape before merge: {left_df.shape}")
        print(f"[PROGRESS] Right DataFrame shape before merge: {right_df.shape}")
        print(f"[PROGRESS] Merge columns: {merge_columns}")

        left_duplicates = int(left_df.duplicated(subset=merge_columns).sum())
        right_duplicates = int(right_df.duplicated(subset=merge_columns).sum())

        print(f"[PROGRESS] Duplicate merge keys in left_df: {left_duplicates}")
        print(f"[PROGRESS] Duplicate merge keys in right_df: {right_duplicates}")

        if left_duplicates > 0 or right_duplicates > 0:
            print("[WARNING] Using occurrence-index safe merge in ensemble")

            left_temp = left_df.copy(deep=False)
            right_temp = right_df.copy(deep=False)

            left_temp["__merge_occurrence"] = left_temp.groupby(merge_columns).cumcount()
            right_temp["__merge_occurrence"] = right_temp.groupby(merge_columns).cumcount()

            merged = left_temp.merge(
                right_temp,
                on=merge_columns + ["__merge_occurrence"],
                how="left",
                validate="one_to_one",
                sort=False,
            ).drop(columns=["__merge_occurrence"])

            del left_temp
            del right_temp
            gc.collect()

        else:
            merged = left_df.merge(
                right_df,
                on=merge_columns,
                how="left",
                validate="one_to_one",
                sort=False,
            )

        print(f"[PROGRESS] Safe merge completed. Merged shape: {merged.shape}")
        return merged

    # ==================================================================================
    # Header and validation helpers
    # ==================================================================================

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV header columns without loading full file.

        Args:
            path: CSV path.

        Returns:
            List of column names.
        """
        print(f"[PROGRESS] Reading CSV header only from: {path}")
        columns = list(pd.read_csv(path, nrows=0).columns)
        print(f"[PROGRESS] Header column count: {len(columns)}")
        return columns

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read CSV header as an empty DataFrame.

        Args:
            path: CSV path.

        Returns:
            Empty DataFrame containing only columns.
        """
        print(f"[PROGRESS] Reading CSV header DataFrame only from: {path}")
        return pd.read_csv(path, nrows=0)

    def _validate_required_prediction_columns(
        self,
        columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate required columns exist in a CSV header.

        Args:
            columns: Available columns.
            required_columns: Required columns.
            label: Human-readable source label.
        """
        print(f"[PROGRESS] Validating required columns for {label}")

        missing = [column for column in required_columns if column not in columns]

        if missing:
            print(f"[ERROR] Missing columns for {label}: {missing}")
            raise KeyError(f"Missing columns for {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _verify_key_alignment(
        self,
        base_chunk: pd.DataFrame,
        other_chunk: pd.DataFrame,
        merge_columns: List[str],
        label: str,
    ) -> None:
        """
        Verify chunks are row-aligned by unit_id, cycle, and split.

        This avoids full-memory merges. If files are not aligned, the method
        stops instead of producing incorrect ensemble output.

        Args:
            base_chunk: Base chunk.
            other_chunk: Other chunk to compare.
            merge_columns: Key columns.
            label: Source label for error messages.
        """
        if len(base_chunk) != len(other_chunk):
            print(
                f"[ERROR] Chunk row count mismatch for {label}: "
                f"base={len(base_chunk)}, other={len(other_chunk)}"
            )
            raise ValueError(f"Chunk row count mismatch for {label}")

        base_keys = base_chunk[merge_columns].reset_index(drop=True)
        other_keys = other_chunk[merge_columns].reset_index(drop=True)

        if not base_keys.equals(other_keys):
            print(f"[ERROR] Row-key alignment failed for {label}")
            print("[ERROR] Prediction files are not aligned by unit_id/cycle/split")
            raise ValueError(
                f"Prediction files are not row-aligned for {label}. "
                "Regenerate RF, XGB, LightGBM, and MLP prediction CSVs using the same input order."
            )

    def _infer_target_sensors_from_rf_columns(self, rf_columns: List[str]) -> List[str]:
        """
        Infer raw target sensors from RF prediction column names.

        Args:
            rf_columns: RF prediction CSV columns.

        Returns:
            Raw sensor names.
        """
        print("[PROGRESS] Inferring ensemble target sensors from RF prediction columns")

        sensors = [
            column.replace("rf_predicted_", "")
            for column in rf_columns
            if column.startswith("rf_predicted_")
        ]

        if not sensors:
            print("[ERROR] No RF prediction columns found")
            raise ValueError("No RF prediction columns found.")

        engineered_tokens = [
            "rolling",
            "trend",
            "lag",
            "delta",
            "diff",
            "mean_",
            "std_",
            "var_",
        ]

        raw_sensors = [
            sensor
            for sensor in sensors
            if not any(token in sensor.lower() for token in engineered_tokens)
        ]

        if not raw_sensors:
            print("[ERROR] No raw RF prediction sensors found")
            raise ValueError("No raw RF prediction sensors found.")

        print(f"[PROGRESS] Ensemble raw target sensors count: {len(raw_sensors)}")
        print(f"[PROGRESS] Ensemble raw target sensors: {raw_sensors}")

        return raw_sensors

    def _infer_target_sensors_from_rf(self, rf_df: pd.DataFrame) -> List[str]:
        """
        Backward-compatible method using an already loaded RF DataFrame.

        Args:
            rf_df: RF predictions DataFrame.

        Returns:
            Raw sensor names.
        """
        return self._infer_target_sensors_from_rf_columns(list(rf_df.columns))

    # ==================================================================================
    # Weight calculation
    # ==================================================================================

    def compute_validation_weights(self) -> Dict[str, float]:
        """
        Compute ensemble weights from dev RMSE using raw X_s sensors only.

        Memory-safe behavior:
        - Does not load full scaled/RF/XGB/LGBM/MLP CSVs.
        - Reads aligned chunks.
        - Accumulates SSE for dev rows only.

        Returns:
            Dictionary of ensemble weights.
        """
        print("[PROGRESS] Entering EnsembleDigitalTwin.compute_validation_weights")

        try:
            stage_start = perf_counter()
            merge_columns = ["unit_id", "cycle", "split"]

            print("[PROGRESS] Reading headers for scaled and prediction CSVs")
            scaled_header_df = self._read_header_df(Config.SCALED_CSV)
            scaled_columns = list(scaled_header_df.columns)

            rf_columns = self._read_header_columns(Config.RF_PREDICTIONS_CSV)
            xgb_columns = self._read_header_columns(Config.XGB_PREDICTIONS_CSV)
            lgbm_columns = self._read_header_columns(Config.LGBM_PREDICTIONS_CSV)
            mlp_columns = self._read_header_columns(self.mlp_predictions_csv)

            print("[PROGRESS] Extracting raw X_s target sensors from scaled CSV header")
            target_sensors = get_raw_xs_columns(scaled_header_df)

            if not target_sensors:
                print("[ERROR] No raw X_s target columns found in scaled CSV")
                raise ValueError("No raw X_s target columns found in scaled CSV.")

            print(f"[PROGRESS] Raw X_s target sensor count: {len(target_sensors)}")
            print(f"[PROGRESS] Raw X_s target sensors: {target_sensors}")

            rf_prediction_columns = [
                f"rf_predicted_{sensor}" for sensor in target_sensors
            ]
            xgb_prediction_columns = [
                f"xgb_predicted_{sensor}" for sensor in target_sensors
            ]
            lgbm_prediction_columns = [
                f"lgbm_predicted_{sensor}" for sensor in target_sensors
            ]
            mlp_prediction_columns = [
                f"tf_predicted_{sensor}" for sensor in target_sensors
            ]

            self._validate_required_prediction_columns(
                columns=scaled_columns,
                required_columns=merge_columns + target_sensors,
                label="scaled CSV",
            )
            self._validate_required_prediction_columns(
                columns=rf_columns,
                required_columns=merge_columns + rf_prediction_columns,
                label="RF predictions CSV",
            )
            self._validate_required_prediction_columns(
                columns=xgb_columns,
                required_columns=merge_columns + xgb_prediction_columns,
                label="XGB predictions CSV",
            )
            self._validate_required_prediction_columns(
                columns=lgbm_columns,
                required_columns=merge_columns + lgbm_prediction_columns,
                label="LightGBM predictions CSV",
            )
            self._validate_required_prediction_columns(
                columns=mlp_columns,
                required_columns=merge_columns + mlp_prediction_columns,
                label="MLP predictions CSV",
            )

            scaled_usecols = merge_columns + target_sensors
            rf_usecols = merge_columns + rf_prediction_columns
            xgb_usecols = merge_columns + xgb_prediction_columns
            lgbm_usecols = merge_columns + lgbm_prediction_columns
            mlp_usecols = merge_columns + mlp_prediction_columns

            print("[PROGRESS] Starting chunked dev RMSE calculation")
            print(f"[PROGRESS] Chunk size: {self.chunk_size}")

            scaled_iter = pd.read_csv(
                Config.SCALED_CSV,
                usecols=scaled_usecols,
                chunksize=self.chunk_size,
            )
            rf_iter = pd.read_csv(
                Config.RF_PREDICTIONS_CSV,
                usecols=rf_usecols,
                chunksize=self.chunk_size,
            )
            xgb_iter = pd.read_csv(
                Config.XGB_PREDICTIONS_CSV,
                usecols=xgb_usecols,
                chunksize=self.chunk_size,
            )
            lgbm_iter = pd.read_csv(
                Config.LGBM_PREDICTIONS_CSV,
                usecols=lgbm_usecols,
                chunksize=self.chunk_size,
            )
            mlp_iter = pd.read_csv(
                self.mlp_predictions_csv,
                usecols=mlp_usecols,
                chunksize=self.chunk_size,
            )

            sse = {
                "rf": 0.0,
                "xgb": 0.0,
                "lgbm": 0.0,
                "mlp": 0.0,
            }

            value_count = {
                "rf": 0,
                "xgb": 0,
                "lgbm": 0,
                "mlp": 0,
            }

            total_rows_seen = 0
            total_dev_rows_seen = 0
            chunk_index = 0

            for scaled_chunk, rf_chunk, xgb_chunk, lgbm_chunk, mlp_chunk in zip(
                scaled_iter,
                rf_iter,
                xgb_iter,
                lgbm_iter,
                mlp_iter,
            ):
                chunk_index += 1
                chunk_rows = len(scaled_chunk)
                total_rows_seen += chunk_rows

                print("=" * 100)
                print(f"[PROGRESS] Processing validation-weight chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {chunk_rows}")
                print(f"[PROGRESS] Total rows seen so far: {total_rows_seen}")

                self._verify_key_alignment(
                    base_chunk=scaled_chunk,
                    other_chunk=rf_chunk,
                    merge_columns=merge_columns,
                    label="RF",
                )
                self._verify_key_alignment(
                    base_chunk=scaled_chunk,
                    other_chunk=xgb_chunk,
                    merge_columns=merge_columns,
                    label="XGB",
                )
                self._verify_key_alignment(
                    base_chunk=scaled_chunk,
                    other_chunk=lgbm_chunk,
                    merge_columns=merge_columns,
                    label="LightGBM",
                )
                self._verify_key_alignment(
                    base_chunk=scaled_chunk,
                    other_chunk=mlp_chunk,
                    merge_columns=merge_columns,
                    label="MLP",
                )

                dev_mask = scaled_chunk["split"] == Config.DEV_SPLIT_NAME
                dev_rows = int(dev_mask.sum())
                total_dev_rows_seen += dev_rows

                print(f"[PROGRESS] Dev rows in this chunk: {dev_rows}")
                print(f"[PROGRESS] Total dev rows seen so far: {total_dev_rows_seen}")

                if dev_rows == 0:
                    print("[PROGRESS] No dev rows in this chunk. Skipping RMSE accumulation.")
                    del scaled_chunk
                    del rf_chunk
                    del xgb_chunk
                    del lgbm_chunk
                    del mlp_chunk
                    gc.collect()
                    continue

                true_values = scaled_chunk.loc[dev_mask, target_sensors].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )

                model_chunks = {
                    "rf": (rf_chunk, rf_prediction_columns),
                    "xgb": (xgb_chunk, xgb_prediction_columns),
                    "lgbm": (lgbm_chunk, lgbm_prediction_columns),
                    "mlp": (mlp_chunk, mlp_prediction_columns),
                }

                for model_name, (prediction_chunk, prediction_columns) in model_chunks.items():
                    pred_values = prediction_chunk.loc[
                        dev_mask,
                        prediction_columns,
                    ].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )

                    diff = true_values - pred_values
                    sse[model_name] += float(np.square(diff, dtype=np.float64).sum())
                    value_count[model_name] += int(diff.size)

                    print(
                        f"[PROGRESS] {model_name} accumulated values: "
                        f"{value_count[model_name]}, SSE: {sse[model_name]:.6f}"
                    )

                    del pred_values
                    del diff

                del true_values
                del scaled_chunk
                del rf_chunk
                del xgb_chunk
                del lgbm_chunk
                del mlp_chunk
                gc.collect()

            if total_dev_rows_seen == 0:
                print("[ERROR] No dev rows found for ensemble weight calculation")
                raise ValueError("No dev rows found for ensemble weight calculation.")

            model_rmse: Dict[str, float] = {}

            for model_name in self.ensemble_members:
                if value_count[model_name] == 0:
                    print(f"[ERROR] No prediction values accumulated for {model_name}")
                    raise ValueError(f"No prediction values accumulated for {model_name}")

                mse = sse[model_name] / max(value_count[model_name], 1)
                rmse = float(np.sqrt(mse))
                model_rmse[model_name] = rmse

            inverse_rmse = {
                model_name: 1.0 / max(rmse, 1e-9)
                for model_name, rmse in model_rmse.items()
            }

            total_inverse = sum(inverse_rmse.values())

            weights = {
                model_name: value / total_inverse
                for model_name, value in inverse_rmse.items()
            }

            weight_payload = {
                "weights": weights,
                "dev_rmse": model_rmse,
                "target_type": "raw_X_s_only",
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "inference_only",
                "calculation_mode": "chunked_streaming",
                "chunk_size": self.chunk_size,
                "dev_rows_used": total_dev_rows_seen,
                "total_rows_scanned": total_rows_seen,
                "ensemble_members": [
                    "random_forest",
                    "xgboost",
                    "lightgbm",
                    "mlp_digital_twin",
                ],
                "prediction_columns": {
                    "rf": rf_prediction_columns,
                    "xgb": xgb_prediction_columns,
                    "lgbm": lgbm_prediction_columns,
                    "mlp": mlp_prediction_columns,
                },
            }

            print("[PROGRESS] Writing ensemble weights JSON")
            atomic_write_json(weight_payload, Config.ENSEMBLE_WEIGHTS_PATH)

            print("[PROGRESS] Writing ensemble metadata JSON")
            atomic_write_json(weight_payload, self.ensemble_metadata_path)

            duration = perf_counter() - stage_start

            print(f"[PROGRESS] Ensemble dev RMSE: {model_rmse}")
            print(f"[PROGRESS] Ensemble weights: {weights}")
            print(f"[PROGRESS] Dev rows used for weights: {total_dev_rows_seen}")
            print(f"[PROGRESS] Weight calculation duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Weight calculation duration minutes: {duration / 60.0:.2f}")

            return weights

        except Exception as exc:
            print(f"[ERROR] Ensemble weight calculation failed: {exc}")
            logger.exception("Ensemble weight calculation failed.")
            raise RuntimeError("Ensemble weight calculation failed.") from exc

    def load_or_create_weights(self) -> Dict[str, float]:
        """
        Load existing ensemble weights if valid; otherwise calculate them.

        Returns:
            Dictionary with rf, xgb, lgbm, and mlp weights.
        """
        print("[PROGRESS] Entering EnsembleDigitalTwin.load_or_create_weights")

        try:
            if Config.ENSEMBLE_WEIGHTS_PATH.exists():
                print(
                    f"[PROGRESS] Existing ensemble weights file found: "
                    f"{Config.ENSEMBLE_WEIGHTS_PATH}"
                )

                data = read_json_required(Config.ENSEMBLE_WEIGHTS_PATH)
                weights = data.get("weights", {})

                if all(key in weights for key in self.ensemble_members):
                    loaded_weights = {
                        "rf": float(weights["rf"]),
                        "xgb": float(weights["xgb"]),
                        "lgbm": float(weights["lgbm"]),
                        "mlp": float(weights["mlp"]),
                    }

                    print(f"[PROGRESS] Loaded 4-model ensemble weights: {loaded_weights}")
                    return loaded_weights

                print("[WARNING] Existing weights file is missing 4-model keys. Recomputing weights.")

            else:
                print("[PROGRESS] No existing ensemble weights file found. Computing new weights.")

            return self.compute_validation_weights()

        except Exception as exc:
            print(f"[ERROR] Failed to load or create ensemble weights: {exc}")
            logger.exception("Failed to load or create ensemble weights.")
            raise RuntimeError("Failed to load or create ensemble weights.") from exc

    # ==================================================================================
    # Ensemble prediction generation
    # ==================================================================================

    def ensemble_predictions(self) -> int:
        """
        Generate 4-model ensemble predictions.

        Memory-safe behavior:
        - Does not load all RF/XGB/LGBM/MLP predictions into RAM.
        - Reads prediction CSVs in aligned chunks.
        - Writes ensemble output batch by batch to temporary CSV.
        - Replaces final CSV only when all chunks complete.

        Returns:
            Number of ensemble prediction rows written.
        """
        print("[PROGRESS] Entering EnsembleDigitalTwin.ensemble_predictions")

        try:
            stage_start = perf_counter()
            merge_columns = ["unit_id", "cycle", "split"]

            print("[PROGRESS] Reading prediction CSV headers")
            rf_columns = self._read_header_columns(Config.RF_PREDICTIONS_CSV)
            xgb_columns = self._read_header_columns(Config.XGB_PREDICTIONS_CSV)
            lgbm_columns = self._read_header_columns(Config.LGBM_PREDICTIONS_CSV)
            mlp_columns = self._read_header_columns(self.mlp_predictions_csv)

            sensors = self._infer_target_sensors_from_rf_columns(rf_columns)

            rf_prediction_columns = [
                f"rf_predicted_{sensor}" for sensor in sensors
            ]
            xgb_prediction_columns = [
                f"xgb_predicted_{sensor}" for sensor in sensors
            ]
            lgbm_prediction_columns = [
                f"lgbm_predicted_{sensor}" for sensor in sensors
            ]
            mlp_prediction_columns = [
                f"tf_predicted_{sensor}" for sensor in sensors
            ]

            self._validate_required_prediction_columns(
                columns=rf_columns,
                required_columns=merge_columns + rf_prediction_columns,
                label="RF predictions CSV",
            )
            self._validate_required_prediction_columns(
                columns=xgb_columns,
                required_columns=merge_columns + xgb_prediction_columns,
                label="XGB predictions CSV",
            )
            self._validate_required_prediction_columns(
                columns=lgbm_columns,
                required_columns=merge_columns + lgbm_prediction_columns,
                label="LightGBM predictions CSV",
            )
            self._validate_required_prediction_columns(
                columns=mlp_columns,
                required_columns=merge_columns + mlp_prediction_columns,
                label="MLP predictions CSV",
            )

            print("[PROGRESS] Loading or creating 4-model ensemble weights")
            weights = self.load_or_create_weights()
            print(f"[PROGRESS] Ensemble weights used for prediction: {weights}")

            rf_usecols = merge_columns + rf_prediction_columns
            xgb_usecols = merge_columns + xgb_prediction_columns
            lgbm_usecols = merge_columns + lgbm_prediction_columns
            mlp_usecols = merge_columns + mlp_prediction_columns

            print("[PROGRESS] Creating chunk iterators for RF, XGB, LightGBM, and MLP predictions")
            print(f"[PROGRESS] Chunk size: {self.chunk_size}")

            rf_iter = pd.read_csv(
                Config.RF_PREDICTIONS_CSV,
                usecols=rf_usecols,
                chunksize=self.chunk_size,
            )
            xgb_iter = pd.read_csv(
                Config.XGB_PREDICTIONS_CSV,
                usecols=xgb_usecols,
                chunksize=self.chunk_size,
            )
            lgbm_iter = pd.read_csv(
                Config.LGBM_PREDICTIONS_CSV,
                usecols=lgbm_usecols,
                chunksize=self.chunk_size,
            )
            mlp_iter = pd.read_csv(
                self.mlp_predictions_csv,
                usecols=mlp_usecols,
                chunksize=self.chunk_size,
            )

            output_path = Config.ENSEMBLE_PREDICTIONS_CSV
            temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

            print(f"[PROGRESS] Final ensemble predictions CSV path: {output_path}")
            print(f"[PROGRESS] Temporary ensemble predictions CSV path: {temp_output_path}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary ensemble prediction file")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            print("[PROGRESS] Starting memory-safe streaming 4-model ensemble prediction")

            for rf_chunk, xgb_chunk, lgbm_chunk, mlp_chunk in zip(
                rf_iter,
                xgb_iter,
                lgbm_iter,
                mlp_iter,
            ):
                chunk_index += 1

                print("=" * 100)
                print(f"[PROGRESS] Processing ensemble prediction chunk #{chunk_index}")
                print(f"[PROGRESS] RF chunk shape: {rf_chunk.shape}")
                print(f"[PROGRESS] XGB chunk shape: {xgb_chunk.shape}")
                print(f"[PROGRESS] LightGBM chunk shape: {lgbm_chunk.shape}")
                print(f"[PROGRESS] MLP chunk shape: {mlp_chunk.shape}")

                self._verify_key_alignment(
                    base_chunk=rf_chunk,
                    other_chunk=xgb_chunk,
                    merge_columns=merge_columns,
                    label="XGB",
                )
                self._verify_key_alignment(
                    base_chunk=rf_chunk,
                    other_chunk=lgbm_chunk,
                    merge_columns=merge_columns,
                    label="LightGBM",
                )
                self._verify_key_alignment(
                    base_chunk=rf_chunk,
                    other_chunk=mlp_chunk,
                    merge_columns=merge_columns,
                    label="MLP",
                )

                result_chunk = rf_chunk[merge_columns].copy()

                for sensor in sensors:
                    rf_col = f"rf_predicted_{sensor}"
                    xgb_col = f"xgb_predicted_{sensor}"
                    lgbm_col = f"lgbm_predicted_{sensor}"
                    mlp_col = f"tf_predicted_{sensor}"

                    ensemble_values = (
                        weights["rf"] * rf_chunk[rf_col].to_numpy(dtype=np.float32, copy=False)
                        + weights["xgb"] * xgb_chunk[xgb_col].to_numpy(dtype=np.float32, copy=False)
                        + weights["lgbm"] * lgbm_chunk[lgbm_col].to_numpy(dtype=np.float32, copy=False)
                        + weights["mlp"] * mlp_chunk[mlp_col].to_numpy(dtype=np.float32, copy=False)
                    ).astype(np.float32, copy=False)

                    result_chunk[f"ensemble_predicted_{sensor}"] = ensemble_values

                    del ensemble_values

                print(f"[PROGRESS] Ensemble result chunk shape: {result_chunk.shape}")

                print("[PROGRESS] Writing ensemble chunk to temporary CSV")
                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                print(f"[PROGRESS] Total ensemble rows written so far: {total_rows_written}")

                del rf_chunk
                del xgb_chunk
                del lgbm_chunk
                del mlp_chunk
                del result_chunk
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All ensemble prediction chunks completed")

            print("[PROGRESS] Replacing final ensemble predictions CSV with completed temporary CSV")
            os.replace(temp_output_path, output_path)
            print("[PROGRESS] Ensemble predictions CSV written successfully")

            duration = perf_counter() - stage_start

            print(f"[PROGRESS] Ensemble prediction records written: {total_rows_written}")
            print(f"[PROGRESS] Ensemble prediction duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Ensemble prediction duration minutes: {duration / 60.0:.2f}")

            logger.info("4-model ensemble prediction completed. rows=%s", total_rows_written)

            return total_rows_written

        except Exception as exc:
            print(f"[ERROR] Ensemble prediction failed: {exc}")
            logger.exception("Ensemble prediction failed.")
            raise RuntimeError("Ensemble prediction failed.") from exc

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run ensemble weight calculation and ensemble prediction generation.

        Returns:
            Stage response dictionary.
        """
        print("[PROGRESS] Entering EnsembleDigitalTwin.run")

        try:
            print("[PROGRESS] Starting 4-model ensemble validation weight calculation")
            self.compute_validation_weights()
            print("[PROGRESS] Ensemble validation weight calculation completed")

            print("[PROGRESS] Starting memory-safe 4-model ensemble prediction generation")
            records_count = self.ensemble_predictions()
            print("[PROGRESS] Ensemble prediction generation completed")

            response = {
                "status": "success",
                "message": (
                    "4-model Ensemble Digital Twin predictions generated for raw X_s sensors only."
                ),
                "output_file": str(Config.ENSEMBLE_PREDICTIONS_CSV),
                "records_count": records_count,
                "target_type": "raw_X_s_only",
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "inference_only",
                "ensemble_members": [
                    "random_forest",
                    "xgboost",
                    "lightgbm",
                    "mlp_digital_twin",
                ],
            }

            print(f"[PROGRESS] Ensemble digital twin response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Ensemble digital twin stage failed: {exc}")
            logger.exception("Ensemble digital twin stage failed.")
            raise RuntimeError("Ensemble digital twin stage failed.") from exc


def run_ensemble_twin() -> Dict[str, object]:
    """
    Execute 4-model ensemble digital twin stage.

    Returns:
        Stage response dictionary.
    """
    service = EnsembleDigitalTwin()
    return service.run()


if __name__ == "__main__":
    print("[PROGRESS] ensemble_twin.py execution started from __main__")
    result = run_ensemble_twin()
    print("[PROGRESS] ensemble_twin.py execution finished successfully")
    print(result)