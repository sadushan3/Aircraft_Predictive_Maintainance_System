"""
Digital Twin Comparator for CA-EDT-AHMA.

Evaluates RF, XGBoost, LightGBM, MLP Digital Twin, and Ensemble predictions
against raw X_s measured sensors only.

Memory-safe version:
- Does not load all prediction CSVs into RAM.
- Does not perform giant full-DataFrame merges.
- Reads aligned CSV chunks.
- Accumulates MAE, RMSE, and R2 statistics chunk by chunk.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
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
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_raw_xs_columns


logger = get_logger(__name__)


class TwinComparator:
    """
    Compare digital twin models against actual raw measured sensor values.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize comparator.

        Args:
            chunk_size: Number of rows per CSV chunk.
        """
        Config.create_directories()

        self.chunk_size = int(getattr(Config, "TWIN_COMPARATOR_CHUNK_SIZE", chunk_size))

        if self.chunk_size <= 0:
            raise ValueError("Twin comparator chunk size must be greater than 0.")

        self.mlp_predictions_csv: Path = getattr(
            Config,
            "MLP_TWIN_PREDICTIONS_CSV",
            getattr(Config, "TF_PREDICTIONS_CSV", Config.OUTPUT_DIR / "mlp_twin_predictions.csv"),
        )

        print("[PROGRESS] TwinComparator initialized")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] MLP predictions CSV: {self.mlp_predictions_csv}")

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read CSV header only as an empty DataFrame.
        """
        print(f"[PROGRESS] Reading header only from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV header column names only.
        """
        print(f"[PROGRESS] Reading header columns only from: {path}")
        columns = list(pd.read_csv(path, nrows=0).columns)
        print(f"[PROGRESS] Column count for {path}: {len(columns)}")
        return columns

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate required columns.
        """
        print(f"[PROGRESS] Validating required columns for {label}")
        missing = [column for column in required_columns if column not in available_columns]

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
        Verify chunk row alignment by unit_id, cycle, split.
        """
        if len(base_chunk) != len(other_chunk):
            print(
                f"[ERROR] Row count mismatch for {label}: "
                f"base={len(base_chunk)}, other={len(other_chunk)}"
            )
            raise ValueError(f"Row count mismatch for {label}")

        base_keys = base_chunk[merge_columns].reset_index(drop=True)
        other_keys = other_chunk[merge_columns].reset_index(drop=True)

        if not base_keys.equals(other_keys):
            print(f"[ERROR] Row-key alignment failed for {label}")
            print("[ERROR] Prediction files are not aligned by unit_id/cycle/split")
            raise ValueError(
                f"Prediction files are not row-aligned for {label}. "
                "Regenerate all digital twin predictions using the same input order."
            )

    def evaluate(self) -> pd.DataFrame:
        """
        Memory-safe evaluation.

        Metrics:
        - MAE  = sum(abs(error)) / n
        - RMSE = sqrt(sum(error^2) / n)
        - R2   = 1 - SSE / SST
        """
        print("[PROGRESS] Entering TwinComparator.evaluate")

        try:
            stage_start = perf_counter()

            merge_columns = ["unit_id", "cycle", "split"]
            splits = [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]

            print("[PROGRESS] Reading CSV headers")
            scaled_header_df = self._read_header_df(Config.SCALED_CSV)
            scaled_columns = list(scaled_header_df.columns)

            rf_columns = self._read_header_columns(Config.RF_PREDICTIONS_CSV)
            xgb_columns = self._read_header_columns(Config.XGB_PREDICTIONS_CSV)
            lgbm_columns = self._read_header_columns(Config.LGBM_PREDICTIONS_CSV)
            mlp_columns = self._read_header_columns(self.mlp_predictions_csv)
            ensemble_columns = self._read_header_columns(Config.ENSEMBLE_PREDICTIONS_CSV)

            print("[PROGRESS] Extracting raw X_s sensors from scaled CSV header")
            sensors = get_raw_xs_columns(scaled_header_df)

            if not sensors:
                print("[ERROR] No raw X_s sensors found")
                raise ValueError("No raw X_s sensors found.")

            print(f"[PROGRESS] Raw X_s sensors count: {len(sensors)}")
            print(f"[PROGRESS] Raw X_s sensors: {sensors}")

            model_prefixes = {
                "random_forest": "rf_predicted_",
                "xgboost": "xgb_predicted_",
                "lightgbm": "lgbm_predicted_",
                "mlp_digital_twin": "tf_predicted_",
                "ensemble": "ensemble_predicted_",
            }

            prediction_file_columns = {
                "random_forest": rf_columns,
                "xgboost": xgb_columns,
                "lightgbm": lgbm_columns,
                "mlp_digital_twin": mlp_columns,
                "ensemble": ensemble_columns,
            }

            prediction_paths = {
                "random_forest": Config.RF_PREDICTIONS_CSV,
                "xgboost": Config.XGB_PREDICTIONS_CSV,
                "lightgbm": Config.LGBM_PREDICTIONS_CSV,
                "mlp_digital_twin": self.mlp_predictions_csv,
                "ensemble": Config.ENSEMBLE_PREDICTIONS_CSV,
            }

            print("[PROGRESS] Validating scaled CSV columns")
            self._validate_columns(
                available_columns=scaled_columns,
                required_columns=merge_columns + sensors,
                label="scaled CSV",
            )

            for model_name, prefix in model_prefixes.items():
                required_prediction_columns = merge_columns + [
                    f"{prefix}{sensor}" for sensor in sensors
                ]

                self._validate_columns(
                    available_columns=prediction_file_columns[model_name],
                    required_columns=required_prediction_columns,
                    label=f"{model_name} predictions CSV",
                )

            scaled_usecols = merge_columns + sensors

            prediction_usecols = {
                model_name: merge_columns + [f"{prefix}{sensor}" for sensor in sensors]
                for model_name, prefix in model_prefixes.items()
            }

            print("[PROGRESS] Initializing metric accumulators")

            accumulators: Dict[Tuple[str, str, str], Dict[str, float]] = {}

            for split in splits:
                for sensor in sensors:
                    for model_name in model_prefixes.keys():
                        accumulators[(split, sensor, model_name)] = {
                            "count": 0.0,
                            "sum_abs_error": 0.0,
                            "sum_squared_error": 0.0,
                            "sum_y": 0.0,
                            "sum_y_squared": 0.0,
                        }

            print("[PROGRESS] Creating chunk iterators")
            print(f"[PROGRESS] Chunk size: {self.chunk_size}")

            scaled_iter = pd.read_csv(
                Config.SCALED_CSV,
                usecols=scaled_usecols,
                chunksize=self.chunk_size,
            )

            prediction_iters = {
                model_name: pd.read_csv(
                    prediction_paths[model_name],
                    usecols=prediction_usecols[model_name],
                    chunksize=self.chunk_size,
                )
                for model_name in model_prefixes.keys()
            }

            total_rows_seen = 0
            chunk_index = 0

            for scaled_chunk in scaled_iter:
                chunk_index += 1
                chunk_rows = len(scaled_chunk)
                total_rows_seen += chunk_rows

                print("=" * 100)
                print(f"[PROGRESS] Processing comparator chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {chunk_rows}")
                print(f"[PROGRESS] Total rows seen so far: {total_rows_seen}")

                prediction_chunks = {
                    model_name: next(prediction_iters[model_name])
                    for model_name in model_prefixes.keys()
                }

                for model_name, prediction_chunk in prediction_chunks.items():
                    self._verify_key_alignment(
                        base_chunk=scaled_chunk,
                        other_chunk=prediction_chunk,
                        merge_columns=merge_columns,
                        label=model_name,
                    )

                for split in splits:
                    split_mask = scaled_chunk["split"] == split
                    split_rows = int(split_mask.sum())

                    print(f"[PROGRESS] Split={split}, rows in chunk={split_rows}")

                    if split_rows == 0:
                        continue

                    for sensor in sensors:
                        y_true = scaled_chunk.loc[split_mask, sensor].to_numpy(
                            dtype=np.float32,
                            copy=False,
                        )

                        for model_name, prefix in model_prefixes.items():
                            prediction_column = f"{prefix}{sensor}"

                            y_pred = prediction_chunks[model_name].loc[
                                split_mask,
                                prediction_column,
                            ].to_numpy(dtype=np.float32, copy=False)

                            error = y_true - y_pred

                            key = (split, sensor, model_name)

                            accumulators[key]["count"] += float(error.size)
                            accumulators[key]["sum_abs_error"] += float(
                                np.abs(error).sum(dtype=np.float64)
                            )
                            accumulators[key]["sum_squared_error"] += float(
                                np.square(error, dtype=np.float64).sum()
                            )
                            accumulators[key]["sum_y"] += float(
                                y_true.sum(dtype=np.float64)
                            )
                            accumulators[key]["sum_y_squared"] += float(
                                np.square(y_true, dtype=np.float64).sum()
                            )

                            del y_pred
                            del error

                        del y_true

                del scaled_chunk
                for prediction_chunk in prediction_chunks.values():
                    del prediction_chunk

                del prediction_chunks
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All chunks processed. Building metrics DataFrame.")

            records: List[Dict[str, object]] = []

            for split in splits:
                for sensor in sensors:
                    for model_name in model_prefixes.keys():
                        key = (split, sensor, model_name)
                        stats = accumulators[key]
                        count = stats["count"]

                        if count <= 0:
                            print(
                                f"[WARNING] No rows accumulated for "
                                f"split={split}, sensor={sensor}, model={model_name}"
                            )
                            continue

                        mae = stats["sum_abs_error"] / count
                        rmse = float(np.sqrt(stats["sum_squared_error"] / count))

                        sst = stats["sum_y_squared"] - (
                            (stats["sum_y"] ** 2) / max(count, 1.0)
                        )

                        if abs(sst) < 1e-12:
                            r2 = 0.0
                        else:
                            r2 = 1.0 - (stats["sum_squared_error"] / sst)

                        records.append(
                            {
                                "split": split,
                                "sensor": sensor,
                                "model": model_name,
                                "mae": float(mae),
                                "rmse": float(rmse),
                                "r2": float(r2),
                                "r2_percent": float(r2 * 100.0),
                                "count": int(count),
                                "target_type": "raw_X_s_only",
                            }
                        )

            metrics_df = pd.DataFrame(records)

            # Add overall ALL_RAW_XS rows.
            overall_records: List[Dict[str, object]] = []

            for split in splits:
                for model_name in model_prefixes.keys():
                    model_rows = metrics_df[
                        (metrics_df["split"] == split)
                        & (metrics_df["model"] == model_name)
                    ]

                    if model_rows.empty:
                        continue

                    overall_records.append(
                        {
                            "split": split,
                            "sensor": "ALL_RAW_XS",
                            "model": model_name,
                            "mae": float(model_rows["mae"].mean()),
                            "rmse": float(model_rows["rmse"].mean()),
                            "r2": float(model_rows["r2"].mean()),
                            "r2_percent": float(model_rows["r2_percent"].mean()),
                            "count": int(model_rows["count"].sum()),
                            "target_type": "raw_X_s_only",
                        }
                    )

            if overall_records:
                metrics_df = pd.concat(
                    [metrics_df, pd.DataFrame(overall_records)],
                    ignore_index=True,
                )

            duration = perf_counter() - stage_start

            print(f"[PROGRESS] Digital twin comparison completed. rows={len(metrics_df)}")
            print(f"[PROGRESS] Total rows scanned: {total_rows_seen}")
            print(f"[PROGRESS] Comparator duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Comparator duration minutes: {duration / 60.0:.2f}")

            print("[PROGRESS] Overall digital twin comparison rows:")
            print(metrics_df[metrics_df["sensor"] == "ALL_RAW_XS"].to_string(index=False))

            logger.info("Digital twin comparison completed. rows=%s", len(metrics_df))
            return metrics_df

        except Exception as exc:
            print(f"[ERROR] Digital twin comparison failed: {exc}")
            logger.exception("Digital twin comparison failed.")
            raise RuntimeError("Digital twin comparison failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run digital twin comparison.
        """
        print("[PROGRESS] Entering TwinComparator.run")

        try:
            metrics_df = self.evaluate()

            output_path: Path = getattr(
                Config,
                "ENSEMBLE_METRICS_CSV",
                Config.METRIC_DIR / "digital_twin_comparison.csv",
            )

            print(f"[PROGRESS] Writing digital twin comparison metrics to: {output_path}")
            atomic_write_csv(metrics_df, output_path)
            print("[PROGRESS] Digital twin comparison metrics written successfully")

            response = {
                "status": "success",
                "message": (
                    "Digital twin comparison completed for RF, XGBoost, LightGBM, "
                    "MLP Digital Twin, and Ensemble using raw X_s sensors only."
                ),
                "output_file": str(output_path),
                "records_count": len(metrics_df),
            }

            print(f"[PROGRESS] Twin comparator response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Twin comparator stage failed: {exc}")
            logger.exception("Twin comparator stage failed.")
            raise RuntimeError("Twin comparator stage failed.") from exc


def run_twin_comparison() -> Dict[str, object]:
    """
    Execute twin comparison.
    """
    service = TwinComparator()
    return service.run()


if __name__ == "__main__":
    print("[PROGRESS] twin_comparator.py execution started")
    result = run_twin_comparison()
    print("[PROGRESS] twin_comparator.py execution finished successfully")
    print(result)