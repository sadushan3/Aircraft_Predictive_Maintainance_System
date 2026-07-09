"""
XGBoost Digital Twin Regressor for CA-EDT-AHMA.

CA-EDT-AHMA:
Context-Aware Ensemble Digital Twin for Explainable Health Monitoring
and Anomaly Reasoning.

Model:
- XGBoost Regressor

Library:
- XGBoost

Research-correct data usage:
- Input  = W + X_v + gmm_context_id
- Target = raw measured X_s sensors only
- Train  = dev split only
- Test   = prediction/evaluation only
- Ignore = Y_dev/Y_test
- Ignore = T_dev/T_test as model input
- Do not use X_s columns as input features

Memory-safety:
- Does not load full scaled_features.csv into one pandas DataFrame.
- Reads scaled_features.csv and context_clusters.csv in chunks.
- Builds disk-backed NumPy memmap arrays for full dev training data.
- Trains XGBoost MultiOutputRegressor on full dev memmaps.
- Predicts all rows chunk-by-chunk directly to a temporary CSV.
- Replaces final prediction CSV only after successful completion.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
import gc
import json
import os
import shutil
import sys
import uuid

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    if BACKEND_ROOT not in sys.path:
        sys.path.append(BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_save_joblib,
    load_joblib_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import (
    get_raw_xs_columns,
    get_w_columns,
    get_xv_columns,
)


logger = get_logger(__name__)


class XGBoostTwin:
    """
    Full-dev memory-safe XGBoost Digital Twin Regressor.
    """

    def __init__(
        self,
        train_chunk_size: int = 50_000,
        prediction_chunk_size: int = 50_000,
    ) -> None:
        """
        Initialize XGBoost digital twin.

        Args:
            train_chunk_size: Number of rows read per training/memmap chunk.
            prediction_chunk_size: Number of rows read per prediction chunk.
        """
        Config.create_directories()

        self.train_chunk_size = int(
            getattr(Config, "XGB_TRAIN_CHUNK_SIZE", train_chunk_size)
        )
        self.prediction_chunk_size = int(
            getattr(Config, "XGB_PREDICTION_BATCH_SIZE", prediction_chunk_size)
        )

        self.model_path: Path = getattr(
            Config,
            "XGB_MODEL_PATH",
            Config.DIGITAL_TWIN_MODEL_DIR / "xgboost_twin.pkl",
        )
        self.predictions_csv: Path = getattr(
            Config,
            "XGB_PREDICTIONS_CSV",
            Config.OUTPUT_DIR / "xgb_predictions.csv",
        )
        self.metadata_path: Path = getattr(
            Config,
            "XGB_MODEL_METADATA_PATH",
            Config.DIGITAL_TWIN_MODEL_DIR / "xgboost_twin_metadata.json",
        )

        self.memmap_dir: Path = getattr(
            Config,
            "XGB_TRAIN_MEMMAP_DIR",
            Config.DIGITAL_TWIN_MODEL_DIR / "xgb_full_dev_memmap",
        )

        self.rebuild_memmap = bool(getattr(Config, "XGB_REBUILD_MEMMAP", True))
        self.cleanup_memmap_after_training = bool(
            getattr(Config, "XGB_CLEANUP_MEMMAP_AFTER_TRAINING", False)
        )

        self.xgb_train_n_jobs = int(getattr(Config, "XGB_TRAIN_N_JOBS", 2))
        self.run_id = uuid.uuid4().hex[:12]

        self._validate_runtime_config()

        print("[PROGRESS] XGBoostTwin initialized")
        print(f"[PROGRESS] Run ID: {self.run_id}")
        print(f"[PROGRESS] Train chunk size: {self.train_chunk_size}")
        print(f"[PROGRESS] Prediction chunk size: {self.prediction_chunk_size}")
        print(f"[PROGRESS] Model path: {self.model_path}")
        print(f"[PROGRESS] Metadata path: {self.metadata_path}")
        print(f"[PROGRESS] Predictions CSV: {self.predictions_csv}")
        print(f"[PROGRESS] Memmap directory: {self.memmap_dir}")
        print(f"[PROGRESS] XGBoost training n_jobs: {self.xgb_train_n_jobs}")

    def _validate_runtime_config(self) -> None:
        """
        Validate runtime settings.
        """
        if self.train_chunk_size <= 0:
            raise ValueError("XGB train_chunk_size must be positive.")

        if self.prediction_chunk_size <= 0:
            raise ValueError("XGB prediction_chunk_size must be positive.")

        if self.xgb_train_n_jobs == 0:
            raise ValueError("XGB_TRAIN_N_JOBS cannot be 0.")

    def _atomic_write_json(self, payload: Dict[str, object], output_path: Path) -> None:
        """
        Atomically write JSON metadata.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + f".{self.run_id}.tmp")

        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        os.replace(temp_path, output_path)

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read only CSV header as empty DataFrame.
        """
        print(f"[PROGRESS] Reading CSV header from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read only CSV column names.
        """
        print(f"[PROGRESS] Reading CSV columns from: {path}")
        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate required columns.
        """
        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            raise KeyError(f"Missing required columns for {label}: {missing}")

    def _verify_key_alignment(
        self,
        scaled_chunk: pd.DataFrame,
        context_chunk: pd.DataFrame,
        merge_columns: List[str],
    ) -> None:
        """
        Verify scaled and context chunks are row-aligned.
        """
        if len(scaled_chunk) != len(context_chunk):
            raise ValueError(
                "Scaled/context chunk row count mismatch: "
                f"scaled={len(scaled_chunk)}, context={len(context_chunk)}"
            )

        scaled_keys = scaled_chunk[merge_columns].reset_index(drop=True)
        context_keys = context_chunk[merge_columns].reset_index(drop=True)

        if not scaled_keys.equals(context_keys):
            raise ValueError(
                "scaled_features.csv and context_clusters.csv are not row-aligned. "
                "Regenerate context_clusters.csv using the same scaled_features.csv order."
            )

    def prepare_columns(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Prepare merge, feature, and target columns.

        Returns:
            merge_columns, feature_columns, target_columns
        """
        print("[PROGRESS] Preparing XGBoost columns")

        merge_columns = ["unit_id", "cycle", "split"]

        scaled_header_df = self._read_header_df(Config.SCALED_CSV)
        scaled_columns = list(scaled_header_df.columns)
        context_columns = self._read_header_columns(Config.CONTEXT_CSV)

        self._validate_columns(
            available_columns=scaled_columns,
            required_columns=merge_columns,
            label="scaled_features.csv",
        )
        self._validate_columns(
            available_columns=context_columns,
            required_columns=merge_columns + ["gmm_context_id"],
            label="context_clusters.csv",
        )

        w_columns = get_w_columns(scaled_header_df)
        xv_columns = get_xv_columns(scaled_header_df)
        target_columns = get_raw_xs_columns(scaled_header_df)

        if not w_columns:
            raise ValueError("No W operating-condition columns found.")

        if not xv_columns:
            raise ValueError("No X_v virtual sensor columns found.")

        if not target_columns:
            raise ValueError("No raw X_s target columns found.")

        feature_columns = w_columns + xv_columns + ["gmm_context_id"]

        leakage_features = [
            column
            for column in feature_columns
            if column.startswith(("Xs_", "X_s_", "Y_", "T_"))
        ]

        if leakage_features:
            raise ValueError(f"Leakage-risk XGBoost input columns found: {leakage_features}")

        engineered_target_tokens = (
            "rolling",
            "trend",
            "lag",
            "delta",
            "diff",
            "mean",
            "std",
            "var",
            "min",
            "max",
        )

        engineered_targets = [
            column
            for column in target_columns
            if any(token in column.lower() for token in engineered_target_tokens)
        ]

        if engineered_targets:
            raise ValueError(
                "Engineered target columns found. XGBoost target must be raw X_s only: "
                f"{engineered_targets}"
            )

        print(f"[PROGRESS] XGBoost feature columns count: {len(feature_columns)}")
        print(f"[PROGRESS] XGBoost target columns count: {len(target_columns)}")
        print(f"[PROGRESS] XGBoost feature columns: {feature_columns}")
        print(f"[PROGRESS] XGBoost target columns: {target_columns}")

        return merge_columns, feature_columns, target_columns

    def count_rows(self, merge_columns: List[str]) -> Dict[str, int]:
        """
        Count total/dev/test rows.
        """
        print("[PROGRESS] Counting rows for XGBoost training")

        counts = {
            "total": 0,
            "dev": 0,
            "test": 0,
        }

        chunk_index = 0

        for chunk in pd.read_csv(
            Config.SCALED_CSV,
            usecols=merge_columns,
            chunksize=self.train_chunk_size,
            low_memory=True,
        ):
            chunk_index += 1
            counts["total"] += len(chunk)

            dev_rows = int((chunk["split"] == Config.DEV_SPLIT_NAME).sum())
            test_rows = int((chunk["split"] == Config.TEST_SPLIT_NAME).sum())

            counts["dev"] += dev_rows
            counts["test"] += test_rows

            print(
                f"[PROGRESS] Count chunk {chunk_index}: "
                f"total={counts['total']} dev={counts['dev']} test={counts['test']}"
            )

            del chunk
            gc.collect()

        if counts["dev"] <= 0:
            raise ValueError("No dev rows found for XGBoost training.")

        print(f"[PROGRESS] Final XGBoost row counts: {counts}")

        return counts

    def _get_memmap_paths(self) -> Dict[str, Path]:
        """
        Get memmap file paths.
        """
        return {
            "x": self.memmap_dir / "X_dev_full.float32.memmap",
            "y": self.memmap_dir / "y_dev_full.float32.memmap",
            "metadata": self.memmap_dir / "memmap_metadata.json",
        }

    def build_full_dev_memmaps(
        self,
        merge_columns: List[str],
        feature_columns: List[str],
        target_columns: List[str],
        expected_dev_rows: int,
    ) -> Tuple[np.memmap, np.memmap]:
        """
        Build full dev X/y memmaps from chunked CSVs.
        """
        print("[PROGRESS] Building full-dev XGBoost memmaps")

        memmap_paths = self._get_memmap_paths()

        if self.rebuild_memmap and self.memmap_dir.exists():
            print("[PROGRESS] Removing old XGBoost memmap cache directory")
            shutil.rmtree(self.memmap_dir)

        self.memmap_dir.mkdir(parents=True, exist_ok=True)

        n_features = len(feature_columns)
        n_targets = len(target_columns)

        x_memmap = np.memmap(
            memmap_paths["x"],
            dtype=np.float32,
            mode="w+",
            shape=(expected_dev_rows, n_features),
        )

        y_memmap = np.memmap(
            memmap_paths["y"],
            dtype=np.float32,
            mode="w+",
            shape=(expected_dev_rows, n_targets),
        )

        scaled_usecols = list(
            dict.fromkeys(
                merge_columns
                + [column for column in feature_columns if column != "gmm_context_id"]
                + target_columns
            )
        )
        context_usecols = merge_columns + ["gmm_context_id"]

        scaled_iter = pd.read_csv(
            Config.SCALED_CSV,
            usecols=scaled_usecols,
            chunksize=self.train_chunk_size,
            low_memory=True,
        )
        context_iter = pd.read_csv(
            Config.CONTEXT_CSV,
            usecols=context_usecols,
            chunksize=self.train_chunk_size,
            low_memory=True,
        )

        write_index = 0
        total_rows_seen = 0
        chunk_index = 0
        started = perf_counter()

        for scaled_chunk, context_chunk in zip(scaled_iter, context_iter):
            chunk_index += 1
            total_rows_seen += len(scaled_chunk)

            print("=" * 100)
            print(f"[PROGRESS] XGBoost memmap chunk #{chunk_index}")
            print(f"[PROGRESS] Chunk rows: {len(scaled_chunk)}")
            print(f"[PROGRESS] Total rows scanned: {total_rows_seen}")

            self._verify_key_alignment(
                scaled_chunk=scaled_chunk,
                context_chunk=context_chunk,
                merge_columns=merge_columns,
            )

            scaled_chunk["gmm_context_id"] = context_chunk["gmm_context_id"].values

            dev_mask = scaled_chunk["split"] == Config.DEV_SPLIT_NAME
            dev_rows = int(dev_mask.sum())

            print(f"[PROGRESS] Dev rows in chunk: {dev_rows}")

            if dev_rows == 0:
                del scaled_chunk
                del context_chunk
                gc.collect()
                continue

            dev_chunk = scaled_chunk.loc[
                dev_mask,
                feature_columns + target_columns,
            ].copy()

            x_chunk = (
                dev_chunk[feature_columns]
                .replace([np.inf, -np.inf], np.nan)
                .astype(np.float32)
            )
            y_chunk = (
                dev_chunk[target_columns]
                .replace([np.inf, -np.inf], np.nan)
                .astype(np.float32)
            )

            x_nan_count = int(x_chunk.isna().sum().sum())
            y_nan_count = int(y_chunk.isna().sum().sum())

            if x_nan_count > 0:
                raise ValueError(
                    f"XGBoost training features contain NaN in chunk {chunk_index}: "
                    f"{x_nan_count}"
                )

            if y_nan_count > 0:
                raise ValueError(
                    f"XGBoost training targets contain NaN in chunk {chunk_index}: "
                    f"{y_nan_count}"
                )

            end_index = write_index + dev_rows

            if end_index > expected_dev_rows:
                raise ValueError(
                    "XGBoost memmap write exceeded expected dev rows. "
                    f"end_index={end_index}, expected_dev_rows={expected_dev_rows}"
                )

            print(f"[PROGRESS] Writing memmap rows {write_index}:{end_index}")

            x_memmap[write_index:end_index, :] = x_chunk.to_numpy(
                dtype=np.float32,
                copy=False,
            )
            y_memmap[write_index:end_index, :] = y_chunk.to_numpy(
                dtype=np.float32,
                copy=False,
            )

            write_index = end_index

            print(f"[PROGRESS] Total dev rows written to memmap: {write_index}")

            del scaled_chunk
            del context_chunk
            del dev_chunk
            del x_chunk
            del y_chunk
            gc.collect()

        if write_index != expected_dev_rows:
            raise ValueError(
                "XGBoost memmap written row count mismatch. "
                f"written={write_index}, expected={expected_dev_rows}"
            )

        x_memmap.flush()
        y_memmap.flush()

        duration = perf_counter() - started

        memmap_metadata = {
            "run_id": self.run_id,
            "training_mode": "full_dev_memmap",
            "expected_dev_rows": int(expected_dev_rows),
            "written_dev_rows": int(write_index),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "x_shape": [int(expected_dev_rows), int(n_features)],
            "y_shape": [int(expected_dev_rows), int(n_targets)],
            "x_path": str(memmap_paths["x"]),
            "y_path": str(memmap_paths["y"]),
            "duration_seconds": float(duration),
        }

        self._atomic_write_json(memmap_metadata, memmap_paths["metadata"])

        print("[PROGRESS] XGBoost full-dev memmaps built successfully")
        print(f"[PROGRESS] Memmap build duration seconds: {duration:.2f}")
        print(f"[PROGRESS] Memmap build duration minutes: {duration / 60.0:.2f}")

        x_memmap = np.memmap(
            memmap_paths["x"],
            dtype=np.float32,
            mode="r+",
            shape=(expected_dev_rows, n_features),
        )
        y_memmap = np.memmap(
            memmap_paths["y"],
            dtype=np.float32,
            mode="r+",
            shape=(expected_dev_rows, n_targets),
        )

        return x_memmap, y_memmap

    def train(
        self,
        x_dev: np.memmap,
        y_dev: np.memmap,
        feature_columns: List[str],
        target_columns: List[str],
        row_counts: Dict[str, int],
    ) -> Dict[str, object]:
        """
        Train XGBoost Regressor on full dev memmap.
        """
        print("[TRAINING] XGBoost full-dev training started")
        print("[TRAINING] Model: XGBRegressor wrapped by MultiOutputRegressor")
        print("[TRAINING] Library: XGBoost")
        print("[TRAINING] Training mode: full_dev_memmap")
        print(f"[TRAINING] X_dev shape: {x_dev.shape}")
        print(f"[TRAINING] y_dev shape: {y_dev.shape}")
        print(f"[TRAINING] Full dev rows used for training: {row_counts['dev']}")
        print("[TRAINING] Test rows used for training: 0")

        try:
            if XGBRegressor is None:
                raise ImportError("xgboost is not installed. Install xgboost.")

            started = perf_counter()

            xgb_params = dict(Config.XGB_PARAMS)
            xgb_params["n_jobs"] = self.xgb_train_n_jobs

            if "verbosity" not in xgb_params:
                xgb_params["verbosity"] = int(getattr(Config, "XGB_VERBOSITY", 1))

            print(f"[TRAINING] XGBoost params used: {xgb_params}")

            base_model = XGBRegressor(**xgb_params)
            model = MultiOutputRegressor(base_model, n_jobs=1)

            fit_started = perf_counter()
            print("[TRAINING] model.fit(X_dev_memmap, y_dev_memmap) started")

            model.fit(x_dev, y_dev)

            fit_duration = perf_counter() - fit_started
            print("[TRAINING] model.fit completed successfully")
            print(f"[TRAINING] XGBoost fit duration seconds: {fit_duration:.2f}")
            print(f"[TRAINING] XGBoost fit duration minutes: {fit_duration / 60.0:.2f}")

            trained_output_models = int(len(model.estimators_))
            print(f"[TRAINING] Number of output XGBoost models trained: {trained_output_models}")

            total_duration = perf_counter() - started

            payload: Dict[str, object] = {
                "model": model,
                "model_name": "xgboost_digital_twin_regressor",
                "model_type": "XGBRegressor_MultiOutputRegressor",
                "library": "XGBoost",
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "prediction_prefix": "xgb_predicted_",
                "target_type": "raw_X_s_only",
                "input_type": "W_plus_Xv_plus_gmm_context_id",
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "predict_only",
                "training_mode": "full_dev_memmap",
                "training_rows": int(row_counts["dev"]),
                "total_dev_rows": int(row_counts["dev"]),
                "test_rows": int(row_counts["test"]),
                "test_rows_used_for_training": 0,
                "total_rows_available": int(row_counts["total"]),
                "feature_count": int(len(feature_columns)),
                "target_count": int(len(target_columns)),
                "xgb_params": xgb_params,
                "output_models_trained": trained_output_models,
                "fit_duration_seconds": float(fit_duration),
                "total_training_duration_seconds": float(total_duration),
                "run_id": self.run_id,
                "leakage_audit": {
                    "uses_y_targets": False,
                    "uses_t_degradation_as_input": False,
                    "uses_xs_as_input": False,
                    "target_is_raw_xs_only": True,
                    "train_on_test": False,
                },
            }

            print(f"[TRAINING] Saving XGBoost model payload safely to: {self.model_path}")
            atomic_save_joblib(payload, self.model_path)

            metadata_payload = {
                key: value
                for key, value in payload.items()
                if key != "model"
            }
            self._atomic_write_json(metadata_payload, self.metadata_path)

            print("[TRAINING] XGBoost model payload saved successfully")
            print(f"[TRAINING] XGBoost metadata saved to: {self.metadata_path}")
            print(f"[TRAINING] Total XGBoost training duration seconds: {total_duration:.2f}")
            print(f"[TRAINING] Total XGBoost training duration minutes: {total_duration / 60.0:.2f}")

            logger.info(
                "XGBoost full-dev trained. rows=%s targets=%s output_models=%s",
                row_counts["dev"],
                len(target_columns),
                trained_output_models,
            )

            return payload

        except Exception as exc:
            print(f"[ERROR] XGBoost training failed: {exc}")
            logger.exception("XGBoost training failed.")
            raise RuntimeError("XGBoost training failed.") from exc

    def predict(self, expected_total_rows: int) -> int:
        """
        Predict raw X_s values for all rows.

        Args:
            expected_total_rows: Expected total row count.

        Returns:
            Number of prediction rows written.
        """
        print("[PROGRESS] XGBoost streaming prediction started")

        try:
            payload = load_joblib_required(self.model_path)

            model: MultiOutputRegressor = payload["model"]
            feature_columns: List[str] = payload["feature_columns"]
            target_columns: List[str] = payload["target_columns"]
            prediction_prefix = str(payload.get("prediction_prefix", "xgb_predicted_"))

            merge_columns = ["unit_id", "cycle", "split"]

            scaled_usecols = list(
                dict.fromkeys(
                    merge_columns
                    + [column for column in feature_columns if column != "gmm_context_id"]
                )
            )
            context_usecols = merge_columns + ["gmm_context_id"]

            output_path = self.predictions_csv
            temp_output_path = output_path.with_suffix(output_path.suffix + f".{self.run_id}.tmp")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                temp_output_path.unlink()

            print(f"[PROGRESS] Final XGBoost predictions CSV path: {output_path}")
            print(f"[PROGRESS] Temporary XGBoost predictions CSV path: {temp_output_path}")

            scaled_iter = pd.read_csv(
                Config.SCALED_CSV,
                usecols=scaled_usecols,
                chunksize=self.prediction_chunk_size,
                low_memory=True,
            )
            context_iter = pd.read_csv(
                Config.CONTEXT_CSV,
                usecols=context_usecols,
                chunksize=self.prediction_chunk_size,
                low_memory=True,
            )

            prediction_columns = [
                f"{prediction_prefix}{sensor}" for sensor in target_columns
            ]

            first_batch = True
            total_rows_written = 0
            chunk_index = 0
            started = perf_counter()

            for scaled_chunk, context_chunk in zip(scaled_iter, context_iter):
                chunk_index += 1
                start_row = total_rows_written
                end_row = total_rows_written + len(scaled_chunk)

                print("=" * 100)
                print(f"[PROGRESS] XGBoost prediction chunk #{chunk_index}")
                print(f"[PROGRESS] Rows {start_row} to {end_row}")
                print(f"[PROGRESS] Chunk shape: {scaled_chunk.shape}")

                self._verify_key_alignment(
                    scaled_chunk=scaled_chunk,
                    context_chunk=context_chunk,
                    merge_columns=merge_columns,
                )

                scaled_chunk["gmm_context_id"] = context_chunk["gmm_context_id"].values

                x_batch = (
                    scaled_chunk[feature_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                predictions = model.predict(x_batch).astype(np.float32, copy=False)

                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)

                if predictions.shape[1] != len(target_columns):
                    raise ValueError(
                        "XGBoost prediction target count mismatch. "
                        f"predicted={predictions.shape[1]}, expected={len(target_columns)}"
                    )

                result_chunk = scaled_chunk[merge_columns].copy()

                for column_index, prediction_column in enumerate(prediction_columns):
                    result_chunk[prediction_column] = predictions[:, column_index]

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                print(f"[PROGRESS] Total XGBoost prediction rows written: {total_rows_written}")

                del scaled_chunk
                del context_chunk
                del x_batch
                del predictions
                del result_chunk
                gc.collect()

            if total_rows_written != expected_total_rows:
                raise ValueError(
                    "XGBoost prediction row count mismatch. "
                    f"written={total_rows_written}, expected={expected_total_rows}"
                )

            os.replace(temp_output_path, output_path)

            duration = perf_counter() - started

            print("[PROGRESS] XGBoost predictions CSV written successfully")
            print(f"[PROGRESS] XGBoost prediction records written: {total_rows_written}")
            print(f"[PROGRESS] XGBoost prediction duration seconds: {duration:.2f}")
            print(f"[PROGRESS] XGBoost prediction duration minutes: {duration / 60.0:.2f}")

            logger.info("XGBoost prediction completed. rows=%s", total_rows_written)

            return total_rows_written

        except Exception as exc:
            print(f"[ERROR] XGBoost prediction failed: {exc}")
            logger.exception("XGBoost prediction failed.")
            raise RuntimeError("XGBoost prediction failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run XGBoost full-dev training and streaming prediction.
        """
        print("[PROGRESS] XGBoostTwin.run started")

        try:
            merge_columns, feature_columns, target_columns = self.prepare_columns()
            row_counts = self.count_rows(merge_columns=merge_columns)

            x_dev, y_dev = self.build_full_dev_memmaps(
                merge_columns=merge_columns,
                feature_columns=feature_columns,
                target_columns=target_columns,
                expected_dev_rows=row_counts["dev"],
            )

            self.train(
                x_dev=x_dev,
                y_dev=y_dev,
                feature_columns=feature_columns,
                target_columns=target_columns,
                row_counts=row_counts,
            )

            print("[PROGRESS] Releasing XGBoost training memmap references before prediction")
            del x_dev
            del y_dev
            gc.collect()

            if self.cleanup_memmap_after_training and self.memmap_dir.exists():
                print("[PROGRESS] Cleaning up XGBoost memmap directory after successful training")
                shutil.rmtree(self.memmap_dir)

            records_count = self.predict(expected_total_rows=row_counts["total"])

            response = {
                "status": "success",
                "message": (
                    "XGBoost Digital Twin trained on full dev split using "
                    "disk-backed memmaps and inferred for all splits. "
                    "Targets are raw X_s sensors only."
                ),
                "output_file": str(self.predictions_csv),
                "records_count": int(records_count),
                "model_name": "xgboost_digital_twin_regressor",
                "library": "XGBoost",
                "training_mode": "full_dev_memmap",
                "training_rows": int(row_counts["dev"]),
                "test_rows_used_for_training": 0,
                "target_type": "raw_X_s_only",
                "run_id": self.run_id,
            }

            print(f"[PROGRESS] XGBoost twin response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] XGBoost twin stage failed: {exc}")
            logger.exception("XGBoost twin stage failed.")
            raise RuntimeError("XGBoost twin stage failed.") from exc


def run_xgboost_twin() -> Dict[str, object]:
    """
    Execute XGBoost Digital Twin stage.
    """
    service = XGBoostTwin()
    return service.run()


if __name__ == "__main__":
    print("[PROGRESS] xgboost_twin.py execution started")
    result = run_xgboost_twin()
    print("[PROGRESS] xgboost_twin.py execution finished successfully")
    print(result)