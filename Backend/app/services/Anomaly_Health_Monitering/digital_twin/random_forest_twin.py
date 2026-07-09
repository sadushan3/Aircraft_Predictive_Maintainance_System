"""
Random Forest Digital Twin Regressor for CA-EDT-AHMA.

CA-EDT-AHMA:
Context-Aware Ensemble Digital Twin for Explainable Health Monitoring
and Anomaly Reasoning.

Model:
- Random Forest Regressor

Library:
- scikit-learn

Research-correct data usage:
- Input  = W + X_v + gmm_context_id
- Target = raw measured X_s sensors only
- Train  = dev split only
- Test   = prediction/evaluation only
- Ignore = Y_dev/Y_test
- Ignore = T_dev/T_test as model input
- Do not use X_s columns as input features

Memory-safety:
- Does not load full scaled_features.csv into pandas RAM.
- Reads scaled_features.csv and context_clusters.csv in chunks.
- Builds disk-backed NumPy memmap arrays for full dev training data.
- Fits sklearn RandomForestRegressor on full dev memmaps.
- Predicts all rows chunk-by-chunk directly to temporary CSV.
- Replaces final prediction CSV only after successful completion.

Important:
scikit-learn RandomForestRegressor does not support partial_fit.
So this is the safest full-dev Random Forest version while still using sklearn.
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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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


class RandomForestTwin:
    """
    Full-dev memory-safe Random Forest Digital Twin Regressor.
    """

    def __init__(
        self,
        train_chunk_size: int = 50_000,
        prediction_chunk_size: int = 50_000,
    ) -> None:
        """
        Initialize Random Forest digital twin.

        Args:
            train_chunk_size: Number of CSV rows read per training/memmap chunk.
            prediction_chunk_size: Number of CSV rows read per prediction chunk.
        """
        Config.create_directories()

        self.train_chunk_size = int(
            getattr(Config, "RF_TRAIN_CHUNK_SIZE", train_chunk_size)
        )
        self.prediction_chunk_size = int(
            getattr(Config, "RF_PREDICTION_BATCH_SIZE", prediction_chunk_size)
        )

        self.model_path: Path = getattr(
            Config,
            "RF_MODEL_PATH",
            Config.DIGITAL_TWIN_MODEL_DIR / "random_forest_twin.pkl",
        )
        self.predictions_csv: Path = getattr(
            Config,
            "RF_PREDICTIONS_CSV",
            Config.OUTPUT_DIR / "rf_predictions.csv",
        )
        self.metadata_path: Path = getattr(
            Config,
            "RF_MODEL_METADATA_PATH",
            Config.DIGITAL_TWIN_MODEL_DIR / "random_forest_twin_metadata.json",
        )

        self.memmap_dir: Path = getattr(
            Config,
            "RF_TRAIN_MEMMAP_DIR",
            Config.DIGITAL_TWIN_MODEL_DIR / "rf_full_dev_memmap",
        )

        self.rebuild_memmap = bool(getattr(Config, "RF_REBUILD_MEMMAP", True))
        self.cleanup_memmap_after_training = bool(
            getattr(Config, "RF_CLEANUP_MEMMAP_AFTER_TRAINING", False)
        )

        self.rf_train_n_jobs = int(getattr(Config, "RF_TRAIN_N_JOBS", 1))
        self.rf_verbose = int(getattr(Config, "RF_VERBOSE", 2))

        self.run_id = uuid.uuid4().hex[:12]

        self._validate_runtime_config()

        print("[PROGRESS] RandomForestTwin initialized")
        print(f"[PROGRESS] Run ID: {self.run_id}")
        print(f"[PROGRESS] Train chunk size: {self.train_chunk_size}")
        print(f"[PROGRESS] Prediction chunk size: {self.prediction_chunk_size}")
        print(f"[PROGRESS] Model path: {self.model_path}")
        print(f"[PROGRESS] Metadata path: {self.metadata_path}")
        print(f"[PROGRESS] Predictions CSV: {self.predictions_csv}")
        print(f"[PROGRESS] Memmap directory: {self.memmap_dir}")
        print(f"[PROGRESS] Rebuild memmap: {self.rebuild_memmap}")
        print(f"[PROGRESS] Cleanup memmap after training: {self.cleanup_memmap_after_training}")
        print(f"[PROGRESS] RF training n_jobs: {self.rf_train_n_jobs}")

    def _validate_runtime_config(self) -> None:
        """
        Validate runtime settings.
        """
        if self.train_chunk_size <= 0:
            raise ValueError("RF train_chunk_size must be positive.")

        if self.prediction_chunk_size <= 0:
            raise ValueError("RF prediction_chunk_size must be positive.")

        if self.rf_train_n_jobs == 0:
            raise ValueError("RF_TRAIN_N_JOBS cannot be 0.")

    # ==================================================================================
    # Small safe writers
    # ==================================================================================

    def _atomic_write_json(self, payload: Dict[str, object], output_path: Path) -> None:
        """
        Atomically write JSON metadata.

        Args:
            payload: JSON-serializable dictionary.
            output_path: Final JSON path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + f".{self.run_id}.tmp")

        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        os.replace(temp_path, output_path)

    # ==================================================================================
    # Header and column helpers
    # ==================================================================================

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read only CSV header as an empty DataFrame.

        Args:
            path: CSV file path.

        Returns:
            Empty DataFrame with columns.
        """
        print(f"[PROGRESS] Reading CSV header from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read only CSV column names.

        Args:
            path: CSV file path.

        Returns:
            Column names.
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
        Validate required columns are available.

        Args:
            available_columns: Existing columns.
            required_columns: Required columns.
            label: Human-readable source label.
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

        Args:
            scaled_chunk: Chunk from scaled_features.csv.
            context_chunk: Chunk from context_clusters.csv.
            merge_columns: Key columns.
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
        Prepare metadata, feature, and target columns.

        Returns:
            Tuple of merge columns, feature columns, target columns.
        """
        print("[PROGRESS] Preparing Random Forest columns")

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
            raise ValueError(f"Leakage-risk RF input columns found: {leakage_features}")

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
                "Engineered target columns found. RF target must be raw X_s only: "
                f"{engineered_targets}"
            )

        print(f"[PROGRESS] RF feature columns count: {len(feature_columns)}")
        print(f"[PROGRESS] RF target columns count: {len(target_columns)}")
        print(f"[PROGRESS] RF feature columns: {feature_columns}")
        print(f"[PROGRESS] RF target columns: {target_columns}")

        return merge_columns, feature_columns, target_columns

    # ==================================================================================
    # Row counting
    # ==================================================================================

    def count_rows(self, merge_columns: List[str]) -> Dict[str, int]:
        """
        Count total/dev/test rows from scaled_features.csv.

        Args:
            merge_columns: Key columns.

        Returns:
            Row-count dictionary.
        """
        print("[PROGRESS] Counting rows for Random Forest training")

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
            raise ValueError("No dev rows found for Random Forest training.")

        print(f"[PROGRESS] Final RF row counts: {counts}")

        return counts

    # ==================================================================================
    # Memmap training-data construction
    # ==================================================================================

    def _get_memmap_paths(self) -> Dict[str, Path]:
        """
        Get memmap paths for current run.

        Returns:
            Dictionary of memmap paths.
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

        Args:
            merge_columns: Key columns.
            feature_columns: RF input columns.
            target_columns: Raw X_s target columns.
            expected_dev_rows: Counted dev rows.

        Returns:
            Tuple of X_dev memmap and y_dev memmap.
        """
        print("[PROGRESS] Building full-dev Random Forest memmaps")

        memmap_paths = self._get_memmap_paths()

        if self.rebuild_memmap and self.memmap_dir.exists():
            print("[PROGRESS] Removing old RF memmap cache directory")
            shutil.rmtree(self.memmap_dir)

        self.memmap_dir.mkdir(parents=True, exist_ok=True)

        n_features = len(feature_columns)
        n_targets = len(target_columns)

        print(f"[PROGRESS] Creating X memmap shape: ({expected_dev_rows}, {n_features})")
        print(f"[PROGRESS] Creating y memmap shape: ({expected_dev_rows}, {n_targets})")

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
            print(f"[PROGRESS] RF memmap chunk #{chunk_index}")
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
                    f"RF training features contain NaN in chunk {chunk_index}: {x_nan_count}"
                )

            if y_nan_count > 0:
                raise ValueError(
                    f"RF training targets contain NaN in chunk {chunk_index}: {y_nan_count}"
                )

            end_index = write_index + dev_rows

            if end_index > expected_dev_rows:
                raise ValueError(
                    "RF memmap write exceeded expected dev rows. "
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
                "RF memmap written row count mismatch. "
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

        print("[PROGRESS] RF full-dev memmaps built successfully")
        print(f"[PROGRESS] Memmap build duration seconds: {duration:.2f}")
        print(f"[PROGRESS] Memmap build duration minutes: {duration / 60.0:.2f}")

        # Reopen memmaps in read/write mode for sklearn fit.
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

    # ==================================================================================
    # Training
    # ==================================================================================

    def train(
        self,
        x_dev: np.memmap,
        y_dev: np.memmap,
        feature_columns: List[str],
        target_columns: List[str],
        row_counts: Dict[str, int],
    ) -> Dict[str, object]:
        """
        Train Random Forest Regressor on full dev memmap.

        Args:
            x_dev: Feature memmap.
            y_dev: Target memmap.
            feature_columns: Feature names.
            target_columns: Target names.
            row_counts: Count dictionary.

        Returns:
            Model payload dictionary.
        """
        print("[TRAINING] Random Forest full-dev training started")
        print("[TRAINING] Model: RandomForestRegressor")
        print("[TRAINING] Library: scikit-learn")
        print("[TRAINING] Training mode: full_dev_memmap")
        print(f"[TRAINING] X_dev shape: {x_dev.shape}")
        print(f"[TRAINING] y_dev shape: {y_dev.shape}")
        print(f"[TRAINING] Full dev rows used for training: {row_counts['dev']}")
        print("[TRAINING] Test rows used for training: 0")

        try:
            started = perf_counter()

            rf_params = dict(Config.RF_PARAMS)
            rf_params["n_jobs"] = self.rf_train_n_jobs
            rf_params["verbose"] = self.rf_verbose

            print(f"[TRAINING] RF params used: {rf_params}")

            model = RandomForestRegressor(**rf_params)

            fit_started = perf_counter()
            print("[TRAINING] model.fit(X_dev_memmap, y_dev_memmap) started")

            model.fit(x_dev, y_dev)

            fit_duration = perf_counter() - fit_started
            print("[TRAINING] model.fit completed successfully")
            print(f"[TRAINING] RF fit duration seconds: {fit_duration:.2f}")
            print(f"[TRAINING] RF fit duration minutes: {fit_duration / 60.0:.2f}")

            trained_tree_count = int(len(model.estimators_))

            tree_depths = [int(tree.tree_.max_depth) for tree in model.estimators_]
            tree_nodes = [int(tree.tree_.node_count) for tree in model.estimators_]
            tree_leaves = [int(tree.tree_.n_leaves) for tree in model.estimators_]

            print("[TRAINING] Random Forest tree summary")
            print(
                f"[TRAINING] depth_min={min(tree_depths)}, "
                f"depth_max={max(tree_depths)}, "
                f"depth_mean={float(np.mean(tree_depths)):.2f}"
            )
            print(
                f"[TRAINING] nodes_min={min(tree_nodes)}, "
                f"nodes_max={max(tree_nodes)}, "
                f"nodes_mean={float(np.mean(tree_nodes)):.2f}"
            )
            print(
                f"[TRAINING] leaves_min={min(tree_leaves)}, "
                f"leaves_max={max(tree_leaves)}, "
                f"leaves_mean={float(np.mean(tree_leaves)):.2f}"
            )

            feature_importance_records = []

            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame(
                    {
                        "feature": feature_columns,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)

                feature_importance_records = importance_df.to_dict(orient="records")

                print("[TRAINING] Top 20 RF feature importances")
                print(importance_df.head(20).to_string(index=False))

            total_duration = perf_counter() - started

            payload: Dict[str, object] = {
                "model": model,
                "model_name": "random_forest_digital_twin_regressor",
                "model_type": "RandomForestRegressor",
                "library": "scikit-learn",
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "prediction_prefix": "rf_predicted_",
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
                "rf_params": rf_params,
                "trained_tree_count": trained_tree_count,
                "tree_depth_summary": {
                    "min": int(min(tree_depths)),
                    "max": int(max(tree_depths)),
                    "mean": float(np.mean(tree_depths)),
                },
                "tree_node_summary": {
                    "min": int(min(tree_nodes)),
                    "max": int(max(tree_nodes)),
                    "mean": float(np.mean(tree_nodes)),
                },
                "tree_leaf_summary": {
                    "min": int(min(tree_leaves)),
                    "max": int(max(tree_leaves)),
                    "mean": float(np.mean(tree_leaves)),
                },
                "feature_importances": feature_importance_records,
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

            print(f"[TRAINING] Saving RF model payload safely to: {self.model_path}")
            atomic_save_joblib(payload, self.model_path)

            metadata_payload = {
                key: value
                for key, value in payload.items()
                if key != "model"
            }
            self._atomic_write_json(metadata_payload, self.metadata_path)

            print("[TRAINING] RF model payload saved successfully")
            print(f"[TRAINING] RF metadata saved to: {self.metadata_path}")
            print(f"[TRAINING] Total RF training duration seconds: {total_duration:.2f}")
            print(f"[TRAINING] Total RF training duration minutes: {total_duration / 60.0:.2f}")

            logger.info(
                "Random Forest full-dev trained. rows=%s targets=%s trees=%s",
                row_counts["dev"],
                len(target_columns),
                trained_tree_count,
            )

            return payload

        except Exception as exc:
            print(f"[ERROR] Random Forest training failed: {exc}")
            logger.exception("Random Forest training failed.")
            raise RuntimeError("Random Forest training failed.") from exc

    # ==================================================================================
    # Prediction
    # ==================================================================================

    def predict(self, expected_total_rows: int) -> int:
        """
        Predict raw X_s values for all rows.

        Args:
            expected_total_rows: Expected total row count from scaled_features.csv.

        Returns:
            Number of prediction rows written.
        """
        print("[PROGRESS] Random Forest streaming prediction started")

        try:
            payload = load_joblib_required(self.model_path)

            model: RandomForestRegressor = payload["model"]
            feature_columns: List[str] = payload["feature_columns"]
            target_columns: List[str] = payload["target_columns"]
            prediction_prefix = str(payload.get("prediction_prefix", "rf_predicted_"))

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

            print(f"[PROGRESS] Final RF predictions CSV path: {output_path}")
            print(f"[PROGRESS] Temporary RF predictions CSV path: {temp_output_path}")

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
                print(f"[PROGRESS] RF prediction chunk #{chunk_index}")
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
                        "RF prediction target count mismatch. "
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

                print(f"[PROGRESS] Total RF prediction rows written: {total_rows_written}")

                del scaled_chunk
                del context_chunk
                del x_batch
                del predictions
                del result_chunk
                gc.collect()

            if total_rows_written != expected_total_rows:
                raise ValueError(
                    "RF prediction row count mismatch. "
                    f"written={total_rows_written}, expected={expected_total_rows}"
                )

            os.replace(temp_output_path, output_path)

            duration = perf_counter() - started

            print("[PROGRESS] RF predictions CSV written successfully")
            print(f"[PROGRESS] RF prediction records written: {total_rows_written}")
            print(f"[PROGRESS] RF prediction duration seconds: {duration:.2f}")
            print(f"[PROGRESS] RF prediction duration minutes: {duration / 60.0:.2f}")

            logger.info("Random Forest prediction completed. rows=%s", total_rows_written)

            return total_rows_written

        except Exception as exc:
            print(f"[ERROR] Random Forest prediction failed: {exc}")
            logger.exception("Random Forest prediction failed.")
            raise RuntimeError("Random Forest prediction failed.") from exc

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run Random Forest full-dev training and streaming prediction.

        Returns:
            Stage response dictionary.
        """
        print("[PROGRESS] RandomForestTwin.run started")

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

            print("[PROGRESS] Releasing RF training memmap references before prediction")
            del x_dev
            del y_dev
            gc.collect()

            if self.cleanup_memmap_after_training and self.memmap_dir.exists():
                print("[PROGRESS] Cleaning up RF memmap directory after successful training")
                shutil.rmtree(self.memmap_dir)

            records_count = self.predict(expected_total_rows=row_counts["total"])

            response = {
                "status": "success",
                "message": (
                    "Random Forest Digital Twin trained on full dev split using "
                    "disk-backed memmaps and inferred for all splits. "
                    "Targets are raw X_s sensors only."
                ),
                "output_file": str(self.predictions_csv),
                "records_count": int(records_count),
                "model_name": "random_forest_digital_twin_regressor",
                "library": "scikit-learn",
                "training_mode": "full_dev_memmap",
                "training_rows": int(row_counts["dev"]),
                "test_rows_used_for_training": 0,
                "target_type": "raw_X_s_only",
                "run_id": self.run_id,
            }

            print(f"[PROGRESS] Random Forest twin response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Random Forest twin stage failed: {exc}")
            logger.exception("Random Forest twin stage failed.")
            raise RuntimeError("Random Forest twin stage failed.") from exc


def run_random_forest_twin() -> Dict[str, object]:
    """
    Execute Random Forest Digital Twin stage.

    Returns:
        Stage response dictionary.
    """
    service = RandomForestTwin()
    return service.run()


if __name__ == "__main__":
    print("[PROGRESS] random_forest_twin.py execution started")
    result = run_random_forest_twin()
    print("[PROGRESS] random_forest_twin.py execution finished successfully")
    print(result)