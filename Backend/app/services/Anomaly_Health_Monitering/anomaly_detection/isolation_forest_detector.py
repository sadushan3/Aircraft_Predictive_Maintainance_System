"""
Isolation Forest detector for CA-EDT-AHMA.

Updated high-record memory-safe version.

Correct research version:
Input features:
1. raw residual_Xs_* columns
2. residual temporal resfeat_* columns

Training:
- Fit scaler on dev residual/anomaly features only using chunked partial_fit.
- Collect a large bounded dev-only reservoir sample.
- Train Isolation Forest on the maximum available sample records.
- Uses explicit max_samples instead of sklearn default auto behavior.

Inference:
- Score dev and test.
- Does not fit on test.
- Does not use Y_dev/Y_test.
- Does not use T_dev/T_test.
- Does not use actual Xs_* or ensemble_predicted_* as model features.

Memory-safe behavior:
- Does not load full residuals.csv into RAM.
- Reads residuals.csv in chunks.
- Uses bounded NumPy reservoir sample.
- Scores all rows using two streaming passes.
- Writes to temporary CSV first, then replaces final CSV only after success.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "anomaly_detection/isolation_forest_detector.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
import gc
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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
    atomic_write_json,
    load_joblib_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class IsolationForestDetector:
    """
    Memory-safe Isolation Forest residual-pattern detector.

    Uses:
    - residual_Xs_* raw residual features
    - resfeat_* residual temporal features

    Excludes:
    - abs_residual_* threshold magnitude columns
    - actual Xs_* sensor values
    - ensemble_predicted_* values
    - context IDs as model features
    """

    def __init__(
        self,
        chunk_size: int = 50_000,
        train_sample_size: int = 1_000_000,
        n_estimators: int = 200,
    ) -> None:
        """
        Initialize Isolation Forest detector.

        Args:
            chunk_size: Number of CSV rows processed per chunk.
            train_sample_size: Maximum dev rows stored for Isolation Forest fitting.
            n_estimators: Number of Isolation Forest trees.
        """
        print("[PROGRESS] Entering IsolationForestDetector.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "IFOREST_CHUNK_SIZE", chunk_size)
        )

        self.train_sample_size = int(
            getattr(Config, "IFOREST_TRAIN_SAMPLE_SIZE", train_sample_size)
        )

        self.n_estimators = int(
            getattr(Config, "IFOREST_N_ESTIMATORS", n_estimators)
        )

        self.n_jobs = int(
            getattr(Config, "IFOREST_N_JOBS", 2)
        )

        self.contamination = getattr(
            Config,
            "IFOREST_CONTAMINATION",
            "auto",
        )

        # Important:
        # "all" means every tree uses all collected training sample rows.
        # This avoids sklearn's default max_samples="auto" behavior.
        self.max_samples_config = getattr(
            Config,
            "IFOREST_MAX_SAMPLES",
            "all",
        )

        self.random_seed = int(
            getattr(Config, "RANDOM_SEED", 42)
        )

        if self.chunk_size <= 0:
            raise ValueError("IFOREST_CHUNK_SIZE must be positive.")

        if self.train_sample_size <= 0:
            raise ValueError("IFOREST_TRAIN_SAMPLE_SIZE must be positive.")

        if self.n_estimators <= 0:
            raise ValueError("IFOREST_N_ESTIMATORS must be positive.")

        if self.n_jobs == 0:
            raise ValueError("IFOREST_N_JOBS cannot be 0.")

        self.model_path: Path = Config.IFOREST_MODEL_PATH
        self.output_csv: Path = Config.IFOREST_CSV
        self.summary_json: Path = Config.REPORT_DIR / "isolation_forest_summary.json"

        self.rng = np.random.default_rng(self.random_seed)

        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Train sample size: {self.train_sample_size}")
        print(f"[PROGRESS] Number of trees: {self.n_estimators}")
        print(f"[PROGRESS] n_jobs: {self.n_jobs}")
        print(f"[PROGRESS] Contamination: {self.contamination}")
        print(f"[PROGRESS] max_samples config: {self.max_samples_config}")
        print(f"[PROGRESS] Residual CSV: {Config.RESIDUALS_CSV}")
        print(f"[PROGRESS] Model path: {self.model_path}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")

    # ==================================================================================
    # Helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows without loading file into memory.

        Args:
            path: CSV path.

        Returns:
            Number of data rows excluding header.
        """
        print(f"[PROGRESS] Counting rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)

        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read CSV header only as empty DataFrame.
        """
        print(f"[PROGRESS] Reading header DataFrame from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV column names only.
        """
        print(f"[PROGRESS] Reading header columns from: {path}")
        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate that required columns exist.

        Args:
            available_columns: Existing column names.
            required_columns: Required column names.
            label: Source label.
        """
        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def get_anomaly_feature_columns_from_header(self) -> List[str]:
        """
        Select anomaly feature columns from residuals.csv header.

        Uses:
        - residual_Xs_* raw residual columns
        - resfeat_* temporal residual columns

        Excludes:
        - abs_residual_Xs_* direct threshold magnitude columns
        - Xs_* actual values
        - ensemble_predicted_* values
        - metadata/context columns

        Returns:
            Anomaly feature columns.
        """
        print("[PROGRESS] Selecting Isolation Forest anomaly feature columns")

        residual_header_df = self._read_header_df(Config.RESIDUALS_CSV)
        columns = list(residual_header_df.columns)

        merge_columns = ["unit_id", "cycle", "split"]

        self._validate_columns(
            available_columns=columns,
            required_columns=merge_columns,
            label="residuals.csv",
        )

        raw_residual_columns = [
            column
            for column in columns
            if column.startswith("residual_Xs_")
        ]

        resfeat_columns = [
            column
            for column in columns
            if column.startswith("resfeat_")
        ]

        feature_columns = raw_residual_columns + resfeat_columns

        invalid_features = [
            column
            for column in feature_columns
            if column.startswith("abs_residual_")
            or column.startswith("Xs_")
            or column.startswith("ensemble_predicted_")
            or column in {"unit_id", "cycle", "split", "gmm_context_id"}
        ]

        if invalid_features:
            raise ValueError(
                "Leakage or invalid Isolation Forest feature columns found: "
                f"{invalid_features}"
            )

        if not feature_columns:
            raise ValueError(
                "No Isolation Forest anomaly features found. "
                "Expected residual_Xs_* and resfeat_* columns."
            )

        print(f"[PROGRESS] Raw residual feature count: {len(raw_residual_columns)}")
        print(f"[PROGRESS] Residual temporal feature count: {len(resfeat_columns)}")
        print(f"[PROGRESS] Total Isolation Forest feature count: {len(feature_columns)}")
        print(f"[PROGRESS] Isolation Forest feature columns: {feature_columns}")

        return feature_columns

    def _prepare_feature_array(
        self,
        chunk: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """
        Convert chunk feature columns to clean float32 array.

        Args:
            chunk: Input chunk.
            feature_columns: Feature columns.

        Returns:
            Clean feature array.
        """
        x = (
            chunk[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

        return x

    def _resolve_max_samples(self, sample_rows: int) -> int | str:
        """
        Resolve IsolationForest max_samples.

        Args:
            sample_rows: Number of training sample rows collected.

        Returns:
            max_samples value for IsolationForest.

        Notes:
            - "all" or "max" means use all collected training sample rows.
            - "auto" keeps sklearn auto behavior.
            - integer uses min(integer, sample_rows).
        """
        value = self.max_samples_config

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in {"all", "max", "full"}:
                return int(sample_rows)

            if normalized == "auto":
                return "auto"

            try:
                parsed = int(normalized)
                return int(min(parsed, sample_rows))
            except ValueError as exc:
                raise ValueError(
                    "Invalid IFOREST_MAX_SAMPLES. Use 'all', 'auto', or an integer."
                ) from exc

        parsed_int = int(value)
        if parsed_int <= 0:
            raise ValueError("IFOREST_MAX_SAMPLES integer must be positive.")

        return int(min(parsed_int, sample_rows))

    def _create_reservoir_buffer(
        self,
        n_features: int,
    ) -> np.ndarray:
        """
        Create bounded reservoir training buffer.

        Args:
            n_features: Number of anomaly features.

        Returns:
            Empty reservoir buffer.
        """
        print("[TRAINING] Creating bounded dev reservoir buffer")
        print(
            "[TRAINING] Reservoir shape: "
            f"({self.train_sample_size}, {n_features})"
        )

        buffer = np.empty(
            (self.train_sample_size, n_features),
            dtype=np.float32,
        )

        return buffer

    def _reservoir_update(
        self,
        reservoir: np.ndarray,
        sample_count: int,
        dev_seen_before: int,
        x_dev: np.ndarray,
    ) -> int:
        """
        Update reservoir sample using streaming reservoir sampling.

        Args:
            reservoir: Preallocated sample buffer.
            sample_count: Current filled sample count.
            dev_seen_before: Number of dev rows seen before this chunk.
            x_dev: Current dev feature rows.

        Returns:
            Updated sample count.
        """
        if len(x_dev) == 0:
            return sample_count

        capacity = len(reservoir)

        for local_index in range(len(x_dev)):
            global_dev_index = dev_seen_before + local_index

            if sample_count < capacity:
                reservoir[sample_count, :] = x_dev[local_index, :]
                sample_count += 1
            else:
                replacement_index = self.rng.integers(
                    low=0,
                    high=global_dev_index + 1,
                )

                if replacement_index < capacity:
                    reservoir[replacement_index, :] = x_dev[local_index, :]

        return sample_count

    # ==================================================================================
    # Training
    # ==================================================================================

    def fit(self) -> Dict[str, object]:
        """
        Fit Isolation Forest using dev data only.

        Memory-safe:
        - StandardScaler is partial-fitted on all dev rows chunk-by-chunk.
        - Isolation Forest is trained on a large bounded dev-only reservoir sample.
        - max_samples is explicitly set, usually to all collected sample rows.

        Returns:
            Model payload.
        """
        print("[TRAINING] Entering IsolationForestDetector.fit")

        try:
            started = perf_counter()

            if not Config.RESIDUALS_CSV.exists():
                raise FileNotFoundError(f"Residual CSV not found: {Config.RESIDUALS_CSV}")

            feature_columns = self.get_anomaly_feature_columns_from_header()
            n_features = len(feature_columns)

            merge_columns = ["unit_id", "cycle", "split"]
            usecols = merge_columns + feature_columns

            scaler = StandardScaler()

            reservoir = self._create_reservoir_buffer(n_features=n_features)
            sample_count = 0

            total_rows_seen = 0
            total_dev_rows_seen = 0
            chunk_index = 0

            print("[TRAINING] Starting chunked dev-only scaler fitting and reservoir sampling")

            for chunk in pd.read_csv(
                Config.RESIDUALS_CSV,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                total_rows_seen += len(chunk)

                print("=" * 100)
                print(f"[TRAINING] Isolation Forest training chunk #{chunk_index}")
                print(f"[TRAINING] Chunk rows: {len(chunk)}")
                print(f"[TRAINING] Total rows scanned: {total_rows_seen}")

                dev_mask = chunk["split"] == Config.DEV_SPLIT_NAME
                dev_rows = int(dev_mask.sum())

                print(f"[TRAINING] Dev rows in chunk: {dev_rows}")
                print(f"[TRAINING] Total dev rows before chunk: {total_dev_rows_seen}")

                if dev_rows == 0:
                    del chunk
                    gc.collect()
                    continue

                dev_chunk = chunk.loc[dev_mask, feature_columns]

                x_dev = (
                    dev_chunk
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                scaler.partial_fit(x_dev)

                sample_count = self._reservoir_update(
                    reservoir=reservoir,
                    sample_count=sample_count,
                    dev_seen_before=total_dev_rows_seen,
                    x_dev=x_dev,
                )

                total_dev_rows_seen += dev_rows

                print(f"[TRAINING] Total dev rows seen after chunk: {total_dev_rows_seen}")
                print(f"[TRAINING] Current reservoir sample count: {sample_count}")

                del chunk
                del dev_chunk
                del x_dev
                gc.collect()

            if total_dev_rows_seen <= 0:
                raise ValueError("No dev rows found. Cannot fit Isolation Forest.")

            if sample_count <= 0:
                raise ValueError("No dev sample rows collected for Isolation Forest.")

            training_sample = reservoir[:sample_count, :]

            print("[TRAINING] Scaling final reservoir training sample")
            sample_scaled = scaler.transform(training_sample).astype(np.float32, copy=False)

            max_samples_for_model = self._resolve_max_samples(sample_rows=sample_count)

            print("[TRAINING] Starting Isolation Forest model fit")
            print(f"[TRAINING] Full dev rows seen: {total_dev_rows_seen}")
            print(f"[TRAINING] Reservoir sample rows used for model fit: {sample_count}")
            print(f"[TRAINING] Feature count: {len(feature_columns)}")
            print(f"[TRAINING] IsolationForest max_samples used: {max_samples_for_model}")
            print(f"[TRAINING] IsolationForest n_estimators: {self.n_estimators}")

            model = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=max_samples_for_model,
                contamination=self.contamination,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
                verbose=1,
            )

            fit_started = perf_counter()
            model.fit(sample_scaled)
            fit_duration = perf_counter() - fit_started

            duration = perf_counter() - started

            payload: Dict[str, object] = {
                "model": model,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "feature_type": "raw_residual_plus_residual_temporal_features",
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "training_mode": "dev_only_chunked_scaler_large_reservoir_iforest",
                "total_rows_scanned": int(total_rows_seen),
                "dev_rows_seen": int(total_dev_rows_seen),
                "train_sample_rows": int(sample_count),
                "feature_count": int(len(feature_columns)),
                "n_estimators": int(self.n_estimators),
                "max_samples_config": self.max_samples_config,
                "max_samples_used": max_samples_for_model,
                "contamination": self.contamination,
                "n_jobs": int(self.n_jobs),
                "fit_duration_seconds": float(fit_duration),
                "duration_seconds": float(duration),
                "leakage_audit": {
                    "fit_on_dev_only": True,
                    "test_rows_used_for_fit": 0,
                    "uses_y_targets": False,
                    "uses_t_degradation_as_input": False,
                    "uses_actual_xs_as_features": False,
                    "uses_predicted_xs_as_features": False,
                },
            }

            print(f"[TRAINING] Saving Isolation Forest model to: {self.model_path}")
            atomic_save_joblib(payload, self.model_path)

            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"model", "scaler"}
            }

            metadata_path = Config.REPORT_DIR / "isolation_forest_training_summary.json"
            atomic_write_json(metadata, metadata_path)

            print("[TRAINING] Isolation Forest fit completed successfully")
            print(f"[TRAINING] Metadata saved to: {metadata_path}")
            print(f"[TRAINING] Fit duration seconds: {fit_duration:.2f}")
            print(f"[TRAINING] Fit duration minutes: {fit_duration / 60.0:.2f}")
            print(f"[TRAINING] Total duration seconds: {duration:.2f}")
            print(f"[TRAINING] Total duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Isolation Forest fitted. dev_rows=%s sample_rows=%s features=%s max_samples=%s",
                total_dev_rows_seen,
                sample_count,
                len(feature_columns),
                max_samples_for_model,
            )

            del reservoir
            del training_sample
            del sample_scaled
            gc.collect()

            return payload

        except Exception as exc:
            print(f"[ERROR] Isolation Forest fitting failed: {exc}")
            logger.exception("Isolation Forest fitting failed.")
            raise RuntimeError("Isolation Forest fitting failed.") from exc

    # ==================================================================================
    # Scoring
    # ==================================================================================

    def _score_chunk_raw(
        self,
        model: IsolationForest,
        scaler: StandardScaler,
        chunk: pd.DataFrame,
        feature_columns: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score one chunk with Isolation Forest.

        Args:
            model: Fitted IsolationForest.
            scaler: Fitted StandardScaler.
            chunk: Residual chunk.
            feature_columns: Feature columns.

        Returns:
            Tuple:
            - raw anomaly scores, where higher means more anomalous
            - anomaly labels, 1 means anomaly, 0 means normal
        """
        x = self._prepare_feature_array(chunk, feature_columns)
        x_scaled = scaler.transform(x).astype(np.float32, copy=False)

        decision_values = model.decision_function(x_scaled)

        raw_scores = (-decision_values).astype(np.float32, copy=False)
        labels = (decision_values < 0.0).astype(np.int8)

        del x
        del x_scaled
        del decision_values

        return raw_scores, labels

    def _first_score_pass(
        self,
        model: IsolationForest,
        scaler: StandardScaler,
        feature_columns: List[str],
    ) -> Dict[str, object]:
        """
        First scoring pass.

        Computes raw score min/max and label counts without storing all scores.

        Args:
            model: Fitted IsolationForest.
            scaler: Fitted StandardScaler.
            feature_columns: Feature columns.

        Returns:
            Score range metadata.
        """
        print("[PROGRESS] Starting Isolation Forest first scoring pass")

        merge_columns = ["unit_id", "cycle", "split"]
        usecols = merge_columns + feature_columns

        score_min = np.inf
        score_max = -np.inf
        total_rows = 0
        anomaly_count = 0
        normal_count = 0
        chunk_index = 0

        for chunk in pd.read_csv(
            Config.RESIDUALS_CSV,
            usecols=usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        ):
            chunk_index += 1
            total_rows += len(chunk)

            print("=" * 100)
            print(f"[PROGRESS] First scoring pass chunk #{chunk_index}")
            print(f"[PROGRESS] Chunk rows: {len(chunk)}")
            print(f"[PROGRESS] Total rows scanned: {total_rows}")

            raw_scores, labels = self._score_chunk_raw(
                model=model,
                scaler=scaler,
                chunk=chunk,
                feature_columns=feature_columns,
            )

            if len(raw_scores) > 0:
                score_min = min(score_min, float(np.min(raw_scores)))
                score_max = max(score_max, float(np.max(raw_scores)))

            anomaly_count += int(labels.sum())
            normal_count += int(len(labels) - labels.sum())

            print(f"[PROGRESS] Running raw score min: {score_min}")
            print(f"[PROGRESS] Running raw score max: {score_max}")
            print(f"[PROGRESS] Running normal count: {normal_count}")
            print(f"[PROGRESS] Running anomaly count: {anomaly_count}")

            del chunk
            del raw_scores
            del labels
            gc.collect()

        if total_rows <= 0:
            raise ValueError("No rows found during Isolation Forest scoring.")

        if not np.isfinite(score_min) or not np.isfinite(score_max):
            raise ValueError("Invalid Isolation Forest raw score range.")

        return {
            "score_min": float(score_min),
            "score_max": float(score_max),
            "total_rows": int(total_rows),
            "normal_count": int(normal_count),
            "anomaly_count": int(anomaly_count),
        }

    def _normalize_scores(
        self,
        raw_scores: np.ndarray,
        score_min: float,
        score_max: float,
    ) -> np.ndarray:
        """
        Normalize raw scores to 0-1 using global streaming min/max.

        Args:
            raw_scores: Raw anomaly scores.
            score_min: Global min.
            score_max: Global max.

        Returns:
            Normalized scores.
        """
        denominator = max(float(score_max - score_min), 1e-12)

        normalized = ((raw_scores - score_min) / denominator).astype(
            np.float32,
            copy=False,
        )

        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized

    def score(self) -> int:
        """
        Score all residual rows with dev-fitted Isolation Forest.

        Memory-safe:
        - First pass computes global score range.
        - Second pass writes normalized scores chunk-by-chunk.

        Returns:
            Number of scored rows written.
        """
        print("[PROGRESS] Entering IsolationForestDetector.score")

        try:
            started = perf_counter()

            if not Config.RESIDUALS_CSV.exists():
                raise FileNotFoundError(f"Residual CSV not found: {Config.RESIDUALS_CSV}")

            expected_rows = self._count_csv_rows(Config.RESIDUALS_CSV)

            payload = load_joblib_required(self.model_path)

            model: IsolationForest = payload["model"]
            scaler: StandardScaler = payload["scaler"]
            feature_columns: List[str] = payload["feature_columns"]

            residual_columns = self._read_header_columns(Config.RESIDUALS_CSV)

            self._validate_columns(
                available_columns=residual_columns,
                required_columns=["unit_id", "cycle", "split"] + feature_columns,
                label="residuals.csv",
            )

            score_range = self._first_score_pass(
                model=model,
                scaler=scaler,
                feature_columns=feature_columns,
            )

            score_min = float(score_range["score_min"])
            score_max = float(score_range["score_max"])

            print("[PROGRESS] First scoring pass completed")
            print(f"[PROGRESS] Score min: {score_min}")
            print(f"[PROGRESS] Score max: {score_max}")

            merge_columns = ["unit_id", "cycle", "split"]
            usecols = merge_columns + feature_columns

            output_path = self.output_csv
            temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary Isolation Forest output")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            label_counts = {
                0: 0,
                1: 0,
            }

            split_label_counts: Dict[str, Dict[int, int]] = {
                Config.DEV_SPLIT_NAME: {0: 0, 1: 0},
                Config.TEST_SPLIT_NAME: {0: 0, 1: 0},
            }

            raw_score_sum = 0.0
            normalized_score_sum = 0.0

            print("[PROGRESS] Starting Isolation Forest second scoring/write pass")

            for chunk in pd.read_csv(
                Config.RESIDUALS_CSV,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1

                print("=" * 100)
                print(f"[PROGRESS] Second scoring pass chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                raw_scores, labels = self._score_chunk_raw(
                    model=model,
                    scaler=scaler,
                    chunk=chunk,
                    feature_columns=feature_columns,
                )

                normalized_scores = self._normalize_scores(
                    raw_scores=raw_scores,
                    score_min=score_min,
                    score_max=score_max,
                )

                result_chunk = chunk[merge_columns].copy()
                result_chunk["iforest_raw_score"] = raw_scores
                result_chunk["iforest_anomaly_score"] = normalized_scores
                result_chunk["iforest_anomaly_label"] = labels
                result_chunk["iforest_feature_count"] = int(len(feature_columns))
                result_chunk["iforest_decision_threshold_raw"] = 0.0

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_labels, unique_counts = np.unique(labels, return_counts=True)

                for label, count in zip(unique_labels, unique_counts):
                    label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_labels = labels[split_mask.to_numpy()]
                    split_unique, split_counts = np.unique(split_labels, return_counts=True)

                    for label, count in zip(split_unique, split_counts):
                        split_label_counts[split][int(label)] = (
                            split_label_counts[split].get(int(label), 0) + int(count)
                        )

                raw_score_sum += float(np.sum(raw_scores, dtype=np.float64))
                normalized_score_sum += float(np.sum(normalized_scores, dtype=np.float64))

                print(f"[PROGRESS] Total Isolation Forest rows written: {total_rows_written}")
                print(f"[PROGRESS] Running label counts: {label_counts}")

                del chunk
                del raw_scores
                del labels
                del normalized_scores
                del result_chunk
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All Isolation Forest scoring chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Isolation Forest output row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final CSV will not be replaced."
                )

            os.replace(temp_output_path, output_path)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "output_file": str(output_path),
                "records_count": int(total_rows_written),
                "feature_count": int(len(feature_columns)),
                "score_min": float(score_min),
                "score_max": float(score_max),
                "raw_score_mean": float(raw_score_sum / max(total_rows_written, 1)),
                "normalized_score_mean": float(
                    normalized_score_sum / max(total_rows_written, 1)
                ),
                "label_counts": {
                    "normal_0": int(label_counts.get(0, 0)),
                    "anomaly_1": int(label_counts.get(1, 0)),
                },
                "split_label_counts": {
                    split: {
                        "normal_0": int(counts.get(0, 0)),
                        "anomaly_1": int(counts.get(1, 0)),
                    }
                    for split, counts in split_label_counts.items()
                },
                "training_mode": payload.get("training_mode"),
                "fit_split": payload.get("fit_split"),
                "test_usage": payload.get("test_usage"),
                "train_sample_rows": payload.get("train_sample_rows"),
                "max_samples_used": payload.get("max_samples_used"),
                "n_estimators": payload.get("n_estimators"),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
            }

            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Isolation Forest scores CSV written successfully")
            print(f"[PROGRESS] Summary JSON written to: {self.summary_json}")
            print(f"[PROGRESS] Label counts: {summary['label_counts']}")
            print(f"[PROGRESS] Split label counts: {summary['split_label_counts']}")
            print(f"[PROGRESS] Scoring duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Scoring duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Isolation Forest scoring completed. rows=%s labels=%s",
                total_rows_written,
                summary["label_counts"],
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Isolation Forest scoring failed: {exc}")
            logger.exception("Isolation Forest scoring failed.")
            raise RuntimeError("Isolation Forest scoring failed.") from exc

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run Isolation Forest detector.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering IsolationForestDetector.run")

        try:
            payload = self.fit()
            records_count = self.score()

            response = {
                "status": "success",
                "message": (
                    "Isolation Forest scores generated using dev-fitted model. "
                    "Features include raw residuals and residual temporal features."
                ),
                "output_file": str(self.output_csv),
                "model_file": str(self.model_path),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "feature_type": payload["feature_type"],
                "training_mode": payload["training_mode"],
                "train_sample_rows": payload["train_sample_rows"],
                "max_samples_used": payload["max_samples_used"],
                "dev_rows_seen": payload["dev_rows_seen"],
                "feature_count": payload["feature_count"],
                "n_estimators": payload["n_estimators"],
            }

            print(f"[PROGRESS] Isolation Forest detector response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Isolation Forest detector stage failed: {exc}")
            logger.exception("Isolation Forest detector stage failed.")
            raise RuntimeError("Isolation Forest detector stage failed.") from exc


def run_isolation_forest_detection() -> Dict[str, object]:
    """
    Execute Isolation Forest detection.
    """
    print("[PROGRESS] Entering run_isolation_forest_detection")

    detector = IsolationForestDetector()
    return detector.run()


if __name__ == "__main__":
    print("[PROGRESS] isolation_forest_detector.py execution started")
    result = run_isolation_forest_detection()
    print("[PROGRESS] isolation_forest_detector.py execution finished successfully")
    print(result)