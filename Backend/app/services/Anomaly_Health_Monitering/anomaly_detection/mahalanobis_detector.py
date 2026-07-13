"""
Mahalanobis distance detector for CA-EDT-AHMA.

Correct research version:
Input features:
1. raw residual_Xs_* columns
2. residual temporal resfeat_* columns

Training:
- Fit StandardScaler on dev rows only using chunked partial_fit.
- Fit LedoitWolf covariance on a large dev-only reservoir sample.
- Fit Mahalanobis threshold from dev rows only.

Inference:
- Score dev and test.
- Does not fit on test.
- Does not use Y_dev/Y_test.
- Does not use T_dev/T_test.
- Does not use actual Xs_* or ensemble_predicted_* as anomaly features.

Memory-safe:
- Does not load full residuals.csv into RAM.
- Reads residuals.csv in chunks.
- Writes output to temporary CSV first.
- Replaces final CSV only after successful completion.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "anomaly_detection/mahalanobis_detector.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
import gc
import os
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    if BACKEND_ROOT not in sys.path:
        sys.path.insert(0, BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_save_joblib,
    atomic_write_json,
    load_joblib_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class MahalanobisDetector:
    """
    Memory-safe Mahalanobis residual distance detector.

    Uses:
    - residual_Xs_* raw residual features
    - resfeat_* residual temporal features

    Excludes:
    - abs_residual_Xs_* direct threshold features
    - actual Xs_* sensor values
    - ensemble_predicted_* values
    - Y and T targets
    """

    def __init__(
        self,
        chunk_size: int = 50_000,
        train_sample_size: int = 1_000_000,
    ) -> None:
        """
        Initialize Mahalanobis detector.

        Args:
            chunk_size: Number of CSV rows processed per chunk.
            train_sample_size: Maximum dev rows used for LedoitWolf covariance fitting.
        """
        print("[PROGRESS] Entering MahalanobisDetector.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "MAHALANOBIS_CHUNK_SIZE", chunk_size)
        )

        self.train_sample_size = int(
            getattr(Config, "MAHALANOBIS_TRAIN_SAMPLE_SIZE", train_sample_size)
        )

        self.regularization = float(
            getattr(Config, "MAHALANOBIS_REGULARIZATION", 1e-6)
        )

        self.threshold_percentile = float(
            getattr(Config, "MAHALANOBIS_THRESHOLD_PERCENTILE", 99.0)
        )

        self.random_seed = int(
            getattr(Config, "RANDOM_SEED", 42)
        )

        if self.chunk_size <= 0:
            raise ValueError("MAHALANOBIS_CHUNK_SIZE must be positive.")

        if self.train_sample_size <= 0:
            raise ValueError("MAHALANOBIS_TRAIN_SAMPLE_SIZE must be positive.")

        if self.regularization < 0:
            raise ValueError("MAHALANOBIS_REGULARIZATION cannot be negative.")

        if not (0.0 < self.threshold_percentile < 100.0):
            raise ValueError("MAHALANOBIS_THRESHOLD_PERCENTILE must be between 0 and 100.")

        self.params_path: Path = Config.MAHALANOBIS_PARAMS_PATH
        self.output_csv: Path = Config.MAHALANOBIS_CSV
        self.summary_json: Path = Config.REPORT_DIR / "mahalanobis_summary.json"

        self.rng = np.random.default_rng(self.random_seed)

        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Train sample size: {self.train_sample_size}")
        print(f"[PROGRESS] Regularization: {self.regularization}")
        print(f"[PROGRESS] Threshold percentile: {self.threshold_percentile}")
        print(f"[PROGRESS] Residual CSV: {Config.RESIDUALS_CSV}")
        print(f"[PROGRESS] Params path: {self.params_path}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")

    # ==================================================================================
    # Helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows without loading the file.

        Args:
            path: CSV path.

        Returns:
            Data row count excluding header.
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
        Read CSV header only.
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
        Validate required columns.
        """
        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def get_anomaly_feature_columns_from_header(self) -> List[str]:
        """
        Select Mahalanobis anomaly feature columns from residuals.csv header.

        Includes:
        - residual_Xs_*
        - resfeat_*

        Excludes:
        - abs_residual_Xs_* direct magnitude columns
        - actual Xs_* columns
        - ensemble_predicted_* columns
        - metadata/context columns

        Returns:
            Feature column list.
        """
        print("[PROGRESS] Selecting Mahalanobis anomaly feature columns")

        header_df = self._read_header_df(Config.RESIDUALS_CSV)
        columns = list(header_df.columns)

        self._validate_columns(
            available_columns=columns,
            required_columns=["unit_id", "cycle", "split"],
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
                "Invalid/leakage Mahalanobis feature columns found: "
                f"{invalid_features}"
            )

        if not feature_columns:
            raise ValueError(
                "No Mahalanobis anomaly features found. "
                "Expected residual_Xs_* and resfeat_* columns."
            )

        print(f"[PROGRESS] Raw residual feature count: {len(raw_residual_columns)}")
        print(f"[PROGRESS] Residual temporal feature count: {len(resfeat_columns)}")
        print(f"[PROGRESS] Total Mahalanobis feature count: {len(feature_columns)}")
        print(f"[PROGRESS] Mahalanobis feature columns: {feature_columns}")

        return feature_columns

    def _prepare_feature_array(
        self,
        chunk: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """
        Convert feature columns into clean float32 array.
        """
        x = (
            chunk[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

        return x

    def _create_reservoir_buffer(self, n_features: int) -> np.ndarray:
        """
        Create bounded dev reservoir buffer.
        """
        print("[TRAINING] Creating bounded Mahalanobis dev reservoir buffer")
        print(
            "[TRAINING] Reservoir shape: "
            f"({self.train_sample_size}, {n_features})"
        )

        return np.empty(
            (self.train_sample_size, n_features),
            dtype=np.float32,
        )

    def _reservoir_update(
        self,
        reservoir: np.ndarray,
        sample_count: int,
        dev_seen_before: int,
        x_dev: np.ndarray,
    ) -> int:
        """
        Streaming reservoir sampling update.

        Args:
            reservoir: Preallocated sample buffer.
            sample_count: Current filled sample count.
            dev_seen_before: Number of dev rows seen before this chunk.
            x_dev: Current dev rows.

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

    def _calculate_distances(
        self,
        values: np.ndarray,
        mean_vector: np.ndarray,
        precision_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Mahalanobis distances.

        Args:
            values: Scaled feature matrix.
            mean_vector: Mean vector.
            precision_matrix: Precision matrix.

        Returns:
            Distance values.
        """
        delta = values - mean_vector

        squared_distances = np.einsum(
            "ij,jk,ik->i",
            delta,
            precision_matrix,
            delta,
        )

        squared_distances = np.maximum(squared_distances, 0.0)
        distances = np.sqrt(squared_distances)

        return distances.astype(np.float32, copy=False)

    # ==================================================================================
    # Training
    # ==================================================================================

    def fit(self) -> Dict[str, object]:
        """
        Fit Mahalanobis parameters using dev anomaly features only.

        Memory-safe:
        - scaler.partial_fit uses all dev rows
        - LedoitWolf covariance uses large dev-only reservoir sample
        - threshold uses all dev rows through streaming distance pass

        Returns:
            Parameter payload.
        """
        print("[TRAINING] Entering MahalanobisDetector.fit")

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

            print("[TRAINING] Starting chunked dev scaler fitting and reservoir sampling")

            for chunk in pd.read_csv(
                Config.RESIDUALS_CSV,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                total_rows_seen += len(chunk)

                print("=" * 100)
                print(f"[TRAINING] Mahalanobis training chunk #{chunk_index}")
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
                x_dev = self._prepare_feature_array(dev_chunk, feature_columns)

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
                raise ValueError("No dev rows found. Cannot fit Mahalanobis detector.")

            if sample_count <= 0:
                raise ValueError("No dev sample rows collected for Mahalanobis detector.")

            training_sample = reservoir[:sample_count, :]

            print("[TRAINING] Scaling final Mahalanobis reservoir sample")
            training_sample_scaled = scaler.transform(training_sample).astype(
                np.float32,
                copy=False,
            )

            covariance_model = LedoitWolf()

            print("[TRAINING] LedoitWolf covariance fit started")
            covariance_started = perf_counter()
            covariance_model.fit(training_sample_scaled)
            covariance_duration = perf_counter() - covariance_started
            print("[TRAINING] LedoitWolf covariance fit completed")

            mean_vector = covariance_model.location_.astype(np.float32, copy=False)
            precision_matrix = covariance_model.precision_.astype(np.float32, copy=False)

            if self.regularization > 0:
                print("[TRAINING] Applying precision matrix regularization")
                precision_matrix = precision_matrix + (
                    np.eye(precision_matrix.shape[0], dtype=np.float32)
                    * self.regularization
                )

            threshold_payload = self.fit_threshold(
                scaler=scaler,
                feature_columns=feature_columns,
                mean_vector=mean_vector,
                precision_matrix=precision_matrix,
            )

            duration = perf_counter() - started

            payload: Dict[str, object] = {
                "mean_vector": mean_vector,
                "precision_matrix": precision_matrix,
                "threshold": float(threshold_payload["threshold"]),
                "threshold_percentile": float(self.threshold_percentile),
                "scaler": scaler,
                "feature_columns": feature_columns,
                "feature_type": "raw_residual_plus_residual_temporal_features",
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "training_mode": "dev_only_chunked_scaler_large_reservoir_ledoitwolf",
                "total_rows_scanned": int(total_rows_seen),
                "dev_rows_seen": int(total_dev_rows_seen),
                "train_sample_rows": int(sample_count),
                "feature_count": int(len(feature_columns)),
                "regularization": float(self.regularization),
                "covariance_duration_seconds": float(covariance_duration),
                "duration_seconds": float(duration),
                "threshold_payload": threshold_payload,
                "leakage_audit": {
                    "fit_on_dev_only": True,
                    "threshold_fit_on_dev_only": True,
                    "test_rows_used_for_fit": 0,
                    "uses_y_targets": False,
                    "uses_t_degradation_as_input": False,
                    "uses_actual_xs_as_features": False,
                    "uses_predicted_xs_as_features": False,
                },
            }

            print(f"[TRAINING] Saving Mahalanobis params to: {self.params_path}")
            atomic_save_joblib(payload, self.params_path)

            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"scaler", "mean_vector", "precision_matrix"}
            }

            metadata_path = Config.REPORT_DIR / "mahalanobis_training_summary.json"
            atomic_write_json(metadata, metadata_path)

            print("[TRAINING] Mahalanobis fitting completed successfully")
            print(f"[TRAINING] Training summary saved to: {metadata_path}")
            print(f"[TRAINING] Duration seconds: {duration:.2f}")
            print(f"[TRAINING] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Mahalanobis fitted. dev_rows=%s sample_rows=%s features=%s threshold=%s",
                total_dev_rows_seen,
                sample_count,
                len(feature_columns),
                threshold_payload["threshold"],
            )

            del reservoir
            del training_sample
            del training_sample_scaled
            gc.collect()

            return payload

        except Exception as exc:
            print(f"[ERROR] Mahalanobis fitting failed: {exc}")
            logger.exception("Mahalanobis fitting failed.")
            raise RuntimeError("Mahalanobis fitting failed.") from exc

    def fit_threshold(
        self,
        scaler: StandardScaler,
        feature_columns: List[str],
        mean_vector: np.ndarray,
        precision_matrix: np.ndarray,
    ) -> Dict[str, object]:
        """
        Fit Mahalanobis threshold from all dev distances using streaming chunks.

        Returns:
            Threshold payload.
        """
        print("[TRAINING] Fitting Mahalanobis threshold from dev rows only")

        started = perf_counter()

        merge_columns = ["unit_id", "cycle", "split"]
        usecols = merge_columns + feature_columns

        dev_distance_arrays: List[np.ndarray] = []

        total_rows_seen = 0
        dev_rows_seen = 0
        chunk_index = 0

        for chunk in pd.read_csv(
            Config.RESIDUALS_CSV,
            usecols=usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        ):
            chunk_index += 1
            total_rows_seen += len(chunk)

            print("=" * 100)
            print(f"[TRAINING] Threshold chunk #{chunk_index}")
            print(f"[TRAINING] Chunk rows: {len(chunk)}")
            print(f"[TRAINING] Total rows scanned: {total_rows_seen}")

            dev_mask = chunk["split"] == Config.DEV_SPLIT_NAME
            dev_rows = int(dev_mask.sum())
            dev_rows_seen += dev_rows

            print(f"[TRAINING] Dev rows in threshold chunk: {dev_rows}")
            print(f"[TRAINING] Total dev threshold rows: {dev_rows_seen}")

            if dev_rows == 0:
                del chunk
                gc.collect()
                continue

            dev_chunk = chunk.loc[dev_mask, feature_columns]
            x_dev = self._prepare_feature_array(dev_chunk, feature_columns)
            x_dev_scaled = scaler.transform(x_dev).astype(np.float32, copy=False)

            distances = self._calculate_distances(
                values=x_dev_scaled,
                mean_vector=mean_vector,
                precision_matrix=precision_matrix,
            )

            dev_distance_arrays.append(distances.astype(np.float32, copy=True))

            del chunk
            del dev_chunk
            del x_dev
            del x_dev_scaled
            del distances
            gc.collect()

        if dev_rows_seen <= 0:
            raise ValueError("No dev rows found for Mahalanobis threshold fitting.")

        all_dev_distances = np.concatenate(dev_distance_arrays).astype(
            np.float32,
            copy=False,
        )

        threshold = float(np.percentile(all_dev_distances, self.threshold_percentile))

        duration = perf_counter() - started

        payload = {
            "threshold": threshold,
            "threshold_percentile": float(self.threshold_percentile),
            "dev_rows_used": int(dev_rows_seen),
            "total_rows_scanned": int(total_rows_seen),
            "distance_min": float(np.min(all_dev_distances)),
            "distance_max": float(np.max(all_dev_distances)),
            "distance_mean": float(np.mean(all_dev_distances)),
            "distance_std": float(np.std(all_dev_distances)),
            "fit_split": Config.DEV_SPLIT_NAME,
            "test_usage": "score_only",
            "duration_seconds": float(duration),
        }

        print("[TRAINING] Mahalanobis threshold payload:")
        print(payload)

        del all_dev_distances
        del dev_distance_arrays
        gc.collect()

        return payload

    # ==================================================================================
    # Scoring
    # ==================================================================================

    def _score_chunk_distances(
        self,
        chunk: pd.DataFrame,
        scaler: StandardScaler,
        feature_columns: List[str],
        mean_vector: np.ndarray,
        precision_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Mahalanobis distances for one chunk.
        """
        x = self._prepare_feature_array(chunk, feature_columns)
        x_scaled = scaler.transform(x).astype(np.float32, copy=False)

        distances = self._calculate_distances(
            values=x_scaled,
            mean_vector=mean_vector,
            precision_matrix=precision_matrix,
        )

        del x
        del x_scaled

        return distances

    def _first_score_pass(
        self,
        scaler: StandardScaler,
        feature_columns: List[str],
        mean_vector: np.ndarray,
        precision_matrix: np.ndarray,
        threshold: float,
    ) -> Dict[str, object]:
        """
        First scoring pass to calculate global distance min/max and label counts.
        """
        print("[PROGRESS] Starting Mahalanobis first scoring pass")

        merge_columns = ["unit_id", "cycle", "split"]
        usecols = merge_columns + feature_columns

        distance_min = np.inf
        distance_max = -np.inf
        total_rows = 0
        normal_count = 0
        anomaly_count = 0
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
            print(f"[PROGRESS] Mahalanobis first scoring pass chunk #{chunk_index}")
            print(f"[PROGRESS] Chunk rows: {len(chunk)}")
            print(f"[PROGRESS] Total rows scanned: {total_rows}")

            distances = self._score_chunk_distances(
                chunk=chunk,
                scaler=scaler,
                feature_columns=feature_columns,
                mean_vector=mean_vector,
                precision_matrix=precision_matrix,
            )

            distance_min = min(distance_min, float(np.min(distances)))
            distance_max = max(distance_max, float(np.max(distances)))

            labels = (distances >= threshold).astype(np.int8)

            anomaly_count += int(labels.sum())
            normal_count += int(len(labels) - labels.sum())

            print(f"[PROGRESS] Running distance min: {distance_min}")
            print(f"[PROGRESS] Running distance max: {distance_max}")
            print(f"[PROGRESS] Running normal count: {normal_count}")
            print(f"[PROGRESS] Running anomaly count: {anomaly_count}")

            del chunk
            del distances
            del labels
            gc.collect()

        if total_rows <= 0:
            raise ValueError("No rows found during Mahalanobis scoring.")

        return {
            "distance_min": float(distance_min),
            "distance_max": float(distance_max),
            "total_rows": int(total_rows),
            "normal_count": int(normal_count),
            "anomaly_count": int(anomaly_count),
        }

    def _normalize_distances(
        self,
        distances: np.ndarray,
        distance_min: float,
        distance_max: float,
    ) -> np.ndarray:
        """
        Normalize distances to 0-1.
        """
        denominator = max(float(distance_max - distance_min), 1e-12)

        scores = ((distances - distance_min) / denominator).astype(
            np.float32,
            copy=False,
        )

        return np.clip(scores, 0.0, 1.0)

    def score(self) -> int:
        """
        Score dev and test rows using dev-fitted Mahalanobis parameters.

        Returns:
            Number of rows written.
        """
        print("[PROGRESS] Entering MahalanobisDetector.score")

        try:
            started = perf_counter()

            expected_rows = self._count_csv_rows(Config.RESIDUALS_CSV)

            payload = load_joblib_required(self.params_path)

            mean_vector = payload["mean_vector"]
            precision_matrix = payload["precision_matrix"]
            threshold = float(payload["threshold"])
            scaler: StandardScaler = payload["scaler"]
            feature_columns: List[str] = payload["feature_columns"]

            residual_columns = self._read_header_columns(Config.RESIDUALS_CSV)

            required_columns = ["unit_id", "cycle", "split"] + feature_columns

            self._validate_columns(
                available_columns=residual_columns,
                required_columns=required_columns,
                label="residuals.csv",
            )

            score_range = self._first_score_pass(
                scaler=scaler,
                feature_columns=feature_columns,
                mean_vector=mean_vector,
                precision_matrix=precision_matrix,
                threshold=threshold,
            )

            distance_min = float(score_range["distance_min"])
            distance_max = float(score_range["distance_max"])

            print("[PROGRESS] First Mahalanobis scoring pass completed")
            print(f"[PROGRESS] Distance min: {distance_min}")
            print(f"[PROGRESS] Distance max: {distance_max}")

            merge_columns = ["unit_id", "cycle", "split"]
            optional_context_columns = []

            if "gmm_context_id" in residual_columns:
                optional_context_columns.append("gmm_context_id")

            usecols = merge_columns + optional_context_columns + feature_columns

            output_path = self.output_csv
            temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary Mahalanobis output")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            label_counts = {0: 0, 1: 0}

            split_label_counts: Dict[str, Dict[int, int]] = {
                Config.DEV_SPLIT_NAME: {0: 0, 1: 0},
                Config.TEST_SPLIT_NAME: {0: 0, 1: 0},
            }

            distance_sum = 0.0
            score_sum = 0.0

            print("[PROGRESS] Starting Mahalanobis second scoring/write pass")

            for chunk in pd.read_csv(
                Config.RESIDUALS_CSV,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1

                print("=" * 100)
                print(f"[PROGRESS] Mahalanobis second scoring pass chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                distances = self._score_chunk_distances(
                    chunk=chunk,
                    scaler=scaler,
                    feature_columns=feature_columns,
                    mean_vector=mean_vector,
                    precision_matrix=precision_matrix,
                )

                scores = self._normalize_distances(
                    distances=distances,
                    distance_min=distance_min,
                    distance_max=distance_max,
                )

                labels = (distances >= threshold).astype(np.int8)

                result_chunk = chunk[merge_columns].copy()

                if optional_context_columns:
                    result_chunk["gmm_context_id"] = chunk["gmm_context_id"].astype(int).values

                result_chunk["mahalanobis_distance"] = distances
                result_chunk["mahalanobis_score"] = scores
                result_chunk["mahalanobis_anomaly_label"] = labels
                result_chunk["mahalanobis_threshold"] = float(threshold)
                result_chunk["mahalanobis_feature_count"] = int(len(feature_columns))

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

                distance_sum += float(np.sum(distances, dtype=np.float64))
                score_sum += float(np.sum(scores, dtype=np.float64))

                print(f"[PROGRESS] Total Mahalanobis rows written: {total_rows_written}")
                print(f"[PROGRESS] Running label counts: {label_counts}")

                del chunk
                del distances
                del scores
                del labels
                del result_chunk
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All Mahalanobis scoring chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Mahalanobis output row count mismatch. "
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
                "threshold": float(threshold),
                "threshold_percentile": float(payload.get("threshold_percentile", self.threshold_percentile)),
                "distance_min": float(distance_min),
                "distance_max": float(distance_max),
                "distance_mean": float(distance_sum / max(total_rows_written, 1)),
                "score_mean": float(score_sum / max(total_rows_written, 1)),
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
                "dev_rows_seen": payload.get("dev_rows_seen"),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
            }

            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Mahalanobis scores CSV written successfully")
            print(f"[PROGRESS] Summary JSON written to: {self.summary_json}")
            print(f"[PROGRESS] Label counts: {summary['label_counts']}")
            print(f"[PROGRESS] Split label counts: {summary['split_label_counts']}")
            print(f"[PROGRESS] Scoring duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Scoring duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Mahalanobis scoring completed. rows=%s labels=%s",
                total_rows_written,
                summary["label_counts"],
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Mahalanobis scoring failed: {exc}")
            logger.exception("Mahalanobis scoring failed.")
            raise RuntimeError("Mahalanobis scoring failed.") from exc

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run Mahalanobis detector.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering MahalanobisDetector.run")

        try:
            payload = self.fit()
            records_count = self.score()

            response = {
                "status": "success",
                "message": (
                    "Mahalanobis scores generated using dev-fitted parameters. "
                    "Features include raw residuals and residual temporal features."
                ),
                "output_file": str(self.output_csv),
                "params_file": str(self.params_path),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "feature_type": payload["feature_type"],
                "training_mode": payload["training_mode"],
                "train_sample_rows": payload["train_sample_rows"],
                "dev_rows_seen": payload["dev_rows_seen"],
                "feature_count": payload["feature_count"],
                "threshold": payload["threshold"],
            }

            print(f"[PROGRESS] Mahalanobis detector response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Mahalanobis detector stage failed: {exc}")
            logger.exception("Mahalanobis detector stage failed.")
            raise RuntimeError("Mahalanobis detector stage failed.") from exc


def run_mahalanobis_detection() -> Dict[str, object]:
    """
    Execute Mahalanobis detection.
    """
    print("[PROGRESS] Entering run_mahalanobis_detection")

    detector = MahalanobisDetector()
    return detector.run()


if __name__ == "__main__":
    print("[PROGRESS] mahalanobis_detector.py execution started")
    result = run_mahalanobis_detection()
    print("[PROGRESS] mahalanobis_detector.py execution finished successfully")
    print(result)
