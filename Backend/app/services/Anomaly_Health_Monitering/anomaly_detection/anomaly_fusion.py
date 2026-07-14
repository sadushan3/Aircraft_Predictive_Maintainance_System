"""
Anomaly fusion for CA-EDT-AHMA.

Final 4-detector anomaly fusion.

Inputs:
1. residual_anomaly_scores.csv
2. isolation_forest_scores.csv
3. mahalanobis_scores.csv
4. lstm_autoencoder_scores.csv

Formula:
final_anomaly_score =
    residual_weight          * residual_anomaly_score
  + iforest_weight           * iforest_anomaly_score
  + mahalanobis_weight       * mahalanobis_score
  + lstm_autoencoder_weight  * lstm_autoencoder_score

Default weights from Config.FUSION_WEIGHTS:
residual          = 0.35
iforest           = 0.20
mahalanobis       = 0.20
lstm_autoencoder  = 0.25

Memory-safe:
- Does not load full CSV files into RAM.
- Reads all detector outputs in aligned chunks.
- Validates row counts before processing.
- Validates unit_id/cycle/split alignment per chunk.
- Writes to temporary CSV first.
- Replaces final anomaly_fusion.csv only after successful completion.

Important:
- This file only performs fusion.
- It does not rerun residual, Isolation Forest, Mahalanobis, or LSTM detectors.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "anomaly_detection/anomaly_fusion.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List
import gc
import json
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
        sys.path.insert(0, BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class AnomalyFusion:
    """
    Memory-safe 4-detector anomaly fusion engine.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize anomaly fusion engine.

        Args:
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering AnomalyFusion.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "ANOMALY_FUSION_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("ANOMALY_FUSION_CHUNK_SIZE must be positive.")

        self.output_csv: Path = Config.ANOMALY_FUSION_CSV
        self.weights_path: Path = Config.FUSION_WEIGHTS_PATH
        self.metadata_path: Path = getattr(
            Config,
            "ANOMALY_FUSION_METADATA_PATH",
            Config.ANOMALY_MODEL_DIR / "anomaly_fusion_metadata.json",
        )
        self.summary_json: Path = Config.REPORT_DIR / "anomaly_fusion_summary.json"

        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Residual anomaly CSV: {Config.RESIDUAL_ANOMALY_CSV}")
        print(f"[PROGRESS] Isolation Forest CSV: {Config.IFOREST_CSV}")
        print(f"[PROGRESS] Mahalanobis CSV: {Config.MAHALANOBIS_CSV}")
        print(f"[PROGRESS] LSTM Autoencoder CSV: {Config.LSTM_AUTOENCODER_CSV}")
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
        print(f"[PROGRESS] Counting CSV rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)

        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV columns only.
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
        Validate required columns exist.
        """
        missing = [
            column
            for column in required_columns
            if column not in available_columns
        ]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _build_usecols(
        self,
        available_columns: List[str],
        required_columns: List[str],
        optional_columns: List[str],
    ) -> List[str]:
        """
        Build read_csv usecols from required + available optional columns.
        """
        usecols = list(required_columns)

        for column in optional_columns:
            if column in available_columns and column not in usecols:
                usecols.append(column)

        return usecols

    def _validate_input_files(self) -> int:
        """
        Validate all input files exist and have matching row counts.

        Returns:
            Expected total rows.
        """
        print("[PROGRESS] Validating anomaly fusion input files")

        files = {
            "residual": Config.RESIDUAL_ANOMALY_CSV,
            "iforest": Config.IFOREST_CSV,
            "mahalanobis": Config.MAHALANOBIS_CSV,
            "lstm_autoencoder": Config.LSTM_AUTOENCODER_CSV,
        }

        row_counts: Dict[str, int] = {}

        for label, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f"Required fusion input missing: {label} -> {path}")

            row_counts[label] = self._count_csv_rows(path)

        print(f"[PROGRESS] Fusion input row counts: {row_counts}")

        if len(set(row_counts.values())) != 1:
            raise ValueError(
                "Fusion input row counts do not match. "
                f"row_counts={row_counts}. "
                "Regenerate detector outputs using the same row order."
            )

        expected_rows = next(iter(row_counts.values()))

        if expected_rows <= 0:
            raise ValueError("Fusion input files contain zero rows.")

        print("[PROGRESS] Fusion input row counts validated successfully")
        return int(expected_rows)

    def _verify_key_alignment(
        self,
        base_chunk: pd.DataFrame,
        other_chunk: pd.DataFrame,
        merge_columns: List[str],
        label: str,
    ) -> None:
        """
        Verify row alignment using unit_id, cycle, split.
        """
        if len(base_chunk) != len(other_chunk):
            raise ValueError(
                f"Chunk row count mismatch for {label}: "
                f"base={len(base_chunk)}, other={len(other_chunk)}"
            )

        base_keys = base_chunk[merge_columns].reset_index(drop=True)
        other_keys = other_chunk[merge_columns].reset_index(drop=True)

        if not base_keys.equals(other_keys):
            raise ValueError(
                f"Row-key alignment failed for {label}. "
                "Regenerate detector outputs using the same base residual row order."
            )

    def _resolve_weights(self) -> Dict[str, float]:
        """
        Resolve and validate fusion weights.

        Returns:
            Normalized fusion weights.
        """
        print("[PROGRESS] Resolving anomaly fusion weights")

        weights = dict(
            getattr(
                Config,
                "FUSION_WEIGHTS",
                {
                    "residual": 0.35,
                    "iforest": 0.20,
                    "mahalanobis": 0.20,
                    "lstm_autoencoder": 0.25,
                },
            )
        )

        required_keys = [
            "residual",
            "iforest",
            "mahalanobis",
            "lstm_autoencoder",
        ]

        missing = [key for key in required_keys if key not in weights]

        if missing:
            raise KeyError(
                "Config.FUSION_WEIGHTS is missing required keys: "
                f"{missing}. Required keys: {required_keys}"
            )

        weights = {
            key: float(weights[key])
            for key in required_keys
        }

        if any(value < 0.0 for value in weights.values()):
            raise ValueError(f"Fusion weights cannot be negative: {weights}")

        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError("Fusion weights sum to zero.")

        normalized_weights = {
            key: value / total_weight
            for key, value in weights.items()
        }

        print(f"[PROGRESS] Raw fusion weights: {weights}")
        print(f"[PROGRESS] Normalized fusion weights: {normalized_weights}")
        print(f"[PROGRESS] Normalized weight sum: {sum(normalized_weights.values())}")

        return normalized_weights

    def _classify_alert(self, score: float) -> str:
        """
        Classify final anomaly score into alert level.

        Thresholds are intentionally simple and dashboard-friendly.
        """
        value = float(score)

        if value >= 0.85:
            return "Critical"

        if value >= 0.65:
            return "Warning"

        if value >= 0.40:
            return "Watch"

        return "Normal"

    def _dominant_detector(
        self,
        residual_contribution: np.ndarray,
        iforest_contribution: np.ndarray,
        mahalanobis_contribution: np.ndarray,
        lstm_contribution: np.ndarray,
    ) -> np.ndarray:
        """
        Identify dominant detector contribution per row.
        """
        detector_names = np.asarray(
            [
                "residual",
                "iforest",
                "mahalanobis",
                "lstm_autoencoder",
            ],
            dtype=object,
        )

        contribution_matrix = np.vstack(
            [
                residual_contribution,
                iforest_contribution,
                mahalanobis_contribution,
                lstm_contribution,
            ]
        ).T

        dominant_indices = np.argmax(contribution_matrix, axis=1)
        return detector_names[dominant_indices]

    # ==================================================================================
    # Main fusion
    # ==================================================================================

    def fuse(self) -> int:
        """
        Fuse anomaly scores chunk-by-chunk.

        Returns:
            Number of fused rows written.
        """
        print("[PROGRESS] Entering AnomalyFusion.fuse")

        try:
            started = perf_counter()

            expected_rows = self._validate_input_files()
            weights = self._resolve_weights()

            merge_columns = ["unit_id", "cycle", "split"]

            residual_columns = self._read_header_columns(Config.RESIDUAL_ANOMALY_CSV)
            iforest_columns = self._read_header_columns(Config.IFOREST_CSV)
            mahalanobis_columns = self._read_header_columns(Config.MAHALANOBIS_CSV)
            lstm_columns = self._read_header_columns(Config.LSTM_AUTOENCODER_CSV)

            self._validate_columns(
                residual_columns,
                merge_columns + ["residual_anomaly_score", "residual_alert_level"],
                "residual_anomaly_scores.csv",
            )
            self._validate_columns(
                iforest_columns,
                merge_columns + ["iforest_anomaly_score", "iforest_anomaly_label"],
                "isolation_forest_scores.csv",
            )
            self._validate_columns(
                mahalanobis_columns,
                merge_columns + ["mahalanobis_score", "mahalanobis_anomaly_label"],
                "mahalanobis_scores.csv",
            )
            self._validate_columns(
                lstm_columns,
                merge_columns + ["lstm_autoencoder_score", "lstm_autoencoder_label"],
                "lstm_autoencoder_scores.csv",
            )

            residual_usecols = self._build_usecols(
                available_columns=residual_columns,
                required_columns=merge_columns
                + [
                    "residual_anomaly_score",
                    "residual_alert_level",
                ],
                optional_columns=[
                    "gmm_context_id",
                    "total_abs_residual",
                    "watch_threshold",
                    "warning_threshold",
                    "critical_threshold",
                ],
            )

            iforest_usecols = self._build_usecols(
                available_columns=iforest_columns,
                required_columns=merge_columns
                + [
                    "iforest_anomaly_score",
                    "iforest_anomaly_label",
                ],
                optional_columns=[
                    "iforest_raw_score",
                    "iforest_feature_count",
                ],
            )

            mahalanobis_usecols = self._build_usecols(
                available_columns=mahalanobis_columns,
                required_columns=merge_columns
                + [
                    "mahalanobis_score",
                    "mahalanobis_anomaly_label",
                ],
                optional_columns=[
                    "gmm_context_id",
                    "mahalanobis_distance",
                    "mahalanobis_threshold",
                    "mahalanobis_feature_count",
                ],
            )

            lstm_usecols = self._build_usecols(
                available_columns=lstm_columns,
                required_columns=merge_columns
                + [
                    "lstm_autoencoder_score",
                    "lstm_autoencoder_label",
                ],
                optional_columns=[
                    "gmm_context_id",
                    "lstm_reconstruction_error",
                    "lstm_sequence_ready",
                    "lstm_threshold",
                    "lstm_sequence_length",
                    "lstm_feature_count",
                ],
            )

            residual_iter = pd.read_csv(
                Config.RESIDUAL_ANOMALY_CSV,
                usecols=residual_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            iforest_iter = pd.read_csv(
                Config.IFOREST_CSV,
                usecols=iforest_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            mahalanobis_iter = pd.read_csv(
                Config.MAHALANOBIS_CSV,
                usecols=mahalanobis_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            lstm_iter = pd.read_csv(
                Config.LSTM_AUTOENCODER_CSV,
                usecols=lstm_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            output_path = self.output_csv
            temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary anomaly fusion CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            alert_counts = {
                "Normal": 0,
                "Watch": 0,
                "Warning": 0,
                "Critical": 0,
            }

            split_alert_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {
                    "Normal": 0,
                    "Watch": 0,
                    "Warning": 0,
                    "Critical": 0,
                },
                Config.TEST_SPLIT_NAME: {
                    "Normal": 0,
                    "Watch": 0,
                    "Warning": 0,
                    "Critical": 0,
                },
            }

            score_sum = 0.0
            detector_agreement_sum = 0.0

            print("[PROGRESS] Starting chunked 4-detector anomaly fusion")

            for residual_chunk, iforest_chunk, mahalanobis_chunk, lstm_chunk in zip(
                residual_iter,
                iforest_iter,
                mahalanobis_iter,
                lstm_iter,
            ):
                chunk_index += 1

                print("=" * 100)
                print(f"[PROGRESS] Fusion chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(residual_chunk)}")

                self._verify_key_alignment(
                    base_chunk=residual_chunk,
                    other_chunk=iforest_chunk,
                    merge_columns=merge_columns,
                    label="Isolation Forest scores",
                )
                self._verify_key_alignment(
                    base_chunk=residual_chunk,
                    other_chunk=mahalanobis_chunk,
                    merge_columns=merge_columns,
                    label="Mahalanobis scores",
                )
                self._verify_key_alignment(
                    base_chunk=residual_chunk,
                    other_chunk=lstm_chunk,
                    merge_columns=merge_columns,
                    label="LSTM Autoencoder scores",
                )

                residual_score = (
                    residual_chunk["residual_anomaly_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )
                iforest_score = (
                    iforest_chunk["iforest_anomaly_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )
                mahalanobis_score = (
                    mahalanobis_chunk["mahalanobis_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )
                lstm_score = (
                    lstm_chunk["lstm_autoencoder_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                residual_score = np.clip(residual_score, 0.0, 1.0)
                iforest_score = np.clip(iforest_score, 0.0, 1.0)
                mahalanobis_score = np.clip(mahalanobis_score, 0.0, 1.0)
                lstm_score = np.clip(lstm_score, 0.0, 1.0)

                residual_contribution = weights["residual"] * residual_score
                iforest_contribution = weights["iforest"] * iforest_score
                mahalanobis_contribution = weights["mahalanobis"] * mahalanobis_score
                lstm_contribution = weights["lstm_autoencoder"] * lstm_score

                final_score = (
                    residual_contribution
                    + iforest_contribution
                    + mahalanobis_contribution
                    + lstm_contribution
                )

                final_score = np.clip(final_score, 0.0, 1.0).astype(
                    np.float32,
                    copy=False,
                )

                alert_levels = np.asarray(
                    [self._classify_alert(score) for score in final_score],
                    dtype=object,
                )

                residual_label = (
                    residual_chunk["residual_alert_level"]
                    .astype(str)
                    .ne("Normal")
                    .to_numpy(dtype=np.int8, copy=False)
                )
                iforest_label = (
                    iforest_chunk["iforest_anomaly_label"]
                    .fillna(0)
                    .astype(np.int8)
                    .to_numpy(copy=False)
                )
                mahalanobis_label = (
                    mahalanobis_chunk["mahalanobis_anomaly_label"]
                    .fillna(0)
                    .astype(np.int8)
                    .to_numpy(copy=False)
                )
                lstm_label = (
                    lstm_chunk["lstm_autoencoder_label"]
                    .fillna(0)
                    .astype(np.int8)
                    .to_numpy(copy=False)
                )

                detector_agreement_count = (
                    residual_label
                    + iforest_label
                    + mahalanobis_label
                    + lstm_label
                ).astype(np.int8, copy=False)

                detector_agreement_ratio = (
                    detector_agreement_count.astype(np.float32) / 4.0
                )

                dominant_detector = self._dominant_detector(
                    residual_contribution=residual_contribution,
                    iforest_contribution=iforest_contribution,
                    mahalanobis_contribution=mahalanobis_contribution,
                    lstm_contribution=lstm_contribution,
                )

                result_chunk = residual_chunk[merge_columns].copy()

                if "gmm_context_id" in residual_chunk.columns:
                    result_chunk["gmm_context_id"] = residual_chunk["gmm_context_id"].astype(int).values
                elif "gmm_context_id" in lstm_chunk.columns:
                    result_chunk["gmm_context_id"] = lstm_chunk["gmm_context_id"].astype(int).values
                elif "gmm_context_id" in mahalanobis_chunk.columns:
                    result_chunk["gmm_context_id"] = mahalanobis_chunk["gmm_context_id"].astype(int).values

                result_chunk["residual_anomaly_score"] = residual_score
                result_chunk["iforest_anomaly_score"] = iforest_score
                result_chunk["mahalanobis_score"] = mahalanobis_score
                result_chunk["lstm_autoencoder_score"] = lstm_score

                result_chunk["residual_contribution"] = residual_contribution
                result_chunk["iforest_contribution"] = iforest_contribution
                result_chunk["mahalanobis_contribution"] = mahalanobis_contribution
                result_chunk["lstm_autoencoder_contribution"] = lstm_contribution

                result_chunk["residual_alert_level"] = residual_chunk["residual_alert_level"].values
                result_chunk["iforest_anomaly_label"] = iforest_label
                result_chunk["mahalanobis_anomaly_label"] = mahalanobis_label
                result_chunk["lstm_autoencoder_label"] = lstm_label

                result_chunk["detector_agreement_count"] = detector_agreement_count
                result_chunk["detector_agreement_ratio"] = detector_agreement_ratio
                result_chunk["dominant_detector"] = dominant_detector

                result_chunk["final_anomaly_score"] = final_score
                result_chunk["alert_level"] = alert_levels

                result_chunk["fusion_weight_residual"] = float(weights["residual"])
                result_chunk["fusion_weight_iforest"] = float(weights["iforest"])
                result_chunk["fusion_weight_mahalanobis"] = float(weights["mahalanobis"])
                result_chunk["fusion_weight_lstm_autoencoder"] = float(
                    weights["lstm_autoencoder"]
                )

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_alerts, unique_counts = np.unique(alert_levels, return_counts=True)

                for level, count in zip(unique_alerts, unique_counts):
                    alert_counts[str(level)] = alert_counts.get(str(level), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_alerts = alert_levels[split_mask.to_numpy()]
                    split_unique, split_counts = np.unique(split_alerts, return_counts=True)

                    for level, count in zip(split_unique, split_counts):
                        split_alert_counts[split][str(level)] = (
                            split_alert_counts[split].get(str(level), 0) + int(count)
                        )

                score_sum += float(np.sum(final_score, dtype=np.float64))
                detector_agreement_sum += float(
                    np.sum(detector_agreement_ratio, dtype=np.float64)
                )

                print(f"[PROGRESS] Total fused rows written: {total_rows_written}")
                print(f"[PROGRESS] Running alert counts: {alert_counts}")

                del residual_chunk
                del iforest_chunk
                del mahalanobis_chunk
                del lstm_chunk
                del result_chunk
                del residual_score
                del iforest_score
                del mahalanobis_score
                del lstm_score
                del residual_contribution
                del iforest_contribution
                del mahalanobis_contribution
                del lstm_contribution
                del final_score
                del alert_levels
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All anomaly fusion chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Anomaly fusion output row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final CSV will not be replaced."
                )

            os.replace(temp_output_path, output_path)

            duration = perf_counter() - started

            weights_payload = {
                "weights": weights,
                "formula": (
                    "final_anomaly_score = "
                    f"{weights['residual']:.4f}*residual_anomaly_score + "
                    f"{weights['iforest']:.4f}*iforest_anomaly_score + "
                    f"{weights['mahalanobis']:.4f}*mahalanobis_score + "
                    f"{weights['lstm_autoencoder']:.4f}*lstm_autoencoder_score"
                ),
                "detectors": [
                    "residual_threshold",
                    "isolation_forest",
                    "mahalanobis",
                    "lstm_autoencoder",
                ],
                "score_range": [0.0, 1.0],
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
            }

            summary = {
                "status": "success",
                "output_file": str(output_path),
                "records_count": int(total_rows_written),
                "weights": weights,
                "alert_counts": alert_counts,
                "split_alert_counts": split_alert_counts,
                "final_anomaly_score_mean": float(
                    score_sum / max(total_rows_written, 1)
                ),
                "detector_agreement_ratio_mean": float(
                    detector_agreement_sum / max(total_rows_written, 1)
                ),
                "chunk_size": int(self.chunk_size),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "uses_residual_threshold_output": True,
                    "uses_iforest_output": True,
                    "uses_mahalanobis_output": True,
                    "uses_lstm_autoencoder_output": True,
                    "does_not_refit_detectors": True,
                    "does_not_use_y_targets": True,
                    "does_not_use_t_degradation_as_input": True,
                },
            }

            print(f"[PROGRESS] Writing fusion weights JSON to: {self.weights_path}")
            atomic_write_json(weights_payload, self.weights_path)

            print(f"[PROGRESS] Writing fusion metadata JSON to: {self.metadata_path}")
            atomic_write_json(weights_payload, self.metadata_path)

            print(f"[PROGRESS] Writing fusion summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Anomaly fusion CSV written successfully")
            print(f"[PROGRESS] Alert counts: {alert_counts}")
            print(f"[PROGRESS] Split alert counts: {split_alert_counts}")
            print(f"[PROGRESS] Fusion duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Fusion duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Anomaly fusion completed. rows=%s alerts=%s",
                total_rows_written,
                alert_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Anomaly fusion failed: {exc}")
            logger.exception("Anomaly fusion failed.")
            raise RuntimeError("Anomaly fusion failed.") from exc

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run anomaly fusion only.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering AnomalyFusion.run")

        try:
            records_count = self.fuse()

            response = {
                "status": "success",
                "message": (
                    "4-detector anomaly fusion completed using residual, "
                    "Isolation Forest, Mahalanobis, and LSTM Autoencoder scores."
                ),
                "output_file": str(self.output_csv),
                "weights_file": str(self.weights_path),
                "metadata_file": str(self.metadata_path),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
            }

            print(f"[PROGRESS] Anomaly fusion response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Anomaly fusion stage failed: {exc}")
            logger.exception("Anomaly fusion stage failed.")
            raise RuntimeError("Anomaly fusion stage failed.") from exc


def run_anomaly_fusion() -> Dict[str, object]:
    """
    Execute anomaly fusion only.
    """
    print("[PROGRESS] Entering run_anomaly_fusion")

    service = AnomalyFusion()
    return service.run()


if __name__ == "__main__":
    print("[PROGRESS] anomaly_fusion.py execution started")
    result = run_anomaly_fusion()
    print("[PROGRESS] anomaly_fusion.py execution finished successfully")
    print(result)