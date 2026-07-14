"""
Anomaly detection evaluation for CA-EDT-AHMA.

Evaluates anomaly detection outputs without requiring true anomaly labels.

Metrics:
- Alert distribution
- Final anomaly score summary
- Detector agreement summary
- Detection persistence proxy
- Early warning score summary
- Label-safe false-alarm proxy

Important:
- Does not use Y_dev/Y_test.
- Does not use T_dev/T_test.
- Precision, recall, F1, and ROC-AUC are not calculated because true anomaly
  labels are not available.

Reads:
outputs/Anomaly_Health_Monitering/anomaly_fusion.csv
outputs/Anomaly_Health_Monitering/early_warning_scores.csv, if available

Writes:
metrics/evaluate_anomaly.csv
reports/evaluate_anomaly_summary.json

Memory-safe:
- Does not load full CSV files into RAM.
- Reads anomaly_fusion.csv in chunks.
- Reads early_warning_scores.csv in aligned chunks if available.
- Maintains rolling persistence state per split/unit_id.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "evaluation/evaluate_anomaly.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple
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
    atomic_write_csv,
    atomic_write_json,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class AnomalyEvaluator:
    """
    Memory-safe anomaly detection evaluator.

    Evaluates fused anomaly outputs and optional early-warning outputs.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize anomaly evaluator.

        Args:
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering AnomalyEvaluator.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "ANOMALY_EVALUATION_CHUNK_SIZE", chunk_size)
        )

        self.persistence_window = int(
            getattr(Config, "ANOMALY_EVALUATION_PERSISTENCE_WINDOW", 5)
        )

        self.anomaly_threshold = float(
            getattr(Config, "ANOMALY_EVALUATION_ALERT_THRESHOLD", 0.40)
        )

        self.fusion_csv: Path = Config.ANOMALY_FUSION_CSV
        self.early_warning_csv: Path = getattr(
            Config,
            "EARLY_WARNING_CSV",
            Config.OUTPUT_DIR / "early_warning_scores.csv",
        )

        self.metrics_csv: Path = Config.METRIC_DIR / "evaluate_anomaly.csv"
        self.summary_json: Path = Config.REPORT_DIR / "evaluate_anomaly_summary.json"

        if self.chunk_size <= 0:
            raise ValueError("ANOMALY_EVALUATION_CHUNK_SIZE must be positive.")

        if self.persistence_window <= 1:
            raise ValueError("ANOMALY_EVALUATION_PERSISTENCE_WINDOW must be greater than 1.")

        if not (0.0 <= self.anomaly_threshold <= 1.0):
            raise ValueError("ANOMALY_EVALUATION_ALERT_THRESHOLD must be between 0 and 1.")

        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Persistence window: {self.persistence_window}")
        print(f"[PROGRESS] Anomaly threshold: {self.anomaly_threshold}")
        print(f"[PROGRESS] Fusion CSV: {self.fusion_csv}")
        print(f"[PROGRESS] Early warning CSV: {self.early_warning_csv}")
        print(f"[PROGRESS] Metrics CSV: {self.metrics_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")

    # ==================================================================================
    # File/header helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows without loading the full file.

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

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV header columns only.
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

    def _build_fusion_usecols(self, columns: List[str]) -> List[str]:
        """
        Build anomaly fusion usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "final_anomaly_score",
            "alert_level",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="anomaly_fusion.csv",
        )

        optional_columns = [
            "gmm_context_id",
            "detector_agreement_count",
            "detector_agreement_ratio",
            "dominant_detector",
            "residual_anomaly_score",
            "iforest_anomaly_score",
            "mahalanobis_score",
            "lstm_autoencoder_score",
            "severity_rank",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Fusion evaluation usecols: {usecols}")
        return usecols

    def _build_early_warning_usecols(self, columns: List[str]) -> List[str]:
        """
        Build early-warning usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "early_warning_score",
            "early_warning_label",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="early_warning_scores.csv",
        )

        optional_columns = [
            "rolling_anomaly_mean",
            "rolling_anomaly_slope",
            "positive_rolling_anomaly_slope",
            "early_warning_rank",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Early warning evaluation usecols: {usecols}")
        return usecols

    def _verify_key_alignment(
        self,
        base_chunk: pd.DataFrame,
        other_chunk: pd.DataFrame,
        merge_columns: List[str],
        label: str,
    ) -> None:
        """
        Verify row alignment by unit_id/cycle/split.
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
                "Regenerate outputs using the same base row order."
            )

    # ==================================================================================
    # Metric state
    # ==================================================================================

    def _empty_split_state(self) -> Dict[str, object]:
        """
        Initialize per-split metric state.
        """
        return {
            "row_count": 0,
            "normal_count": 0,
            "watch_count": 0,
            "warning_count": 0,
            "critical_count": 0,
            "anomaly_alert_count": 0,
            "score_sum": 0.0,
            "score_sq_sum": 0.0,
            "score_min": np.inf,
            "score_max": -np.inf,
            "persistence_sum": 0.0,
            "false_alarm_proxy_count": 0,
            "detector_agreement_ratio_sum": 0.0,
            "detector_agreement_count_sum": 0.0,
            "dominant_detector_counts": {},
            "residual_score_sum": 0.0,
            "iforest_score_sum": 0.0,
            "mahalanobis_score_sum": 0.0,
            "lstm_score_sum": 0.0,
            "early_warning_score_sum": 0.0,
            "early_warning_row_count": 0,
            "stable_count": 0,
            "watch_risk_count": 0,
            "increasing_risk_count": 0,
            "rolling_anomaly_mean_sum": 0.0,
            "positive_rolling_slope_sum": 0.0,
        }

    def _initialize_states(self) -> Dict[str, Dict[str, object]]:
        """
        Initialize split states.
        """
        return {
            Config.DEV_SPLIT_NAME: self._empty_split_state(),
            Config.TEST_SPLIT_NAME: self._empty_split_state(),
        }

    # ==================================================================================
    # Persistence proxy
    # ==================================================================================

    def _calculate_persistence_for_chunk(
        self,
        chunk: pd.DataFrame,
        state: Dict[Tuple[object, object], Dict[str, object]],
    ) -> np.ndarray:
        """
        Calculate anomaly persistence proxy chunk-by-chunk.

        Persistence proxy:
        rolling mean of (final_anomaly_score >= anomaly_threshold)
        per split/unit_id.

        Args:
            chunk: Fusion chunk.
            state: Rolling state per split/unit_id.

        Returns:
            Persistence values for rows in chunk.
        """
        local_chunk = chunk.reset_index(drop=True)
        persistence_values = np.zeros(len(local_chunk), dtype=np.float32)

        for group_key, group_index in local_chunk.groupby(["split", "unit_id"], sort=False).groups.items():
            group_positions = np.asarray(group_index, dtype=np.int64)
            scores = (
                local_chunk.loc[group_positions, "final_anomaly_score"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )

            flags = (scores >= self.anomaly_threshold).astype(np.float32)

            group_state = state.get(
                group_key,
                {
                    "flag_window": [],
                },
            )

            previous_flags = np.asarray(
                group_state["flag_window"],
                dtype=np.float32,
            )

            combined_flags = np.concatenate([previous_flags, flags])

            rolling_persistence = (
                pd.Series(combined_flags)
                .rolling(window=self.persistence_window, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )[-len(flags):]

            persistence_values[group_positions] = rolling_persistence

            keep_count = max(self.persistence_window - 1, 1)

            state[group_key] = {
                "flag_window": combined_flags[-keep_count:].tolist(),
            }

            del scores
            del flags
            del previous_flags
            del combined_flags
            del rolling_persistence

        return persistence_values

    # ==================================================================================
    # Update metric state
    # ==================================================================================

    def _update_fusion_metrics(
        self,
        metrics_state: Dict[str, Dict[str, object]],
        fusion_chunk: pd.DataFrame,
        persistence_values: np.ndarray,
    ) -> None:
        """
        Update metrics using anomaly fusion chunk.
        """
        scores = (
            fusion_chunk["final_anomaly_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

        scores = np.clip(scores, 0.0, 1.0)

        alert_levels = fusion_chunk["alert_level"].astype(str).to_numpy(dtype=object)

        for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
            split_mask = fusion_chunk["split"] == split

            if not split_mask.any():
                continue

            state = metrics_state[split]

            mask_np = split_mask.to_numpy()
            split_scores = scores[mask_np]
            split_alerts = alert_levels[mask_np]
            split_persistence = persistence_values[mask_np]

            row_count = len(split_scores)

            state["row_count"] += int(row_count)
            state["score_sum"] += float(np.sum(split_scores, dtype=np.float64))
            state["score_sq_sum"] += float(np.sum(np.square(split_scores), dtype=np.float64))
            state["score_min"] = min(float(state["score_min"]), float(np.min(split_scores)))
            state["score_max"] = max(float(state["score_max"]), float(np.max(split_scores)))
            state["persistence_sum"] += float(np.sum(split_persistence, dtype=np.float64))

            normal_count = int(np.sum(split_alerts == "Normal"))
            watch_count = int(np.sum(split_alerts == "Watch"))
            warning_count = int(np.sum(split_alerts == "Warning"))
            critical_count = int(np.sum(split_alerts == "Critical"))

            anomaly_count = watch_count + warning_count + critical_count

            state["normal_count"] += normal_count
            state["watch_count"] += watch_count
            state["warning_count"] += warning_count
            state["critical_count"] += critical_count
            state["anomaly_alert_count"] += anomaly_count

            false_alarm_proxy = (
                np.isin(split_alerts, ["Watch", "Warning", "Critical"])
                & (split_scores < self.anomaly_threshold)
            )

            state["false_alarm_proxy_count"] += int(np.sum(false_alarm_proxy))

            if "detector_agreement_ratio" in fusion_chunk.columns:
                agreement_ratio = (
                    fusion_chunk.loc[split_mask, "detector_agreement_ratio"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )
                state["detector_agreement_ratio_sum"] += float(
                    np.sum(agreement_ratio, dtype=np.float64)
                )

            if "detector_agreement_count" in fusion_chunk.columns:
                agreement_count = (
                    fusion_chunk.loc[split_mask, "detector_agreement_count"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )
                state["detector_agreement_count_sum"] += float(
                    np.sum(agreement_count, dtype=np.float64)
                )

            if "dominant_detector" in fusion_chunk.columns:
                dominant_values = (
                    fusion_chunk.loc[split_mask, "dominant_detector"]
                    .astype(str)
                    .to_numpy(dtype=object)
                )

                dominant_counts = state["dominant_detector_counts"]

                unique_values, unique_counts = np.unique(dominant_values, return_counts=True)

                for value, count in zip(unique_values, unique_counts):
                    dominant_counts[str(value)] = dominant_counts.get(str(value), 0) + int(count)

            if "residual_anomaly_score" in fusion_chunk.columns:
                state["residual_score_sum"] += float(
                    fusion_chunk.loc[split_mask, "residual_anomaly_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

            if "iforest_anomaly_score" in fusion_chunk.columns:
                state["iforest_score_sum"] += float(
                    fusion_chunk.loc[split_mask, "iforest_anomaly_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

            if "mahalanobis_score" in fusion_chunk.columns:
                state["mahalanobis_score_sum"] += float(
                    fusion_chunk.loc[split_mask, "mahalanobis_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

            if "lstm_autoencoder_score" in fusion_chunk.columns:
                state["lstm_score_sum"] += float(
                    fusion_chunk.loc[split_mask, "lstm_autoencoder_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

    def _update_early_warning_metrics(
        self,
        metrics_state: Dict[str, Dict[str, object]],
        early_warning_chunk: pd.DataFrame,
    ) -> None:
        """
        Update metrics using early warning chunk.
        """
        ew_scores = (
            early_warning_chunk["early_warning_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

        ew_scores = np.clip(ew_scores, 0.0, 1.0)

        labels = early_warning_chunk["early_warning_label"].astype(str).to_numpy(dtype=object)

        for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
            split_mask = early_warning_chunk["split"] == split

            if not split_mask.any():
                continue

            state = metrics_state[split]
            mask_np = split_mask.to_numpy()

            split_scores = ew_scores[mask_np]
            split_labels = labels[mask_np]

            state["early_warning_row_count"] += int(len(split_scores))
            state["early_warning_score_sum"] += float(np.sum(split_scores, dtype=np.float64))

            state["stable_count"] += int(np.sum(split_labels == "Stable"))
            state["watch_risk_count"] += int(np.sum(split_labels == "Watch_Risk"))
            state["increasing_risk_count"] += int(np.sum(split_labels == "Increasing_Risk"))

            if "rolling_anomaly_mean" in early_warning_chunk.columns:
                state["rolling_anomaly_mean_sum"] += float(
                    early_warning_chunk.loc[split_mask, "rolling_anomaly_mean"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

            if "positive_rolling_anomaly_slope" in early_warning_chunk.columns:
                state["positive_rolling_slope_sum"] += float(
                    early_warning_chunk.loc[split_mask, "positive_rolling_anomaly_slope"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

    # ==================================================================================
    # Evaluation
    # ==================================================================================

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate anomaly detection outputs.

        Returns:
            Anomaly evaluation metrics DataFrame.
        """
        print("[PROGRESS] Entering AnomalyEvaluator.evaluate")

        try:
            started = perf_counter()

            if not self.fusion_csv.exists():
                raise FileNotFoundError(f"Anomaly fusion CSV not found: {self.fusion_csv}")

            fusion_rows = self._count_csv_rows(self.fusion_csv)

            if fusion_rows <= 0:
                raise ValueError("anomaly_fusion.csv contains zero rows.")

            fusion_columns = self._read_header_columns(self.fusion_csv)
            fusion_usecols = self._build_fusion_usecols(fusion_columns)

            early_warning_available = self.early_warning_csv.exists()
            early_warning_usecols: Optional[List[str]] = None

            if early_warning_available:
                early_warning_rows = self._count_csv_rows(self.early_warning_csv)

                if early_warning_rows != fusion_rows:
                    print(
                        "[WARNING] early_warning_scores.csv row count does not match "
                        "anomaly_fusion.csv. Early warning metrics will be skipped."
                    )
                    early_warning_available = False
                else:
                    early_warning_columns = self._read_header_columns(self.early_warning_csv)
                    early_warning_usecols = self._build_early_warning_usecols(
                        early_warning_columns
                    )

            metrics_state = self._initialize_states()
            persistence_state: Dict[Tuple[object, object], Dict[str, object]] = {}

            fusion_iter = pd.read_csv(
                self.fusion_csv,
                usecols=fusion_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            if early_warning_available and early_warning_usecols is not None:
                early_warning_iter = pd.read_csv(
                    self.early_warning_csv,
                    usecols=early_warning_usecols,
                    chunksize=self.chunk_size,
                    low_memory=True,
                )
            else:
                early_warning_iter = None

            chunk_index = 0
            total_rows_seen = 0

            print("[PROGRESS] Starting memory-safe anomaly evaluation")

            if early_warning_iter is not None:
                iterator = zip(fusion_iter, early_warning_iter)
            else:
                iterator = ((fusion_chunk, None) for fusion_chunk in fusion_iter)

            for fusion_chunk, early_warning_chunk in iterator:
                chunk_index += 1
                total_rows_seen += len(fusion_chunk)

                print("=" * 100)
                print(f"[PROGRESS] Anomaly evaluation chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(fusion_chunk)}")
                print(f"[PROGRESS] Total rows seen: {total_rows_seen}")

                if early_warning_chunk is not None:
                    self._verify_key_alignment(
                        base_chunk=fusion_chunk,
                        other_chunk=early_warning_chunk,
                        merge_columns=["unit_id", "cycle", "split"],
                        label="early_warning_scores.csv",
                    )

                persistence_values = self._calculate_persistence_for_chunk(
                    chunk=fusion_chunk,
                    state=persistence_state,
                )

                self._update_fusion_metrics(
                    metrics_state=metrics_state,
                    fusion_chunk=fusion_chunk,
                    persistence_values=persistence_values,
                )

                if early_warning_chunk is not None:
                    self._update_early_warning_metrics(
                        metrics_state=metrics_state,
                        early_warning_chunk=early_warning_chunk,
                    )

                del fusion_chunk
                del early_warning_chunk
                del persistence_values
                gc.collect()

            if total_rows_seen != fusion_rows:
                raise ValueError(
                    "Anomaly evaluation row scan mismatch. "
                    f"seen={total_rows_seen}, expected={fusion_rows}"
                )

            records: List[Dict[str, object]] = []

            for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                state = metrics_state[split]
                row_count = int(state["row_count"])

                if row_count <= 0:
                    continue

                score_mean = float(state["score_sum"] / row_count)

                score_variance = max(
                    float(state["score_sq_sum"] / row_count) - (score_mean ** 2),
                    0.0,
                )

                score_std = float(np.sqrt(score_variance))

                dominant_counts = state["dominant_detector_counts"]
                dominant_detector = None

                if dominant_counts:
                    dominant_detector = max(
                        dominant_counts.items(),
                        key=lambda item: item[1],
                    )[0]

                ew_count = int(state["early_warning_row_count"])

                record = {
                    "split": split,
                    "row_count": row_count,

                    "normal_count": int(state["normal_count"]),
                    "watch_count": int(state["watch_count"]),
                    "warning_count": int(state["warning_count"]),
                    "critical_count": int(state["critical_count"]),
                    "anomaly_alert_count": int(state["anomaly_alert_count"]),

                    "normal_ratio": float(state["normal_count"] / row_count),
                    "watch_ratio": float(state["watch_count"] / row_count),
                    "warning_ratio": float(state["warning_count"] / row_count),
                    "critical_ratio": float(state["critical_count"] / row_count),
                    "anomaly_alert_ratio": float(state["anomaly_alert_count"] / row_count),

                    "average_final_anomaly_score": score_mean,
                    "std_final_anomaly_score": score_std,
                    "min_final_anomaly_score": float(state["score_min"]),
                    "max_final_anomaly_score": float(state["score_max"]),

                    "average_anomaly_persistence_proxy": float(
                        state["persistence_sum"] / row_count
                    ),
                    "false_alarm_proxy_rate": float(
                        state["false_alarm_proxy_count"] / row_count
                    ),

                    "average_detector_agreement_ratio": float(
                        state["detector_agreement_ratio_sum"] / row_count
                    ),
                    "average_detector_agreement_count": float(
                        state["detector_agreement_count_sum"] / row_count
                    ),
                    "dominant_detector": dominant_detector,

                    "average_residual_anomaly_score": float(
                        state["residual_score_sum"] / row_count
                    ),
                    "average_iforest_anomaly_score": float(
                        state["iforest_score_sum"] / row_count
                    ),
                    "average_mahalanobis_score": float(
                        state["mahalanobis_score_sum"] / row_count
                    ),
                    "average_lstm_autoencoder_score": float(
                        state["lstm_score_sum"] / row_count
                    ),

                    "early_warning_rows": ew_count,
                    "average_early_warning_score": float(
                        state["early_warning_score_sum"] / max(ew_count, 1)
                    ),
                    "stable_count": int(state["stable_count"]),
                    "watch_risk_count": int(state["watch_risk_count"]),
                    "increasing_risk_count": int(state["increasing_risk_count"]),
                    "stable_ratio": float(state["stable_count"] / max(ew_count, 1)),
                    "watch_risk_ratio": float(state["watch_risk_count"] / max(ew_count, 1)),
                    "increasing_risk_ratio": float(
                        state["increasing_risk_count"] / max(ew_count, 1)
                    ),
                    "average_rolling_anomaly_mean": float(
                        state["rolling_anomaly_mean_sum"] / max(ew_count, 1)
                    ),
                    "average_positive_rolling_slope": float(
                        state["positive_rolling_slope_sum"] / max(ew_count, 1)
                    ),

                    "true_labels_available": False,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "roc_auc": None,
                }

                records.append(record)

            metrics_df = pd.DataFrame(records)

            if metrics_df.empty:
                raise ValueError("No anomaly evaluation metrics were generated.")

            duration = perf_counter() - started

            metrics_df["evaluation_mode"] = "memory_safe_label_free_anomaly_evaluation"
            metrics_df["duration_seconds"] = float(duration)
            metrics_df["duration_minutes"] = float(duration / 60.0)

            print("[PROGRESS] Anomaly evaluation completed")
            print(f"[PROGRESS] Metrics rows: {len(metrics_df)}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info("Anomaly evaluation completed. rows=%s", len(metrics_df))
            return metrics_df

        except Exception as exc:
            print(f"[ERROR] Anomaly evaluation failed: {exc}")
            logger.exception("Anomaly evaluation failed.")
            raise RuntimeError("Anomaly evaluation failed.") from exc

    def summarize(self, metrics_df: pd.DataFrame) -> Dict[str, object]:
        """
        Summarize anomaly evaluation.

        Args:
            metrics_df: Metrics DataFrame.

        Returns:
            Summary dictionary.
        """
        print("[PROGRESS] Entering AnomalyEvaluator.summarize")

        try:
            split_summary = {}

            for _, row in metrics_df.iterrows():
                split_summary[str(row["split"])] = {
                    "row_count": int(row["row_count"]),
                    "anomaly_alert_ratio": float(row["anomaly_alert_ratio"]),
                    "watch_ratio": float(row["watch_ratio"]),
                    "warning_ratio": float(row["warning_ratio"]),
                    "critical_ratio": float(row["critical_ratio"]),
                    "average_final_anomaly_score": float(row["average_final_anomaly_score"]),
                    "average_anomaly_persistence_proxy": float(
                        row["average_anomaly_persistence_proxy"]
                    ),
                    "average_early_warning_score": float(
                        row["average_early_warning_score"]
                    ),
                    "watch_risk_ratio": float(row["watch_risk_ratio"]),
                    "increasing_risk_ratio": float(row["increasing_risk_ratio"]),
                    "dominant_detector": row.get("dominant_detector"),
                }

            dev_ratio = split_summary.get(Config.DEV_SPLIT_NAME, {}).get(
                "anomaly_alert_ratio",
                0.0,
            )
            test_ratio = split_summary.get(Config.TEST_SPLIT_NAME, {}).get(
                "anomaly_alert_ratio",
                0.0,
            )

            test_vs_dev_alert_ratio_multiplier = (
                float(test_ratio / dev_ratio)
                if dev_ratio > 0
                else None
            )

            return {
                "status": "success",
                "evaluation_mode": "memory_safe_label_free_anomaly_evaluation",
                "true_labels_available": False,
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_t_dev_t_test": False,
                    "note": (
                        "Y_dev/Y_test are RUL targets and are intentionally not used "
                        "as anomaly labels."
                    ),
                },
                "average_anomaly_alert_ratio": float(
                    metrics_df["anomaly_alert_ratio"].mean()
                ),
                "average_final_anomaly_score": float(
                    metrics_df["average_final_anomaly_score"].mean()
                ),
                "average_early_warning_score": float(
                    metrics_df["average_early_warning_score"].mean()
                ),
                "test_vs_dev_alert_ratio_multiplier": test_vs_dev_alert_ratio_multiplier,
                "split_summary": split_summary,
                "label_based_metrics": {
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "roc_auc": None,
                    "reason": (
                        "External true anomaly labels are required. "
                        "The N-CMAPSS Y groups are RUL targets, not anomaly labels."
                    ),
                },
            }

        except Exception as exc:
            print(f"[ERROR] Anomaly summary generation failed: {exc}")
            logger.exception("Anomaly summary generation failed.")
            raise RuntimeError("Anomaly summary generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run anomaly evaluation.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering AnomalyEvaluator.run")

        try:
            metrics_df = self.evaluate()

            print(f"[PROGRESS] Writing anomaly evaluation metrics to: {self.metrics_csv}")
            atomic_write_csv(metrics_df, self.metrics_csv)

            summary = self.summarize(metrics_df)

            print(f"[PROGRESS] Writing anomaly evaluation summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            response = {
                "status": "success",
                "message": "Anomaly detection evaluation completed.",
                "output_file": str(self.metrics_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(len(metrics_df)),
                "metrics": summary,
            }

            print(f"[PROGRESS] Anomaly evaluator response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Anomaly evaluator stage failed: {exc}")
            logger.exception("Anomaly evaluator stage failed.")
            raise RuntimeError("Anomaly evaluator stage failed.") from exc


def run_anomaly_evaluation() -> Dict[str, object]:
    """
    Execute anomaly evaluation.
    """
    print("[PROGRESS] Entering run_anomaly_evaluation")

    evaluator = AnomalyEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    print("[PROGRESS] evaluate_anomaly.py execution started")
    result = run_anomaly_evaluation()
    print("[PROGRESS] evaluate_anomaly.py execution finished successfully")
    print(result)
