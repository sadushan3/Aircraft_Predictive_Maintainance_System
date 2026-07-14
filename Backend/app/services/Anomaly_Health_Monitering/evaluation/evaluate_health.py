"""
Health monitoring evaluation for CA-EDT-AHMA.

Metrics:
- Health state distribution
- Average health index
- Health trend smoothness
- Health deterioration consistency
- Optional health trend summary
- Optional health alert priority summary
- Optional correlation with T health/degradation columns if available

Important:
- This module does not use Y_dev or Y_test.
- This module does not predict RUL.
- T columns, if available, are used only for optional interpretation/correlation,
  not for model training.

Reads:
outputs/Anomaly_Health_Monitering/health_states.csv
outputs/Anomaly_Health_Monitering/health_trends.csv, if available
outputs/Anomaly_Health_Monitering/health_alerts.csv, if available
processed/scaled_features.csv, only for optional T columns if available

Writes:
metrics/evaluate_health.csv
reports/evaluate_health_summary.json

Memory-safe:
- Does not load full CSVs into RAM.
- Reads files in chunks.
- Maintains delta state per split/unit_id.
- Validates aligned row counts for optional files.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "evaluation/evaluate_health.py"
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


class HealthEvaluator:
    """
    Memory-safe health monitoring evaluator.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize health evaluator.

        Args:
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering HealthEvaluator.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "HEALTH_EVALUATION_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("HEALTH_EVALUATION_CHUNK_SIZE must be positive.")

        self.health_states_csv: Path = Config.HEALTH_STATES_CSV

        self.health_trends_csv: Path = getattr(
            Config,
            "HEALTH_TRENDS_CSV",
            Config.OUTPUT_DIR / "health_trends.csv",
        )

        self.health_alerts_csv: Path = getattr(
            Config,
            "HEALTH_ALERTS_CSV",
            Config.OUTPUT_DIR / "health_alerts.csv",
        )

        self.scaled_csv: Path = Config.SCALED_CSV

        self.metrics_csv: Path = Config.METRIC_DIR / "evaluate_health.csv"
        self.summary_json: Path = Config.REPORT_DIR / "evaluate_health_summary.json"

        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Health states CSV: {self.health_states_csv}")
        print(f"[PROGRESS] Health trends CSV: {self.health_trends_csv}")
        print(f"[PROGRESS] Health alerts CSV: {self.health_alerts_csv}")
        print(f"[PROGRESS] Scaled CSV: {self.scaled_csv}")
        print(f"[PROGRESS] Metrics CSV: {self.metrics_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")

    # ==================================================================================
    # File helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows without loading full file.
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
        Read CSV header only.
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
        missing = [
            column
            for column in required_columns
            if column not in available_columns
        ]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _verify_key_alignment(
        self,
        base_chunk: pd.DataFrame,
        other_chunk: pd.DataFrame,
        merge_columns: List[str],
        label: str,
    ) -> None:
        """
        Verify row-key alignment.
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
    # Column builders
    # ==================================================================================

    def _build_health_state_usecols(self, columns: List[str]) -> List[str]:
        """
        Build health_states.csv usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "health_index",
            "health_state",
            "final_anomaly_score",
            "alert_level",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="health_states.csv",
        )

        optional_columns = [
            "gmm_context_id",
            "remaining_health_percentage",
            "residual_trend_score",
            "anomaly_persistence_score",
            "health_state_rank",
            "detector_agreement_ratio",
            "detector_agreement_count",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Health-state evaluation usecols: {usecols}")
        return usecols

    def _build_health_trend_usecols(self, columns: List[str]) -> List[str]:
        """
        Build health_trends.csv usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "health_trend_label",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="health_trends.csv",
        )

        optional_columns = [
            "health_index_rolling_mean",
            "health_index_delta",
            "health_deterioration_score",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Health-trend evaluation usecols: {usecols}")
        return usecols

    def _build_health_alert_usecols(self, columns: List[str]) -> List[str]:
        """
        Build health_alerts.csv usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "inspection_priority",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="health_alerts.csv",
        )

        optional_columns = [
            "inspection_priority_rank",
            "inspection_focus",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Health-alert evaluation usecols: {usecols}")
        return usecols

    def _get_t_columns_from_header(self, columns: List[str]) -> List[str]:
        """
        Get optional T health/degradation parameter columns.

        Accepts common naming patterns:
        - T_*
        - T.*
        - HPT/LPT/fan/LPC/HPC health modifier columns if present
        """
        t_columns: List[str] = []

        for column in columns:
            column_lower = column.lower()

            if column.startswith("T_"):
                t_columns.append(column)
                continue

            if any(
                token in column_lower
                for token in [
                    "eff_mod",
                    "flow_mod",
                    "hpt",
                    "lpt",
                    "hpc",
                    "lpc",
                    "fan_eff",
                    "fan_flow",
                ]
            ):
                if column not in {"split", "unit_id", "cycle"}:
                    t_columns.append(column)

        # Preserve order and remove duplicates
        t_columns = list(dict.fromkeys(t_columns))

        print(f"[PROGRESS] Optional T/health parameter columns found: {t_columns}")
        return t_columns

    # ==================================================================================
    # Metric state
    # ==================================================================================

    def _empty_split_state(self) -> Dict[str, object]:
        """
        Create empty split metric state.
        """
        return {
            "rows": 0,

            "health_sum": 0.0,
            "health_sq_sum": 0.0,
            "health_min": np.inf,
            "health_max": -np.inf,

            "anomaly_score_sum": 0.0,
            "residual_trend_sum": 0.0,
            "persistence_sum": 0.0,

            "delta_abs_sum": 0.0,
            "delta_count": 0,
            "deterioration_nonpositive_count": 0,

            "healthy_count": 0,
            "degrading_count": 0,
            "warning_count": 0,
            "critical_count": 0,

            "stable_trend_count": 0,
            "deteriorating_trend_count": 0,
            "recovering_trend_count": 0,
            "trend_rows": 0,
            "health_deterioration_score_sum": 0.0,
            "health_index_delta_sum": 0.0,

            "routine_priority_count": 0,
            "low_priority_count": 0,
            "medium_priority_count": 0,
            "high_priority_count": 0,
            "alert_rows": 0,

            "detector_agreement_ratio_sum": 0.0,
            "detector_agreement_count_sum": 0.0,
        }

    def _initialize_states(self) -> Dict[str, Dict[str, object]]:
        """
        Initialize dev/test states.
        """
        return {
            Config.DEV_SPLIT_NAME: self._empty_split_state(),
            Config.TEST_SPLIT_NAME: self._empty_split_state(),
        }

    def _initialize_correlation_state(
        self,
        t_columns: List[str],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Initialize streaming correlation state for health_index vs optional T columns.
        """
        correlation_state: Dict[str, Dict[str, Dict[str, float]]] = {}

        for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
            correlation_state[split] = {}

            for column in t_columns:
                correlation_state[split][column] = {
                    "n": 0.0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_x2": 0.0,
                    "sum_y2": 0.0,
                    "sum_xy": 0.0,
                }

        return correlation_state

    # ==================================================================================
    # Delta/smoothness
    # ==================================================================================

    def _update_delta_metrics(
        self,
        metrics_state: Dict[str, Dict[str, object]],
        health_chunk: pd.DataFrame,
        rolling_state: Dict[Tuple[object, object], Dict[str, object]],
    ) -> None:
        """
        Update smoothness and deterioration consistency using streaming deltas.
        """
        for group_key, group_index in health_chunk.groupby(["split", "unit_id"], sort=False).groups.items():
            split = str(group_key[0])

            if split not in metrics_state:
                continue

            values = (
                health_chunk.loc[group_index, "health_index"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )

            group_state = rolling_state.get(
                group_key,
                {
                    "last_health": None,
                },
            )

            deltas = np.zeros(len(values), dtype=np.float32)

            if len(values) > 0:
                if group_state["last_health"] is None:
                    deltas[0] = 0.0
                else:
                    deltas[0] = values[0] - float(group_state["last_health"])

                if len(values) > 1:
                    deltas[1:] = np.diff(values)

                rolling_state[group_key] = {
                    "last_health": float(values[-1]),
                }

            state = metrics_state[split]

            state["delta_abs_sum"] += float(np.sum(np.abs(deltas), dtype=np.float64))
            state["delta_count"] += int(len(deltas))
            state["deterioration_nonpositive_count"] += int(np.sum(deltas <= 0.0))

            del values
            del deltas

    # ==================================================================================
    # State updates
    # ==================================================================================

    def _update_health_state_metrics(
        self,
        metrics_state: Dict[str, Dict[str, object]],
        health_chunk: pd.DataFrame,
    ) -> None:
        """
        Update health-state metrics.
        """
        health_values = (
            health_chunk["health_index"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

        health_values = np.clip(health_values, 0.0, 100.0)

        anomaly_scores = (
            health_chunk["final_anomaly_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

        health_states = health_chunk["health_state"].astype(str).to_numpy(dtype=object)

        for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
            split_mask = health_chunk["split"] == split

            if not split_mask.any():
                continue

            mask_np = split_mask.to_numpy()

            split_health = health_values[mask_np]
            split_anomaly = anomaly_scores[mask_np]
            split_states = health_states[mask_np]

            state = metrics_state[split]

            rows = len(split_health)

            state["rows"] += int(rows)
            state["health_sum"] += float(np.sum(split_health, dtype=np.float64))
            state["health_sq_sum"] += float(np.sum(np.square(split_health), dtype=np.float64))
            state["health_min"] = min(float(state["health_min"]), float(np.min(split_health)))
            state["health_max"] = max(float(state["health_max"]), float(np.max(split_health)))
            state["anomaly_score_sum"] += float(np.sum(split_anomaly, dtype=np.float64))

            state["healthy_count"] += int(np.sum(split_states == "Healthy"))
            state["degrading_count"] += int(np.sum(split_states == "Degrading"))
            state["warning_count"] += int(np.sum(split_states == "Warning"))
            state["critical_count"] += int(np.sum(split_states == "Critical"))

            if "residual_trend_score" in health_chunk.columns:
                state["residual_trend_sum"] += float(
                    health_chunk.loc[split_mask, "residual_trend_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

            if "anomaly_persistence_score" in health_chunk.columns:
                state["persistence_sum"] += float(
                    health_chunk.loc[split_mask, "anomaly_persistence_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

            if "detector_agreement_ratio" in health_chunk.columns:
                state["detector_agreement_ratio_sum"] += float(
                    health_chunk.loc[split_mask, "detector_agreement_ratio"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

            if "detector_agreement_count" in health_chunk.columns:
                state["detector_agreement_count_sum"] += float(
                    health_chunk.loc[split_mask, "detector_agreement_count"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

    def _update_trend_metrics(
        self,
        metrics_state: Dict[str, Dict[str, object]],
        trend_chunk: pd.DataFrame,
    ) -> None:
        """
        Update optional health trend metrics.
        """
        trend_labels = trend_chunk["health_trend_label"].astype(str).to_numpy(dtype=object)

        for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
            split_mask = trend_chunk["split"] == split

            if not split_mask.any():
                continue

            mask_np = split_mask.to_numpy()
            split_labels = trend_labels[mask_np]

            state = metrics_state[split]

            rows = len(split_labels)

            state["trend_rows"] += int(rows)
            state["stable_trend_count"] += int(np.sum(split_labels == "Stable"))
            state["deteriorating_trend_count"] += int(np.sum(split_labels == "Deteriorating"))
            state["recovering_trend_count"] += int(np.sum(split_labels == "Recovering"))

            if "health_deterioration_score" in trend_chunk.columns:
                state["health_deterioration_score_sum"] += float(
                    trend_chunk.loc[split_mask, "health_deterioration_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

            if "health_index_delta" in trend_chunk.columns:
                state["health_index_delta_sum"] += float(
                    trend_chunk.loc[split_mask, "health_index_delta"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum()
                )

    def _update_alert_metrics(
        self,
        metrics_state: Dict[str, Dict[str, object]],
        alert_chunk: pd.DataFrame,
    ) -> None:
        """
        Update optional health alert metrics.
        """
        priorities = alert_chunk["inspection_priority"].astype(str).to_numpy(dtype=object)

        for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
            split_mask = alert_chunk["split"] == split

            if not split_mask.any():
                continue

            mask_np = split_mask.to_numpy()
            split_priorities = priorities[mask_np]

            state = metrics_state[split]

            rows = len(split_priorities)

            state["alert_rows"] += int(rows)
            state["routine_priority_count"] += int(np.sum(split_priorities == "Routine"))
            state["low_priority_count"] += int(np.sum(split_priorities == "Low"))
            state["medium_priority_count"] += int(np.sum(split_priorities == "Medium"))
            state["high_priority_count"] += int(np.sum(split_priorities == "High"))

    def _update_t_correlations(
        self,
        correlation_state: Dict[str, Dict[str, Dict[str, float]]],
        health_chunk: pd.DataFrame,
        scaled_chunk: pd.DataFrame,
        t_columns: List[str],
    ) -> None:
        """
        Update streaming correlations between health_index and optional T columns.
        """
        if not t_columns:
            return

        x_all = (
            health_chunk["health_index"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=False)
        )

        for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
            split_mask = health_chunk["split"] == split

            if not split_mask.any():
                continue

            mask_np = split_mask.to_numpy()

            x = x_all[mask_np]

            for column in t_columns:
                y = (
                    scaled_chunk.loc[split_mask, column]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64, copy=False)
                )

                state = correlation_state[split][column]

                state["n"] += float(len(x))
                state["sum_x"] += float(np.sum(x))
                state["sum_y"] += float(np.sum(y))
                state["sum_x2"] += float(np.sum(x * x))
                state["sum_y2"] += float(np.sum(y * y))
                state["sum_xy"] += float(np.sum(x * y))

    def _finalize_correlation(self, state: Dict[str, float]) -> float:
        """
        Finalize Pearson correlation from streaming sums.
        """
        n = float(state["n"])

        if n <= 1:
            return 0.0

        numerator = (n * state["sum_xy"]) - (state["sum_x"] * state["sum_y"])

        denominator_x = (n * state["sum_x2"]) - (state["sum_x"] ** 2)
        denominator_y = (n * state["sum_y2"]) - (state["sum_y"] ** 2)

        denominator = np.sqrt(max(denominator_x, 0.0) * max(denominator_y, 0.0))

        if denominator <= 1e-12:
            return 0.0

        corr = numerator / denominator

        if not np.isfinite(corr):
            return 0.0

        return float(corr)

    # ==================================================================================
    # Evaluation
    # ==================================================================================

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate health monitoring outputs.

        Returns:
            Health evaluation metrics DataFrame.
        """
        print("[PROGRESS] Entering HealthEvaluator.evaluate")

        try:
            started = perf_counter()

            if not self.health_states_csv.exists():
                raise FileNotFoundError(
                    f"Health states CSV not found: {self.health_states_csv}"
                )

            expected_rows = self._count_csv_rows(self.health_states_csv)

            if expected_rows <= 0:
                raise ValueError("health_states.csv contains zero rows.")

            health_columns = self._read_header_columns(self.health_states_csv)
            health_usecols = self._build_health_state_usecols(health_columns)

            trends_available = False
            trend_usecols: Optional[List[str]] = None

            if self.health_trends_csv.exists():
                trend_rows = self._count_csv_rows(self.health_trends_csv)

                if trend_rows == expected_rows:
                    trend_columns = self._read_header_columns(self.health_trends_csv)
                    trend_usecols = self._build_health_trend_usecols(trend_columns)
                    trends_available = True
                else:
                    print(
                        "[WARNING] health_trends.csv row count mismatch. "
                        "Trend metrics will be skipped."
                    )

            alerts_available = False
            alert_usecols: Optional[List[str]] = None

            if self.health_alerts_csv.exists():
                alert_rows = self._count_csv_rows(self.health_alerts_csv)

                if alert_rows == expected_rows:
                    alert_columns = self._read_header_columns(self.health_alerts_csv)
                    alert_usecols = self._build_health_alert_usecols(alert_columns)
                    alerts_available = True
                else:
                    print(
                        "[WARNING] health_alerts.csv row count mismatch. "
                        "Alert priority metrics will be skipped."
                    )

            scaled_available = False
            scaled_usecols: Optional[List[str]] = None
            t_columns: List[str] = []

            if self.scaled_csv.exists():
                scaled_rows = self._count_csv_rows(self.scaled_csv)

                if scaled_rows == expected_rows:
                    scaled_columns = self._read_header_columns(self.scaled_csv)
                    t_columns = self._get_t_columns_from_header(scaled_columns)

                    if t_columns:
                        scaled_usecols = ["unit_id", "cycle", "split"] + t_columns
                        scaled_available = True
                    else:
                        print("[PROGRESS] No optional T columns found in scaled_features.csv.")
                else:
                    print(
                        "[WARNING] scaled_features.csv row count does not match "
                        "health_states.csv. Optional T correlation will be skipped."
                    )

            metrics_state = self._initialize_states()
            delta_state: Dict[Tuple[object, object], Dict[str, object]] = {}

            correlation_state = self._initialize_correlation_state(t_columns)

            health_iter = pd.read_csv(
                self.health_states_csv,
                usecols=health_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            trend_iter = None
            if trends_available and trend_usecols is not None:
                trend_iter = pd.read_csv(
                    self.health_trends_csv,
                    usecols=trend_usecols,
                    chunksize=self.chunk_size,
                    low_memory=True,
                )

            alert_iter = None
            if alerts_available and alert_usecols is not None:
                alert_iter = pd.read_csv(
                    self.health_alerts_csv,
                    usecols=alert_usecols,
                    chunksize=self.chunk_size,
                    low_memory=True,
                )

            scaled_iter = None
            if scaled_available and scaled_usecols is not None:
                scaled_iter = pd.read_csv(
                    self.scaled_csv,
                    usecols=scaled_usecols,
                    chunksize=self.chunk_size,
                    low_memory=True,
                )

            total_rows_seen = 0
            chunk_index = 0

            print("[PROGRESS] Starting memory-safe health evaluation")

            while True:
                try:
                    health_chunk = next(health_iter)
                except StopIteration:
                    break

                chunk_index += 1
                total_rows_seen += len(health_chunk)

                health_chunk = health_chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Health evaluation chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(health_chunk)}")
                print(f"[PROGRESS] Total rows seen: {total_rows_seen}")

                trend_chunk = None
                alert_chunk = None
                scaled_chunk = None

                if trend_iter is not None:
                    trend_chunk = next(trend_iter).reset_index(drop=True)
                    self._verify_key_alignment(
                        base_chunk=health_chunk,
                        other_chunk=trend_chunk,
                        merge_columns=["unit_id", "cycle", "split"],
                        label="health_trends.csv",
                    )

                if alert_iter is not None:
                    alert_chunk = next(alert_iter).reset_index(drop=True)
                    self._verify_key_alignment(
                        base_chunk=health_chunk,
                        other_chunk=alert_chunk,
                        merge_columns=["unit_id", "cycle", "split"],
                        label="health_alerts.csv",
                    )

                if scaled_iter is not None:
                    scaled_chunk = next(scaled_iter).reset_index(drop=True)
                    self._verify_key_alignment(
                        base_chunk=health_chunk,
                        other_chunk=scaled_chunk,
                        merge_columns=["unit_id", "cycle", "split"],
                        label="scaled_features.csv",
                    )

                self._update_health_state_metrics(
                    metrics_state=metrics_state,
                    health_chunk=health_chunk,
                )

                self._update_delta_metrics(
                    metrics_state=metrics_state,
                    health_chunk=health_chunk,
                    rolling_state=delta_state,
                )

                if trend_chunk is not None:
                    self._update_trend_metrics(
                        metrics_state=metrics_state,
                        trend_chunk=trend_chunk,
                    )

                if alert_chunk is not None:
                    self._update_alert_metrics(
                        metrics_state=metrics_state,
                        alert_chunk=alert_chunk,
                    )

                if scaled_chunk is not None and t_columns:
                    self._update_t_correlations(
                        correlation_state=correlation_state,
                        health_chunk=health_chunk,
                        scaled_chunk=scaled_chunk,
                        t_columns=t_columns,
                    )

                del health_chunk
                del trend_chunk
                del alert_chunk
                del scaled_chunk
                gc.collect()

            if total_rows_seen != expected_rows:
                raise ValueError(
                    "Health evaluation row scan mismatch. "
                    f"seen={total_rows_seen}, expected={expected_rows}"
                )

            records: List[Dict[str, object]] = []

            for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                state = metrics_state[split]
                rows = int(state["rows"])

                if rows <= 0:
                    continue

                health_mean = float(state["health_sum"] / rows)

                health_variance = max(
                    float(state["health_sq_sum"] / rows) - (health_mean ** 2),
                    0.0,
                )

                health_std = float(np.sqrt(health_variance))

                delta_count = max(int(state["delta_count"]), 1)
                mean_abs_delta = float(state["delta_abs_sum"] / delta_count)
                smoothness = float(1.0 / (1.0 + mean_abs_delta))

                deterioration_consistency = float(
                    state["deterioration_nonpositive_count"] / delta_count
                )

                trend_rows = max(int(state["trend_rows"]), 1)
                alert_rows = max(int(state["alert_rows"]), 1)

                record: Dict[str, object] = {
                    "split": split,
                    "row_count": rows,

                    "average_health_index": health_mean,
                    "std_health_index": health_std,
                    "min_health_index": float(state["health_min"]),
                    "max_health_index": float(state["health_max"]),

                    "average_final_anomaly_score": float(state["anomaly_score_sum"] / rows),
                    "average_residual_trend_score": float(state["residual_trend_sum"] / rows),
                    "average_anomaly_persistence_score": float(state["persistence_sum"] / rows),

                    "health_trend_smoothness": smoothness,
                    "mean_abs_health_delta": mean_abs_delta,
                    "health_deterioration_consistency": deterioration_consistency,

                    "healthy_count": int(state["healthy_count"]),
                    "degrading_count": int(state["degrading_count"]),
                    "warning_count": int(state["warning_count"]),
                    "critical_count": int(state["critical_count"]),

                    "healthy_ratio": float(state["healthy_count"] / rows),
                    "degrading_ratio": float(state["degrading_count"] / rows),
                    "warning_ratio": float(state["warning_count"] / rows),
                    "critical_ratio": float(state["critical_count"] / rows),

                    "health_trends_available": bool(trends_available),
                    "stable_trend_ratio": float(state["stable_trend_count"] / trend_rows),
                    "deteriorating_trend_ratio": float(
                        state["deteriorating_trend_count"] / trend_rows
                    ),
                    "recovering_trend_ratio": float(
                        state["recovering_trend_count"] / trend_rows
                    ),
                    "average_health_deterioration_score": float(
                        state["health_deterioration_score_sum"] / trend_rows
                    ),
                    "average_health_index_delta": float(
                        state["health_index_delta_sum"] / trend_rows
                    ),

                    "health_alerts_available": bool(alerts_available),
                    "routine_priority_ratio": float(state["routine_priority_count"] / alert_rows),
                    "low_priority_ratio": float(state["low_priority_count"] / alert_rows),
                    "medium_priority_ratio": float(state["medium_priority_count"] / alert_rows),
                    "high_priority_ratio": float(state["high_priority_count"] / alert_rows),

                    "average_detector_agreement_ratio": float(
                        state["detector_agreement_ratio_sum"] / rows
                    ),
                    "average_detector_agreement_count": float(
                        state["detector_agreement_count_sum"] / rows
                    ),

                    "optional_t_correlation_available": bool(t_columns),
                    "t_correlation_column_count": int(len(t_columns)),

                    "uses_rul_targets": False,
                    "uses_y_dev_y_test": False,
                    "uses_t_dev_t_test_for_training": False,
                    "rul_prediction": False,
                }

                for t_column in t_columns:
                    corr = self._finalize_correlation(
                        correlation_state[split][t_column]
                    )
                    record[f"correlation_health_index_{t_column}"] = corr

                records.append(record)

            metrics_df = pd.DataFrame(records)

            if metrics_df.empty:
                raise ValueError("No health evaluation metrics were generated.")

            duration = perf_counter() - started

            metrics_df["evaluation_mode"] = "memory_safe_health_monitoring_evaluation"
            metrics_df["duration_seconds"] = float(duration)
            metrics_df["duration_minutes"] = float(duration / 60.0)

            print("[PROGRESS] Health evaluation completed")
            print(f"[PROGRESS] Metrics rows: {len(metrics_df)}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info("Health evaluation completed. rows=%s", len(metrics_df))
            return metrics_df

        except Exception as exc:
            print(f"[ERROR] Health evaluation failed: {exc}")
            logger.exception("Health evaluation failed.")
            raise RuntimeError("Health evaluation failed.") from exc

    def summarize(self, metrics_df: pd.DataFrame) -> Dict[str, object]:
        """
        Summarize health evaluation.
        """
        print("[PROGRESS] Entering HealthEvaluator.summarize")

        try:
            split_summary: Dict[str, Dict[str, object]] = {}

            for _, row in metrics_df.iterrows():
                split_summary[str(row["split"])] = {
                    "row_count": int(row["row_count"]),
                    "average_health_index": float(row["average_health_index"]),
                    "healthy_ratio": float(row["healthy_ratio"]),
                    "degrading_ratio": float(row["degrading_ratio"]),
                    "warning_ratio": float(row["warning_ratio"]),
                    "critical_ratio": float(row["critical_ratio"]),
                    "health_trend_smoothness": float(row["health_trend_smoothness"]),
                    "health_deterioration_consistency": float(
                        row["health_deterioration_consistency"]
                    ),
                    "deteriorating_trend_ratio": float(
                        row["deteriorating_trend_ratio"]
                    ),
                    "high_priority_ratio": float(row["high_priority_ratio"]),
                }

            dev_health = split_summary.get(Config.DEV_SPLIT_NAME, {}).get(
                "average_health_index"
            )
            test_health = split_summary.get(Config.TEST_SPLIT_NAME, {}).get(
                "average_health_index"
            )

            health_drop_test_vs_dev = None

            if dev_health is not None and test_health is not None:
                health_drop_test_vs_dev = float(dev_health - test_health)

            dev_critical = split_summary.get(Config.DEV_SPLIT_NAME, {}).get(
                "critical_ratio"
            )
            test_critical = split_summary.get(Config.TEST_SPLIT_NAME, {}).get(
                "critical_ratio"
            )

            test_vs_dev_critical_multiplier = None

            if dev_critical is not None and dev_critical > 0 and test_critical is not None:
                test_vs_dev_critical_multiplier = float(test_critical / dev_critical)

            return {
                "status": "success",
                "evaluation_mode": "memory_safe_health_monitoring_evaluation",
                "average_health_trend_smoothness": float(
                    metrics_df["health_trend_smoothness"].mean()
                ),
                "average_health_deterioration_consistency": float(
                    metrics_df["health_deterioration_consistency"].mean()
                ),
                "average_health_index": float(
                    metrics_df["average_health_index"].mean()
                ),
                "health_drop_test_vs_dev": health_drop_test_vs_dev,
                "test_vs_dev_critical_multiplier": test_vs_dev_critical_multiplier,
                "split_summary": split_summary,
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_t_dev_t_test_for_training": False,
                    "uses_t_only_for_optional_correlation": bool(
                        metrics_df["optional_t_correlation_available"].any()
                    ),
                    "rul_prediction": False,
                },
                "note": (
                    "Health monitoring is evaluated without RUL labels. "
                    "Optional T correlations, if present, are used only for "
                    "interpretability, not for model training."
                ),
            }

        except Exception as exc:
            print(f"[ERROR] Health summary generation failed: {exc}")
            logger.exception("Health summary generation failed.")
            raise RuntimeError("Health summary generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run health evaluation.
        """
        print("[PROGRESS] Entering HealthEvaluator.run")

        try:
            metrics_df = self.evaluate()

            print(f"[PROGRESS] Writing health evaluation metrics to: {self.metrics_csv}")
            atomic_write_csv(metrics_df, self.metrics_csv)

            summary = self.summarize(metrics_df)

            print(f"[PROGRESS] Writing health evaluation summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            response = {
                "status": "success",
                "message": "Health monitoring evaluation completed.",
                "output_file": str(self.metrics_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(len(metrics_df)),
                "metrics": summary,
            }

            print(f"[PROGRESS] Health evaluator response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Health evaluator stage failed: {exc}")
            logger.exception("Health evaluator stage failed.")
            raise RuntimeError("Health evaluator stage failed.") from exc


def run_health_evaluation() -> Dict[str, object]:
    """
    Execute health evaluation.
    """
    print("[PROGRESS] Entering run_health_evaluation")

    evaluator = HealthEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    print("[PROGRESS] evaluate_health.py execution started")
    result = run_health_evaluation()
    print("[PROGRESS] evaluate_health.py execution finished successfully")
    print(result)