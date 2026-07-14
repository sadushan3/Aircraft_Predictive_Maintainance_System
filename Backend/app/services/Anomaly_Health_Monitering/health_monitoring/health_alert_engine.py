"""
Health alert engine for CA-EDT-AHMA.

Role:
Generate dashboard-ready health alert summaries.

Important:
- This component does not make maintenance decisions.
- It only provides health intelligence and inspection-focus support.
- It does not predict RUL.
- It does not use Y_dev/Y_test.
- It does not use T_dev/T_test.

Reads:
outputs/Anomaly_Health_Monitering/health_states.csv
outputs/Anomaly_Health_Monitering/health_trends.csv, if available

Writes:
outputs/Anomaly_Health_Monitering/health_alerts.csv
reports/health_alert_summary.json

Memory-safe:
- Does not load full health_states.csv into RAM.
- Reads in chunks.
- Optionally reads health_trends.csv in aligned chunks.
- Writes to temporary CSV first.
- Replaces final health_alerts.csv only after successful completion.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "health_monitoring/health_alert_engine.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional
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
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class HealthAlertEngine:
    """
    Memory-safe dashboard health alert engine.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize health alert engine.

        Args:
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering HealthAlertEngine.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "HEALTH_ALERT_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("HEALTH_ALERT_CHUNK_SIZE must be positive.")

        self.health_states_csv: Path = Config.HEALTH_STATES_CSV

        self.health_trends_csv: Path = getattr(
            Config,
            "HEALTH_TRENDS_CSV",
            Config.OUTPUT_DIR / "health_trends.csv",
        )

        self.output_csv: Path = getattr(
            Config,
            "HEALTH_ALERTS_CSV",
            Config.OUTPUT_DIR / "health_alerts.csv",
        )

        self.summary_json: Path = getattr(
            Config,
            "HEALTH_ALERT_SUMMARY_JSON",
            Config.REPORT_DIR / "health_alert_summary.json",
        )

        print(f"[PROGRESS] Health states CSV: {self.health_states_csv}")
        print(f"[PROGRESS] Health trends CSV: {self.health_trends_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")

    # ==================================================================================
    # Helpers
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
            "health_state_explanation",
            "severity_rank",
            "severity_description",
            "detector_agreement_count",
            "detector_agreement_ratio",
            "dominant_detector",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Health alert state usecols: {usecols}")
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

        print(f"[PROGRESS] Health alert trend usecols: {usecols}")
        return usecols

    def _verify_key_alignment(
        self,
        base_chunk: pd.DataFrame,
        other_chunk: pd.DataFrame,
        merge_columns: List[str],
        label: str,
    ) -> None:
        """
        Verify aligned row keys.
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
                "Regenerate health states and trends using the same row order."
            )

    def _inspection_priority_array(
        self,
        health_state: np.ndarray,
        alert_level: np.ndarray,
        trend_label: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Vectorized inspection priority.

        Note:
        This is inspection priority only, not maintenance decision priority.
        """
        priority = np.full(len(health_state), "Routine", dtype=object)

        low_mask = (
            (health_state == "Degrading")
            | (alert_level == "Watch")
        )

        if trend_label is not None:
            low_mask = low_mask | (trend_label == "Deteriorating")

        priority[low_mask] = "Low"

        medium_mask = (
            (health_state == "Warning")
            | (alert_level == "Warning")
        )

        priority[medium_mask] = "Medium"

        high_mask = (
            (health_state == "Critical")
            | (alert_level == "Critical")
        )

        priority[high_mask] = "High"

        return priority

    def _priority_rank_array(self, priority: np.ndarray) -> np.ndarray:
        """
        Convert priority to numeric rank.
        """
        ranks = np.zeros(len(priority), dtype=np.int8)

        ranks[priority == "Routine"] = 0
        ranks[priority == "Low"] = 1
        ranks[priority == "Medium"] = 2
        ranks[priority == "High"] = 3

        return ranks

    def _build_alert_messages(
        self,
        health_state: np.ndarray,
        alert_level: np.ndarray,
        health_index: np.ndarray,
        anomaly_score: np.ndarray,
        priority: np.ndarray,
        trend_label: Optional[np.ndarray],
    ) -> List[str]:
        """
        Build dashboard-ready alert messages.
        """
        messages: List[str] = []

        if trend_label is None:
            trend_values = ["Unknown"] * len(health_state)
        else:
            trend_values = trend_label.tolist()

        for state, alert, health, anomaly, prio, trend in zip(
            health_state,
            alert_level,
            health_index,
            anomaly_score,
            priority,
            trend_values,
        ):
            messages.append(
                (
                    f"Health state is {state} with {alert} anomaly alert. "
                    f"Health index is {float(health):.1f}/100 and final anomaly score is "
                    f"{float(anomaly):.3f}. Health trend is {trend}. "
                    f"Inspection priority is {prio}. "
                    f"This module provides inspection-focus support only; "
                    f"maintenance scheduling decisions belong to the autonomous maintenance supervisor."
                )
            )

        return messages

    def _inspection_focus_array(
        self,
        health_state: np.ndarray,
        alert_level: np.ndarray,
        trend_label: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Generate compact inspection focus category.
        """
        focus = np.full(len(health_state), "Routine monitoring", dtype=object)

        focus[
            (health_state == "Degrading")
            | (alert_level == "Watch")
        ] = "Monitor degradation indicators"

        if trend_label is not None:
            focus[trend_label == "Deteriorating"] = "Track deteriorating health trend"

        focus[
            (health_state == "Warning")
            | (alert_level == "Warning")
        ] = "Inspect affected subsystem indicators"

        focus[
            (health_state == "Critical")
            | (alert_level == "Critical")
        ] = "Immediate inspection focus required"

        return focus

    # ==================================================================================
    # Main generation
    # ==================================================================================

    def generate_file(self) -> int:
        """
        Generate health alerts chunk-by-chunk.

        Returns:
            Number of rows written.
        """
        print("[PROGRESS] Entering HealthAlertEngine.generate_file")

        try:
            started = perf_counter()

            if not self.health_states_csv.exists():
                raise FileNotFoundError(
                    f"Health states CSV not found: {self.health_states_csv}"
                )

            expected_rows = self._count_csv_rows(self.health_states_csv)

            if expected_rows <= 0:
                raise ValueError("health_states.csv contains zero rows.")

            health_state_columns = self._read_header_columns(self.health_states_csv)
            health_state_usecols = self._build_health_state_usecols(health_state_columns)

            trends_available = False
            health_trend_usecols: Optional[List[str]] = None

            if self.health_trends_csv.exists():
                trend_rows = self._count_csv_rows(self.health_trends_csv)

                if trend_rows == expected_rows:
                    trend_columns = self._read_header_columns(self.health_trends_csv)
                    health_trend_usecols = self._build_health_trend_usecols(trend_columns)
                    trends_available = True
                else:
                    print(
                        "[WARNING] health_trends.csv exists but row count does not match. "
                        "Health alert engine will continue without trend features."
                    )

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary health alerts CSV")
                temp_output_path.unlink()

            health_state_iter = pd.read_csv(
                self.health_states_csv,
                usecols=health_state_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            if trends_available and health_trend_usecols is not None:
                health_trend_iter = pd.read_csv(
                    self.health_trends_csv,
                    usecols=health_trend_usecols,
                    chunksize=self.chunk_size,
                    low_memory=True,
                )
                iterator = zip(health_state_iter, health_trend_iter)
            else:
                iterator = ((state_chunk, None) for state_chunk in health_state_iter)

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            priority_counts = {
                "Routine": 0,
                "Low": 0,
                "Medium": 0,
                "High": 0,
            }

            split_priority_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {
                    "Routine": 0,
                    "Low": 0,
                    "Medium": 0,
                    "High": 0,
                },
                Config.TEST_SPLIT_NAME: {
                    "Routine": 0,
                    "Low": 0,
                    "Medium": 0,
                    "High": 0,
                },
            }

            health_sum = 0.0
            anomaly_sum = 0.0

            print("[PROGRESS] Starting memory-safe health alert generation")

            for state_chunk, trend_chunk in iterator:
                chunk_index += 1
                state_chunk = state_chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Health alert chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(state_chunk)}")

                if trend_chunk is not None:
                    trend_chunk = trend_chunk.reset_index(drop=True)

                    self._verify_key_alignment(
                        base_chunk=state_chunk,
                        other_chunk=trend_chunk,
                        merge_columns=["unit_id", "cycle", "split"],
                        label="health_trends.csv",
                    )

                health_index = (
                    state_chunk["health_index"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                health_index = np.clip(health_index, 0.0, 100.0)

                anomaly_score = (
                    state_chunk["final_anomaly_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                anomaly_score = np.clip(anomaly_score, 0.0, 1.0)

                health_state = state_chunk["health_state"].astype(str).to_numpy(dtype=object)
                alert_level = state_chunk["alert_level"].astype(str).to_numpy(dtype=object)

                if trend_chunk is not None:
                    trend_label = trend_chunk["health_trend_label"].astype(str).to_numpy(dtype=object)
                else:
                    trend_label = None

                inspection_priority = self._inspection_priority_array(
                    health_state=health_state,
                    alert_level=alert_level,
                    trend_label=trend_label,
                )

                priority_rank = self._priority_rank_array(inspection_priority)

                inspection_focus = self._inspection_focus_array(
                    health_state=health_state,
                    alert_level=alert_level,
                    trend_label=trend_label,
                )

                alert_messages = self._build_alert_messages(
                    health_state=health_state,
                    alert_level=alert_level,
                    health_index=health_index,
                    anomaly_score=anomaly_score,
                    priority=inspection_priority,
                    trend_label=trend_label,
                )

                result_chunk = state_chunk[
                    [
                        "unit_id",
                        "cycle",
                        "split",
                    ]
                ].copy()

                if "gmm_context_id" in state_chunk.columns:
                    result_chunk["gmm_context_id"] = state_chunk["gmm_context_id"].values

                result_chunk["health_index"] = health_index
                result_chunk["health_state"] = health_state
                result_chunk["final_anomaly_score"] = anomaly_score
                result_chunk["alert_level"] = alert_level

                if trend_chunk is not None:
                    result_chunk["health_trend_label"] = trend_label

                    if "health_index_delta" in trend_chunk.columns:
                        result_chunk["health_index_delta"] = (
                            trend_chunk["health_index_delta"]
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                            .to_numpy(dtype=np.float32, copy=False)
                        )

                    if "health_deterioration_score" in trend_chunk.columns:
                        result_chunk["health_deterioration_score"] = (
                            trend_chunk["health_deterioration_score"]
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                            .to_numpy(dtype=np.float32, copy=False)
                        )

                result_chunk["inspection_priority"] = inspection_priority
                result_chunk["inspection_priority_rank"] = priority_rank
                result_chunk["inspection_focus"] = inspection_focus
                result_chunk["health_alert_message"] = alert_messages
                result_chunk["maintenance_decision"] = "Not generated by this component"
                result_chunk["component_role"] = "Health intelligence and inspection-focus support"

                optional_state_columns = [
                    "remaining_health_percentage",
                    "residual_trend_score",
                    "anomaly_persistence_score",
                    "health_state_rank",
                    "health_state_explanation",
                    "severity_rank",
                    "severity_description",
                    "detector_agreement_count",
                    "detector_agreement_ratio",
                    "dominant_detector",
                ]

                for column in optional_state_columns:
                    if column in state_chunk.columns and column not in result_chunk.columns:
                        result_chunk[column] = state_chunk[column].values

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_priorities, unique_counts = np.unique(
                    inspection_priority,
                    return_counts=True,
                )

                for priority, count in zip(unique_priorities, unique_counts):
                    priority_counts[str(priority)] = (
                        priority_counts.get(str(priority), 0) + int(count)
                    )

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_priorities = inspection_priority[split_mask.to_numpy()]
                    split_unique, split_counts = np.unique(
                        split_priorities,
                        return_counts=True,
                    )

                    for priority, count in zip(split_unique, split_counts):
                        split_priority_counts[split][str(priority)] = (
                            split_priority_counts[split].get(str(priority), 0)
                            + int(count)
                        )

                health_sum += float(np.sum(health_index, dtype=np.float64))
                anomaly_sum += float(np.sum(anomaly_score, dtype=np.float64))

                print(f"[PROGRESS] Total health alert rows written: {total_rows_written}")
                print(f"[PROGRESS] Running priority counts: {priority_counts}")

                del state_chunk
                del trend_chunk
                del result_chunk
                del health_index
                del anomaly_score
                del health_state
                del alert_level
                del trend_label
                del inspection_priority
                del priority_rank
                del inspection_focus
                del alert_messages
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All health alert chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Health alert row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final health_alerts.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "uses_health_trends": bool(trends_available),
                "priority_counts": priority_counts,
                "priority_ratios": {
                    priority: float(count / max(total_rows_written, 1))
                    for priority, count in priority_counts.items()
                },
                "split_priority_counts": split_priority_counts,
                "average_health_index": float(health_sum / max(total_rows_written, 1)),
                "average_final_anomaly_score": float(
                    anomaly_sum / max(total_rows_written, 1)
                ),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_health_states": True,
                    "uses_health_trends_if_available": bool(trends_available),
                },
            }

            print(f"[PROGRESS] Writing health alert summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Health alert generation completed successfully")
            print(f"[PROGRESS] Priority counts: {priority_counts}")
            print(f"[PROGRESS] Split priority counts: {split_priority_counts}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Health alert generation completed. rows=%s priorities=%s",
                total_rows_written,
                priority_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Health alert generation failed: {exc}")
            logger.exception("Health alert generation failed.")
            raise RuntimeError("Health alert generation failed.") from exc

    # ==================================================================================
    # Compatibility helper
    # ==================================================================================

    def generate_alerts(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        In-memory helper for small DataFrames only.

        Production path is generate_file().
        """
        print("[PROGRESS] Entering HealthAlertEngine.generate_alerts")

        try:
            required_columns = [
                "unit_id",
                "cycle",
                "split",
                "health_index",
                "health_state",
                "final_anomaly_score",
                "alert_level",
            ]

            missing = [
                column for column in required_columns if column not in health_df.columns
            ]

            if missing:
                raise KeyError(f"Missing required health alert columns: {missing}")

            result = health_df[required_columns].copy()

            health_index = result["health_index"].to_numpy(dtype=np.float32)
            anomaly_score = result["final_anomaly_score"].to_numpy(dtype=np.float32)
            health_state = result["health_state"].astype(str).to_numpy(dtype=object)
            alert_level = result["alert_level"].astype(str).to_numpy(dtype=object)

            trend_label = None
            if "health_trend_label" in health_df.columns:
                trend_label = health_df["health_trend_label"].astype(str).to_numpy(dtype=object)

            inspection_priority = self._inspection_priority_array(
                health_state=health_state,
                alert_level=alert_level,
                trend_label=trend_label,
            )

            result["inspection_priority"] = inspection_priority
            result["inspection_priority_rank"] = self._priority_rank_array(
                inspection_priority
            )
            result["inspection_focus"] = self._inspection_focus_array(
                health_state=health_state,
                alert_level=alert_level,
                trend_label=trend_label,
            )
            result["health_alert_message"] = self._build_alert_messages(
                health_state=health_state,
                alert_level=alert_level,
                health_index=health_index,
                anomaly_score=anomaly_score,
                priority=inspection_priority,
                trend_label=trend_label,
            )
            result["maintenance_decision"] = "Not generated by this component"

            return result

        except Exception as exc:
            logger.exception("Health alert generation failed.")
            raise RuntimeError("Health alert generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run health alert generation.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering HealthAlertEngine.run")

        try:
            records_count = self.generate_file()

            response = {
                "status": "success",
                "message": "Health alerts generated without maintenance decisions.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Health alert engine response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Health alert engine stage failed: {exc}")
            logger.exception("Health alert engine stage failed.")
            raise RuntimeError("Health alert engine stage failed.") from exc


def run_health_alert_engine() -> Dict[str, object]:
    """
    Execute health alert generation.
    """
    print("[PROGRESS] Entering run_health_alert_engine")

    engine = HealthAlertEngine()
    return engine.run()


if __name__ == "__main__":
    print("[PROGRESS] health_alert_engine.py execution started")
    result = run_health_alert_engine()
    print("[PROGRESS] health_alert_engine.py execution finished successfully")
    print(result)