"""
Health state classifier for CA-EDT-AHMA.

Rules:
85-100 = Healthy
65-84  = Degrading
40-64  = Warning
0-39   = Critical

Reads:
outputs/Anomaly_Health_Monitering/health_index.csv

Writes:
outputs/Anomaly_Health_Monitering/health_states.csv

Saves:
models/health/health_state_thresholds.json
reports/health_state_summary.json

Memory-safe:
- Does not load full health_index.csv into RAM.
- Reads health_index.csv in chunks.
- Writes to temporary CSV first.
- Replaces final health_states.csv only after successful completion.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "health_index/health_states.py"
)

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
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class HealthStateClassifier:
    """
    Memory-safe health state classifier.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize health state classifier.

        Args:
            chunk_size: Number of health-index rows processed per chunk.
        """
        print("[PROGRESS] Entering HealthStateClassifier.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "HEALTH_STATE_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("HEALTH_STATE_CHUNK_SIZE must be positive.")

        self.input_csv: Path = Config.HEALTH_INDEX_CSV
        self.output_csv: Path = Config.HEALTH_STATES_CSV
        self.thresholds_path: Path = Config.HEALTH_STATE_THRESHOLDS_PATH
        self.summary_json: Path = getattr(
            Config,
            "HEALTH_STATE_SUMMARY_JSON",
            Config.REPORT_DIR / "health_state_summary.json",
        )

        self.thresholds: Dict[str, Dict[str, float]] = {
            "Healthy": {"min": 85.0, "max": 100.0},
            "Degrading": {"min": 65.0, "max": 84.999999},
            "Warning": {"min": 40.0, "max": 64.999999},
            "Critical": {"min": 0.0, "max": 39.999999},
        }

        print(f"[PROGRESS] Input CSV: {self.input_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Thresholds path: {self.thresholds_path}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Thresholds: {self.thresholds}")

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

    def _build_usecols(self, columns: List[str]) -> List[str]:
        """
        Build read_csv usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "health_index",
            "remaining_health_percentage",
            "final_anomaly_score",
            "alert_level",
            "residual_trend_score",
            "anomaly_persistence_score",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="health_index.csv",
        )

        optional_columns = [
            "gmm_context_id",
            "severity_rank",
            "severity_description",
            "detector_agreement_count",
            "detector_agreement_ratio",
            "dominant_detector",
            "residual_anomaly_score",
            "iforest_anomaly_score",
            "mahalanobis_score",
            "lstm_autoencoder_score",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Health state usecols: {usecols}")
        return usecols

    def _classify_health_state_array(self, health_values: np.ndarray) -> np.ndarray:
        """
        Vectorized health-state classification.
        """
        states = np.full(len(health_values), "Critical", dtype=object)

        states[health_values >= 40.0] = "Warning"
        states[health_values >= 65.0] = "Degrading"
        states[health_values >= 85.0] = "Healthy"

        return states

    def _health_state_rank_array(self, states: np.ndarray) -> np.ndarray:
        """
        Convert health state to numeric rank.

        Higher rank = healthier.
        """
        ranks = np.zeros(len(states), dtype=np.int8)

        ranks[states == "Critical"] = 0
        ranks[states == "Warning"] = 1
        ranks[states == "Degrading"] = 2
        ranks[states == "Healthy"] = 3

        return ranks

    def _health_state_explanation_array(self, states: np.ndarray) -> np.ndarray:
        """
        Human-readable health state explanations.
        """
        explanations = np.full(
            len(states),
            "Health index indicates severe anomaly behavior.",
            dtype=object,
        )

        explanations[states == "Warning"] = (
            "Health index indicates significant degradation behavior."
        )
        explanations[states == "Degrading"] = (
            "Health index shows early degradation signs."
        )
        explanations[states == "Healthy"] = (
            "Health index is high and anomaly severity is low."
        )

        return explanations

    # ==================================================================================
    # Main classification
    # ==================================================================================

    def classify_file(self) -> int:
        """
        Classify health states chunk-by-chunk.

        Returns:
            Number of rows written.
        """
        print("[PROGRESS] Entering HealthStateClassifier.classify_file")

        try:
            started = perf_counter()

            if not self.input_csv.exists():
                raise FileNotFoundError(f"Health index CSV not found: {self.input_csv}")

            expected_rows = self._count_csv_rows(self.input_csv)

            if expected_rows <= 0:
                raise ValueError("health_index.csv contains zero rows.")

            columns = self._read_header_columns(self.input_csv)
            usecols = self._build_usecols(columns)

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary health states CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            state_counts = {
                "Healthy": 0,
                "Degrading": 0,
                "Warning": 0,
                "Critical": 0,
            }

            split_state_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {
                    "Healthy": 0,
                    "Degrading": 0,
                    "Warning": 0,
                    "Critical": 0,
                },
                Config.TEST_SPLIT_NAME: {
                    "Healthy": 0,
                    "Degrading": 0,
                    "Warning": 0,
                    "Critical": 0,
                },
            }

            health_sum = 0.0
            health_min = np.inf
            health_max = -np.inf

            split_health_stats: Dict[str, Dict[str, float]] = {
                Config.DEV_SPLIT_NAME: {
                    "rows": 0,
                    "health_sum": 0.0,
                    "health_min": np.inf,
                    "health_max": -np.inf,
                },
                Config.TEST_SPLIT_NAME: {
                    "rows": 0,
                    "health_sum": 0.0,
                    "health_min": np.inf,
                    "health_max": -np.inf,
                },
            }

            print("[PROGRESS] Starting memory-safe health state classification")

            for chunk in pd.read_csv(
                self.input_csv,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                chunk = chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Health state chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                health_values = (
                    chunk["health_index"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                health_values = np.clip(health_values, 0.0, 100.0)

                health_states = self._classify_health_state_array(health_values)
                health_state_ranks = self._health_state_rank_array(health_states)
                explanations = self._health_state_explanation_array(health_states)

                chunk["health_index"] = health_values
                chunk["remaining_health_percentage"] = health_values
                chunk["health_state"] = health_states
                chunk["health_state_rank"] = health_state_ranks
                chunk["health_state_explanation"] = explanations

                chunk["health_state_healthy_min"] = 85.0
                chunk["health_state_degrading_min"] = 65.0
                chunk["health_state_warning_min"] = 40.0
                chunk["health_state_critical_min"] = 0.0

                output_columns = [
                    "unit_id",
                    "cycle",
                    "split",
                ]

                if "gmm_context_id" in chunk.columns:
                    output_columns.append("gmm_context_id")

                output_columns.extend(
                    [
                        "final_anomaly_score",
                        "alert_level",
                        "residual_trend_score",
                        "anomaly_persistence_score",
                        "health_index",
                        "remaining_health_percentage",
                        "health_state",
                        "health_state_rank",
                        "health_state_explanation",
                        "health_state_healthy_min",
                        "health_state_degrading_min",
                        "health_state_warning_min",
                        "health_state_critical_min",
                    ]
                )

                optional_output_columns = [
                    "severity_rank",
                    "severity_description",
                    "detector_agreement_count",
                    "detector_agreement_ratio",
                    "dominant_detector",
                    "residual_anomaly_score",
                    "iforest_anomaly_score",
                    "mahalanobis_score",
                    "lstm_autoencoder_score",
                ]

                for column in optional_output_columns:
                    if column in chunk.columns and column not in output_columns:
                        output_columns.append(column)

                result_chunk = chunk[output_columns]

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_states, unique_counts = np.unique(health_states, return_counts=True)

                for state, count in zip(unique_states, unique_counts):
                    state_counts[str(state)] = state_counts.get(str(state), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_health = result_chunk.loc[split_mask, "health_index"].to_numpy(
                        dtype=np.float32
                    )
                    split_states = result_chunk.loc[split_mask, "health_state"].to_numpy(
                        dtype=object
                    )

                    split_unique, split_counts = np.unique(split_states, return_counts=True)

                    for state, count in zip(split_unique, split_counts):
                        split_state_counts[split][str(state)] = (
                            split_state_counts[split].get(str(state), 0) + int(count)
                        )

                    split_health_stats[split]["rows"] += int(len(split_health))
                    split_health_stats[split]["health_sum"] += float(
                        np.sum(split_health, dtype=np.float64)
                    )
                    split_health_stats[split]["health_min"] = min(
                        float(split_health_stats[split]["health_min"]),
                        float(np.min(split_health)),
                    )
                    split_health_stats[split]["health_max"] = max(
                        float(split_health_stats[split]["health_max"]),
                        float(np.max(split_health)),
                    )

                health_sum += float(np.sum(health_values, dtype=np.float64))
                health_min = min(health_min, float(np.min(health_values)))
                health_max = max(health_max, float(np.max(health_values)))

                print(f"[PROGRESS] Total health state rows written: {total_rows_written}")
                print(f"[PROGRESS] Running health state counts: {state_counts}")

                del chunk
                del result_chunk
                del health_values
                del health_states
                del health_state_ranks
                del explanations
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All health state chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Health state row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final health_states.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            split_summary: Dict[str, Dict[str, object]] = {}

            for split, stats in split_health_stats.items():
                rows = int(stats["rows"])

                if rows <= 0:
                    continue

                counts = split_state_counts[split]

                split_summary[split] = {
                    "rows": rows,
                    "average_health_index": float(stats["health_sum"] / rows),
                    "min_health_index": float(stats["health_min"]),
                    "max_health_index": float(stats["health_max"]),
                    "state_counts": counts,
                    "state_ratios": {
                        state: float(count / rows)
                        for state, count in counts.items()
                    },
                }

            thresholds_payload = {
                "Healthy": {"min": 85.0, "max": 100.0},
                "Degrading": {"min": 65.0, "max": 84.999999},
                "Warning": {"min": 40.0, "max": 64.999999},
                "Critical": {"min": 0.0, "max": 39.999999},
                "higher_is_healthier": True,
                "rul_prediction": False,
                "uses_y_targets": False,
                "uses_t_targets": False,
            }

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "state_counts": state_counts,
                "state_ratios": {
                    state: float(count / max(total_rows_written, 1))
                    for state, count in state_counts.items()
                },
                "average_health_index": float(health_sum / max(total_rows_written, 1)),
                "min_health_index": float(health_min),
                "max_health_index": float(health_max),
                "split_summary": split_summary,
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_predict_rul": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_health_index_only": True,
                },
            }

            print(f"[PROGRESS] Writing health state thresholds to: {self.thresholds_path}")
            atomic_write_json(thresholds_payload, self.thresholds_path)

            print(f"[PROGRESS] Writing health state summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Health state classification completed successfully")
            print(f"[PROGRESS] State counts: {state_counts}")
            print(f"[PROGRESS] Split state counts: {split_state_counts}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Health state classification completed. rows=%s states=%s",
                total_rows_written,
                state_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Health state classification failed: {exc}")
            logger.exception("Health state classification failed.")
            raise RuntimeError("Health state classification failed.") from exc

    def classify(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        In-memory helper for small DataFrames only.

        Kept for compatibility. Production path is classify_file().
        """
        print("[PROGRESS] Entering HealthStateClassifier.classify")

        try:
            result = health_df.copy()

            if "health_index" not in result.columns:
                raise KeyError("health_index is required for health state classification.")

            health_values = (
                result["health_index"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )

            health_values = np.clip(health_values, 0.0, 100.0)

            health_states = self._classify_health_state_array(health_values)

            result["health_index"] = health_values
            result["remaining_health_percentage"] = health_values
            result["health_state"] = health_states
            result["health_state_rank"] = self._health_state_rank_array(health_states)
            result["health_state_explanation"] = self._health_state_explanation_array(
                health_states
            )

            return result

        except Exception as exc:
            logger.exception("Health state classification failed.")
            raise RuntimeError("Health state classification failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run health state classification.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering HealthStateClassifier.run")

        try:
            records_count = self.classify_file()

            response = {
                "status": "success",
                "message": "Health state classification completed.",
                "output_file": str(self.output_csv),
                "threshold_file": str(self.thresholds_path),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Health state classifier response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Health state classifier stage failed: {exc}")
            logger.exception("Health state classifier stage failed.")
            raise RuntimeError("Health state classifier stage failed.") from exc


def run_health_state_classification() -> Dict[str, object]:
    """
    Execute health state classification.
    """
    print("[PROGRESS] Entering run_health_state_classification")

    classifier = HealthStateClassifier()
    return classifier.run()


if __name__ == "__main__":
    print("[PROGRESS] health_states.py execution started")
    result = run_health_state_classification()
    print("[PROGRESS] health_states.py execution finished successfully")
    print(result)