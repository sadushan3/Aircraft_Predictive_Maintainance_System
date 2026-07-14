"""
Health trend tracker for CA-EDT-AHMA.

Role:
Track health deterioration trend over time.

Reads:
outputs/Anomaly_Health_Monitering/health_states.csv

Writes:
outputs/Anomaly_Health_Monitering/health_trends.csv

Memory-safe:
- Does not load full health_states.csv into RAM.
- Reads in chunks.
- Maintains rolling state across chunks per split/unit_id.
- Uses two passes:
  1. First pass calculates global max deterioration score.
  2. Second pass writes normalized deterioration score.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "health_monitoring/health_trend_tracker.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
import gc
import os
import sys

import numpy as np
import pandas as pd


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


class HealthTrendTracker:
    """
    Memory-safe health trend tracker by split and unit.
    """

    def __init__(self, window: int | None = None, chunk_size: int = 25_000) -> None:
        print("[PROGRESS] Entering HealthTrendTracker.__init__")

        Config.create_directories()

        self.window = int(
            window
            if window is not None
            else getattr(Config, "HEALTH_TREND_TRACKING_WINDOW", getattr(Config, "ROLLING_WINDOW", 5))
        )

        self.chunk_size = int(
            getattr(Config, "HEALTH_TREND_CHUNK_SIZE", chunk_size)
        )

        if self.window <= 1:
            raise ValueError("HEALTH_TREND_TRACKING_WINDOW must be greater than 1.")

        if self.chunk_size <= 0:
            raise ValueError("HEALTH_TREND_CHUNK_SIZE must be positive.")

        self.delta_threshold = float(
            getattr(Config, "HEALTH_TREND_DELTA_THRESHOLD", 2.0)
        )

        self.input_csv: Path = Config.HEALTH_STATES_CSV
        self.output_csv: Path = getattr(
            Config,
            "HEALTH_TRENDS_CSV",
            Config.OUTPUT_DIR / "health_trends.csv",
        )
        self.summary_json: Path = getattr(
            Config,
            "HEALTH_TREND_SUMMARY_JSON",
            Config.REPORT_DIR / "health_trend_summary.json",
        )

        print(f"[PROGRESS] Input CSV: {self.input_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Rolling window: {self.window}")
        print(f"[PROGRESS] Delta threshold: {self.delta_threshold}")

    def _count_csv_rows(self, path: Path) -> int:
        print(f"[PROGRESS] Counting rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)
        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_columns(self, path: Path) -> List[str]:
        print(f"[PROGRESS] Reading header columns from: {path}")
        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
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
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "health_index",
            "health_state",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="health_states.csv",
        )

        optional_columns = [
            "gmm_context_id",
            "remaining_health_percentage",
            "final_anomaly_score",
            "alert_level",
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

        print(f"[PROGRESS] Health trend usecols: {usecols}")
        return usecols

    def _calculate_chunk_trends(
        self,
        chunk: pd.DataFrame,
        state: Dict[Tuple[object, object], Dict[str, object]],
    ) -> pd.DataFrame:
        result = chunk.copy()

        result["health_index"] = (
            result["health_index"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .clip(0.0, 100.0)
        )

        result["health_index_rolling_mean"] = np.zeros(len(result), dtype=np.float32)
        result["health_index_delta"] = np.zeros(len(result), dtype=np.float32)
        result["health_deterioration_score_raw"] = np.zeros(len(result), dtype=np.float32)

        for group_key, group_index in result.groupby(["split", "unit_id"], sort=False).groups.items():
            health_values = result.loc[group_index, "health_index"].to_numpy(
                dtype=np.float32,
                copy=False,
            )

            group_state = state.get(
                group_key,
                {
                    "health_window": [],
                    "deterioration_window": [],
                    "last_health": None,
                },
            )

            previous_health_window = np.asarray(
                group_state["health_window"],
                dtype=np.float32,
            )

            previous_deterioration_window = np.asarray(
                group_state["deterioration_window"],
                dtype=np.float32,
            )

            combined_health = np.concatenate([previous_health_window, health_values])

            rolling_mean = (
                pd.Series(combined_health)
                .rolling(window=self.window, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )[-len(health_values):]

            deltas = np.zeros(len(health_values), dtype=np.float32)

            if len(health_values) > 0:
                last_health = group_state["last_health"]

                if last_health is None:
                    deltas[0] = 0.0
                else:
                    deltas[0] = health_values[0] - float(last_health)

                if len(health_values) > 1:
                    deltas[1:] = np.diff(health_values)

            deterioration_values = np.where(
                deltas < 0.0,
                np.abs(deltas),
                0.0,
            ).astype(np.float32)

            combined_deterioration = np.concatenate(
                [previous_deterioration_window, deterioration_values]
            )

            rolling_deterioration = (
                pd.Series(combined_deterioration)
                .rolling(window=self.window, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )[-len(health_values):]

            result.loc[group_index, "health_index_rolling_mean"] = rolling_mean
            result.loc[group_index, "health_index_delta"] = deltas
            result.loc[group_index, "health_deterioration_score_raw"] = rolling_deterioration

            keep_count = max(self.window - 1, 1)

            state[group_key] = {
                "health_window": combined_health[-keep_count:].tolist(),
                "deterioration_window": combined_deterioration[-keep_count:].tolist(),
                "last_health": float(health_values[-1]) if len(health_values) > 0 else group_state["last_health"],
            }

            del health_values
            del previous_health_window
            del previous_deterioration_window
            del combined_health
            del rolling_mean
            del deltas
            del deterioration_values
            del combined_deterioration
            del rolling_deterioration

        return result

    def _first_pass_max_deterioration(self, usecols: List[str]) -> Dict[str, object]:
        print("[PROGRESS] Starting first pass for max deterioration")

        state: Dict[Tuple[object, object], Dict[str, object]] = {}

        max_deterioration = 0.0
        total_rows = 0
        chunk_index = 0

        for chunk in pd.read_csv(
            self.input_csv,
            usecols=usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        ):
            chunk_index += 1
            total_rows += len(chunk)

            print("=" * 100)
            print(f"[PROGRESS] Health trend first pass chunk #{chunk_index}")
            print(f"[PROGRESS] Chunk rows: {len(chunk)}")
            print(f"[PROGRESS] Total rows scanned: {total_rows}")

            trend_chunk = self._calculate_chunk_trends(chunk, state)

            chunk_max = float(trend_chunk["health_deterioration_score_raw"].max())

            if chunk_max > max_deterioration:
                max_deterioration = chunk_max

            print(f"[PROGRESS] Running max deterioration raw: {max_deterioration}")

            del chunk
            del trend_chunk
            gc.collect()

        return {
            "rows_seen": int(total_rows),
            "max_deterioration_raw": float(max_deterioration),
        }

    def track_file(self) -> int:
        print("[PROGRESS] Entering HealthTrendTracker.track_file")

        try:
            started = perf_counter()

            if not self.input_csv.exists():
                raise FileNotFoundError(f"Health states CSV not found: {self.input_csv}")

            expected_rows = self._count_csv_rows(self.input_csv)

            if expected_rows <= 0:
                raise ValueError("health_states.csv contains zero rows.")

            columns = self._read_header_columns(self.input_csv)
            usecols = self._build_usecols(columns)

            first_pass = self._first_pass_max_deterioration(usecols)

            if first_pass["rows_seen"] != expected_rows:
                raise ValueError(
                    "First pass row count mismatch. "
                    f"seen={first_pass['rows_seen']}, expected={expected_rows}"
                )

            max_deterioration = float(first_pass["max_deterioration_raw"])
            safe_max_deterioration = max(max_deterioration, 1e-12)

            print(f"[PROGRESS] Final max deterioration raw: {max_deterioration}")

            temp_output_path = self.output_csv.with_suffix(self.output_csv.suffix + ".tmp")
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary health trends CSV")
                temp_output_path.unlink()

            second_pass_state: Dict[Tuple[object, object], Dict[str, object]] = {}

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            trend_counts = {
                "Stable": 0,
                "Deteriorating": 0,
                "Recovering": 0,
            }

            split_trend_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {
                    "Stable": 0,
                    "Deteriorating": 0,
                    "Recovering": 0,
                },
                Config.TEST_SPLIT_NAME: {
                    "Stable": 0,
                    "Deteriorating": 0,
                    "Recovering": 0,
                },
            }

            health_sum = 0.0
            deterioration_sum = 0.0
            delta_sum = 0.0

            split_summary_state: Dict[str, Dict[str, float]] = {
                Config.DEV_SPLIT_NAME: {
                    "rows": 0,
                    "health_sum": 0.0,
                    "deterioration_sum": 0.0,
                    "delta_sum": 0.0,
                },
                Config.TEST_SPLIT_NAME: {
                    "rows": 0,
                    "health_sum": 0.0,
                    "deterioration_sum": 0.0,
                    "delta_sum": 0.0,
                },
            }

            print("[PROGRESS] Starting second pass for normalized health trend output")

            for chunk in pd.read_csv(
                self.input_csv,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                chunk = chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Health trend second pass chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                trend_chunk = self._calculate_chunk_trends(chunk, second_pass_state)

                trend_chunk["health_deterioration_score"] = (
                    trend_chunk["health_deterioration_score_raw"] / safe_max_deterioration
                ).astype(np.float32).clip(0.0, 1.0)

                delta_values = trend_chunk["health_index_delta"].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )

                trend_labels = np.full(len(trend_chunk), "Stable", dtype=object)
                trend_labels[delta_values < -self.delta_threshold] = "Deteriorating"
                trend_labels[delta_values > self.delta_threshold] = "Recovering"

                trend_chunk["health_trend_label"] = trend_labels

                output_columns = [
                    "unit_id",
                    "cycle",
                    "split",
                ]

                if "gmm_context_id" in trend_chunk.columns:
                    output_columns.append("gmm_context_id")

                output_columns.extend(
                    [
                        "health_index",
                        "health_state",
                        "health_index_rolling_mean",
                        "health_index_delta",
                        "health_deterioration_score",
                        "health_trend_label",
                    ]
                )

                optional_output_columns = [
                    "remaining_health_percentage",
                    "final_anomaly_score",
                    "alert_level",
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

                for column in optional_output_columns:
                    if column in trend_chunk.columns and column not in output_columns:
                        output_columns.append(column)

                result_chunk = trend_chunk[output_columns]

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_labels, unique_counts = np.unique(trend_labels, return_counts=True)

                for label, count in zip(unique_labels, unique_counts):
                    trend_counts[str(label)] = trend_counts.get(str(label), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_health = result_chunk.loc[split_mask, "health_index"].to_numpy(dtype=np.float32)
                    split_deterioration = result_chunk.loc[split_mask, "health_deterioration_score"].to_numpy(dtype=np.float32)
                    split_delta = result_chunk.loc[split_mask, "health_index_delta"].to_numpy(dtype=np.float32)
                    split_labels = result_chunk.loc[split_mask, "health_trend_label"].to_numpy(dtype=object)

                    split_summary_state[split]["rows"] += int(len(split_health))
                    split_summary_state[split]["health_sum"] += float(np.sum(split_health, dtype=np.float64))
                    split_summary_state[split]["deterioration_sum"] += float(np.sum(split_deterioration, dtype=np.float64))
                    split_summary_state[split]["delta_sum"] += float(np.sum(split_delta, dtype=np.float64))

                    split_unique, split_counts = np.unique(split_labels, return_counts=True)

                    for label, count in zip(split_unique, split_counts):
                        split_trend_counts[split][str(label)] = (
                            split_trend_counts[split].get(str(label), 0) + int(count)
                        )

                health_sum += float(result_chunk["health_index"].sum())
                deterioration_sum += float(result_chunk["health_deterioration_score"].sum())
                delta_sum += float(result_chunk["health_index_delta"].sum())

                print(f"[PROGRESS] Total health trend rows written: {total_rows_written}")
                print(f"[PROGRESS] Running trend counts: {trend_counts}")

                del chunk
                del trend_chunk
                del result_chunk
                del trend_labels
                del delta_values
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All health trend chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Health trend row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final health_trends.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            split_summary: Dict[str, Dict[str, object]] = {}

            for split, state in split_summary_state.items():
                rows = int(state["rows"])

                if rows <= 0:
                    continue

                split_summary[split] = {
                    "rows": rows,
                    "average_health_index": float(state["health_sum"] / rows),
                    "average_health_deterioration_score": float(
                        state["deterioration_sum"] / rows
                    ),
                    "average_health_index_delta": float(state["delta_sum"] / rows),
                    "trend_counts": split_trend_counts[split],
                    "trend_ratios": {
                        label: float(count / rows)
                        for label, count in split_trend_counts[split].items()
                    },
                }

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "window": int(self.window),
                "delta_threshold": float(self.delta_threshold),
                "max_deterioration_raw": float(max_deterioration),
                "trend_counts": trend_counts,
                "trend_ratios": {
                    label: float(count / max(total_rows_written, 1))
                    for label, count in trend_counts.items()
                },
                "average_health_index": float(health_sum / max(total_rows_written, 1)),
                "average_health_deterioration_score": float(
                    deterioration_sum / max(total_rows_written, 1)
                ),
                "average_health_index_delta": float(
                    delta_sum / max(total_rows_written, 1)
                ),
                "split_summary": split_summary,
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_health_states_only": True,
                },
            }

            print(f"[PROGRESS] Writing health trend summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Health trend tracking completed successfully")
            print(f"[PROGRESS] Trend counts: {trend_counts}")
            print(f"[PROGRESS] Split trend counts: {split_trend_counts}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Health trend tracking completed. rows=%s trends=%s",
                total_rows_written,
                trend_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Health trend tracking failed: {exc}")
            logger.exception("Health trend tracking failed.")
            raise RuntimeError("Health trend tracking failed.") from exc

    def track(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        In-memory helper for small DataFrames only.
        """
        print("[PROGRESS] Entering HealthTrendTracker.track")

        try:
            if "health_index" not in health_df.columns:
                raise KeyError("health_index is required for health trend tracking.")

            result = health_df.copy()
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            result["health_index_rolling_mean"] = (
                result.groupby(["split", "unit_id"])["health_index"]
                .transform(lambda series: series.rolling(self.window, min_periods=1).mean())
            )

            result["health_index_delta"] = (
                result.groupby(["split", "unit_id"])["health_index"]
                .transform(lambda series: series.diff().fillna(0.0))
            )

            result["health_deterioration_score"] = (
                result.groupby(["split", "unit_id"])["health_index_delta"]
                .transform(
                    lambda series: (
                        series.apply(lambda value: abs(value) if value < 0 else 0.0)
                        .rolling(self.window, min_periods=1)
                        .mean()
                    )
                )
            )

            max_deterioration = float(result["health_deterioration_score"].max())

            if max_deterioration > 1e-12:
                result["health_deterioration_score"] = (
                    result["health_deterioration_score"] / max_deterioration
                ).clip(0.0, 1.0)
            else:
                result["health_deterioration_score"] = 0.0

            result["health_trend_label"] = np.where(
                result["health_index_delta"] < -self.delta_threshold,
                "Deteriorating",
                np.where(
                    result["health_index_delta"] > self.delta_threshold,
                    "Recovering",
                    "Stable",
                ),
            )

            return result

        except Exception as exc:
            logger.exception("Health trend tracking failed.")
            raise RuntimeError("Health trend tracking failed.") from exc

    def run(self) -> Dict[str, object]:
        print("[PROGRESS] Entering HealthTrendTracker.run")

        try:
            records_count = self.track_file()

            response = {
                "status": "success",
                "message": "Health trend tracking completed.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Health trend tracker response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Health trend tracker stage failed: {exc}")
            logger.exception("Health trend tracker stage failed.")
            raise RuntimeError("Health trend tracker stage failed.") from exc


def run_health_trend_tracking() -> Dict[str, object]:
    print("[PROGRESS] Entering run_health_trend_tracking")

    tracker = HealthTrendTracker()
    return tracker.run()


if __name__ == "__main__":
    print("[PROGRESS] health_trend_tracker.py execution started")
    result = run_health_trend_tracking()
    print("[PROGRESS] health_trend_tracker.py execution finished successfully")
    print(result)