"""
Root-cause tracker for CA-EDT-AHMA.

Role:
Track whether similar root-cause patterns occurred previously.

Reads:
outputs/Anomaly_Health_Monitering/root_cause_analysis.csv

Writes:
outputs/Anomaly_Health_Monitering/root_cause_memory.csv
reports/root_cause_memory_summary.json

Fast version:
- Memory-safe chunk reading.
- Uses cycle-level recurrence tracking.
- Uses O(1) rolling pattern counters per split/unit_id.
- Does not scan all recent rows for every row.
- Does not count same-cycle rows as previous recurrence.
- Avoids writing long text notes by default.

Important:
- Does not train a model.
- Does not predict RUL.
- Does not make maintenance decisions.
- Does not use Y_dev/Y_test.
- Does not use T_dev/T_test.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "reasoning/root_cause_tracker.py"
)

from collections import Counter, deque
from pathlib import Path
from time import perf_counter
from typing import Any, Deque, Dict, List, Tuple
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


class RootCauseTracker:
    """
    Fast memory-safe recurring root-cause pattern tracker.

    Meaning of similar_pattern_count_recent:
    Number of previous cycles within the lookback window where the same
    root_cause_pattern appeared for the same split/unit_id.

    This is intentionally cycle-level, not row-level.
    """

    def __init__(
        self,
        lookback_window: int | None = None,
        chunk_size: int = 500_000,
    ) -> None:
        """
        Initialize root-cause tracker.

        Args:
            lookback_window:
                Number of recent cycles to inspect for recurring patterns.
            chunk_size:
                Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering RootCauseTracker.__init__")

        Config.create_directories()

        self.lookback_window = int(
            lookback_window
            if lookback_window is not None
            else getattr(Config, "ROOT_CAUSE_LOOKBACK_WINDOW", 20)
        )

        self.chunk_size = int(
            getattr(Config, "ROOT_CAUSE_TRACKER_CHUNK_SIZE", chunk_size)
        )

        if self.lookback_window <= 1:
            raise ValueError("ROOT_CAUSE_LOOKBACK_WINDOW must be greater than 1.")

        if self.chunk_size <= 0:
            raise ValueError("ROOT_CAUSE_TRACKER_CHUNK_SIZE must be positive.")

        self.input_csv: Path = Config.ROOT_CAUSE_CSV

        self.output_csv: Path = getattr(
            Config,
            "ROOT_CAUSE_MEMORY_CSV",
            Config.OUTPUT_DIR / "root_cause_memory.csv",
        )

        self.summary_json: Path = getattr(
            Config,
            "ROOT_CAUSE_MEMORY_SUMMARY_JSON",
            Config.REPORT_DIR / "root_cause_memory_summary.json",
        )

        self.write_memory_note: bool = bool(
            getattr(Config, "ROOT_CAUSE_WRITE_MEMORY_NOTE", False)
        )

        print(f"[PROGRESS] Input CSV: {self.input_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Lookback window: {self.lookback_window}")
        print(f"[PROGRESS] Write memory note: {self.write_memory_note}")

    # ==================================================================================
    # File helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows without loading the full file.
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

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

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
        Build root_cause_analysis.csv usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "root_cause_pattern",
            "top_sensor_1",
            "top_sensor_2",
            "top_sensor_3",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="root_cause_analysis.csv",
        )

        optional_columns = [
            "gmm_context_id",
            "alert_level",
            "final_anomaly_score",
            "health_index",
            "health_state",
            "contribution_1",
            "contribution_2",
            "contribution_3",
            "top3_contribution_sum",
            "inspection_focus",
            "total_abs_residual",
            "detector_agreement_count",
            "detector_agreement_ratio",
            "dominant_detector",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Root-cause tracker usecols: {usecols}")
        return usecols

    # ==================================================================================
    # Cycle-level rolling memory
    # ==================================================================================

    def _new_unit_state(self) -> Dict[str, Any]:
        """
        Create state for one split/unit_id.

        committed_cycles:
            Deque of previous cycles already committed into rolling_counts.

        rolling_counts:
            Counter of how many previous cycles contained each root_cause_pattern.

        pending_cycle:
            Current cycle not yet committed.

        pending_patterns:
            Set of root_cause_pattern labels seen in the current cycle.

        Important:
            Each pattern contributes at most 1 count per cycle.
            This prevents row-level overcounting.
        """
        return {
            "committed_cycles": deque(),
            "rolling_counts": Counter(),
            "pending_cycle": None,
            "pending_patterns": set(),
            "last_cycle": None,
        }

    def _commit_pending_cycle(self, state: Dict[str, Any]) -> None:
        """
        Commit the completed cycle into rolling memory.

        Each root-cause pattern contributes only once per cycle.
        """
        pending_cycle = state["pending_cycle"]
        pending_patterns = state["pending_patterns"]

        if pending_cycle is None or not pending_patterns:
            return

        committed_cycles: Deque[Tuple[float, set[str]]] = state["committed_cycles"]
        rolling_counts: Counter = state["rolling_counts"]

        committed_cycles.append((float(pending_cycle), set(pending_patterns)))

        for pattern in pending_patterns:
            rolling_counts[str(pattern)] += 1

        state["pending_patterns"] = set()

    def _prune_old_cycles(
        self,
        state: Dict[str, Any],
        current_cycle: float,
    ) -> None:
        """
        Remove cycles outside the lookback window.
        """
        min_cycle = float(current_cycle) - float(self.lookback_window)

        committed_cycles: Deque[Tuple[float, set[str]]] = state["committed_cycles"]
        rolling_counts: Counter = state["rolling_counts"]

        while committed_cycles and committed_cycles[0][0] < min_cycle:
            _, old_patterns = committed_cycles.popleft()

            for pattern in old_patterns:
                pattern_key = str(pattern)
                rolling_counts[pattern_key] -= 1

                if rolling_counts[pattern_key] <= 0:
                    del rolling_counts[pattern_key]

    def _track_chunk_fast(
        self,
        chunk: pd.DataFrame,
        memory_state: Dict[Tuple[str, object], Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Fast cycle-level recurrence tracker.

        For each row:
        - Query rolling previous-cycle counts for the same pattern.
        - Do not count same-cycle rows as previous recurrence.
        - Add current pattern to the pending current-cycle set.
        """
        result = chunk.copy()

        unit_values = result["unit_id"].to_numpy(copy=False)
        cycle_values = result["cycle"].to_numpy(dtype=np.float64, copy=False)
        split_values = result["split"].astype(str).to_numpy(dtype=object)
        pattern_values = result["root_cause_pattern"].astype(str).to_numpy(dtype=object)

        recurrence_counts = np.zeros(len(result), dtype=np.int32)
        recurrence_statuses = np.empty(len(result), dtype=object)

        non_monotonic_count = 0

        for index, (unit_id, cycle, split, pattern) in enumerate(
            zip(unit_values, cycle_values, split_values, pattern_values)
        ):
            key = (str(split), unit_id)
            state = memory_state.setdefault(key, self._new_unit_state())

            last_cycle = state["last_cycle"]

            if last_cycle is not None and float(cycle) < float(last_cycle):
                non_monotonic_count += 1

            pending_cycle = state["pending_cycle"]

            if pending_cycle is None:
                state["pending_cycle"] = float(cycle)

            elif float(cycle) != float(pending_cycle):
                self._commit_pending_cycle(state)
                state["pending_cycle"] = float(cycle)

            self._prune_old_cycles(state, float(cycle))

            pattern_key = str(pattern)
            count = int(state["rolling_counts"].get(pattern_key, 0))

            recurrence_counts[index] = count

            if count >= 2:
                recurrence_statuses[index] = "Recurring"
            elif count == 1:
                recurrence_statuses[index] = "Previously_Seen"
            else:
                recurrence_statuses[index] = "New_or_Isolated"

            state["pending_patterns"].add(pattern_key)
            state["last_cycle"] = float(cycle)

        result["similar_pattern_count_recent"] = recurrence_counts
        result["root_cause_recurrence_status"] = recurrence_statuses
        result["root_cause_lookback_window"] = int(self.lookback_window)
        result["root_cause_tracker_non_monotonic_cycle_flag"] = (
            "Check_Order" if non_monotonic_count > 0 else "OK"
        )

        if self.write_memory_note:
            result["root_cause_memory_note"] = (
                "Recent previous cycle count="
                + result["similar_pattern_count_recent"].astype(str)
                + " within "
                + str(self.lookback_window)
                + " cycles."
            )

        return result

    # ==================================================================================
    # Main tracking
    # ==================================================================================

    def track_file(self) -> int:
        """
        Track root-cause recurrence from root_cause_analysis.csv.
        """
        print("[PROGRESS] Entering RootCauseTracker.track_file")

        try:
            started = perf_counter()

            if not self.input_csv.exists():
                raise FileNotFoundError(f"Root-cause CSV not found: {self.input_csv}")

            expected_rows = self._count_csv_rows(self.input_csv)

            if expected_rows <= 0:
                raise ValueError("root_cause_analysis.csv contains zero rows.")

            columns = self._read_header_columns(self.input_csv)
            usecols = self._build_usecols(columns)

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary root-cause memory CSV")
                temp_output_path.unlink()

            memory_state: Dict[Tuple[str, object], Dict[str, Any]] = {}

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            status_counts = {
                "New_or_Isolated": 0,
                "Previously_Seen": 0,
                "Recurring": 0,
            }

            split_status_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {
                    "New_or_Isolated": 0,
                    "Previously_Seen": 0,
                    "Recurring": 0,
                },
                Config.TEST_SPLIT_NAME: {
                    "New_or_Isolated": 0,
                    "Previously_Seen": 0,
                    "Recurring": 0,
                },
            }

            recurring_pattern_counts: Dict[str, int] = {}
            max_recent_count = 0
            recent_count_sum = 0.0
            non_monotonic_chunk_count = 0

            print("[PROGRESS] Starting FAST cycle-level root-cause recurrence tracking")

            for chunk in pd.read_csv(
                self.input_csv,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                chunk = chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Root-cause tracker chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                tracked_chunk = self._track_chunk_fast(
                    chunk=chunk,
                    memory_state=memory_state,
                )

                output_columns = [
                    "unit_id",
                    "cycle",
                    "split",
                ]

                if "gmm_context_id" in tracked_chunk.columns:
                    output_columns.append("gmm_context_id")

                output_columns.extend(
                    [
                        "root_cause_pattern",
                        "top_sensor_1",
                        "top_sensor_2",
                        "top_sensor_3",
                        "similar_pattern_count_recent",
                        "root_cause_recurrence_status",
                        "root_cause_lookback_window",
                        "root_cause_tracker_non_monotonic_cycle_flag",
                    ]
                )

                if self.write_memory_note and "root_cause_memory_note" in tracked_chunk.columns:
                    output_columns.append("root_cause_memory_note")

                optional_output_columns = [
                    "alert_level",
                    "final_anomaly_score",
                    "health_index",
                    "health_state",
                    "contribution_1",
                    "contribution_2",
                    "contribution_3",
                    "top3_contribution_sum",
                    "inspection_focus",
                    "total_abs_residual",
                    "detector_agreement_count",
                    "detector_agreement_ratio",
                    "dominant_detector",
                ]

                for column in optional_output_columns:
                    if column in tracked_chunk.columns and column not in output_columns:
                        output_columns.append(column)

                result_chunk = tracked_chunk[output_columns]

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                statuses = result_chunk[
                    "root_cause_recurrence_status"
                ].astype(str).to_numpy(dtype=object)

                unique_statuses, unique_counts = np.unique(
                    statuses,
                    return_counts=True,
                )

                for status, count in zip(unique_statuses, unique_counts):
                    status_counts[str(status)] = (
                        status_counts.get(str(status), 0) + int(count)
                    )

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_statuses = statuses[split_mask.to_numpy()]
                    split_unique, split_counts = np.unique(
                        split_statuses,
                        return_counts=True,
                    )

                    for status, count in zip(split_unique, split_counts):
                        split_status_counts[split][str(status)] = (
                            split_status_counts[split].get(str(status), 0)
                            + int(count)
                        )

                recurring_mask = (
                    result_chunk["root_cause_recurrence_status"] == "Recurring"
                )

                if recurring_mask.any():
                    recurring_patterns = (
                        result_chunk.loc[recurring_mask, "root_cause_pattern"]
                        .astype(str)
                        .to_numpy(dtype=object)
                    )

                    unique_patterns, pattern_counts = np.unique(
                        recurring_patterns,
                        return_counts=True,
                    )

                    for pattern, count in zip(unique_patterns, pattern_counts):
                        recurring_pattern_counts[str(pattern)] = (
                            recurring_pattern_counts.get(str(pattern), 0)
                            + int(count)
                        )

                recent_counts = result_chunk[
                    "similar_pattern_count_recent"
                ].to_numpy(dtype=np.float32)

                if len(recent_counts) > 0:
                    max_recent_count = max(max_recent_count, int(np.max(recent_counts)))
                    recent_count_sum += float(np.sum(recent_counts, dtype=np.float64))

                non_monotonic_flags = (
                    result_chunk["root_cause_tracker_non_monotonic_cycle_flag"]
                    .astype(str)
                    .eq("Check_Order")
                    .sum()
                )

                if non_monotonic_flags > 0:
                    non_monotonic_chunk_count += 1

                print(f"[PROGRESS] Total root-cause memory rows written: {total_rows_written}")
                print(f"[PROGRESS] Running recurrence status counts: {status_counts}")

                del chunk
                del tracked_chunk
                del result_chunk
                del statuses
                del recent_counts
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All root-cause tracker chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Root-cause memory row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final root_cause_memory.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "lookback_window": int(self.lookback_window),
                "write_memory_note": bool(self.write_memory_note),
                "recurrence_status_counts": status_counts,
                "recurrence_status_ratios": {
                    status: float(count / max(total_rows_written, 1))
                    for status, count in status_counts.items()
                },
                "split_status_counts": split_status_counts,
                "recurring_pattern_counts": recurring_pattern_counts,
                "max_similar_pattern_count_recent": int(max_recent_count),
                "average_similar_pattern_count_recent": float(
                    recent_count_sum / max(total_rows_written, 1)
                ),
                "tracked_unit_context_count": int(len(memory_state)),
                "non_monotonic_chunk_count": int(non_monotonic_chunk_count),
                "chunk_size": int(self.chunk_size),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "tracking_mode": "fast_cycle_level_counter_tracker",
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_root_cause_analysis_only": True,
                },
            }

            print(f"[PROGRESS] Writing root-cause memory summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Root-cause recurrence tracking completed successfully")
            print(f"[PROGRESS] Recurrence status counts: {status_counts}")
            print(f"[PROGRESS] Split status counts: {split_status_counts}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Root-cause tracking completed. rows=%s statuses=%s",
                total_rows_written,
                status_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Root-cause tracking failed: {exc}")
            logger.exception("Root-cause tracking failed.")
            raise RuntimeError("Root-cause tracking failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run root-cause tracking.
        """
        print("[PROGRESS] Entering RootCauseTracker.run")

        try:
            records_count = self.track_file()

            response = {
                "status": "success",
                "message": "Root-cause recurrence tracking completed.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Root-cause tracker response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Root-cause tracker stage failed: {exc}")
            logger.exception("Root-cause tracker stage failed.")
            raise RuntimeError("Root-cause tracker stage failed.") from exc


def run_root_cause_tracking() -> Dict[str, object]:
    """
    Execute root-cause tracking.
    """
    print("[PROGRESS] Entering run_root_cause_tracking")

    tracker = RootCauseTracker()
    return tracker.run()


if __name__ == "__main__":
    print("[PROGRESS] root_cause_tracker.py execution started")
    result = run_root_cause_tracking()
    print("[PROGRESS] root_cause_tracker.py execution finished successfully")
    print(result)