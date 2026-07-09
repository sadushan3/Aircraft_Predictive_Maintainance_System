"""
Root-cause tracker for CA-EDT-AHMA.

Role:
Track whether similar root-cause patterns occurred previously.

Reads:
data/outputs/root_cause_analysis.csv

Writes:
data/outputs/root_cause_memory.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_tracker.py")
from pathlib import Path
from typing import Dict

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv, read_csv_required
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class RootCauseTracker:
    """
    Tracks recurring root-cause patterns.
    """

    def __init__(self, lookback_window: int = 20) -> None:
        """
        Initialize root-cause tracker.

        Args:
            lookback_window: Number of recent cycles used for recurrence tracking.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_tracker.py::__init__")
        Config.create_directories()

        if lookback_window <= 1:
            raise ValueError("lookback_window must be greater than 1.")

        self.lookback_window = lookback_window

    def track(self, root_df: pd.DataFrame) -> pd.DataFrame:
        """
        Track recurring root-cause patterns.

        Args:
            root_df: Root-cause DataFrame.

        Returns:
            pd.DataFrame: Root-cause memory DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_tracker.py::track")
        try:
            required = [
                "unit_id",
                "cycle",
                "split",
                "root_cause_pattern",
                "alert_level",
                "top_sensor_1",
                "top_sensor_2",
                "top_sensor_3",
            ]

            missing = [column for column in required if column not in root_df.columns]
            if missing:
                raise KeyError(f"Missing root-cause tracker columns: {missing}")

            result = root_df.copy()
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            recurrence_counts = []
            recurring_flags = []

            for _, row in result.iterrows():
                unit_mask = (
                    (result["split"] == row["split"])
                    & (result["unit_id"] == row["unit_id"])
                    & (result["cycle"] < row["cycle"])
                    & (result["cycle"] >= row["cycle"] - self.lookback_window)
                    & (result["root_cause_pattern"] == row["root_cause_pattern"])
                )

                count = int(unit_mask.sum())
                recurrence_counts.append(count)
                recurring_flags.append("Recurring" if count >= 2 else "New_or_Isolated")

            result["similar_pattern_count_recent"] = recurrence_counts
            result["root_cause_recurrence_status"] = recurring_flags

            result["root_cause_memory_note"] = result.apply(
                lambda row: (
                    f"Pattern {row['root_cause_pattern']} has appeared "
                    f"{int(row['similar_pattern_count_recent'])} time(s) in the recent lookback window."
                ),
                axis=1,
            )

            output_columns = [
                "unit_id",
                "cycle",
                "split",
                "root_cause_pattern",
                "top_sensor_1",
                "top_sensor_2",
                "top_sensor_3",
                "similar_pattern_count_recent",
                "root_cause_recurrence_status",
                "root_cause_memory_note",
            ]

            memory_df = result[output_columns].copy()

            logger.info("Root-cause tracking completed. rows=%s", len(memory_df))
            return memory_df

        except Exception as exc:
            logger.exception("Root-cause tracking failed.")
            raise RuntimeError("Root-cause tracking failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run root-cause tracking.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_tracker.py::run")
        try:
            root_df = read_csv_required(Config.ROOT_CAUSE_CSV)
            memory_df = self.track(root_df)

            output_path: Path = Config.OUTPUT_DIR / "root_cause_memory.csv"
            atomic_write_csv(memory_df, output_path)

            return {
                "status": "success",
                "message": "Root-cause recurrence tracking completed.",
                "output_file": str(output_path),
                "records_count": len(memory_df),
            }

        except Exception as exc:
            logger.exception("Root-cause tracker stage failed.")
            raise RuntimeError("Root-cause tracker stage failed.") from exc


def run_root_cause_tracking() -> Dict[str, object]:
    """
    Execute root-cause tracking.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_tracker.py::run_root_cause_tracking")
    tracker = RootCauseTracker()
    return tracker.run()


if __name__ == "__main__":
    result = run_root_cause_tracking()
    print(result)