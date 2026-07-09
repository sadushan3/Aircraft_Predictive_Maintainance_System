"""
Alert memory for CA-EDT-AHMA.

Role:
Store anomaly alert memory for later comparison and feedback learning.

Reads:
data/outputs/anomaly_fusion.csv
data/outputs/root_cause_analysis.csv

Writes:
data/outputs/alert_memory.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/feedback/alert_memory.py")
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


class AlertMemory:
    """
    Alert memory manager.
    """

    def __init__(self) -> None:
        """
        Initialize alert memory.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/alert_memory.py::__init__")
        Config.create_directories()

    def _empty_alert_memory(self) -> pd.DataFrame:
        """
        Create empty alert memory DataFrame.

        Returns:
            pd.DataFrame: Empty alert memory.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/alert_memory.py::_empty_alert_memory")
        return pd.DataFrame(
            columns=[
                "unit_id",
                "cycle",
                "split",
                "context_id",
                "alert_level",
                "final_anomaly_score",
                "root_cause_pattern",
                "top_sensor_1",
                "top_sensor_2",
                "top_sensor_3",
                "feedback_status",
            ]
        )

    def load_memory(self) -> pd.DataFrame:
        """
        Load alert memory.

        Returns:
            pd.DataFrame: Alert memory DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/alert_memory.py::load_memory")
        try:
            if Config.ALERT_MEMORY_CSV.exists():
                return read_csv_required(Config.ALERT_MEMORY_CSV)

            empty_df = self._empty_alert_memory()
            atomic_write_csv(empty_df, Config.ALERT_MEMORY_CSV)
            return empty_df

        except Exception as exc:
            logger.exception("Failed to load alert memory.")
            raise RuntimeError("Failed to load alert memory.") from exc

    def build_memory_from_current_alerts(self) -> pd.DataFrame:
        """
        Build alert memory from current anomaly and root-cause outputs.

        Returns:
            pd.DataFrame: Alert memory DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/alert_memory.py::build_memory_from_current_alerts")
        try:
            anomaly_df = read_csv_required(Config.ANOMALY_FUSION_CSV)
            root_df = read_csv_required(Config.ROOT_CAUSE_CSV)

            merge_columns = ["unit_id", "cycle", "split"]

            df = anomaly_df.merge(
                root_df[
                    merge_columns
                    + [
                        "root_cause_pattern",
                        "top_sensor_1",
                        "top_sensor_2",
                        "top_sensor_3",
                    ]
                ],
                on=merge_columns,
                how="left",
            )

            if "gmm_context_id" not in df.columns:
                df["gmm_context_id"] = -1

            memory_df = pd.DataFrame(
                {
                    "unit_id": df["unit_id"],
                    "cycle": df["cycle"],
                    "split": df["split"],
                    "context_id": df["gmm_context_id"],
                    "alert_level": df["alert_level"],
                    "final_anomaly_score": df["final_anomaly_score"],
                    "root_cause_pattern": df["root_cause_pattern"].fillna("unknown"),
                    "top_sensor_1": df["top_sensor_1"].fillna("unknown"),
                    "top_sensor_2": df["top_sensor_2"].fillna("unknown"),
                    "top_sensor_3": df["top_sensor_3"].fillna("unknown"),
                    "feedback_status": "no_feedback",
                }
            )

            logger.info("Alert memory built from current alerts. rows=%s", len(memory_df))
            return memory_df

        except Exception as exc:
            logger.exception("Failed to build alert memory from current alerts.")
            raise RuntimeError("Failed to build alert memory from current alerts.") from exc

    def update_feedback_status(self) -> pd.DataFrame:
        """
        Update alert memory feedback status using stored feedback.

        Returns:
            pd.DataFrame: Updated alert memory.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/alert_memory.py::update_feedback_status")
        try:
            memory_df = self.load_memory()

            if not Config.FEEDBACK_UPDATES_CSV.exists():
                return memory_df

            feedback_df = read_csv_required(Config.FEEDBACK_UPDATES_CSV)

            if feedback_df.empty:
                return memory_df

            updated = memory_df.copy()

            for _, feedback in feedback_df.iterrows():
                mask = (
                    (updated["unit_id"] == int(feedback["unit_id"]))
                    & (updated["cycle"] == int(feedback["cycle"]))
                    & (updated["context_id"] == int(feedback["context_id"]))
                )

                updated.loc[mask, "feedback_status"] = str(feedback["feedback_label"])

            atomic_write_csv(updated, Config.ALERT_MEMORY_CSV)

            logger.info("Alert memory feedback statuses updated. rows=%s", len(updated))
            return updated

        except Exception as exc:
            logger.exception("Failed to update alert memory feedback status.")
            raise RuntimeError("Failed to update alert memory feedback status.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run alert memory update.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/alert_memory.py::run")
        try:
            memory_df = self.build_memory_from_current_alerts()
            atomic_write_csv(memory_df, Config.ALERT_MEMORY_CSV)

            updated_memory = self.update_feedback_status()

            return {
                "status": "success",
                "message": "Alert memory updated safely.",
                "output_file": str(Config.ALERT_MEMORY_CSV),
                "records_count": len(updated_memory),
            }

        except Exception as exc:
            logger.exception("Alert memory stage failed.")
            raise RuntimeError("Alert memory stage failed.") from exc


def run_alert_memory() -> Dict[str, object]:
    """
    Execute alert memory update.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/alert_memory.py::run_alert_memory")
    memory = AlertMemory()
    return memory.run()


if __name__ == "__main__":
    result = run_alert_memory()
    print(result)