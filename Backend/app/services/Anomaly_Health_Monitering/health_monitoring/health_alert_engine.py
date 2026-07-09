"""
Health alert engine for CA-EDT-AHMA.

Role:
Generate dashboard-ready health alert summaries.

Important:
This component does not make maintenance decisions.
It only provides health intelligence and inspection focus support.

Reads:
data/outputs/health_states.csv

Writes:
data/outputs/health_alerts.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_alert_engine.py")
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


class HealthAlertEngine:
    """
    Generates health alert summaries.
    """

    def __init__(self) -> None:
        """
        Initialize health alert engine.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_alert_engine.py::__init__")
        Config.create_directories()

    def generate_alerts(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate health alert summary.

        Args:
            health_df: Health states DataFrame.

        Returns:
            pd.DataFrame: Health alerts DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_alert_engine.py::generate_alerts")
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

            missing = [column for column in required_columns if column not in health_df.columns]
            if missing:
                raise KeyError(f"Missing required health alert columns: {missing}")

            result = health_df[required_columns].copy()

            result["health_alert_message"] = result.apply(
                lambda row: self._build_alert_message(
                    health_state=str(row["health_state"]),
                    alert_level=str(row["alert_level"]),
                    health_index=float(row["health_index"]),
                    anomaly_score=float(row["final_anomaly_score"]),
                ),
                axis=1,
            )

            result["inspection_priority"] = result.apply(
                lambda row: self._inspection_priority(
                    health_state=str(row["health_state"]),
                    alert_level=str(row["alert_level"]),
                ),
                axis=1,
            )

            logger.info("Health alert generation completed. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Health alert generation failed.")
            raise RuntimeError("Health alert generation failed.") from exc

    def _build_alert_message(
        self,
        health_state: str,
        alert_level: str,
        health_index: float,
        anomaly_score: float,
    ) -> str:
        """
        Build a health alert message.

        Args:
            health_state: Health state.
            alert_level: Alert level.
            health_index: Health index.
            anomaly_score: Final anomaly score.

        Returns:
            str: Health alert message.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_alert_engine.py::_build_alert_message")
        return (
            f"Health state is {health_state} with {alert_level} anomaly alert. "
            f"Health index is {health_index:.1f}/100 and final anomaly score is "
            f"{anomaly_score:.3f}. This module recommends inspection focus only; "
            f"maintenance scheduling decisions belong to the autonomous maintenance supervisor."
        )

    def _inspection_priority(self, health_state: str, alert_level: str) -> str:
        """
        Calculate inspection priority.

        Args:
            health_state: Health state.
            alert_level: Alert level.

        Returns:
            str: Inspection priority.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_alert_engine.py::_inspection_priority")
        if health_state == "Critical" or alert_level == "Critical":
            return "High"
        if health_state == "Warning" or alert_level == "Warning":
            return "Medium"
        if health_state == "Degrading" or alert_level == "Watch":
            return "Low"
        return "Routine"

    def run(self) -> Dict[str, object]:
        """
        Run health alert generation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_alert_engine.py::run")
        try:
            health_df = read_csv_required(Config.HEALTH_STATES_CSV)
            alert_df = self.generate_alerts(health_df)

            output_path: Path = Config.OUTPUT_DIR / "health_alerts.csv"
            atomic_write_csv(alert_df, output_path)

            return {
                "status": "success",
                "message": "Health alerts generated without maintenance decisions.",
                "output_file": str(output_path),
                "records_count": len(alert_df),
            }

        except Exception as exc:
            logger.exception("Health alert engine stage failed.")
            raise RuntimeError("Health alert engine stage failed.") from exc


def run_health_alert_engine() -> Dict[str, object]:
    """
    Execute health alert generation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_alert_engine.py::run_health_alert_engine")
    engine = HealthAlertEngine()
    return engine.run()


if __name__ == "__main__":
    result = run_health_alert_engine()
    print(result)