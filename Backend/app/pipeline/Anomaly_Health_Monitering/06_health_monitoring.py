"""
Health monitoring pipeline for CA-EDT-AHMA.

Stages:
1. Health index calculation
2. Health state classification
3. Health trend tracking
4. Health alert generation

Important:
This stage does not predict RUL and does not use Y_dev/Y_test.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/06_health_monitoring.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.health_monitoring.health_alert_engine import HealthAlertEngine
from app.services.Anomaly_Health_Monitering.health_monitoring.health_index_calculator import (
    HealthIndexCalculator,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_state_classifier import (
    HealthStateClassifier,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_trend_tracker import HealthTrendTracker
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class HealthMonitoringPipeline:
    """
    Complete health monitoring pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize health monitoring pipeline.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/06_health_monitoring.py::__init__")
        Config.create_directories()

    def run(self) -> Dict[str, object]:
        """
        Run health monitoring safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/06_health_monitoring.py::run")
        try:
            stages = [
                ("health_index_calculation", HealthIndexCalculator().run),
                ("health_state_classification", HealthStateClassifier().run),
                ("health_trend_tracking", HealthTrendTracker().run),
                ("health_alert_generation", HealthAlertEngine().run),
            ]

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                result: StageResult = run_stage_safely(stage_name, stage_function)
                completed.append(result.__dict__)

                if result.status == "failed":
                    failed.append(result.__dict__)
                    break

            status = "success" if not failed else "partial_failure"

            summary = {
                "status": status,
                "message": (
                    "Health monitoring pipeline completed."
                    if status == "success"
                    else "Health monitoring stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": (
                    str(Config.HEALTH_STATES_CSV)
                    if Config.HEALTH_STATES_CSV.exists()
                    else None
                ),
            }

            atomic_write_json(summary, Config.REPORT_DIR / "06_health_monitoring_summary.json")
            logger.info("Health monitoring pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Health monitoring pipeline failed.")
            raise RuntimeError("Health monitoring pipeline failed.") from exc


def run_health_monitoring_pipeline() -> Dict[str, object]:
    """
    Execute health monitoring pipeline.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/06_health_monitoring.py::run_health_monitoring_pipeline")
    pipeline = HealthMonitoringPipeline()
    return pipeline.run()


if __name__ == "__main__":
    result = run_health_monitoring_pipeline()
    print(result)