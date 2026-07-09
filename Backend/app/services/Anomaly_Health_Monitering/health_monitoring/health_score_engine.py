"""
Health score engine for CA-EDT-AHMA.

Role:
Run the complete health monitoring stage:
1. Health index calculation
2. Health state classification
3. Health trend tracking
4. Health alert summary

Reads:
data/outputs/anomaly_fusion.csv

Writes:
data/outputs/health_index.csv
data/outputs/health_states.csv
data/outputs/health_trends.csv
data/outputs/health_alerts.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_score_engine.py")
from typing import Dict

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.health_monitoring.health_alert_engine import (
    HealthAlertEngine,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_index_calculator import (
    HealthIndexCalculator,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_state_classifier import (
    HealthStateClassifier,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_trend_tracker import (
    HealthTrendTracker,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class HealthScoreEngine:
    """
    Complete health monitoring engine.
    """

    def __init__(self) -> None:
        """
        Initialize health score engine.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_score_engine.py::__init__")
        Config.create_directories()

    def run(self) -> Dict[str, object]:
        """
        Run all health monitoring stages.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_score_engine.py::run")
        try:
            health_index_result = HealthIndexCalculator().run()
            health_state_result = HealthStateClassifier().run()
            trend_result = HealthTrendTracker().run()
            alert_result = HealthAlertEngine().run()

            return {
                "status": "success",
                "message": "Complete health monitoring stage finished.",
                "output_file": str(Config.HEALTH_STATES_CSV),
                "records_count": health_state_result.get("records_count"),
                "data": {
                    "health_index": health_index_result,
                    "health_state": health_state_result,
                    "health_trend": trend_result,
                    "health_alerts": alert_result,
                },
            }

        except Exception as exc:
            logger.exception("Health score engine failed.")
            raise RuntimeError("Health score engine failed.") from exc


def run_health_score_engine() -> Dict[str, object]:
    """
    Execute full health monitoring.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_score_engine.py::run_health_score_engine")
    engine = HealthScoreEngine()
    return engine.run()


if __name__ == "__main__":
    result = run_health_score_engine()
    print(result)