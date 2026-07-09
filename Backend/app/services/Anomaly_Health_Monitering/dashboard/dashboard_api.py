"""
Dashboard API service wrapper for CA-EDT-AHMA.

Role:
Provide function-level API helpers that are used by FastAPI routes.

This file does not define FastAPI routes directly.
Routes are defined in:

app/routers/Anomaly_Health_Monitering/Routes.py
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py")
from typing import Dict

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.dashboard.dashboard_data_generator import (
    DashboardDataGenerator,
)
from app.services.Anomaly_Health_Monitering.dashboard.dashboard_service import DashboardService
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class DashboardAPI:
    """
    Dashboard API wrapper service.
    """

    def __init__(self) -> None:
        """
        Initialize dashboard API wrapper.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::__init__")
        Config.create_directories()
        self.generator = DashboardDataGenerator()
        self.service = DashboardService()

    def generate_dashboard(self) -> Dict[str, object]:
        """
        Generate dashboard_data.csv.

        Returns:
            Dict[str, object]: Generation response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::generate_dashboard")
        try:
            return self.generator.run()

        except Exception as exc:
            logger.exception("Dashboard generation API wrapper failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "errors": [str(exc)],
            }

    def get_summary(self) -> Dict[str, object]:
        """
        Get dashboard summary.

        Returns:
            Dict[str, object]: Summary response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::get_summary")
        try:
            return self.service.summary_counts()

        except Exception as exc:
            logger.exception("Dashboard summary API wrapper failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "errors": [str(exc)],
            }

    def get_latest_unit_health(self, unit_id: int) -> Dict[str, object]:
        """
        Get latest health for one unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: Latest unit health response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::get_latest_unit_health")
        try:
            return self.service.latest_unit_health(unit_id)

        except Exception as exc:
            logger.exception("Latest unit health API wrapper failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "errors": [str(exc)],
            }

    def get_latest_all_units(self) -> Dict[str, object]:
        """
        Get latest health for all units.

        Returns:
            Dict[str, object]: Latest all units response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::get_latest_all_units")
        try:
            return self.service.latest_all_units()

        except Exception as exc:
            logger.exception("Latest all units API wrapper failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "errors": [str(exc)],
            }

    def get_health_trend(self, unit_id: int) -> Dict[str, object]:
        """
        Get health trend for a unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: Health trend response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::get_health_trend")
        try:
            return self.service.health_trend_by_unit(unit_id)

        except Exception as exc:
            logger.exception("Health trend API wrapper failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "errors": [str(exc)],
            }

    def get_anomalies(self, unit_id: int) -> Dict[str, object]:
        """
        Get anomalies for a unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: Anomalies response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::get_anomalies")
        try:
            return self.service.anomalies_by_unit(unit_id)

        except Exception as exc:
            logger.exception("Anomalies API wrapper failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "errors": [str(exc)],
            }

    def get_explanation(self, unit_id: int, cycle: int) -> Dict[str, object]:
        """
        Get explanation for unit and cycle.

        Args:
            unit_id: Unit id.
            cycle: Cycle.

        Returns:
            Dict[str, object]: Explanation response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::get_explanation")
        try:
            return self.service.root_cause_explanation(unit_id, cycle)

        except Exception as exc:
            logger.exception("Explanation API wrapper failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "errors": [str(exc)],
            }

    def get_confidence_uncertainty(self, unit_id: int) -> Dict[str, object]:
        """
        Get confidence and uncertainty for a unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: Confidence response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::get_confidence_uncertainty")
        try:
            return self.service.confidence_uncertainty_by_unit(unit_id)

        except Exception as exc:
            logger.exception("Confidence API wrapper failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "errors": [str(exc)],
            }


def run_dashboard_api_self_check() -> Dict[str, object]:
    """
    Run dashboard API self-check.

    Returns:
        Dict[str, object]: Self-check response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_api.py::run_dashboard_api_self_check")
    api = DashboardAPI()
    return api.get_summary()


if __name__ == "__main__":
    result = run_dashboard_api_self_check()
    print(result)
