"""
Dashboard API service wrapper for CA-EDT-AHMA.

Role:
Provide function-level API helpers used by FastAPI routes.

This file does not define FastAPI routes directly.
Routes are defined in:

app/routers/Anomaly_Health_Monitering/Routes.py

Important:
- This wrapper does not train models.
- This wrapper does not predict RUL.
- This wrapper does not use Y_dev/Y_test.
- This wrapper does not make maintenance decisions.
- Heavy dashboard services are initialized lazily only when needed.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "dashboard/dashboard_api.py"
)

from typing import Dict, Optional
import os
import sys


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
from app.services.Anomaly_Health_Monitering.dashboard.dashboard_data_generator import (
    DashboardDataGenerator,
)
from app.services.Anomaly_Health_Monitering.dashboard.dashboard_service import (
    DashboardService,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class DashboardAPI:
    """
    Dashboard API wrapper service.

    This class is intentionally lightweight. It lazily initializes the dashboard
    generator and dashboard service only when those methods are called.
    """

    def __init__(self) -> None:
        """
        Initialize dashboard API wrapper.
        """
        print("[PROGRESS] Entering DashboardAPI.__init__")

        Config.create_directories()

        self.generator: Optional[DashboardDataGenerator] = None
        self.service: Optional[DashboardService] = None

    # ==================================================================================
    # Lazy service access
    # ==================================================================================

    def _get_generator(self) -> DashboardDataGenerator:
        """
        Lazily initialize dashboard data generator.

        Returns:
            DashboardDataGenerator: Dashboard generator instance.
        """
        print("[PROGRESS] Entering DashboardAPI._get_generator")

        if self.generator is None:
            self.generator = DashboardDataGenerator()

        return self.generator

    def _get_service(self) -> DashboardService:
        """
        Lazily initialize dashboard service.

        Returns:
            DashboardService: Dashboard service instance.
        """
        print("[PROGRESS] Entering DashboardAPI._get_service")

        if self.service is None:
            self.service = DashboardService()

        return self.service

    def _failed_response(self, message: str) -> Dict[str, object]:
        """
        Build a consistent failure response.

        Args:
            message: Error message.

        Returns:
            Dict[str, object]: Standard failed response.
        """
        return {
            "status": "failed",
            "message": message,
            "errors": [message],
        }

    # ==================================================================================
    # Dashboard generation
    # ==================================================================================

    def generate_dashboard(self) -> Dict[str, object]:
        """
        Generate dashboard_data.csv.

        Returns:
            Dict[str, object]: Generation response.
        """
        print("[PROGRESS] Entering DashboardAPI.generate_dashboard")

        try:
            return self._get_generator().run()

        except Exception as exc:
            logger.exception("Dashboard generation API wrapper failed.")
            return self._failed_response(str(exc))

    # ==================================================================================
    # Dashboard query helpers
    # ==================================================================================

    def get_summary(self) -> Dict[str, object]:
        """
        Get dashboard summary counts.

        Returns:
            Dict[str, object]: Summary response.
        """
        print("[PROGRESS] Entering DashboardAPI.get_summary")

        try:
            return self._get_service().summary_counts()

        except Exception as exc:
            logger.exception("Dashboard summary API wrapper failed.")
            return self._failed_response(str(exc))

    def get_latest_unit_health(self, unit_id: int) -> Dict[str, object]:
        """
        Get latest health for one unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: Latest unit health response.
        """
        print("[PROGRESS] Entering DashboardAPI.get_latest_unit_health")

        try:
            return self._get_service().latest_unit_health(unit_id)

        except Exception as exc:
            logger.exception("Latest unit health API wrapper failed.")
            return self._failed_response(str(exc))

    def get_latest_all_units(self) -> Dict[str, object]:
        """
        Get latest health for all units.

        Returns:
            Dict[str, object]: Latest all units response.
        """
        print("[PROGRESS] Entering DashboardAPI.get_latest_all_units")

        try:
            return self._get_service().latest_all_units()

        except Exception as exc:
            logger.exception("Latest all units API wrapper failed.")
            return self._failed_response(str(exc))

    def get_health_trend(self, unit_id: int) -> Dict[str, object]:
        """
        Get health trend for one unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: Health trend response.
        """
        print("[PROGRESS] Entering DashboardAPI.get_health_trend")

        try:
            return self._get_service().health_trend_by_unit(unit_id)

        except Exception as exc:
            logger.exception("Health trend API wrapper failed.")
            return self._failed_response(str(exc))

    def get_anomalies(self, unit_id: int) -> Dict[str, object]:
        """
        Get anomaly records for one unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: Anomalies response.
        """
        print("[PROGRESS] Entering DashboardAPI.get_anomalies")

        try:
            return self._get_service().anomalies_by_unit(unit_id)

        except Exception as exc:
            logger.exception("Anomalies API wrapper failed.")
            return self._failed_response(str(exc))

    def get_explanation(self, unit_id: int, cycle: int) -> Dict[str, object]:
        """
        Get explanation for one unit and cycle.

        Args:
            unit_id: Unit id.
            cycle: Cycle.

        Returns:
            Dict[str, object]: Explanation response.
        """
        print("[PROGRESS] Entering DashboardAPI.get_explanation")

        try:
            return self._get_service().root_cause_explanation(unit_id, cycle)

        except Exception as exc:
            logger.exception("Explanation API wrapper failed.")
            return self._failed_response(str(exc))

    def get_confidence_uncertainty(self, unit_id: int) -> Dict[str, object]:
        """
        Get confidence and uncertainty records for one unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: Confidence and uncertainty response.
        """
        print("[PROGRESS] Entering DashboardAPI.get_confidence_uncertainty")

        try:
            return self._get_service().confidence_uncertainty_by_unit(unit_id)

        except Exception as exc:
            logger.exception("Confidence API wrapper failed.")
            return self._failed_response(str(exc))

    def get_reports(self) -> Dict[str, object]:
        """Get generated report content and artifact metadata."""
        print("[PROGRESS] Entering DashboardAPI.get_reports")

        try:
            return self._get_service().reports_catalog()

        except Exception as exc:
            logger.exception("Reports API wrapper failed.")
            return self._failed_response(str(exc))

    def get_overview(self) -> Dict[str, object]:
        """Get fast overview aggregates from the dashboard summary report."""
        print("[PROGRESS] Entering DashboardAPI.get_overview")

        try:
            return self._get_service().overview_summary()

        except Exception as exc:
            logger.exception("Dashboard overview API wrapper failed.")
            return self._failed_response(str(exc))

    def get_all_anomalies(self, limit: int = 500) -> Dict[str, object]:
        """Get a bounded fleet-wide sample of persisted alerts."""
        print("[PROGRESS] Entering DashboardAPI.get_all_anomalies")

        try:
            return self._get_service().anomalies_all(limit=limit)

        except Exception as exc:
            logger.exception("Fleet-wide anomalies API wrapper failed.")
            return self._failed_response(str(exc))

    def get_pipeline_status(self) -> Dict[str, object]:
        """Get pipeline stage status derived from generated artifacts."""
        print("[PROGRESS] Entering DashboardAPI.get_pipeline_status")

        try:
            return self._get_service().pipeline_status()

        except Exception as exc:
            logger.exception("Pipeline status API wrapper failed.")
            return self._failed_response(str(exc))

    def get_feedback_history(self, limit: int = 500) -> Dict[str, object]:
        """Get bounded operator feedback history and an alert-memory sample."""
        print("[PROGRESS] Entering DashboardAPI.get_feedback_history")

        try:
            return self._get_service().feedback_history(limit=limit)

        except Exception as exc:
            logger.exception("Feedback history API wrapper failed.")
            return self._failed_response(str(exc))

    def get_adaptive_thresholds(self) -> Dict[str, object]:
        """Get the persisted adaptive threshold artifact."""
        print("[PROGRESS] Entering DashboardAPI.get_adaptive_thresholds")

        try:
            return self._get_service().adaptive_thresholds()

        except Exception as exc:
            logger.exception("Adaptive thresholds API wrapper failed.")
            return self._failed_response(str(exc))

    def get_reasoning_summary(self) -> Dict[str, object]:
        """Get aggregate reasoning reports."""
        print("[PROGRESS] Entering DashboardAPI.get_reasoning_summary")

        try:
            return self._get_service().reasoning_summary()

        except Exception as exc:
            logger.exception("Reasoning summary API wrapper failed.")
            return self._failed_response(str(exc))

    def get_explainability_summary(self) -> Dict[str, object]:
        """Get bounded explainability reports and SHAP rows."""
        print("[PROGRESS] Entering DashboardAPI.get_explainability_summary")

        try:
            return self._get_service().explainability_summary()

        except Exception as exc:
            logger.exception("Explainability summary API wrapper failed.")
            return self._failed_response(str(exc))

    def get_analytics(self) -> Dict[str, object]:
        """Get lightweight analytics from existing JSON reports."""
        print("[PROGRESS] Entering DashboardAPI.get_analytics")

        try:
            return self._get_service().analytics_summary()

        except Exception as exc:
            logger.exception("Analytics API wrapper failed.")
            return self._failed_response(str(exc))


def run_dashboard_api_self_check() -> Dict[str, object]:
    """
    Run dashboard API self-check.

    Returns:
        Dict[str, object]: Self-check response.
    """
    print("[PROGRESS] Entering run_dashboard_api_self_check")

    api = DashboardAPI()
    return api.get_summary()


if __name__ == "__main__":
    print("[PROGRESS] dashboard_api.py execution started")
    result = run_dashboard_api_self_check()
    print("[PROGRESS] dashboard_api.py execution finished")
    print(result)
