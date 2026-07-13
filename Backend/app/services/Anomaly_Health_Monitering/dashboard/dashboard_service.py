"""
Dashboard service for CA-EDT-AHMA.

Role:
Read dashboard_data.csv and provide dashboard-ready query outputs.

Supports:
1. Latest unit health
2. Health trend by unit
3. Anomalies by unit
4. Root-cause explanation by unit and cycle
5. Confidence and uncertainty
6. Summary counts
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py")
from typing import Dict, List

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.dashboard.dashboard_data_generator import (
    DashboardDataGenerator,
)
from app.utils.Anomaly_Health_Monitering.file_utils import read_csv_required
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class DashboardService:
    """
    Dashboard query service.
    """

    def __init__(self) -> None:
        """
        Initialize dashboard service.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::__init__")
        Config.create_directories()

    def _load_dashboard_data(self) -> pd.DataFrame:
        """
        Load dashboard_data.csv. Generate it if missing and dependencies exist.

        Returns:
            pd.DataFrame: Dashboard DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::_load_dashboard_data")
        try:
            if not Config.DASHBOARD_CSV.exists():
                logger.warning("dashboard_data.csv not found. Attempting generation.")
                DashboardDataGenerator().run()

            return read_csv_required(Config.DASHBOARD_CSV)

        except Exception as exc:
            logger.exception("Failed to load dashboard data.")
            raise RuntimeError("Failed to load dashboard data.") from exc

    def latest_unit_health(self, unit_id: int) -> Dict[str, object]:
        """
        Return latest health row for a unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: API-ready response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::latest_unit_health")
        try:
            df = self._load_dashboard_data()
            unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")

            if unit_df.empty:
                return {
                    "status": "not_found",
                    "message": f"No dashboard data found for unit_id={unit_id}.",
                    "data": None,
                }

            latest = unit_df.tail(1).to_dict(orient="records")[0]

            return {
                "status": "success",
                "message": "Latest unit health returned.",
                "data": latest,
            }

        except Exception as exc:
            logger.exception("Latest unit health query failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "data": None,
            }

    def health_trend_by_unit(self, unit_id: int) -> Dict[str, object]:
        """
        Return health trend for a unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: API-ready response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::health_trend_by_unit")
        try:
            df = self._load_dashboard_data()

            columns = [
                "unit_id",
                "cycle",
                "split",
                "health_index",
                "remaining_health_percentage",
                "health_state",
                "final_anomaly_score",
                "alert_level",
                "confidence_score",
                "uncertainty_score",
            ]

            trend_df = df[df["unit_id"] == unit_id][columns].sort_values("cycle")

            return {
                "status": "success",
                "message": "Health trend returned.",
                "records_count": len(trend_df),
                "data": trend_df.to_dict(orient="records"),
            }

        except Exception as exc:
            logger.exception("Health trend query failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "records_count": 0,
                "data": [],
            }

    def anomalies_by_unit(self, unit_id: int) -> Dict[str, object]:
        """
        Return non-normal alerts for a unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: API-ready response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::anomalies_by_unit")
        try:
            df = self._load_dashboard_data()

            anomalies = df[
                (df["unit_id"] == unit_id)
                & (df["alert_level"].isin(["Watch", "Warning", "Critical"]))
            ].sort_values("cycle")

            return {
                "status": "success",
                "message": "Anomalies returned.",
                "records_count": len(anomalies),
                "data": anomalies.to_dict(orient="records"),
            }

        except Exception as exc:
            logger.exception("Anomalies by unit query failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "records_count": 0,
                "data": [],
            }

    def root_cause_explanation(self, unit_id: int, cycle: int) -> Dict[str, object]:
        """
        Return root-cause explanation for a unit and cycle.

        Args:
            unit_id: Unit id.
            cycle: Cycle.

        Returns:
            Dict[str, object]: API-ready response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::root_cause_explanation")
        try:
            df = self._load_dashboard_data()

            selected = df[
                (df["unit_id"] == unit_id)
                & (df["cycle"] == cycle)
            ]

            if selected.empty:
                return {
                    "status": "not_found",
                    "message": f"No explanation found for unit_id={unit_id}, cycle={cycle}.",
                    "data": None,
                }

            row = selected.head(1).to_dict(orient="records")[0]

            return {
                "status": "success",
                "message": "Root-cause explanation returned.",
                "data": {
                    "unit_id": row["unit_id"],
                    "cycle": row["cycle"],
                    "alert_level": row["alert_level"],
                    "health_index": row["health_index"],
                    "health_state": row["health_state"],
                    "top_sensor_1": row["top_sensor_1"],
                    "top_sensor_2": row["top_sensor_2"],
                    "top_sensor_3": row["top_sensor_3"],
                    "contribution_1": row["contribution_1"],
                    "contribution_2": row["contribution_2"],
                    "contribution_3": row["contribution_3"],
                    "root_cause_pattern": row["root_cause_pattern"],
                    "explanation_text": row["explanation_text"],
                },
            }

        except Exception as exc:
            logger.exception("Root-cause explanation query failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "data": None,
            }

    def confidence_uncertainty_by_unit(self, unit_id: int) -> Dict[str, object]:
        """
        Return confidence and uncertainty trend for a unit.

        Args:
            unit_id: Unit id.

        Returns:
            Dict[str, object]: API-ready response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::confidence_uncertainty_by_unit")
        try:
            df = self._load_dashboard_data()

            columns = [
                "unit_id",
                "cycle",
                "split",
                "model_agreement_score",
                "confidence_score",
                "uncertainty_score",
                "reliability_score",
                "context_confidence",
            ]

            result_df = df[df["unit_id"] == unit_id][columns].sort_values("cycle")

            return {
                "status": "success",
                "message": "Confidence and uncertainty returned.",
                "records_count": len(result_df),
                "data": result_df.to_dict(orient="records"),
            }

        except Exception as exc:
            logger.exception("Confidence/uncertainty query failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "records_count": 0,
                "data": [],
            }

    def summary_counts(self) -> Dict[str, object]:
        """
        Return dashboard summary counts.

        Returns:
            Dict[str, object]: API-ready response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::summary_counts")
        try:
            df = self._load_dashboard_data()

            alert_counts = df["alert_level"].value_counts().to_dict()
            health_counts = df["health_state"].value_counts().to_dict()

            data = {
                "alert_counts": {
                    "normal": int(alert_counts.get("Normal", 0)),
                    "watch": int(alert_counts.get("Watch", 0)),
                    "warning": int(alert_counts.get("Warning", 0)),
                    "critical": int(alert_counts.get("Critical", 0)),
                },
                "health_state_counts": {
                    "healthy": int(health_counts.get("Healthy", 0)),
                    "degrading": int(health_counts.get("Degrading", 0)),
                    "warning": int(health_counts.get("Warning", 0)),
                    "critical": int(health_counts.get("Critical", 0)),
                },
                "total_records": int(len(df)),
                "unique_units": int(df["unit_id"].nunique()),
                "average_health_index": float(df["health_index"].mean()),
                "average_confidence_score": float(df["confidence_score"].mean()),
                "average_uncertainty_score": float(df["uncertainty_score"].mean()),
            }

            return {
                "status": "success",
                "message": "Dashboard summary counts returned.",
                "data": data,
            }

        except Exception as exc:
            logger.exception("Dashboard summary query failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "data": {},
            }

    def latest_all_units(self) -> Dict[str, object]:
        """
        Return latest health row for every unit.

        Returns:
            Dict[str, object]: API-ready response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::latest_all_units")
        try:
            df = self._load_dashboard_data()

            latest_df = (
                df.sort_values(["unit_id", "cycle"])
                .groupby("unit_id", as_index=False)
                .tail(1)
                .sort_values("unit_id")
            )

            return {
                "status": "success",
                "message": "Latest health for all units returned.",
                "records_count": len(latest_df),
                "data": latest_df.to_dict(orient="records"),
            }

        except Exception as exc:
            logger.exception("Latest all units query failed.")
            return {
                "status": "failed",
                "message": str(exc),
                "records_count": 0,
                "data": [],
            }


def run_dashboard_service_self_check() -> Dict[str, object]:
    """
    Run dashboard service self-check.

    Returns:
        Dict[str, object]: Self-check response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_service.py::run_dashboard_service_self_check")
    service = DashboardService()
    return service.summary_counts()


if __name__ == "__main__":
    result = run_dashboard_service_self_check()
    print(result)