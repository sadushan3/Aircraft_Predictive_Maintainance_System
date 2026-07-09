"""
Dashboard data generator for CA-EDT-AHMA.

Role:
Generate final dashboard_data.csv with all required dashboard columns.

Reads:
data/outputs/context_clusters.csv
data/outputs/health_states.csv
data/outputs/root_cause_analysis.csv
data/outputs/explanation_reports.csv
data/outputs/confidence_scores.csv
data/outputs/feedback_updates.csv, if available

Writes:
data/outputs/dashboard_data.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_data_generator.py")
import gc
from typing import Dict

import numpy as np
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


class DashboardDataGenerator:
    """
    Generates final dashboard-ready CSV.
    """

    def __init__(self) -> None:
        """
        Initialize dashboard data generator.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_data_generator.py::__init__")
        Config.create_directories()

    def _load_optional_feedback(self) -> pd.DataFrame:
        """
        Load feedback CSV if available.

        Returns:
            pd.DataFrame: Feedback DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_data_generator.py::_load_optional_feedback")
        try:
            if Config.FEEDBACK_UPDATES_CSV.exists():
                return read_csv_required(Config.FEEDBACK_UPDATES_CSV)

            return pd.DataFrame(
                columns=[
                    "unit_id",
                    "cycle",
                    "context_id",
                    "feedback_label",
                    "feedback_status",
                ]
            )

        except Exception as exc:
            logger.exception("Failed to load optional feedback.")
            raise RuntimeError("Failed to load optional feedback.") from exc

    def _merge_feedback_status(self, dashboard_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge feedback status into dashboard data.

        Args:
            dashboard_df: Dashboard DataFrame.

        Returns:
            pd.DataFrame: Dashboard DataFrame with feedback_status.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_data_generator.py::_merge_feedback_status")
        try:
            result = dashboard_df.copy()
            feedback_df = self._load_optional_feedback()

            result["feedback_status"] = "no_feedback"

            if feedback_df.empty:
                return result

            required = ["unit_id", "cycle", "context_id", "feedback_label"]
            missing = [column for column in required if column not in feedback_df.columns]
            if missing:
                logger.warning("Feedback file missing columns: %s", missing)
                return result

            latest_feedback = (
                feedback_df.sort_values("feedback_id")
                if "feedback_id" in feedback_df.columns
                else feedback_df.copy()
            )

            latest_feedback = latest_feedback.drop_duplicates(
                subset=["unit_id", "cycle", "context_id"],
                keep="last",
            )

            for _, feedback in latest_feedback.iterrows():
                mask = (
                    (result["unit_id"] == int(feedback["unit_id"]))
                    & (result["cycle"] == int(feedback["cycle"]))
                    & (result["gmm_context_id"] == int(feedback["context_id"]))
                )

                result.loc[mask, "feedback_status"] = str(feedback["feedback_label"])

            return result

        except Exception as exc:
            logger.exception("Feedback status merge failed.")
            raise RuntimeError("Feedback status merge failed.") from exc

    def generate(self) -> pd.DataFrame:
        """
        Generate dashboard_data.csv content.

        Returns:
            pd.DataFrame: Final dashboard DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_data_generator.py::generate")
        try:
            context_df = read_csv_required(Config.CONTEXT_CSV)
            health_df = read_csv_required(Config.HEALTH_STATES_CSV)
            root_df = read_csv_required(Config.ROOT_CAUSE_CSV)
            explanation_df = read_csv_required(Config.EXPLANATION_REPORTS_CSV)
            confidence_df = read_csv_required(Config.CONFIDENCE_CSV)

            merge_columns = ["unit_id", "cycle", "split"]

            df = health_df.merge(
                context_df[
                    merge_columns
                    + [
                        "kmeans_context_id",
                        "context_confidence",
                    ]
                ],
                on=merge_columns,
                how="left",
            )
            del health_df, context_df
            gc.collect()

            df = df.merge(
                root_df[
                    merge_columns
                    + [
                        "top_sensor_1",
                        "top_sensor_2",
                        "top_sensor_3",
                        "contribution_1",
                        "contribution_2",
                        "contribution_3",
                        "root_cause_pattern",
                    ]
                ],
                on=merge_columns,
                how="left",
            )
            del root_df

            df = df.merge(
                explanation_df[
                    merge_columns
                    + [
                        "explanation_text",
                    ]
                ],
                on=merge_columns,
                how="left",
            )
            del explanation_df

            df = df.merge(
                confidence_df[
                    merge_columns
                    + [
                        "model_agreement_score",
                        "confidence_score",
                        "uncertainty_score",
                        "reliability_score",
                    ]
                ],
                on=merge_columns,
                how="left",
            )
            del confidence_df
            gc.collect()

            defaults = {
                "kmeans_context_id": -1,
                "gmm_context_id": -1,
                "context_confidence": 0.0,
                "health_index": 100.0,
                "remaining_health_percentage": 100.0,
                "health_state": "Healthy",
                "final_anomaly_score": 0.0,
                "alert_level": "Normal",
                "top_sensor_1": "none",
                "top_sensor_2": "none",
                "top_sensor_3": "none",
                "contribution_1": 0.0,
                "contribution_2": 0.0,
                "contribution_3": 0.0,
                "root_cause_pattern": "normal_or_no_pattern",
                "explanation_text": "No anomaly explanation available.",
                "model_agreement_score": 0.0,
                "confidence_score": 0.0,
                "uncertainty_score": 1.0,
                "reliability_score": 0.0,
                "feedback_status": "no_feedback",
            }

            for column, default_value in defaults.items():
                if column not in df.columns:
                    df[column] = default_value
                df[column] = df[column].fillna(default_value)

            df = self._merge_feedback_status(df)

            for column in Config.FINAL_DASHBOARD_COLUMNS:
                if column not in df.columns:
                    df[column] = np.nan

            dashboard_df = df[list(Config.FINAL_DASHBOARD_COLUMNS)].copy()
            dashboard_df = dashboard_df.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)
            del df
            gc.collect()

            logger.info("Dashboard data generated. rows=%s", len(dashboard_df))
            return dashboard_df

        except Exception as exc:
            logger.exception("Dashboard data generation failed.")
            raise RuntimeError("Dashboard data generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run dashboard data generation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_data_generator.py::run")
        try:
            dashboard_df = self.generate()
            atomic_write_csv(dashboard_df, Config.DASHBOARD_CSV)

            return {
                "status": "success",
                "message": "Final dashboard_data.csv generated.",
                "output_file": str(Config.DASHBOARD_CSV),
                "records_count": len(dashboard_df),
            }

        except Exception as exc:
            logger.exception("Dashboard data generator stage failed.")
            raise RuntimeError("Dashboard data generator stage failed.") from exc


def run_dashboard_data_generation() -> Dict[str, object]:
    """
    Execute dashboard data generation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/dashboard/dashboard_data_generator.py::run_dashboard_data_generation")
    generator = DashboardDataGenerator()
    return generator.run()


if __name__ == "__main__":
    result = run_dashboard_data_generation()
    print(result)
