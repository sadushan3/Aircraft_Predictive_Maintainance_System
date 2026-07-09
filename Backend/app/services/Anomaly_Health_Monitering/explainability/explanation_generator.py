"""
Human-readable explanation generator for CA-EDT-AHMA.

Role:
Generate dashboard-ready natural language explanations.

Each explanation includes:
- Context ID
- Alert level
- Health index
- Health state
- Top contributing sensors
- Sensor contribution percentages
- Root-cause pattern
- Confidence score when available
- Inspection focus

Reads:
data/outputs/health_states.csv
data/outputs/root_cause_analysis.csv
data/outputs/context_clusters.csv
data/outputs/confidence_scores.csv, if available

Writes:
data/outputs/explanation_reports.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/explainability/explanation_generator.py")
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
from app.services.Anomaly_Health_Monitering.explainability.sensor_residual_ranking import (
    SensorResidualRanking,
)
from app.services.Anomaly_Health_Monitering.explainability.subsystem_explainer import (
    SubsystemExplainer,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv, read_csv_required
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class ExplanationGenerator:
    """
    Generates human-readable explanations.
    """

    def __init__(self) -> None:
        """
        Initialize explanation generator.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/explanation_generator.py::__init__")
        Config.create_directories()

    def _load_optional_confidence(self) -> pd.DataFrame:
        """
        Load confidence scores if available.

        Returns:
            pd.DataFrame: Confidence DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/explanation_generator.py::_load_optional_confidence")
        try:
            if Config.CONFIDENCE_CSV.exists():
                return read_csv_required(Config.CONFIDENCE_CSV)

            logger.warning("Confidence CSV not found. Using default confidence fields.")
            return pd.DataFrame(
                columns=[
                    "unit_id",
                    "cycle",
                    "split",
                    "model_agreement_score",
                    "confidence_score",
                    "uncertainty_score",
                    "reliability_score",
                ]
            )

        except Exception as exc:
            logger.exception("Optional confidence loading failed.")
            raise RuntimeError("Optional confidence loading failed.") from exc

    def build_explanations(self) -> pd.DataFrame:
        """
        Build explanation reports.

        Returns:
            pd.DataFrame: Explanation reports DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/explanation_generator.py::build_explanations")
        try:
            health_df = read_csv_required(Config.HEALTH_STATES_CSV)
            root_df = read_csv_required(Config.ROOT_CAUSE_CSV)
            context_df = read_csv_required(Config.CONTEXT_CSV)
            confidence_df = self._load_optional_confidence()

            merge_columns = ["unit_id", "cycle", "split"]

            df = health_df.merge(root_df, on=merge_columns, how="left")
            df = df.merge(
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

            if not confidence_df.empty:
                confidence_keep = [
                    column
                    for column in [
                        "unit_id",
                        "cycle",
                        "split",
                        "model_agreement_score",
                        "confidence_score",
                        "uncertainty_score",
                        "reliability_score",
                    ]
                    if column in confidence_df.columns
                ]

                df = df.merge(
                    confidence_df[confidence_keep],
                    on=merge_columns,
                    how="left",
                )

            defaults = {
                "kmeans_context_id": -1,
                "context_confidence": 0.0,
                "model_agreement_score": 0.0,
                "confidence_score": 0.0,
                "uncertainty_score": 1.0,
                "reliability_score": 0.0,
                "top_sensor_1": "unknown",
                "top_sensor_2": "unknown",
                "top_sensor_3": "unknown",
                "contribution_1": 0.0,
                "contribution_2": 0.0,
                "contribution_3": 0.0,
                "root_cause_pattern": "unknown_residual_pattern",
                "inspection_focus": "Inspect the top contributing measured sensor channels.",
            }

            for column, value in defaults.items():
                if column not in df.columns:
                    df[column] = value
                df[column] = df[column].fillna(value)

            df["explanation_text"] = df.apply(
                lambda row: self._generate_text(row),
                axis=1,
            )

            logger.info("Explanation text generated. rows=%s", len(df))
            return df

        except Exception as exc:
            logger.exception("Explanation generation failed.")
            raise RuntimeError("Explanation generation failed.") from exc

    def _generate_text(self, row: pd.Series) -> str:
        """
        Generate explanation text for one row.

        Args:
            row: Data row.

        Returns:
            str: Explanation text.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/explanation_generator.py::_generate_text")
        try:
            context_id = int(row.get("gmm_context_id", -1))
            kmeans_context_id = int(row.get("kmeans_context_id", -1))
            context_confidence = float(row.get("context_confidence", 0.0))
            health_index = float(row.get("health_index", 100.0))
            final_anomaly_score = float(row.get("final_anomaly_score", 0.0))
            confidence_score = float(row.get("confidence_score", 0.0))
            uncertainty_score = float(row.get("uncertainty_score", 1.0))
            model_agreement_score = float(row.get("model_agreement_score", 0.0))

            contribution_1 = float(row.get("contribution_1", 0.0)) * 100.0
            contribution_2 = float(row.get("contribution_2", 0.0)) * 100.0
            contribution_3 = float(row.get("contribution_3", 0.0)) * 100.0

            explanation = (
                f"The engine shows a {row.get('alert_level', 'Normal')} alert under "
                f"GMM operating context {context_id}. The baseline K-Means context is "
                f"{kmeans_context_id}. The ensemble digital twin estimated expected "
                f"measured sensor behavior using operating conditions, virtual sensors, "
                f"and context information, then compared it with actual measured sensors. "
                f"The final anomaly score is {final_anomaly_score:.3f}. The health index "
                f"is {health_index:.1f}/100 and the health state is "
                f"{row.get('health_state', 'Healthy')}. The main contributing sensors are "
                f"{row.get('top_sensor_1', 'unknown')} ({contribution_1:.1f}%), "
                f"{row.get('top_sensor_2', 'unknown')} ({contribution_2:.1f}%), and "
                f"{row.get('top_sensor_3', 'unknown')} ({contribution_3:.1f}%). "
                f"The residual pattern is classified as "
                f"{row.get('root_cause_pattern', 'unknown_residual_pattern')}. "
                f"Recommended inspection focus: {row.get('inspection_focus', 'Inspect top sensors.')}. "
                f"Context confidence is {context_confidence * 100:.1f}%, model agreement is "
                f"{model_agreement_score * 100:.1f}%, confidence is "
                f"{confidence_score * 100:.1f}%, and uncertainty is "
                f"{uncertainty_score * 100:.1f}%. This explanation supports inspection "
                f"focus only and does not make maintenance scheduling decisions."
            )

            return explanation

        except Exception as exc:
            logger.exception("Single-row explanation generation failed.")
            raise RuntimeError("Single-row explanation generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run explanation generation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/explanation_generator.py::run")
        try:
            SensorResidualRanking().run()
            SubsystemExplainer().run()

            explanation_df = self.build_explanations()
            atomic_write_csv(explanation_df, Config.EXPLANATION_REPORTS_CSV)

            return {
                "status": "success",
                "message": "Human-readable explanation reports generated.",
                "output_file": str(Config.EXPLANATION_REPORTS_CSV),
                "records_count": len(explanation_df),
            }

        except Exception as exc:
            logger.exception("Explanation generator stage failed.")
            raise RuntimeError("Explanation generator stage failed.") from exc


def run_explanation_generator() -> Dict[str, object]:
    """
    Execute explanation generation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/explanation_generator.py::run_explanation_generator")
    generator = ExplanationGenerator()
    return generator.run()


if __name__ == "__main__":
    result = run_explanation_generator()
    print(result)