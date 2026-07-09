"""
Explainability evaluation for CA-EDT-AHMA.

Metrics:
- Explanation completeness score
- Top sensor explanation availability
- SHAP summary availability
- Human-readable explanation availability

Reads:
data/outputs/explanation_reports.csv
data/outputs/shap_explanations.csv, if available
data/outputs/sensor_residual_ranking.csv, if available

Writes:
metrics/evaluate_explainability.csv
reports/evaluate_explainability_summary.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_explainability.py")
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class ExplainabilityEvaluator:
    """
    Evaluates explanation output completeness.
    """

    def __init__(self) -> None:
        """
        Initialize explainability evaluator.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_explainability.py::__init__")
        Config.create_directories()

    def _optional_csv_exists_and_nonempty(self, path: Path) -> bool:
        """
        Check whether optional CSV exists and is non-empty.

        Args:
            path: CSV path.

        Returns:
            bool: True if exists and non-empty.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_explainability.py::_optional_csv_exists_and_nonempty")
        try:
            if not path.exists():
                return False

            df = read_csv_required(path)
            return not df.empty

        except Exception:
            return False

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate explainability output.

        Returns:
            pd.DataFrame: Explainability metrics.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_explainability.py::evaluate")
        try:
            explanation_df = read_csv_required(Config.EXPLANATION_REPORTS_CSV)

            required_explanation_columns = [
                "gmm_context_id",
                "context_confidence",
                "health_index",
                "health_state",
                "final_anomaly_score",
                "alert_level",
                "top_sensor_1",
                "top_sensor_2",
                "top_sensor_3",
                "root_cause_pattern",
                "explanation_text",
            ]

            records = []

            shap_available = self._optional_csv_exists_and_nonempty(Config.SHAP_CSV)
            ranking_available = self._optional_csv_exists_and_nonempty(
                Config.OUTPUT_DIR / "sensor_residual_ranking.csv"
            )

            for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                split_df = explanation_df[explanation_df["split"] == split]

                if split_df.empty:
                    continue

                completeness_parts = []

                for column in required_explanation_columns:
                    if column in split_df.columns:
                        completeness_parts.append(float(split_df[column].notna().mean()))
                    else:
                        completeness_parts.append(0.0)

                explanation_text_availability = (
                    float(split_df["explanation_text"].astype(str).str.len().gt(20).mean())
                    if "explanation_text" in split_df.columns
                    else 0.0
                )

                top_sensor_availability = float(
                    (
                        split_df["top_sensor_1"].notna()
                        & split_df["top_sensor_2"].notna()
                        & split_df["top_sensor_3"].notna()
                    ).mean()
                )

                records.append(
                    {
                        "split": split,
                        "explanation_completeness_score": float(sum(completeness_parts) / len(completeness_parts)),
                        "explanation_text_availability": explanation_text_availability,
                        "top_sensor_explanation_availability": top_sensor_availability,
                        "shap_available": shap_available,
                        "sensor_residual_ranking_available": ranking_available,
                    }
                )

            metrics_df = pd.DataFrame(records)

            logger.info("Explainability evaluation completed. rows=%s", len(metrics_df))
            return metrics_df

        except Exception as exc:
            logger.exception("Explainability evaluation failed.")
            raise RuntimeError("Explainability evaluation failed.") from exc

    def summarize(self, metrics_df: pd.DataFrame) -> Dict[str, object]:
        """
        Summarize explainability evaluation.

        Args:
            metrics_df: Metrics DataFrame.

        Returns:
            Dict[str, object]: Summary.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_explainability.py::summarize")
        try:
            return {
                "status": "success",
                "average_explanation_completeness_score": float(
                    metrics_df["explanation_completeness_score"].mean()
                ),
                "average_explanation_text_availability": float(
                    metrics_df["explanation_text_availability"].mean()
                ),
                "average_top_sensor_explanation_availability": float(
                    metrics_df["top_sensor_explanation_availability"].mean()
                ),
                "shap_available": bool(metrics_df["shap_available"].any()),
            }

        except Exception as exc:
            logger.exception("Explainability summary generation failed.")
            raise RuntimeError("Explainability summary generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run explainability evaluation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_explainability.py::run")
        try:
            metrics_df = self.evaluate()
            output_path: Path = Config.METRIC_DIR / "evaluate_explainability.csv"
            atomic_write_csv(metrics_df, output_path)

            summary = self.summarize(metrics_df)
            atomic_write_json(summary, Config.REPORT_DIR / "evaluate_explainability_summary.json")

            return {
                "status": "success",
                "message": "Explainability evaluation completed.",
                "output_file": str(output_path),
                "records_count": len(metrics_df),
                "metrics": summary,
            }

        except Exception as exc:
            logger.exception("Explainability evaluator stage failed.")
            raise RuntimeError("Explainability evaluator stage failed.") from exc


def run_explainability_evaluation() -> Dict[str, object]:
    """
    Execute explainability evaluation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_explainability.py::run_explainability_evaluation")
    evaluator = ExplainabilityEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    result = run_explainability_evaluation()
    print(result)