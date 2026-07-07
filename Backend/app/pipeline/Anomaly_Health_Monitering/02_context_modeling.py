"""
Context modeling pipeline for CA-EDT-AHMA.

Stages:
1. K-Means baseline context modeling
2. GMM probabilistic context modeling
3. Combined operating mode detection
4. Context drift detection

Important:
K-Means and GMM are fitted only on W_dev.
Test split is used only for prediction/scoring.
"""

from __future__ import annotations

from typing import Dict, List

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.context_modeling.context_drift import ContextDriftDetector
from app.services.Anomaly_Health_Monitering.context_modeling.operating_mode_detector import (
    OperatingModeDetector,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class ContextModelingPipeline:
    """
    Complete context modeling pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize context modeling pipeline.
        """
        Config.create_directories()

    def run(self, include_drift: bool = True) -> Dict[str, object]:
        """
        Run context modeling pipeline safely.

        Args:
            include_drift: Whether to run context drift scoring.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        try:
            stages = [
                ("operating_mode_detection", OperatingModeDetector().run),
            ]

            if include_drift:
                stages.append(("context_drift_detection", ContextDriftDetector().run))

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
                    "Context modeling pipeline completed."
                    if status == "success"
                    else "Context modeling stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": str(Config.CONTEXT_CSV) if Config.CONTEXT_CSV.exists() else None,
            }

            atomic_write_json(summary, Config.REPORT_DIR / "02_context_modeling_summary.json")
            logger.info("Context modeling pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Context modeling pipeline failed.")
            raise RuntimeError("Context modeling pipeline failed.") from exc


def run_context_modeling_pipeline(include_drift: bool = True) -> Dict[str, object]:
    """
    Execute context modeling pipeline.

    Args:
        include_drift: Whether to include context drift detection.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    pipeline = ContextModelingPipeline()
    return pipeline.run(include_drift=include_drift)


if __name__ == "__main__":
    result = run_context_modeling_pipeline(include_drift=True)
    print(result)