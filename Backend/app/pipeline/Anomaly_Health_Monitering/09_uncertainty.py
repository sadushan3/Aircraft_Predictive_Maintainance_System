"""
Uncertainty pipeline for CA-EDT-AHMA.

Stages:
1. Model agreement calculation
2. Confidence estimation

Outputs:
data/outputs/model_agreement.csv
data/outputs/confidence_scores.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/09_uncertainty.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.uncertainty.confidence_estimator import ConfidenceEstimator
from app.services.Anomaly_Health_Monitering.uncertainty.model_agreement import ModelAgreementCalculator
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class UncertaintyPipeline:
    """
    Complete uncertainty pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize uncertainty pipeline.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/09_uncertainty.py::__init__")
        Config.create_directories()

    def run(self) -> Dict[str, object]:
        """
        Run uncertainty pipeline safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/09_uncertainty.py::run")
        try:
            stages = [
                ("model_agreement", ModelAgreementCalculator().run),
                ("confidence_estimation", ConfidenceEstimator().run),
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
                    "Uncertainty pipeline completed."
                    if status == "success"
                    else "Uncertainty pipeline stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": str(Config.CONFIDENCE_CSV) if Config.CONFIDENCE_CSV.exists() else None,
            }

            atomic_write_json(summary, Config.REPORT_DIR / "09_uncertainty_summary.json")
            logger.info("Uncertainty pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Uncertainty pipeline failed.")
            raise RuntimeError("Uncertainty pipeline failed.") from exc


def run_uncertainty_pipeline() -> Dict[str, object]:
    """
    Execute uncertainty pipeline.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/09_uncertainty.py::run_uncertainty_pipeline")
    pipeline = UncertaintyPipeline()
    return pipeline.run()


if __name__ == "__main__":
    result = run_uncertainty_pipeline()
    print(result)