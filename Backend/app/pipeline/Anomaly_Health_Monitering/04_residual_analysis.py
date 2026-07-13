"""
Residual analysis pipeline for CA-EDT-AHMA.

Stage:
1. Calculate actual X_s minus ensemble-predicted X_s

Output:
data/outputs/residuals.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/04_residual_analysis.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.digital_twin.residual_calculator import ResidualCalculator
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class ResidualAnalysisPipeline:
    """
    Residual analysis pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize residual analysis pipeline.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/04_residual_analysis.py::__init__")
        Config.create_directories()

    def run(self) -> Dict[str, object]:
        """
        Run residual analysis safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/04_residual_analysis.py::run")
        try:
            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            result: StageResult = run_stage_safely(
                "residual_calculation",
                ResidualCalculator().run,
            )
            completed.append(result.__dict__)

            if result.status == "failed":
                failed.append(result.__dict__)

            status = "success" if not failed else "partial_failure"

            summary = {
                "status": status,
                "message": (
                    "Residual analysis completed."
                    if status == "success"
                    else "Residual analysis failed safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": str(Config.RESIDUALS_CSV) if Config.RESIDUALS_CSV.exists() else None,
            }

            atomic_write_json(summary, Config.REPORT_DIR / "04_residual_analysis_summary.json")
            logger.info("Residual analysis pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Residual analysis pipeline failed.")
            raise RuntimeError("Residual analysis pipeline failed.") from exc


def run_residual_analysis_pipeline() -> Dict[str, object]:
    """
    Execute residual analysis pipeline.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/04_residual_analysis.py::run_residual_analysis_pipeline")
    pipeline = ResidualAnalysisPipeline()
    return pipeline.run()


if __name__ == "__main__":
    result = run_residual_analysis_pipeline()
    print(result)