"""
Explainability pipeline for CA-EDT-AHMA.

Stages:
1. Sensor residual ranking
2. Subsystem explanation
3. Human-readable explanation generation
4. Optional SHAP explanation

SHAP can be computationally expensive on large files, so it can be disabled.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/08_explainability.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.explainability.explanation_generator import (
    ExplanationGenerator,
)
from app.services.Anomaly_Health_Monitering.explainability.sensor_residual_ranking import (
    SensorResidualRanking,
)
from app.services.Anomaly_Health_Monitering.explainability.shap_explainer import SHAPExplainer
from app.services.Anomaly_Health_Monitering.explainability.subsystem_explainer import (
    SubsystemExplainer,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class ExplainabilityPipeline:
    """
    Complete explainability pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize explainability pipeline.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/08_explainability.py::__init__")
        Config.create_directories()

    def run(self, include_shap: bool = False) -> Dict[str, object]:
        """
        Run explainability pipeline safely.

        Args:
            include_shap: Whether to calculate SHAP explanations.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/08_explainability.py::run")
        try:
            stages = [
                ("sensor_residual_ranking", SensorResidualRanking().run),
                ("subsystem_explainer", SubsystemExplainer().run),
                ("explanation_generator", ExplanationGenerator().run),
            ]

            if include_shap:
                stages.append(("shap_explainer", SHAPExplainer().run))

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
                    "Explainability pipeline completed."
                    if status == "success"
                    else "Explainability stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": (
                    str(Config.EXPLANATION_REPORTS_CSV)
                    if Config.EXPLANATION_REPORTS_CSV.exists()
                    else None
                ),
                "shap_enabled": include_shap,
            }

            atomic_write_json(summary, Config.REPORT_DIR / "08_explainability_summary.json")
            logger.info("Explainability pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Explainability pipeline failed.")
            raise RuntimeError("Explainability pipeline failed.") from exc


def run_explainability_pipeline(include_shap: bool = False) -> Dict[str, object]:
    """
    Execute explainability pipeline.

    Args:
        include_shap: Whether to run SHAP.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/08_explainability.py::run_explainability_pipeline")
    pipeline = ExplainabilityPipeline()
    return pipeline.run(include_shap=include_shap)


if __name__ == "__main__":
    result = run_explainability_pipeline(include_shap=False)
    print(result)