"""
Digital twin training and inference pipeline for CA-EDT-AHMA.

Stages:
1. Random Forest twin
2. XGBoost twin
3. LightGBM twin
4. Ensemble digital twin
5. Twin comparison metrics

Important:
RF, XGBoost, and LightGBM are trained only on dev split.
Test split is used only for inference/evaluation.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/03_digital_twin.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.digital_twin.ensemble_twin import EnsembleDigitalTwin
from app.services.Anomaly_Health_Monitering.digital_twin.lightgbm_twin import LightGBMTwin
from app.services.Anomaly_Health_Monitering.digital_twin.random_forest_twin import RandomForestTwin
from app.services.Anomaly_Health_Monitering.digital_twin.twin_comparator import TwinComparator
from app.services.Anomaly_Health_Monitering.digital_twin.xgboost_twin import XGBoostTwin
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class DigitalTwinPipeline:
    """
    Complete digital twin pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize digital twin pipeline.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/03_digital_twin.py::__init__")
        Config.create_directories()

    def run(self, include_comparison: bool = True) -> Dict[str, object]:
        """
        Run digital twin pipeline safely.

        Args:
            include_comparison: Whether to run twin comparison metrics.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/03_digital_twin.py::run")
        try:
            stages = [
                ("random_forest_twin", RandomForestTwin().run),
                ("xgboost_twin", XGBoostTwin().run),
                ("lightgbm_twin", LightGBMTwin().run),
                ("ensemble_twin", EnsembleDigitalTwin().run),
            ]

            if include_comparison:
                stages.append(("twin_comparison", TwinComparator().run))

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
                    "Digital twin pipeline completed."
                    if status == "success"
                    else "Digital twin pipeline stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": (
                    str(Config.ENSEMBLE_PREDICTIONS_CSV)
                    if Config.ENSEMBLE_PREDICTIONS_CSV.exists()
                    else None
                ),
            }

            atomic_write_json(summary, Config.REPORT_DIR / "03_digital_twin_summary.json")
            logger.info("Digital twin pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Digital twin pipeline failed.")
            raise RuntimeError("Digital twin pipeline failed.") from exc


def run_digital_twin_pipeline(include_comparison: bool = True) -> Dict[str, object]:
    """
    Execute digital twin pipeline.

    Args:
        include_comparison: Whether to include twin comparison.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/03_digital_twin.py::run_digital_twin_pipeline")
    pipeline = DigitalTwinPipeline()
    return pipeline.run(include_comparison=include_comparison)


if __name__ == "__main__":
    result = run_digital_twin_pipeline(include_comparison=True)
    print(result)