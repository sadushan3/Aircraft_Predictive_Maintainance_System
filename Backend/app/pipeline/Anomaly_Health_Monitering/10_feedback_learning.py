"""
Feedback learning pipeline for CA-EDT-AHMA.

Stages:
1. Feedback store initialization
2. Alert memory update
3. Threshold adaptation

Outputs:
data/outputs/feedback_updates.csv
data/outputs/alert_memory.csv
models/feedback/adaptive_thresholds.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/10_feedback_learning.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.feedback.alert_memory import AlertMemory
from app.services.Anomaly_Health_Monitering.feedback.feedback_store import FeedbackStore
from app.services.Anomaly_Health_Monitering.feedback.threshold_adapter import ThresholdAdapter
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class FeedbackLearningPipeline:
    """
    Complete feedback learning pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize feedback learning pipeline.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/10_feedback_learning.py::__init__")
        Config.create_directories()

    def run(self) -> Dict[str, object]:
        """
        Run feedback learning safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/10_feedback_learning.py::run")
        try:
            stages = [
                ("feedback_store", FeedbackStore().run),
            ]

            if Config.ANOMALY_FUSION_CSV.exists() and Config.ROOT_CAUSE_CSV.exists():
                stages.append(("alert_memory", AlertMemory().run))

            if Config.RESIDUAL_THRESHOLDS_PATH.exists():
                stages.append(("threshold_adapter", ThresholdAdapter().run))

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
                    "Feedback learning pipeline completed."
                    if status == "success"
                    else "Feedback learning stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": (
                    str(Config.FEEDBACK_UPDATES_CSV)
                    if Config.FEEDBACK_UPDATES_CSV.exists()
                    else None
                ),
            }

            atomic_write_json(summary, Config.REPORT_DIR / "10_feedback_learning_summary.json")
            logger.info("Feedback learning pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Feedback learning pipeline failed.")
            raise RuntimeError("Feedback learning pipeline failed.") from exc


def run_feedback_learning_pipeline() -> Dict[str, object]:
    """
    Execute feedback learning pipeline.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/10_feedback_learning.py::run_feedback_learning_pipeline")
    pipeline = FeedbackLearningPipeline()
    return pipeline.run()


if __name__ == "__main__":
    result = run_feedback_learning_pipeline()
    print(result)