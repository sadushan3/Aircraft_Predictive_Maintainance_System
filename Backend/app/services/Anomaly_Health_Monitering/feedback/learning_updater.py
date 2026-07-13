"""
Feedback learning updater for CA-EDT-AHMA.

Role:
Run complete feedback learning stage:
1. Initialize feedback store
2. Update alert memory
3. Adapt thresholds from feedback

Reads/Writes:
data/outputs/feedback_updates.csv
data/outputs/alert_memory.csv
models/feedback/adaptive_thresholds.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/feedback/learning_updater.py")
from typing import Dict, Optional

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.feedback.alert_memory import AlertMemory
from app.services.Anomaly_Health_Monitering.feedback.feedback_store import FeedbackStore
from app.services.Anomaly_Health_Monitering.feedback.threshold_adapter import ThresholdAdapter
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class LearningUpdater:
    """
    Complete feedback learning updater.
    """

    def __init__(self) -> None:
        """
        Initialize feedback learning updater.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/learning_updater.py::__init__")
        Config.create_directories()
        self.feedback_store = FeedbackStore()
        self.alert_memory = AlertMemory()
        self.threshold_adapter = ThresholdAdapter()

    def submit_feedback(
        self,
        unit_id: int,
        cycle: int,
        context_id: int,
        alert_level: str,
        final_anomaly_score: float,
        root_cause_pattern: str,
        feedback_label: str,
        operator_note: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Submit one feedback record and update learning outputs.

        Args:
            unit_id: Unit id.
            cycle: Cycle.
            context_id: Context id.
            alert_level: Alert level.
            final_anomaly_score: Final anomaly score.
            root_cause_pattern: Root-cause pattern.
            feedback_label: Feedback label.
            operator_note: Optional note.

        Returns:
            Dict[str, object]: Update response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/learning_updater.py::submit_feedback")
        try:
            feedback_df = self.feedback_store.store_feedback(
                unit_id=unit_id,
                cycle=cycle,
                context_id=context_id,
                alert_level=alert_level,
                final_anomaly_score=final_anomaly_score,
                root_cause_pattern=root_cause_pattern,
                feedback_label=feedback_label,
                operator_note=operator_note,
            )

            memory_result = self.alert_memory.run()
            threshold_result = self.threshold_adapter.run()

            return {
                "status": "success",
                "message": "Feedback submitted and learning updater executed.",
                "output_file": str(Config.FEEDBACK_UPDATES_CSV),
                "records_count": len(feedback_df),
                "data": {
                    "alert_memory": memory_result,
                    "threshold_adapter": threshold_result,
                },
            }

        except Exception as exc:
            logger.exception("Feedback submission and learning update failed.")
            raise RuntimeError("Feedback submission and learning update failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run feedback learning initialization/update.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/learning_updater.py::run")
        try:
            feedback_result = self.feedback_store.run()

            memory_result = {
                "status": "skipped",
                "message": "Alert memory requires anomaly and root-cause CSV files.",
            }
            if Config.ANOMALY_FUSION_CSV.exists() and Config.ROOT_CAUSE_CSV.exists():
                memory_result = self.alert_memory.run()

            threshold_result = {
                "status": "skipped",
                "message": "Threshold adaptation requires residual thresholds JSON.",
            }
            if Config.RESIDUAL_THRESHOLDS_PATH.exists():
                threshold_result = self.threshold_adapter.run()

            return {
                "status": "success",
                "message": "Feedback learning stage completed safely.",
                "output_file": str(Config.FEEDBACK_UPDATES_CSV),
                "records_count": feedback_result.get("records_count", 0),
                "data": {
                    "feedback_store": feedback_result,
                    "alert_memory": memory_result,
                    "threshold_adapter": threshold_result,
                },
            }

        except Exception as exc:
            logger.exception("Feedback learning updater stage failed.")
            raise RuntimeError("Feedback learning updater stage failed.") from exc


def run_learning_updater() -> Dict[str, object]:
    """
    Execute feedback learning updater.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/learning_updater.py::run_learning_updater")
    updater = LearningUpdater()
    return updater.run()


if __name__ == "__main__":
    result = run_learning_updater()
    print(result)