"""
Feedback learning updater for CA-EDT-AHMA.

Role:
Run complete feedback learning stage:
1. Initialize feedback store
2. Update alert memory
3. Adapt thresholds from feedback

Reads/Writes:
outputs/Anomaly_Health_Monitering/feedback_updates.csv
outputs/Anomaly_Health_Monitering/alert_memory.csv
models/feedback/adaptive_thresholds.json

Important:
- This module orchestrates feedback learning.
- It does not train models.
- It does not predict RUL.
- It does not use Y_dev/Y_test.
- It does not make maintenance decisions.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "feedback/learning_updater.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, Optional
import os
import sys


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    if BACKEND_ROOT not in sys.path:
        sys.path.append(BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.feedback.alert_memory import AlertMemory
from app.services.Anomaly_Health_Monitering.feedback.feedback_store import FeedbackStore
from app.services.Anomaly_Health_Monitering.feedback.threshold_adapter import ThresholdAdapter
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class LearningUpdater:
    """
    Complete feedback learning updater.

    Uses lazy service initialization so heavy stages are only created when needed.
    """

    def __init__(self) -> None:
        """
        Initialize feedback learning updater.
        """
        print("[PROGRESS] Entering LearningUpdater.__init__")

        Config.create_directories()

        self.feedback_store: Optional[FeedbackStore] = None
        self.alert_memory: Optional[AlertMemory] = None
        self.threshold_adapter: Optional[ThresholdAdapter] = None

        self.summary_json: Path = getattr(
            Config,
            "LEARNING_UPDATER_SUMMARY_JSON",
            Config.REPORT_DIR / "learning_updater_summary.json",
        )

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")

    # ==================================================================================
    # Lazy service access
    # ==================================================================================

    def _get_feedback_store(self) -> FeedbackStore:
        if self.feedback_store is None:
            self.feedback_store = FeedbackStore()
        return self.feedback_store

    def _get_alert_memory(self) -> AlertMemory:
        if self.alert_memory is None:
            self.alert_memory = AlertMemory()
        return self.alert_memory

    def _get_threshold_adapter(self) -> ThresholdAdapter:
        if self.threshold_adapter is None:
            self.threshold_adapter = ThresholdAdapter()
        return self.threshold_adapter

    # ==================================================================================
    # Public methods
    # ==================================================================================

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
        print("[PROGRESS] Entering LearningUpdater.submit_feedback")

        try:
            started = perf_counter()

            feedback_result = self._get_feedback_store().store_feedback(
                unit_id=unit_id,
                cycle=cycle,
                context_id=context_id,
                alert_level=alert_level,
                final_anomaly_score=final_anomaly_score,
                root_cause_pattern=root_cause_pattern,
                feedback_label=feedback_label,
                operator_note=operator_note,
            )

            # Update existing alert memory feedback statuses if possible.
            if Config.ALERT_MEMORY_CSV.exists():
                memory_result = self._get_alert_memory().update_feedback_status()
            elif Config.ANOMALY_FUSION_CSV.exists() and Config.ROOT_CAUSE_CSV.exists():
                memory_result = self._get_alert_memory().run()
            else:
                memory_result = {
                    "status": "skipped",
                    "message": "Alert memory requires anomaly_fusion.csv and root_cause_analysis.csv.",
                }

            if Config.RESIDUAL_THRESHOLDS_PATH.exists():
                threshold_result = self._get_threshold_adapter().run()
            else:
                threshold_result = {
                    "status": "skipped",
                    "message": "Threshold adaptation requires residual_thresholds.json.",
                }

            response = {
                "status": "success",
                "message": "Feedback submitted and learning updater executed.",
                "output_file": str(Config.FEEDBACK_UPDATES_CSV),
                "records_count": int(feedback_result.get("records_count", 0)),
                "data": {
                    "feedback_store": feedback_result,
                    "alert_memory": memory_result,
                    "threshold_adapter": threshold_result,
                },
                "duration_seconds": float(perf_counter() - started),
            }

            atomic_write_json(response, self.summary_json)

            return response

        except Exception as exc:
            logger.exception("Feedback submission and learning update failed.")
            raise RuntimeError("Feedback submission and learning update failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run feedback learning initialization/update.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering LearningUpdater.run")

        try:
            started = perf_counter()

            feedback_result = self._get_feedback_store().run()

            if Config.ANOMALY_FUSION_CSV.exists() and Config.ROOT_CAUSE_CSV.exists():
                memory_result = self._get_alert_memory().run()
            else:
                memory_result = {
                    "status": "skipped",
                    "message": "Alert memory requires anomaly_fusion.csv and root_cause_analysis.csv.",
                }

            if Config.RESIDUAL_THRESHOLDS_PATH.exists():
                threshold_result = self._get_threshold_adapter().run()
            else:
                threshold_result = {
                    "status": "skipped",
                    "message": "Threshold adaptation requires residual_thresholds.json.",
                }

            response = {
                "status": "success",
                "message": "Feedback learning stage completed safely.",
                "output_file": str(Config.FEEDBACK_UPDATES_CSV),
                "records_count": int(feedback_result.get("records_count", 0)),
                "data": {
                    "feedback_store": feedback_result,
                    "alert_memory": memory_result,
                    "threshold_adapter": threshold_result,
                },
                "duration_seconds": float(perf_counter() - started),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "operator_feedback_only": True,
                },
            }

            print(f"[PROGRESS] Writing learning updater summary to: {self.summary_json}")
            atomic_write_json(response, self.summary_json)

            return response

        except Exception as exc:
            logger.exception("Feedback learning updater stage failed.")
            raise RuntimeError("Feedback learning updater stage failed.") from exc


def run_learning_updater() -> Dict[str, object]:
    """
    Execute feedback learning updater.
    """
    print("[PROGRESS] Entering run_learning_updater")

    updater = LearningUpdater()
    return updater.run()


if __name__ == "__main__":
    print("[PROGRESS] learning_updater.py execution started")
    result = run_learning_updater()
    print("[PROGRESS] learning_updater.py execution finished successfully")
    print(result)