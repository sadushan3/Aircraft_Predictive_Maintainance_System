"""
Feedback learning pipeline for CA-EDT-AHMA.

Stages:
1. Feedback store initialization
2. Alert memory update
3. Threshold adaptation

Outputs:
outputs/Anomaly_Health_Monitering/feedback_updates.csv
outputs/Anomaly_Health_Monitering/alert_memory.csv
models/feedback/adaptive_thresholds.json

Important:
- Feedback is operator-provided evidence only.
- Alert memory should store alert rows only: Watch, Warning, Critical.
- Threshold adaptation must start from base residual thresholds each run.
- This pipeline does not train models.
- This pipeline does not predict RUL.
- This pipeline does not use Y_dev/Y_test.
- This pipeline does not make maintenance decisions.
- Failed execution must not delete previous outputs.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "10_feedback_learning.py"
)

from typing import Callable, Dict, List
import os
import sys


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    if BACKEND_ROOT not in sys.path:
        sys.path.append(BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.feedback.alert_memory import AlertMemory
from app.services.Anomaly_Health_Monitering.feedback.feedback_store import FeedbackStore
from app.services.Anomaly_Health_Monitering.feedback.threshold_adapter import (
    ThresholdAdapter,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


class FeedbackLearningPipeline:
    """
    Complete feedback learning pipeline.

    This wrapper delegates work to:
    - FeedbackStore
    - AlertMemory
    - ThresholdAdapter

    Each service should remain memory-safe independently.
    """

    def __init__(self) -> None:
        """
        Initialize feedback learning pipeline.
        """
        print("[PROGRESS] Entering FeedbackLearningPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "10_feedback_learning_summary.json"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Feedback updates CSV: {Config.FEEDBACK_UPDATES_CSV}")
        print(f"[PROGRESS] Alert memory CSV: {Config.ALERT_MEMORY_CSV}")
        print(f"[PROGRESS] Residual thresholds JSON: {Config.RESIDUAL_THRESHOLDS_PATH}")
        print(f"[PROGRESS] Adaptive thresholds JSON: {Config.ADAPTIVE_THRESHOLDS_PATH}")
        print(f"[PROGRESS] Anomaly fusion CSV: {Config.ANOMALY_FUSION_CSV}")
        print(f"[PROGRESS] Root-cause CSV: {Config.ROOT_CAUSE_CSV}")

    # ==================================================================================
    # Stage wrappers
    # ==================================================================================

    def _run_feedback_store(self) -> Dict[str, object]:
        """
        Initialize feedback store.

        Expected service responsibility:
        - Create feedback_updates.csv if missing.
        - Preserve existing feedback records.
        - Store operator feedback only.
        """
        print("[PROGRESS] Starting stage wrapper: feedback_store")
        return FeedbackStore().run()

    def _run_alert_memory(self) -> Dict[str, object]:
        """
        Run alert memory update.

        Expected service responsibility:
        - Read anomaly_fusion.csv and root_cause_analysis.csv safely.
        - Store only Watch / Warning / Critical alert rows.
        - Apply feedback status if feedback exists.
        - Write alert_memory.csv.
        """
        print("[PROGRESS] Starting stage wrapper: alert_memory")
        return AlertMemory().run()

    def _run_threshold_adapter(self) -> Dict[str, object]:
        """
        Run threshold adaptation.

        Expected service responsibility:
        - Start from base residual thresholds every run.
        - Apply feedback-driven adjustments.
        - Write adaptive_thresholds.json.
        - Avoid repeated compounding from previously adapted thresholds.
        """
        print("[PROGRESS] Starting stage wrapper: threshold_adapter")
        return ThresholdAdapter().run()

    # ==================================================================================
    # Stage planning
    # ==================================================================================

    def _build_stages(self) -> List[tuple[str, Callable[[], Dict[str, object]]]]:
        """
        Build feedback learning stage list based on available upstream outputs.
        """
        stages: List[tuple[str, Callable[[], Dict[str, object]]]] = [
            ("feedback_store", self._run_feedback_store),
        ]

        if Config.ANOMALY_FUSION_CSV.exists() and Config.ROOT_CAUSE_CSV.exists():
            stages.append(("alert_memory", self._run_alert_memory))
        else:
            print(
                "[PROGRESS] Skipping alert_memory because anomaly_fusion.csv "
                "or root_cause_analysis.csv is missing."
            )

        if Config.RESIDUAL_THRESHOLDS_PATH.exists():
            stages.append(("threshold_adapter", self._run_threshold_adapter))
        else:
            print(
                "[PROGRESS] Skipping threshold_adapter because residual_thresholds.json "
                "is missing."
            )

        return stages

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run feedback learning safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering FeedbackLearningPipeline.run")

        try:
            stages = self._build_stages()

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                print("=" * 100)
                print(f"[PROGRESS] Running feedback learning stage: {stage_name}")

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    print(
                        "[PROGRESS] Feedback learning stopped safely after failed stage. "
                        "Previous outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"

            skipped_stages = []

            if not (
                Config.ANOMALY_FUSION_CSV.exists()
                and Config.ROOT_CAUSE_CSV.exists()
            ):
                skipped_stages.append(
                    {
                        "stage": "alert_memory",
                        "reason": (
                            "Requires anomaly_fusion.csv and root_cause_analysis.csv."
                        ),
                    }
                )

            if not Config.RESIDUAL_THRESHOLDS_PATH.exists():
                skipped_stages.append(
                    {
                        "stage": "threshold_adapter",
                        "reason": "Requires residual_thresholds.json.",
                    }
                )

            final_outputs = {
                "feedback_updates_csv": (
                    str(Config.FEEDBACK_UPDATES_CSV)
                    if Config.FEEDBACK_UPDATES_CSV.exists()
                    else None
                ),
                "alert_memory_csv": (
                    str(Config.ALERT_MEMORY_CSV)
                    if Config.ALERT_MEMORY_CSV.exists()
                    else None
                ),
                "adaptive_thresholds_json": (
                    str(Config.ADAPTIVE_THRESHOLDS_PATH)
                    if Config.ADAPTIVE_THRESHOLDS_PATH.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Feedback learning pipeline completed."
                    if status == "success"
                    else "Feedback learning stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "skipped_stages": skipped_stages,
                "final_output_file": final_outputs["feedback_updates_csv"],
                "final_outputs": final_outputs,
                "pipeline_order": [
                    "feedback_store",
                    "alert_memory_if_upstream_outputs_exist",
                    "threshold_adapter_if_residual_thresholds_exist",
                ],
                "feedback_scope": {
                    "allowed_feedback_labels": list(Config.FEEDBACK_LABELS),
                    "alert_memory_rows": "Watch / Warning / Critical only",
                    "threshold_adaptation": (
                        "Feedback-driven threshold adjustment from base residual thresholds."
                    ),
                    "operator_feedback_only": True,
                },
                "threshold_adaptation_policy": {
                    "rejected_false_alarm": "increase thresholds slightly",
                    "missed_anomaly": "decrease thresholds slightly",
                    "accepted_alert": "keep thresholds",
                    "uncertain": "no change",
                    "starts_from_base_thresholds_each_run": True,
                    "avoids_repeated_compounding": True,
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Feedback learning uses operator feedback, anomaly outputs, "
                        "root-cause outputs, and residual thresholds. Y_dev/Y_test are "
                        "RUL targets and are intentionally ignored."
                    ),
                },
                "decision_boundary": {
                    "makes_maintenance_scheduling_decisions": False,
                    "note": (
                        "Feedback learning adapts anomaly threshold behavior only. "
                        "It does not schedule maintenance."
                    ),
                },
                "leakage_audit": {
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_rul_targets": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "operator_feedback_only": True,
                    "previous_outputs_deleted_on_failure": False,
                },
            }

            print(f"[PROGRESS] Writing feedback learning summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

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
    print("[PROGRESS] Entering run_feedback_learning_pipeline")

    pipeline = FeedbackLearningPipeline()
    return pipeline.run()


if __name__ == "__main__":
    print("[PROGRESS] 10_feedback_learning.py execution started")
    result = run_feedback_learning_pipeline()
    print("[PROGRESS] 10_feedback_learning.py execution finished")
    print(result)