"""
Uncertainty pipeline for CA-EDT-AHMA.

Stages:
1. Model agreement calculation
2. Confidence estimation

Outputs:
outputs/Anomaly_Health_Monitering/model_agreement.csv
outputs/Anomaly_Health_Monitering/confidence_scores.csv

Important:
- Model agreement must use memory-safe chunk processing.
- Model agreement normalization must be fitted on dev split only.
- Confidence estimation must use aligned chunk processing.
- This pipeline does not use Y_dev/Y_test.
- This pipeline does not predict RUL.
- This pipeline does not make maintenance decisions.
- Failed execution must not delete previous outputs.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "09_uncertainty.py"
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
from app.services.Anomaly_Health_Monitering.uncertainty.confidence_estimator import (
    ConfidenceEstimator,
)
from app.services.Anomaly_Health_Monitering.uncertainty.model_agreement import (
    ModelAgreementCalculator,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


class UncertaintyPipeline:
    """
    Complete uncertainty pipeline.

    This wrapper delegates memory-heavy uncertainty processing to:
    - ModelAgreementCalculator
    - ConfidenceEstimator

    Both services should remain memory-safe independently.
    """

    def __init__(self) -> None:
        """
        Initialize uncertainty pipeline.
        """
        print("[PROGRESS] Entering UncertaintyPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "09_uncertainty_summary.json"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] RF predictions CSV: {Config.RF_PREDICTIONS_CSV}")
        print(f"[PROGRESS] XGB predictions CSV: {Config.XGB_PREDICTIONS_CSV}")
        print(f"[PROGRESS] LGBM predictions CSV: {Config.LGBM_PREDICTIONS_CSV}")
        print(f"[PROGRESS] Model agreement CSV: {Config.MODEL_AGREEMENT_CSV}")
        print(f"[PROGRESS] Confidence CSV: {Config.CONFIDENCE_CSV}")
        print(f"[PROGRESS] Dev split name: {Config.DEV_SPLIT_NAME}")
        print(f"[PROGRESS] Test split name: {Config.TEST_SPLIT_NAME}")

    # ==================================================================================
    # Stage wrappers
    # ==================================================================================

    def _run_model_agreement(self) -> Dict[str, object]:
        """
        Run model agreement calculation.

        Expected service responsibility:
        - Read RF/XGB/LGBM predictions in aligned chunks.
        - Fit disagreement normalization threshold on dev split only.
        - Score dev/test using dev-fitted threshold.
        - Write model_agreement.csv.
        """
        print("[PROGRESS] Starting stage wrapper: model_agreement")
        return ModelAgreementCalculator().run()

    def _run_confidence_estimation(self) -> Dict[str, object]:
        """
        Run confidence estimation.

        Expected service responsibility:
        - Read model agreement, context, health, and scaled features in aligned chunks.
        - Calculate confidence_score, uncertainty_score, and reliability_score.
        - Exclude T_*, Y_*, RUL/rul/target/label columns from data quality scoring.
        - Write confidence_scores.csv.
        """
        print("[PROGRESS] Starting stage wrapper: confidence_estimation")
        return ConfidenceEstimator().run()

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run uncertainty pipeline safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering UncertaintyPipeline.run")

        try:
            stages: List[tuple[str, Callable[[], Dict[str, object]]]] = [
                ("model_agreement", self._run_model_agreement),
                ("confidence_estimation", self._run_confidence_estimation),
            ]

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                print("=" * 100)
                print(f"[PROGRESS] Running uncertainty stage: {stage_name}")

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    print(
                        "[PROGRESS] Uncertainty pipeline stopped safely after failed stage. "
                        "Previous outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"

            final_outputs = {
                "model_agreement_csv": (
                    str(Config.MODEL_AGREEMENT_CSV)
                    if Config.MODEL_AGREEMENT_CSV.exists()
                    else None
                ),
                "confidence_scores_csv": (
                    str(Config.CONFIDENCE_CSV)
                    if Config.CONFIDENCE_CSV.exists()
                    else None
                ),
                "confidence_config": (
                    str(Config.CONFIDENCE_CONFIG_PATH)
                    if Config.CONFIDENCE_CONFIG_PATH.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Uncertainty pipeline completed."
                    if status == "success"
                    else "Uncertainty pipeline stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": final_outputs["confidence_scores_csv"],
                "final_outputs": final_outputs,
                "pipeline_order": [
                    "model_agreement",
                    "confidence_estimation",
                ],
                "uncertainty_scope": {
                    "model_agreement": (
                        "Measures disagreement among RF, XGBoost, and LightGBM digital twins."
                    ),
                    "confidence_estimation": (
                        "Combines model agreement, context confidence, anomaly persistence, "
                        "and data quality into confidence/reliability/uncertainty scores."
                    ),
                    "outputs": [
                        "model_disagreement",
                        "model_agreement_score",
                        "confidence_score",
                        "uncertainty_score",
                        "reliability_score",
                    ],
                },
                "normalization_scope": {
                    "model_agreement_fit_split": Config.DEV_SPLIT_NAME,
                    "test_split_usage": "score_only",
                    "test_split_used_for_normalization": False,
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Uncertainty is estimated from digital twin disagreement, "
                        "context confidence, anomaly persistence, and data quality. "
                        "Y_dev/Y_test are RUL targets and are intentionally ignored."
                    ),
                },
                "decision_boundary": {
                    "makes_maintenance_scheduling_decisions": False,
                    "note": (
                        "This component provides uncertainty and confidence intelligence only. "
                        "Final maintenance scheduling decisions belong to the autonomous "
                        "maintenance supervisor component."
                    ),
                },
                "leakage_audit": {
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_rul_targets": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "model_agreement_normalization_fit_split": Config.DEV_SPLIT_NAME,
                    "test_split_used_for_normalization": False,
                    "previous_outputs_deleted_on_failure": False,
                },
            }

            print(f"[PROGRESS] Writing uncertainty summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

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
    print("[PROGRESS] Entering run_uncertainty_pipeline")

    pipeline = UncertaintyPipeline()
    return pipeline.run()


if __name__ == "__main__":
    print("[PROGRESS] 09_uncertainty.py execution started")
    result = run_uncertainty_pipeline()
    print("[PROGRESS] 09_uncertainty.py execution finished")
    print(result)