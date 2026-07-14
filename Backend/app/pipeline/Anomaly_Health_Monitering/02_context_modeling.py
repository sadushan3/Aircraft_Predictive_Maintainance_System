"""
Context modeling pipeline for CA-EDT-AHMA.

Stages:
1. K-Means baseline context modeling
2. GMM probabilistic context modeling
3. Combined operating mode detection
4. Context drift detection

Important:
- K-Means and GMM must be fitted only on W_dev.
- Test split is used only for prediction/scoring.
- This pipeline does not use Y_dev/Y_test.
- This pipeline does not predict RUL.
- This pipeline does not make maintenance decisions.
- Failed execution must not delete previous outputs.

Note:
OperatingModeDetector is expected to run/generate the combined K-Means + GMM
context output. If K-Means and GMM are separate services in your project,
they should be called inside OperatingModeDetector or added as separate stages.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "02_context_modeling.py"
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
from app.services.Anomaly_Health_Monitering.context_modeling.context_drift import (
    ContextDriftDetector,
)
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

    This wrapper delegates memory-heavy work to context modeling services.
    The services themselves must remain memory-safe and dev-fit/test-score only.
    """

    def __init__(self) -> None:
        """
        Initialize context modeling pipeline.
        """
        print("[PROGRESS] Entering ContextModelingPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "02_context_modeling_summary.json"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Scaled CSV: {Config.SCALED_CSV}")
        print(f"[PROGRESS] Context CSV: {Config.CONTEXT_CSV}")
        print(f"[PROGRESS] KMeans model path: {Config.KMEANS_MODEL_PATH}")
        print(f"[PROGRESS] GMM model path: {Config.GMM_MODEL_PATH}")
        print(f"[PROGRESS] Dev split name: {Config.DEV_SPLIT_NAME}")
        print(f"[PROGRESS] Test split name: {Config.TEST_SPLIT_NAME}")

    # ==================================================================================
    # Stage wrappers
    # ==================================================================================

    def _run_operating_mode_detection(self) -> Dict[str, object]:
        """
        Run operating mode detection.

        Expected service responsibility:
        - Fit K-Means/GMM on dev operating conditions only.
        - Predict/score context for dev and test.
        - Write context_clusters.csv.
        """
        print("[PROGRESS] Starting stage wrapper: operating_mode_detection")
        return OperatingModeDetector().run()

    def _run_context_drift_detection(self) -> Dict[str, object]:
        """
        Run context drift detection.

        Expected service responsibility:
        - Fit/reference thresholds using dev split only.
        - Score dev/test without using test to fit thresholds.
        """
        print("[PROGRESS] Starting stage wrapper: context_drift_detection")
        return ContextDriftDetector().run()

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self, include_drift: bool = True) -> Dict[str, object]:
        """
        Run context modeling pipeline safely.

        Args:
            include_drift: Whether to run context drift scoring.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering ContextModelingPipeline.run")
        print(f"[PROGRESS] include_drift={include_drift}")

        try:
            stages: List[tuple[str, Callable[[], Dict[str, object]]]] = [
                ("operating_mode_detection", self._run_operating_mode_detection),
            ]

            if include_drift:
                stages.append(
                    ("context_drift_detection", self._run_context_drift_detection)
                )

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                print("=" * 100)
                print(f"[PROGRESS] Running context modeling stage: {stage_name}")

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    print(
                        "[PROGRESS] Context modeling stopped safely after failed stage. "
                        "Previous outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"

            final_outputs = {
                "context_csv": (
                    str(Config.CONTEXT_CSV) if Config.CONTEXT_CSV.exists() else None
                ),
                "kmeans_model": (
                    str(Config.KMEANS_MODEL_PATH)
                    if Config.KMEANS_MODEL_PATH.exists()
                    else None
                ),
                "gmm_model": (
                    str(Config.GMM_MODEL_PATH)
                    if Config.GMM_MODEL_PATH.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Context modeling pipeline completed."
                    if status == "success"
                    else "Context modeling stopped safely. Previous outputs were not deleted."
                ),
                "include_drift": bool(include_drift),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": final_outputs["context_csv"],
                "final_outputs": final_outputs,
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "note": (
                        "Context modeling uses operating-condition/context features only. "
                        "Y_dev/Y_test are RUL targets and are intentionally ignored."
                    ),
                },
                "leakage_audit": {
                    "kmeans_fit_split": Config.DEV_SPLIT_NAME,
                    "gmm_fit_split": Config.DEV_SPLIT_NAME,
                    "test_split_usage": "predict_or_score_only",
                    "context_drift_threshold_fit_split": Config.DEV_SPLIT_NAME,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                },
            }

            print(f"[PROGRESS] Writing context modeling summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

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
    print("[PROGRESS] Entering run_context_modeling_pipeline")

    pipeline = ContextModelingPipeline()
    return pipeline.run(include_drift=include_drift)


if __name__ == "__main__":
    print("[PROGRESS] 02_context_modeling.py execution started")
    result = run_context_modeling_pipeline(include_drift=True)
    print("[PROGRESS] 02_context_modeling.py execution finished")
    print(result)