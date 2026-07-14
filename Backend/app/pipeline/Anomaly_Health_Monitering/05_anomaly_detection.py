"""
Anomaly detection pipeline for CA-EDT-AHMA.

Stages:
1. Residual threshold detector
2. Isolation Forest detector
3. Mahalanobis detector
4. Anomaly fusion
5. Severity classification
6. Early warning scoring

Important:
- All fitted anomaly models/thresholds use dev residuals only.
- Test split is score/inference only.
- This pipeline does not use Y_dev/Y_test.
- This pipeline does not predict RUL.
- This pipeline does not make maintenance decisions.
- Failed execution must not delete previous outputs.

Note:
AnomalyFusion.run() should only perform fusion. Severity classification and
early-warning scoring are separate stages in this pipeline.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "05_anomaly_detection.py"
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
from app.services.Anomaly_Health_Monitering.anomaly_detection.anomaly_fusion import (
    AnomalyFusion,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.early_warning_score import (
    EarlyWarningScore,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.isolation_forest_detector import (
    IsolationForestDetector,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.mahalanobis_detector import (
    MahalanobisDetector,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.residual_anomaly_detector import (
    ResidualAnomalyDetector,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.severity_classifier import (
    SeverityClassifier,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


class AnomalyDetectionPipeline:
    """
    Complete anomaly detection pipeline.

    This wrapper delegates memory-heavy detector work to individual detector
    services. Each detector should remain memory-safe independently.
    """

    def __init__(self) -> None:
        """
        Initialize anomaly detection pipeline.
        """
        print("[PROGRESS] Entering AnomalyDetectionPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "05_anomaly_detection_summary.json"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Residuals CSV: {Config.RESIDUALS_CSV}")
        print(f"[PROGRESS] Residual anomaly CSV: {Config.RESIDUAL_ANOMALY_CSV}")
        print(f"[PROGRESS] Isolation Forest CSV: {Config.IFOREST_CSV}")
        print(f"[PROGRESS] Mahalanobis CSV: {Config.MAHALANOBIS_CSV}")
        print(f"[PROGRESS] Anomaly fusion CSV: {Config.ANOMALY_FUSION_CSV}")
        print(f"[PROGRESS] Dev split name: {Config.DEV_SPLIT_NAME}")
        print(f"[PROGRESS] Test split name: {Config.TEST_SPLIT_NAME}")

    # ==================================================================================
    # Stage wrappers
    # ==================================================================================

    def _run_residual_anomaly_detector(self) -> Dict[str, object]:
        """
        Run residual threshold anomaly detector.

        Expected service responsibility:
        - Fit residual thresholds using dev split only.
        - Score dev/test residuals.
        - Write residual_anomaly_scores.csv.
        """
        print("[PROGRESS] Starting stage wrapper: residual_anomaly_detector")
        return ResidualAnomalyDetector().run()

    def _run_isolation_forest_detector(self) -> Dict[str, object]:
        """
        Run Isolation Forest anomaly detector.

        Expected service responsibility:
        - Fit Isolation Forest using dev residual behavior only.
        - Score dev/test.
        - Write isolation_forest_scores.csv.
        """
        print("[PROGRESS] Starting stage wrapper: isolation_forest_detector")
        return IsolationForestDetector().run()

    def _run_mahalanobis_detector(self) -> Dict[str, object]:
        """
        Run Mahalanobis anomaly detector.

        Expected service responsibility:
        - Fit mean/covariance or robust statistics using dev residuals only.
        - Score dev/test.
        - Write mahalanobis_scores.csv.
        """
        print("[PROGRESS] Starting stage wrapper: mahalanobis_detector")
        return MahalanobisDetector().run()

    def _run_anomaly_fusion(self) -> Dict[str, object]:
        """
        Run anomaly fusion.

        Expected service responsibility:
        - Combine residual, Isolation Forest, and Mahalanobis scores.
        - Write anomaly_fusion.csv.
        - Should not rerun severity or early-warning internally.
        """
        print("[PROGRESS] Starting stage wrapper: anomaly_fusion")
        return AnomalyFusion().run()

    def _run_severity_classifier(self) -> Dict[str, object]:
        """
        Run severity classification.

        Expected service responsibility:
        - Convert final_anomaly_score into Normal/Watch/Warning/Critical.
        - Update or write anomaly_fusion.csv with alert_level.
        """
        print("[PROGRESS] Starting stage wrapper: severity_classifier")
        return SeverityClassifier().run()

    def _run_early_warning_score(self) -> Dict[str, object]:
        """
        Run early warning scoring.

        Expected service responsibility:
        - Calculate rolling anomaly trend behavior.
        - Write early_warning_scores.csv.
        """
        print("[PROGRESS] Starting stage wrapper: early_warning_score")
        return EarlyWarningScore().run()

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run anomaly detection pipeline safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering AnomalyDetectionPipeline.run")

        try:
            stages: List[tuple[str, Callable[[], Dict[str, object]]]] = [
                ("residual_anomaly_detector", self._run_residual_anomaly_detector),
                ("isolation_forest_detector", self._run_isolation_forest_detector),
                ("mahalanobis_detector", self._run_mahalanobis_detector),
                ("anomaly_fusion", self._run_anomaly_fusion),
                ("severity_classifier", self._run_severity_classifier),
                ("early_warning_score", self._run_early_warning_score),
            ]

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                print("=" * 100)
                print(f"[PROGRESS] Running anomaly detection stage: {stage_name}")

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    print(
                        "[PROGRESS] Anomaly detection stopped safely after failed stage. "
                        "Previous outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"

            early_warning_csv = Config.OUTPUT_DIR / "early_warning_scores.csv"

            final_outputs = {
                "residual_anomaly_csv": (
                    str(Config.RESIDUAL_ANOMALY_CSV)
                    if Config.RESIDUAL_ANOMALY_CSV.exists()
                    else None
                ),
                "isolation_forest_csv": (
                    str(Config.IFOREST_CSV)
                    if Config.IFOREST_CSV.exists()
                    else None
                ),
                "mahalanobis_csv": (
                    str(Config.MAHALANOBIS_CSV)
                    if Config.MAHALANOBIS_CSV.exists()
                    else None
                ),
                "anomaly_fusion_csv": (
                    str(Config.ANOMALY_FUSION_CSV)
                    if Config.ANOMALY_FUSION_CSV.exists()
                    else None
                ),
                "early_warning_csv": (
                    str(early_warning_csv)
                    if early_warning_csv.exists()
                    else None
                ),
                "residual_thresholds": (
                    str(Config.RESIDUAL_THRESHOLDS_PATH)
                    if Config.RESIDUAL_THRESHOLDS_PATH.exists()
                    else None
                ),
                "isolation_forest_model": (
                    str(Config.IFOREST_MODEL_PATH)
                    if Config.IFOREST_MODEL_PATH.exists()
                    else None
                ),
                "mahalanobis_params": (
                    str(Config.MAHALANOBIS_PARAMS_PATH)
                    if Config.MAHALANOBIS_PARAMS_PATH.exists()
                    else None
                ),
                "fusion_weights": (
                    str(Config.FUSION_WEIGHTS_PATH)
                    if Config.FUSION_WEIGHTS_PATH.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Anomaly detection pipeline completed."
                    if status == "success"
                    else "Anomaly detection stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": final_outputs["anomaly_fusion_csv"],
                "final_outputs": final_outputs,
                "pipeline_order": [
                    "residual_anomaly_detector",
                    "isolation_forest_detector",
                    "mahalanobis_detector",
                    "anomaly_fusion",
                    "severity_classifier",
                    "early_warning_score",
                ],
                "anomaly_scope": {
                    "detectors": [
                        "residual_threshold_detector",
                        "isolation_forest_detector",
                        "mahalanobis_detector",
                    ],
                    "fusion_output": "final_anomaly_score",
                    "severity_output": "alert_level",
                    "early_warning_output": "early_warning_score",
                    "alert_levels": ["Normal", "Watch", "Warning", "Critical"],
                },
                "training_scope": {
                    "fit_split": Config.DEV_SPLIT_NAME,
                    "test_usage": "score_or_inference_only",
                    "fit_data": "dev residual behavior only",
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Anomaly detection is residual/score based. Y_dev/Y_test are RUL "
                        "targets and are intentionally ignored."
                    ),
                },
                "leakage_audit": {
                    "residual_threshold_fit_split": Config.DEV_SPLIT_NAME,
                    "isolation_forest_fit_split": Config.DEV_SPLIT_NAME,
                    "mahalanobis_fit_split": Config.DEV_SPLIT_NAME,
                    "test_split_usage": "score_or_inference_only",
                    "does_not_use_y_dev_y_test": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "previous_outputs_deleted_on_failure": False,
                },
            }

            print(f"[PROGRESS] Writing anomaly detection summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info("Anomaly detection pipeline finished with status=%s.", status)

            return summary

        except Exception as exc:
            logger.exception("Anomaly detection pipeline failed.")
            raise RuntimeError("Anomaly detection pipeline failed.") from exc


def run_anomaly_detection_pipeline() -> Dict[str, object]:
    """
    Execute anomaly detection pipeline.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering run_anomaly_detection_pipeline")

    pipeline = AnomalyDetectionPipeline()
    return pipeline.run()


if __name__ == "__main__":
    print("[PROGRESS] 05_anomaly_detection.py execution started")
    result = run_anomaly_detection_pipeline()
    print("[PROGRESS] 05_anomaly_detection.py execution finished")
    print(result)