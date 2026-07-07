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
All fitted anomaly models/thresholds use dev residuals only.
Test split is score/inference only.
"""

from __future__ import annotations

from typing import Dict, List

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitering.anomaly_detection.early_warning_score import EarlyWarningScore
from app.services.Anomaly_Health_Monitering.anomaly_detection.isolation_forest_detector import (
    IsolationForestDetector,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.mahalanobis_detector import MahalanobisDetector
from app.services.Anomaly_Health_Monitering.anomaly_detection.residual_anomaly_detector import (
    ResidualAnomalyDetector,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.severity_classifier import SeverityClassifier
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class AnomalyDetectionPipeline:
    """
    Complete anomaly detection pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize anomaly detection pipeline.
        """
        Config.create_directories()

    def run(self) -> Dict[str, object]:
        """
        Run anomaly detection pipeline safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        try:
            stages = [
                ("residual_anomaly_detector", ResidualAnomalyDetector().run),
                ("isolation_forest_detector", IsolationForestDetector().run),
                ("mahalanobis_detector", MahalanobisDetector().run),
                ("anomaly_fusion", AnomalyFusion().run),
                ("severity_classifier", SeverityClassifier().run),
                ("early_warning_score", EarlyWarningScore().run),
            ]

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
                    "Anomaly detection pipeline completed."
                    if status == "success"
                    else "Anomaly detection stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": (
                    str(Config.ANOMALY_FUSION_CSV)
                    if Config.ANOMALY_FUSION_CSV.exists()
                    else None
                ),
            }

            atomic_write_json(summary, Config.REPORT_DIR / "05_anomaly_detection_summary.json")
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
    pipeline = AnomalyDetectionPipeline()
    return pipeline.run()


if __name__ == "__main__":
    result = run_anomaly_detection_pipeline()
    print(result)