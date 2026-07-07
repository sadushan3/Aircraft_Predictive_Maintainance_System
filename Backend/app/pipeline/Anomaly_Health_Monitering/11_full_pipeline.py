"""
Full CA-EDT-AHMA pipeline.

Full flow:
1. Preprocessing
2. Context modeling
3. Digital twin training/inference
4. Residual analysis
5. Anomaly detection
6. Health monitoring
7. Reasoning
8. Uncertainty
9. Explainability
10. Feedback learning
11. Dashboard generation
12. Evaluation

Safety:
If one stage fails, the pipeline stops safely.
Previously generated CSV files, models, reports, and metrics are not deleted.
"""

from __future__ import annotations

from typing import Dict, List

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.cleaner import DataCleaner
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.data_loader import DataLoader
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.feature_engineering import FeatureEngineer
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.scaler import FeatureScaler
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
from app.services.Anomaly_Health_Monitering.context_modeling.context_drift import ContextDriftDetector
from app.services.Anomaly_Health_Monitering.context_modeling.operating_mode_detector import (
    OperatingModeDetector,
)
from app.services.Anomaly_Health_Monitering.dashboard.dashboard_data_generator import (
    DashboardDataGenerator,
)
from app.services.Anomaly_Health_Monitering.digital_twin.ensemble_twin import EnsembleDigitalTwin
from app.services.Anomaly_Health_Monitering.digital_twin.lightgbm_twin import LightGBMTwin
from app.services.Anomaly_Health_Monitering.digital_twin.random_forest_twin import RandomForestTwin
from app.services.Anomaly_Health_Monitering.digital_twin.residual_calculator import ResidualCalculator
from app.services.Anomaly_Health_Monitering.digital_twin.twin_comparator import TwinComparator
from app.services.Anomaly_Health_Monitering.digital_twin.xgboost_twin import XGBoostTwin
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_anomaly import AnomalyEvaluator
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_context import ContextEvaluator
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_digital_twin import DigitalTwinEvaluator
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_explainability import (
    ExplainabilityEvaluator,
)
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_health import HealthEvaluator
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_reasoning import ReasoningEvaluator
from app.services.Anomaly_Health_Monitering.explainability.explanation_generator import (
    ExplanationGenerator,
)
from app.services.Anomaly_Health_Monitering.explainability.sensor_residual_ranking import (
    SensorResidualRanking,
)
from app.services.Anomaly_Health_Monitering.explainability.subsystem_explainer import (
    SubsystemExplainer,
)
from app.services.Anomaly_Health_Monitering.feedback.alert_memory import AlertMemory
from app.services.Anomaly_Health_Monitering.feedback.feedback_store import FeedbackStore
from app.services.Anomaly_Health_Monitering.feedback.threshold_adapter import ThresholdAdapter
from app.services.Anomaly_Health_Monitering.health_monitoring.health_alert_engine import HealthAlertEngine
from app.services.Anomaly_Health_Monitering.health_monitoring.health_index_calculator import (
    HealthIndexCalculator,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_state_classifier import (
    HealthStateClassifier,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_trend_tracker import HealthTrendTracker
from app.services.Anomaly_Health_Monitering.reasoning.root_cause_analyzer import RootCauseAnalyzer
from app.services.Anomaly_Health_Monitering.reasoning.root_cause_tracker import RootCauseTracker
from app.services.Anomaly_Health_Monitering.reasoning.sensor_dependency_graph import SensorDependencyGraph
from app.services.Anomaly_Health_Monitering.reasoning.temporal_reasoning import TemporalReasoning
from app.services.Anomaly_Health_Monitering.uncertainty.confidence_estimator import ConfidenceEstimator
from app.services.Anomaly_Health_Monitering.uncertainty.model_agreement import ModelAgreementCalculator
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class FullPipeline:
    """
    Full production-grade CA-EDT-AHMA pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize full pipeline.
        """
        Config.create_directories()

    def run(self, include_shap: bool = False) -> Dict[str, object]:
        """
        Run the full pipeline safely.

        Args:
            include_shap: Whether to run SHAP explanations.

        Returns:
            Dict[str, object]: Full pipeline summary.
        """
        try:
            stages = [
                ("01_data_loading", DataLoader().save_raw_data),
                ("02_cleaning", DataCleaner().run),
                ("03_feature_engineering", FeatureEngineer().run),
                ("04_dev_only_scaling", FeatureScaler().run),

                ("05_context_modeling", OperatingModeDetector().run),
                ("06_context_drift", ContextDriftDetector().run),

                ("07_random_forest_twin", RandomForestTwin().run),
                ("08_xgboost_twin", XGBoostTwin().run),
                ("09_lightgbm_twin", LightGBMTwin().run),
                ("10_ensemble_twin", EnsembleDigitalTwin().run),
                ("11_twin_comparison", TwinComparator().run),

                ("12_residual_analysis", ResidualCalculator().run),

                ("13_residual_anomaly_detector", ResidualAnomalyDetector().run),
                ("14_isolation_forest_detector", IsolationForestDetector().run),
                ("15_mahalanobis_detector", MahalanobisDetector().run),
                ("16_anomaly_fusion", AnomalyFusion().run),
                ("17_severity_classifier", SeverityClassifier().run),
                ("18_early_warning_score", EarlyWarningScore().run),

                ("19_health_index", HealthIndexCalculator().run),
                ("20_health_state", HealthStateClassifier().run),
                ("21_health_trend", HealthTrendTracker().run),
                ("22_health_alerts", HealthAlertEngine().run),

                ("23_sensor_dependency_graph", SensorDependencyGraph().run),
                ("24_root_cause_analysis", RootCauseAnalyzer().run),
                ("25_root_cause_tracking", RootCauseTracker().run),
                ("26_temporal_reasoning", TemporalReasoning().run),

                ("27_model_agreement", ModelAgreementCalculator().run),
                ("28_confidence_estimation", ConfidenceEstimator().run),

                ("29_sensor_residual_ranking", SensorResidualRanking().run),
                ("30_subsystem_explainer", SubsystemExplainer().run),
                ("31_explanation_generator", ExplanationGenerator().run),

                ("32_feedback_store", FeedbackStore().run),
                ("33_alert_memory", AlertMemory().run),
                ("34_threshold_adapter", ThresholdAdapter().run),

                ("35_dashboard_data_generation", DashboardDataGenerator().run),

                ("36_evaluate_digital_twin", DigitalTwinEvaluator().run),
                ("37_evaluate_context", ContextEvaluator().run),
                ("38_evaluate_anomaly", AnomalyEvaluator().run),
                ("39_evaluate_health", HealthEvaluator().run),
                ("40_evaluate_reasoning", ReasoningEvaluator().run),
                ("41_evaluate_explainability", ExplainabilityEvaluator().run),
            ]

            if include_shap:
                from app.services.Anomaly_Health_Monitering.explainability.shap_explainer import SHAPExplainer

                stages.insert(31, ("31a_shap_explainer", SHAPExplainer().run))

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                result: StageResult = run_stage_safely(stage_name, stage_function)
                completed.append(result.__dict__)

                if result.status == "failed":
                    failed.append(result.__dict__)
                    logger.error("Full pipeline stopped at failed stage: %s", stage_name)
                    break

            status = "success" if not failed else "partial_failure"

            summary = {
                "status": status,
                "message": (
                    "Full CA-EDT-AHMA pipeline completed successfully."
                    if status == "success"
                    else "Full pipeline stopped safely. Previous successful outputs were not deleted."
                ),
                "completed_stage_count": len(completed),
                "completed_stages": completed,
                "failed_stages": failed,
                "dashboard_file": str(Config.DASHBOARD_CSV) if Config.DASHBOARD_CSV.exists() else None,
                "final_model_name": Config.FULL_MODEL_NAME,
                "rul_prediction_used": False,
                "y_dev_y_test_used": False,
                "fit_rule": (
                    "All scalers, context models, digital twin models, residual thresholds, "
                    "Isolation Forest, and Mahalanobis parameters are fitted on dev only. "
                    "Test split is used for transform, inference, scoring, and evaluation only."
                ),
                "safety_rule": (
                    "No pipeline stage deletes previous successful outputs. "
                    "Writes are atomic where relevant."
                ),
                "shap_enabled": include_shap,
            }

            atomic_write_json(summary, Config.REPORT_DIR / "11_full_pipeline_summary.json")
            logger.info("Full pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Full pipeline failed.")
            raise RuntimeError("Full pipeline failed.") from exc


def run_full_pipeline(include_shap: bool = False) -> Dict[str, object]:
    """
    Execute full CA-EDT-AHMA pipeline.

    Args:
        include_shap: Whether to run SHAP explanations.

    Returns:
        Dict[str, object]: Full pipeline result.
    """
    pipeline = FullPipeline()
    return pipeline.run(include_shap=include_shap)


if __name__ == "__main__":
    result = run_full_pipeline(include_shap=False)
    print(result)