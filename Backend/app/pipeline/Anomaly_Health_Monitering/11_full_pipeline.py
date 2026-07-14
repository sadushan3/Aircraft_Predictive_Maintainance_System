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
- If one stage fails, the pipeline stops safely.
- Previously generated CSV files, models, reports, and metrics are not deleted.
- Heavy services are imported and initialized lazily only when their stage runs.
- This full pipeline does not directly perform heavy dataframe work.
- Memory safety must be handled inside each service stage.
- Y_dev/Y_test are never used because they are RUL targets.
- This component does not predict RUL.
- This component does not make final maintenance scheduling decisions.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "11_full_pipeline.py"
)

from importlib import import_module
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple
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
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


StageSpec = Tuple[str, Callable[[], Dict[str, object]]]


class FullPipeline:
    """
    Full production-grade CA-EDT-AHMA pipeline.

    This class is only an orchestrator. It does not load large CSV files.
    Every heavy stage is delegated to the corresponding service.
    """

    def __init__(self) -> None:
        """
        Initialize full pipeline.
        """
        print("[PROGRESS] Entering FullPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "11_full_pipeline_summary.json"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Dashboard CSV: {Config.DASHBOARD_CSV}")
        print(f"[PROGRESS] Dev split name: {Config.DEV_SPLIT_NAME}")
        print(f"[PROGRESS] Test split name: {Config.TEST_SPLIT_NAME}")

    # ==================================================================================
    # Lazy import helpers
    # ==================================================================================

    def _build_stage_callable(
        self,
        module_path: str,
        class_name: str,
        method_name: str = "run",
    ) -> Callable[[], Dict[str, object]]:
        """
        Build a lazy stage callable.

        Args:
            module_path: Python module path.
            class_name: Class name inside module.
            method_name: Method to call on the class instance.

        Returns:
            Callable stage function.
        """

        def _stage() -> Dict[str, object]:
            print(
                "[PROGRESS] Lazy loading stage class: "
                f"{module_path}.{class_name}.{method_name}"
            )

            module = import_module(module_path)
            stage_class = getattr(module, class_name)
            instance = stage_class()
            method = getattr(instance, method_name)

            return method()

        return _stage

    def _add_stage(
        self,
        stages: List[StageSpec],
        stage_name: str,
        module_path: str,
        class_name: str,
        method_name: str = "run",
        enabled: bool = True,
        skip_reason: Optional[str] = None,
        skipped_stages: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        """
        Add a lazy stage to the stage list, or record it as skipped.

        Args:
            stages: Stage list to mutate.
            stage_name: Stage label.
            module_path: Python module path.
            class_name: Class name.
            method_name: Method name.
            enabled: Whether stage should run.
            skip_reason: Reason when disabled.
            skipped_stages: Optional skipped-stage list.
        """
        if enabled:
            stages.append(
                (
                    stage_name,
                    self._build_stage_callable(
                        module_path=module_path,
                        class_name=class_name,
                        method_name=method_name,
                    ),
                )
            )
            return

        if skipped_stages is not None:
            skipped_stages.append(
                {
                    "stage": stage_name,
                    "reason": skip_reason or "Stage disabled.",
                }
            )

        print(f"[PROGRESS] Skipping stage {stage_name}: {skip_reason}")

    # ==================================================================================
    # Stage planning
    # ==================================================================================

    def _build_stages(
        self,
        include_shap: bool,
        include_evaluation: bool,
        include_dashboard: bool,
        include_feedback: bool,
        include_context_drift: bool,
        include_twin_comparison: bool,
    ) -> Tuple[List[StageSpec], List[Dict[str, object]]]:
        """
        Build full pipeline stage list.

        Returns:
            Tuple of stage specs and skipped-stage records.
        """
        print("[PROGRESS] Entering FullPipeline._build_stages")

        stages: List[StageSpec] = []
        skipped_stages: List[Dict[str, object]] = []

        # ==================================================================================
        # 01 Preprocessing
        # ==================================================================================

        self._add_stage(
            stages,
            "01_data_loading",
            "app.services.Anomaly_Health_Monitering.Data_Preprocessing.data_loader",
            "DataLoader",
            "save_raw_data",
        )
        self._add_stage(
            stages,
            "02_cleaning",
            "app.services.Anomaly_Health_Monitering.Data_Preprocessing.cleaner",
            "DataCleaner",
        )
        self._add_stage(
            stages,
            "03_feature_engineering",
            "app.services.Anomaly_Health_Monitering.Data_Preprocessing.feature_engineering",
            "FeatureEngineer",
        )
        self._add_stage(
            stages,
            "04_dev_only_scaling",
            "app.services.Anomaly_Health_Monitering.Data_Preprocessing.scaler",
            "FeatureScaler",
        )

        # ==================================================================================
        # 02 Context modeling
        # ==================================================================================

        self._add_stage(
            stages,
            "05_context_modeling",
            "app.services.Anomaly_Health_Monitering.context_modeling.operating_mode_detector",
            "OperatingModeDetector",
        )
        self._add_stage(
            stages,
            "06_context_drift",
            "app.services.Anomaly_Health_Monitering.context_modeling.context_drift",
            "ContextDriftDetector",
            enabled=include_context_drift,
            skip_reason="include_context_drift=False",
            skipped_stages=skipped_stages,
        )

        # ==================================================================================
        # 03 Digital twin
        # ==================================================================================

        self._add_stage(
            stages,
            "07_random_forest_twin",
            "app.services.Anomaly_Health_Monitering.digital_twin.random_forest_twin",
            "RandomForestTwin",
        )
        self._add_stage(
            stages,
            "08_xgboost_twin",
            "app.services.Anomaly_Health_Monitering.digital_twin.xgboost_twin",
            "XGBoostTwin",
        )
        self._add_stage(
            stages,
            "09_lightgbm_twin",
            "app.services.Anomaly_Health_Monitering.digital_twin.lightgbm_twin",
            "LightGBMTwin",
        )
        self._add_stage(
            stages,
            "10_ensemble_twin",
            "app.services.Anomaly_Health_Monitering.digital_twin.ensemble_twin",
            "EnsembleDigitalTwin",
        )
        self._add_stage(
            stages,
            "11_twin_comparison",
            "app.services.Anomaly_Health_Monitering.digital_twin.twin_comparator",
            "TwinComparator",
            enabled=include_twin_comparison,
            skip_reason="include_twin_comparison=False",
            skipped_stages=skipped_stages,
        )

        # ==================================================================================
        # 04 Residual analysis
        # ==================================================================================

        self._add_stage(
            stages,
            "12_residual_analysis",
            "app.services.Anomaly_Health_Monitering.digital_twin.residual_calculator",
            "ResidualCalculator",
        )

        # ==================================================================================
        # 05 Anomaly detection
        # ==================================================================================

        self._add_stage(
            stages,
            "13_residual_anomaly_detector",
            "app.services.Anomaly_Health_Monitering.anomaly_detection.residual_anomaly_detector",
            "ResidualAnomalyDetector",
        )
        self._add_stage(
            stages,
            "14_isolation_forest_detector",
            "app.services.Anomaly_Health_Monitering.anomaly_detection.isolation_forest_detector",
            "IsolationForestDetector",
        )
        self._add_stage(
            stages,
            "15_mahalanobis_detector",
            "app.services.Anomaly_Health_Monitering.anomaly_detection.mahalanobis_detector",
            "MahalanobisDetector",
        )
        self._add_stage(
            stages,
            "16_anomaly_fusion",
            "app.services.Anomaly_Health_Monitering.anomaly_detection.anomaly_fusion",
            "AnomalyFusion",
        )
        self._add_stage(
            stages,
            "17_severity_classifier",
            "app.services.Anomaly_Health_Monitering.anomaly_detection.severity_classifier",
            "SeverityClassifier",
        )
        self._add_stage(
            stages,
            "18_early_warning_score",
            "app.services.Anomaly_Health_Monitering.anomaly_detection.early_warning_score",
            "EarlyWarningScore",
        )

        # ==================================================================================
        # 06 Health monitoring
        # ==================================================================================

        self._add_stage(
            stages,
            "19_health_index",
            "app.services.Anomaly_Health_Monitering.health_monitoring.health_index_calculator",
            "HealthIndexCalculator",
        )
        self._add_stage(
            stages,
            "20_health_state",
            "app.services.Anomaly_Health_Monitering.health_monitoring.health_state_classifier",
            "HealthStateClassifier",
        )
        self._add_stage(
            stages,
            "21_health_trend",
            "app.services.Anomaly_Health_Monitering.health_monitoring.health_trend_tracker",
            "HealthTrendTracker",
        )
        self._add_stage(
            stages,
            "22_health_alerts",
            "app.services.Anomaly_Health_Monitering.health_monitoring.health_alert_engine",
            "HealthAlertEngine",
        )

        # ==================================================================================
        # 07 Reasoning
        # ==================================================================================

        self._add_stage(
            stages,
            "23_sensor_dependency_graph",
            "app.services.Anomaly_Health_Monitering.reasoning.sensor_dependency_graph",
            "SensorDependencyGraph",
        )
        self._add_stage(
            stages,
            "24_root_cause_analysis",
            "app.services.Anomaly_Health_Monitering.reasoning.root_cause_analyzer",
            "RootCauseAnalyzer",
        )
        self._add_stage(
            stages,
            "25_root_cause_tracking",
            "app.services.Anomaly_Health_Monitering.reasoning.root_cause_tracker",
            "RootCauseTracker",
        )
        self._add_stage(
            stages,
            "26_temporal_reasoning",
            "app.services.Anomaly_Health_Monitering.reasoning.temporal_reasoning",
            "TemporalReasoning",
        )

        # ==================================================================================
        # 08 Uncertainty
        # ==================================================================================

        self._add_stage(
            stages,
            "27_model_agreement",
            "app.services.Anomaly_Health_Monitering.uncertainty.model_agreement",
            "ModelAgreementCalculator",
        )
        self._add_stage(
            stages,
            "28_confidence_estimation",
            "app.services.Anomaly_Health_Monitering.uncertainty.confidence_estimator",
            "ConfidenceEstimator",
        )

        # ==================================================================================
        # 09 Explainability
        # ==================================================================================

        self._add_stage(
            stages,
            "29_sensor_residual_ranking",
            "app.services.Anomaly_Health_Monitering.explainability.sensor_residual_ranking",
            "SensorResidualRanking",
        )
        self._add_stage(
            stages,
            "30_subsystem_explainer",
            "app.services.Anomaly_Health_Monitering.explainability.subsystem_explainer",
            "SubsystemExplainer",
        )
        self._add_stage(
            stages,
            "31_explanation_generator",
            "app.services.Anomaly_Health_Monitering.explainability.explanation_generator",
            "ExplanationGenerator",
        )
        self._add_stage(
            stages,
            "31a_shap_explainer",
            "app.services.Anomaly_Health_Monitering.explainability.shap_explainer",
            "SHAPExplainer",
            enabled=include_shap,
            skip_reason="include_shap=False. SHAP is optional and expensive.",
            skipped_stages=skipped_stages,
        )

        # ==================================================================================
        # 10 Feedback learning
        # ==================================================================================

        self._add_stage(
            stages,
            "32_feedback_store",
            "app.services.Anomaly_Health_Monitering.feedback.feedback_store",
            "FeedbackStore",
            enabled=include_feedback,
            skip_reason="include_feedback=False",
            skipped_stages=skipped_stages,
        )
        self._add_stage(
            stages,
            "33_alert_memory",
            "app.services.Anomaly_Health_Monitering.feedback.alert_memory",
            "AlertMemory",
            enabled=include_feedback,
            skip_reason="include_feedback=False",
            skipped_stages=skipped_stages,
        )
        self._add_stage(
            stages,
            "34_threshold_adapter",
            "app.services.Anomaly_Health_Monitering.feedback.threshold_adapter",
            "ThresholdAdapter",
            enabled=include_feedback,
            skip_reason="include_feedback=False",
            skipped_stages=skipped_stages,
        )

        # ==================================================================================
        # 11 Dashboard
        # ==================================================================================

        self._add_stage(
            stages,
            "35_dashboard_data_generation",
            "app.services.Anomaly_Health_Monitering.dashboard.dashboard_data_generator",
            "DashboardDataGenerator",
            enabled=include_dashboard,
            skip_reason="include_dashboard=False",
            skipped_stages=skipped_stages,
        )

        # ==================================================================================
        # 12 Evaluation
        # ==================================================================================

        self._add_stage(
            stages,
            "36_evaluate_digital_twin",
            "app.services.Anomaly_Health_Monitering.evaluation.evaluate_digital_twin",
            "DigitalTwinEvaluator",
            enabled=include_evaluation,
            skip_reason="include_evaluation=False",
            skipped_stages=skipped_stages,
        )
        self._add_stage(
            stages,
            "37_evaluate_context",
            "app.services.Anomaly_Health_Monitering.evaluation.evaluate_context",
            "ContextEvaluator",
            enabled=include_evaluation,
            skip_reason="include_evaluation=False",
            skipped_stages=skipped_stages,
        )
        self._add_stage(
            stages,
            "38_evaluate_anomaly",
            "app.services.Anomaly_Health_Monitering.evaluation.evaluate_anomaly",
            "AnomalyEvaluator",
            enabled=include_evaluation,
            skip_reason="include_evaluation=False",
            skipped_stages=skipped_stages,
        )
        self._add_stage(
            stages,
            "39_evaluate_health",
            "app.services.Anomaly_Health_Monitering.evaluation.evaluate_health",
            "HealthEvaluator",
            enabled=include_evaluation,
            skip_reason="include_evaluation=False",
            skipped_stages=skipped_stages,
        )
        self._add_stage(
            stages,
            "40_evaluate_reasoning",
            "app.services.Anomaly_Health_Monitering.evaluation.evaluate_reasoning",
            "ReasoningEvaluator",
            enabled=include_evaluation,
            skip_reason="include_evaluation=False",
            skipped_stages=skipped_stages,
        )
        self._add_stage(
            stages,
            "41_evaluate_explainability",
            "app.services.Anomaly_Health_Monitering.evaluation.evaluate_explainability",
            "ExplainabilityEvaluator",
            enabled=include_evaluation,
            skip_reason="include_evaluation=False",
            skipped_stages=skipped_stages,
        )

        return stages, skipped_stages

    # ==================================================================================
    # Summary helpers
    # ==================================================================================

    def _final_outputs(self) -> Dict[str, object]:
        """
        Collect important final output paths.
        """
        return {
            "raw_csv": str(Config.RAW_CSV) if Config.RAW_CSV.exists() else None,
            "scaled_csv": str(Config.SCALED_CSV) if Config.SCALED_CSV.exists() else None,
            "context_csv": str(Config.CONTEXT_CSV) if Config.CONTEXT_CSV.exists() else None,
            "rf_predictions_csv": (
                str(Config.RF_PREDICTIONS_CSV)
                if Config.RF_PREDICTIONS_CSV.exists()
                else None
            ),
            "xgb_predictions_csv": (
                str(Config.XGB_PREDICTIONS_CSV)
                if Config.XGB_PREDICTIONS_CSV.exists()
                else None
            ),
            "lgbm_predictions_csv": (
                str(Config.LGBM_PREDICTIONS_CSV)
                if Config.LGBM_PREDICTIONS_CSV.exists()
                else None
            ),
            "ensemble_predictions_csv": (
                str(Config.ENSEMBLE_PREDICTIONS_CSV)
                if Config.ENSEMBLE_PREDICTIONS_CSV.exists()
                else None
            ),
            "residuals_csv": (
                str(Config.RESIDUALS_CSV)
                if Config.RESIDUALS_CSV.exists()
                else None
            ),
            "anomaly_fusion_csv": (
                str(Config.ANOMALY_FUSION_CSV)
                if Config.ANOMALY_FUSION_CSV.exists()
                else None
            ),
            "health_index_csv": (
                str(Config.HEALTH_INDEX_CSV)
                if Config.HEALTH_INDEX_CSV.exists()
                else None
            ),
            "health_states_csv": (
                str(Config.HEALTH_STATES_CSV)
                if Config.HEALTH_STATES_CSV.exists()
                else None
            ),
            "root_cause_csv": (
                str(Config.ROOT_CAUSE_CSV)
                if Config.ROOT_CAUSE_CSV.exists()
                else None
            ),
            "model_agreement_csv": (
                str(Config.MODEL_AGREEMENT_CSV)
                if Config.MODEL_AGREEMENT_CSV.exists()
                else None
            ),
            "confidence_csv": (
                str(Config.CONFIDENCE_CSV)
                if Config.CONFIDENCE_CSV.exists()
                else None
            ),
            "explanation_reports_csv": (
                str(Config.EXPLANATION_REPORTS_CSV)
                if Config.EXPLANATION_REPORTS_CSV.exists()
                else None
            ),
            "shap_csv": (
                str(Config.SHAP_CSV)
                if Config.SHAP_CSV.exists()
                else None
            ),
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
            "dashboard_csv": (
                str(Config.DASHBOARD_CSV)
                if Config.DASHBOARD_CSV.exists()
                else None
            ),
        }

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(
        self,
        include_shap: bool = False,
        include_evaluation: bool = True,
        include_dashboard: bool = True,
        include_feedback: bool = True,
        include_context_drift: bool = True,
        include_twin_comparison: bool = True,
    ) -> Dict[str, object]:
        """
        Run the full pipeline safely.

        Args:
            include_shap: Whether to run SHAP explanations.
            include_evaluation: Whether to run evaluation stages.
            include_dashboard: Whether to generate dashboard_data.csv.
            include_feedback: Whether to run feedback learning stages.
            include_context_drift: Whether to run context drift detection.
            include_twin_comparison: Whether to run twin comparison metrics.

        Returns:
            Dict[str, object]: Full pipeline summary.
        """
        print("[PROGRESS] Entering FullPipeline.run")
        print(f"[PROGRESS] include_shap={include_shap}")
        print(f"[PROGRESS] include_evaluation={include_evaluation}")
        print(f"[PROGRESS] include_dashboard={include_dashboard}")
        print(f"[PROGRESS] include_feedback={include_feedback}")
        print(f"[PROGRESS] include_context_drift={include_context_drift}")
        print(f"[PROGRESS] include_twin_comparison={include_twin_comparison}")

        try:
            started = perf_counter()

            stages, skipped_stages = self._build_stages(
                include_shap=include_shap,
                include_evaluation=include_evaluation,
                include_dashboard=include_dashboard,
                include_feedback=include_feedback,
                include_context_drift=include_context_drift,
                include_twin_comparison=include_twin_comparison,
            )

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_index, (stage_name, stage_function) in enumerate(stages, start=1):
                print("=" * 100)
                print(
                    f"[PROGRESS] Running full pipeline stage "
                    f"{stage_index}/{len(stages)}: {stage_name}"
                )

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    logger.error("Full pipeline stopped at failed stage: %s", stage_name)
                    print(
                        "[PROGRESS] Full pipeline stopped safely after failed stage. "
                        "Previous successful outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"
            duration_seconds = perf_counter() - started

            final_outputs = self._final_outputs()

            summary = {
                "status": status,
                "message": (
                    "Full CA-EDT-AHMA pipeline completed successfully."
                    if status == "success"
                    else "Full pipeline stopped safely. Previous successful outputs were not deleted."
                ),
                "completed_stage_count": int(len(completed)),
                "planned_stage_count": int(len(stages)),
                "skipped_stage_count": int(len(skipped_stages)),
                "completed_stages": completed,
                "failed_stages": failed,
                "skipped_stages": skipped_stages,
                "final_output_file": final_outputs["dashboard_csv"],
                "dashboard_file": final_outputs["dashboard_csv"],
                "final_outputs": final_outputs,
                "final_model_name": getattr(Config, "FULL_MODEL_NAME", "CA-EDT-AHMA"),
                "runtime": {
                    "duration_seconds": float(duration_seconds),
                    "duration_minutes": float(duration_seconds / 60.0),
                },
                "run_options": {
                    "include_shap": bool(include_shap),
                    "include_evaluation": bool(include_evaluation),
                    "include_dashboard": bool(include_dashboard),
                    "include_feedback": bool(include_feedback),
                    "include_context_drift": bool(include_context_drift),
                    "include_twin_comparison": bool(include_twin_comparison),
                },
                "pipeline_order": [
                    "preprocessing",
                    "context_modeling",
                    "digital_twin_training_and_inference",
                    "residual_analysis",
                    "anomaly_detection",
                    "health_monitoring",
                    "reasoning",
                    "uncertainty",
                    "explainability",
                    "feedback_learning",
                    "dashboard_generation",
                    "evaluation",
                ],
                "fit_rule": (
                    "All scalers, context models, digital twin models, residual thresholds, "
                    "Isolation Forest, and Mahalanobis parameters must be fitted on dev only. "
                    "Test split is used for transform, inference, scoring, and evaluation only."
                ),
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Y_dev/Y_test are RUL targets and are intentionally ignored by "
                        "this Anomaly and Health Monitoring component."
                    ),
                },
                "decision_boundary": {
                    "makes_maintenance_scheduling_decisions": False,
                    "allowed_outputs": [
                        "context id",
                        "sensor predictions",
                        "residuals",
                        "anomaly scores",
                        "alert levels",
                        "health index",
                        "health state",
                        "root-cause pattern",
                        "inspection focus",
                        "confidence score",
                        "uncertainty score",
                        "dashboard intelligence",
                    ],
                    "note": (
                        "This component provides anomaly, health, reasoning, explanation, "
                        "confidence, and dashboard intelligence only. Final maintenance "
                        "scheduling belongs to the autonomous maintenance supervisor."
                    ),
                },
                "memory_safety_rule": {
                    "full_pipeline_loads_large_csvs": False,
                    "heavy_work_delegated_to_services": True,
                    "service_expectation": (
                        "Large CSV stages must use chunking, aligned key validation, "
                        "and atomic/temp-file writes."
                    ),
                },
                "safety_rule": (
                    "No pipeline stage deletes previous successful outputs. "
                    "Writes should be atomic where relevant."
                ),
                "leakage_audit": {
                    "scaler_fit_split": Config.DEV_SPLIT_NAME,
                    "context_model_fit_split": Config.DEV_SPLIT_NAME,
                    "digital_twin_fit_split": Config.DEV_SPLIT_NAME,
                    "anomaly_threshold_fit_split": Config.DEV_SPLIT_NAME,
                    "model_agreement_normalization_fit_split": Config.DEV_SPLIT_NAME,
                    "test_split_usage": "transform_inference_scoring_evaluation_only",
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_rul_targets": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "previous_outputs_deleted_on_failure": False,
                },
            }

            print(f"[PROGRESS] Writing full pipeline summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info("Full pipeline finished with status=%s.", status)

            return summary

        except Exception as exc:
            logger.exception("Full pipeline failed.")
            raise RuntimeError("Full pipeline failed.") from exc


def run_full_pipeline(
    include_shap: bool = False,
    include_evaluation: bool = True,
    include_dashboard: bool = True,
    include_feedback: bool = True,
    include_context_drift: bool = True,
    include_twin_comparison: bool = True,
) -> Dict[str, object]:
    """
    Execute full CA-EDT-AHMA pipeline.

    Args:
        include_shap: Whether to run SHAP explanations.
        include_evaluation: Whether to run evaluation stages.
        include_dashboard: Whether to generate dashboard_data.csv.
        include_feedback: Whether to run feedback learning stages.
        include_context_drift: Whether to run context drift detection.
        include_twin_comparison: Whether to run twin comparison metrics.

    Returns:
        Dict[str, object]: Full pipeline result.
    """
    print("[PROGRESS] Entering run_full_pipeline")

    pipeline = FullPipeline()
    return pipeline.run(
        include_shap=include_shap,
        include_evaluation=include_evaluation,
        include_dashboard=include_dashboard,
        include_feedback=include_feedback,
        include_context_drift=include_context_drift,
        include_twin_comparison=include_twin_comparison,
    )


if __name__ == "__main__":
    print("[PROGRESS] 11_full_pipeline.py execution started")
    result = run_full_pipeline(
        include_shap=False,
        include_evaluation=True,
        include_dashboard=True,
        include_feedback=True,
        include_context_drift=True,
        include_twin_comparison=True,
    )
    print("[PROGRESS] 11_full_pipeline.py execution finished")
    print(result)