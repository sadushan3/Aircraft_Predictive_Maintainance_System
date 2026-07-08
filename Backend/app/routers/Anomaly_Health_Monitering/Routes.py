"""
FastAPI routes for CA-EDT-AHMA.

CA-EDT-AHMA:
Context-Aware Ensemble Digital Twin for Explainable Health Monitoring
and Anomaly Reasoning.

Endpoints:
- /upload
- /preprocess
- /context-modeling
- /train-digital-twin
- /generate-residuals
- /detect-anomalies
- /generate-health-index
- /generate-health-score
- /classify-health-state
- /root-cause-analysis
- /explain
- /confidence
- /feedback
- /dashboard
- /evaluate
- /full-pipeline

Important:
The numbered pipeline modules cannot be imported with normal Python import
syntax because filenames start with digits. This route file uses service
classes directly and uses importlib for full pipeline loading.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/routers/Anomaly_Health_Monitering/Routes.py")
import importlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.schemas.Anomaly_Health_Monitering.anomaly_schema import FeedbackRequest
from app.schemas.Anomaly_Health_Monitering.common_schema import (
    APIResponse,
    UnitCycleRequest,
    UnitRequest,
)
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.cleaner import DataCleaner
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.data_loader import DataLoader
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.feature_engineering import FeatureEngineer
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.scaler import FeatureScaler
from app.services.Anomaly_Health_Monitering.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitering.anomaly_detection.early_warning_score import EarlyWarningScore
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
from app.services.Anomaly_Health_Monitering.context_modeling.context_drift import (
    ContextDriftDetector,
)
from app.services.Anomaly_Health_Monitering.context_modeling.operating_mode_detector import (
    OperatingModeDetector,
)
from app.services.Anomaly_Health_Monitering.dashboard.dashboard_api import DashboardAPI
from app.services.Anomaly_Health_Monitering.dashboard.dashboard_data_generator import (
    DashboardDataGenerator,
)
from app.services.Anomaly_Health_Monitering.digital_twin.ensemble_twin import EnsembleDigitalTwin
from app.services.Anomaly_Health_Monitering.digital_twin.lightgbm_twin import LightGBMTwin
from app.services.Anomaly_Health_Monitering.digital_twin.random_forest_twin import (
    RandomForestTwin,
)
from app.services.Anomaly_Health_Monitering.digital_twin.residual_calculator import (
    ResidualCalculator,
)
from app.services.Anomaly_Health_Monitering.digital_twin.twin_comparator import TwinComparator
from app.services.Anomaly_Health_Monitering.digital_twin.xgboost_twin import XGBoostTwin
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_anomaly import AnomalyEvaluator
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_context import ContextEvaluator
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_digital_twin import (
    DigitalTwinEvaluator,
)
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_explainability import (
    ExplainabilityEvaluator,
)
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_health import HealthEvaluator
from app.services.Anomaly_Health_Monitering.evaluation.evaluate_reasoning import (
    ReasoningEvaluator,
)
from app.services.Anomaly_Health_Monitering.explainability.explanation_generator import (
    ExplanationGenerator,
)
from app.services.Anomaly_Health_Monitering.explainability.sensor_residual_ranking import (
    SensorResidualRanking,
)
from app.services.Anomaly_Health_Monitering.explainability.subsystem_explainer import (
    SubsystemExplainer,
)
from app.services.Anomaly_Health_Monitering.feedback.learning_updater import LearningUpdater
from app.services.Anomaly_Health_Monitering.health_monitoring.health_alert_engine import (
    HealthAlertEngine,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_index_calculator import (
    HealthIndexCalculator,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_state_classifier import (
    HealthStateClassifier,
)
from app.services.Anomaly_Health_Monitering.health_monitoring.health_trend_tracker import (
    HealthTrendTracker,
)
from app.services.Anomaly_Health_Monitering.reasoning.root_cause_analyzer import (
    RootCauseAnalyzer,
)
from app.services.Anomaly_Health_Monitering.reasoning.root_cause_tracker import (
    RootCauseTracker,
)
from app.services.Anomaly_Health_Monitering.reasoning.sensor_dependency_graph import (
    SensorDependencyGraph,
)
from app.services.Anomaly_Health_Monitering.reasoning.temporal_reasoning import (
    TemporalReasoning,
)
from app.services.Anomaly_Health_Monitering.uncertainty.confidence_estimator import (
    ConfidenceEstimator,
)
from app.services.Anomaly_Health_Monitering.uncertainty.model_agreement import (
    ModelAgreementCalculator,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)

router = APIRouter(
    prefix="/anomaly-health-monitoring",
    tags=["Anomaly Health Monitoring"],
)


def _api_response_from_dict(result: Dict[str, Any]) -> APIResponse:
    """
    Convert any service response dictionary into APIResponse.

    Args:
        result: Service result dictionary.

    Returns:
        APIResponse: Standard API response.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::_api_response_from_dict")
    return APIResponse(
        status=str(result.get("status", "success")),
        message=str(result.get("message", "")),
        output_file=str(result.get("output_file")) if result.get("output_file") else None,
        records_count=(
            int(result["records_count"])
            if result.get("records_count") is not None
            else None
        ),
        metrics=result.get("metrics") if isinstance(result.get("metrics"), dict) else None,
        errors=result.get("errors") if isinstance(result.get("errors"), list) else None,
        data=result.get("data"),
    )


def _failed_response(message: str, exc: Exception) -> APIResponse:
    """
    Build failed API response.

    Args:
        message: User-facing message.
        exc: Exception.

    Returns:
        APIResponse: Failed response.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::_failed_response")
    logger.exception(message)
    return APIResponse(
        status="failed",
        message=f"{message}: {exc}",
        errors=[str(exc)],
    )


def _stage_result_to_dict(stage_result: StageResult) -> Dict[str, Any]:
    """
    Convert StageResult to dictionary.

    Args:
        stage_result: Stage result.

    Returns:
        Dict[str, Any]: Dictionary result.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::_stage_result_to_dict")
    return {
        "status": stage_result.status,
        "message": stage_result.message,
        "output_file": stage_result.output_file,
        "records_count": stage_result.records_count,
        "elapsed_seconds": stage_result.elapsed_seconds,
    }


@router.post("/upload", response_model=APIResponse)
async def upload_dataset(file: UploadFile = File(...)) -> APIResponse:
    """
    Upload the N-CMAPSS HDF5 file.

    The uploaded file is saved as:
    Backend/data/Anomaly_Health_Monitering/N-CMAPSS_DS01-005.h5
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::upload_dataset")
    try:
        Config.create_directories()

        if not file.filename:
            return APIResponse(
                status="failed",
                message="Uploaded file has no filename.",
                errors=["Missing filename."],
            )

        if not file.filename.endswith(".h5"):
            return APIResponse(
                status="failed",
                message="Only .h5 files are accepted.",
                errors=["Invalid file extension."],
            )

        output_path: Path = Config.H5_FILE_PATH

        with output_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("Dataset uploaded to %s.", output_path)

        return APIResponse(
            status="success",
            message="Dataset uploaded successfully.",
            output_file=str(output_path),
        )

    except Exception as exc:
        return _failed_response("Dataset upload failed", exc)


@router.post("/preprocess", response_model=APIResponse)
def preprocess() -> APIResponse:
    """
    Run preprocessing:
    1. Load HDF5
    2. Clean data
    3. Engineer features
    4. Scale features with dev-only fitting
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::preprocess")
    try:
        stages = [
            ("data_loading", DataLoader().save_raw_data),
            ("cleaning", DataCleaner().run),
            ("feature_engineering", FeatureEngineer().run),
            ("dev_only_scaling", FeatureScaler().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Preprocessing completed."
                if status == "success"
                else "Preprocessing stopped safely. Previous outputs were not deleted."
            ),
            output_file=str(Config.SCALED_CSV) if Config.SCALED_CSV.exists() else None,
            data={
                "completed_stages": completed,
                "failed_stages": failed,
                "fit_rule": "Scaler fitted only on dev split.",
            },
        )

    except Exception as exc:
        return _failed_response("Preprocessing failed", exc)


@router.post("/context-modeling", response_model=APIResponse)
def context_modeling() -> APIResponse:
    """
    Train and infer K-Means and GMM operating-context models.

    Fit:
    W_dev only.

    Inference:
    dev and test.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::context_modeling")
    try:
        stages = [
            ("operating_mode_detection", OperatingModeDetector().run),
            ("context_drift_detection", ContextDriftDetector().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Context modeling completed."
                if status == "success"
                else "Context modeling stopped safely."
            ),
            output_file=str(Config.CONTEXT_CSV) if Config.CONTEXT_CSV.exists() else None,
            data={
                "completed_stages": completed,
                "failed_stages": failed,
                "fit_rule": "K-Means and GMM fitted only on W_dev.",
            },
        )

    except Exception as exc:
        return _failed_response("Context modeling failed", exc)


@router.post("/train-digital-twin", response_model=APIResponse)
def train_digital_twin() -> APIResponse:
    """
    Train digital twin models:
    1. Random Forest
    2. XGBoost
    3. LightGBM
    4. Ensemble digital twin

    Fit:
    dev split only.

    Target:
    X_s measured sensor values.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::train_digital_twin")
    try:
        stages = [
            ("random_forest_twin", RandomForestTwin().run),
            ("xgboost_twin", XGBoostTwin().run),
            ("lightgbm_twin", LightGBMTwin().run),
            ("ensemble_twin", EnsembleDigitalTwin().run),
            ("twin_comparator", TwinComparator().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Digital twin training and inference completed."
                if status == "success"
                else "Digital twin stage stopped safely."
            ),
            output_file=(
                str(Config.ENSEMBLE_PREDICTIONS_CSV)
                if Config.ENSEMBLE_PREDICTIONS_CSV.exists()
                else None
            ),
            data={
                "completed_stages": completed,
                "failed_stages": failed,
                "fit_rule": "RF, XGBoost, and LightGBM trained only on dev split.",
            },
        )

    except Exception as exc:
        return _failed_response("Digital twin training failed", exc)


@router.post("/generate-residuals", response_model=APIResponse)
def generate_residuals() -> APIResponse:
    """
    Generate residuals:
    actual X_s - ensemble predicted X_s.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::generate_residuals")
    try:
        result = ResidualCalculator().run()
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Residual generation failed", exc)


@router.post("/detect-anomalies", response_model=APIResponse)
def detect_anomalies() -> APIResponse:
    """
    Run full anomaly detection:
    1. Residual threshold detector
    2. Isolation Forest
    3. Mahalanobis detector
    4. Fusion
    5. Severity classification
    6. Early warning score
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::detect_anomalies")
    try:
        stages = [
            ("residual_anomaly_detector", ResidualAnomalyDetector().run),
            ("isolation_forest_detector", IsolationForestDetector().run),
            ("mahalanobis_detector", MahalanobisDetector().run),
            ("anomaly_fusion", AnomalyFusion().run),
            ("severity_classifier", SeverityClassifier().run),
            ("early_warning_score", EarlyWarningScore().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Anomaly detection completed."
                if status == "success"
                else "Anomaly detection stopped safely."
            ),
            output_file=(
                str(Config.ANOMALY_FUSION_CSV)
                if Config.ANOMALY_FUSION_CSV.exists()
                else None
            ),
            data={
                "completed_stages": completed,
                "failed_stages": failed,
                "fit_rule": (
                    "Residual thresholds, Isolation Forest, and Mahalanobis "
                    "parameters fitted only on dev residuals."
                ),
            },
        )

    except Exception as exc:
        return _failed_response("Anomaly detection failed", exc)


@router.post("/generate-health-index", response_model=APIResponse)
def generate_health_index() -> APIResponse:
    """
    Generate health index from anomaly severity, trend, and persistence.

    This endpoint does not predict RUL.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::generate_health_index")
    try:
        result = HealthIndexCalculator().run()
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Health index generation failed", exc)


@router.post("/generate-health-score", response_model=APIResponse)
def generate_health_score() -> APIResponse:
    """
    Generate complete health score outputs.

    Includes:
    - health index
    - health state
    - health trend
    - health alerts
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::generate_health_score")
    try:
        stages = [
            ("health_index", HealthIndexCalculator().run),
            ("health_state", HealthStateClassifier().run),
            ("health_trend", HealthTrendTracker().run),
            ("health_alerts", HealthAlertEngine().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Health score generation completed."
                if status == "success"
                else "Health score generation stopped safely."
            ),
            output_file=(
                str(Config.HEALTH_STATES_CSV)
                if Config.HEALTH_STATES_CSV.exists()
                else None
            ),
            data={
                "completed_stages": completed,
                "failed_stages": failed,
                "rul_prediction_used": False,
                "y_dev_y_test_used": False,
            },
        )

    except Exception as exc:
        return _failed_response("Health score generation failed", exc)


@router.post("/classify-health-state", response_model=APIResponse)
def classify_health_state() -> APIResponse:
    """
    Classify health state:
    Healthy, Degrading, Warning, Critical.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::classify_health_state")
    try:
        result = HealthStateClassifier().run()
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Health state classification failed", exc)


@router.post("/root-cause-analysis", response_model=APIResponse)
def root_cause_analysis() -> APIResponse:
    """
    Run root-cause reasoning:
    1. Sensor dependency graph
    2. Residual contribution ranking
    3. Root-cause pattern inference
    4. Temporal reasoning
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::root_cause_analysis")
    try:
        stages = [
            ("sensor_dependency_graph", SensorDependencyGraph().run),
            ("root_cause_analysis", RootCauseAnalyzer().run),
            ("root_cause_tracking", RootCauseTracker().run),
            ("temporal_reasoning", TemporalReasoning().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Root-cause analysis completed."
                if status == "success"
                else "Root-cause analysis stopped safely."
            ),
            output_file=str(Config.ROOT_CAUSE_CSV) if Config.ROOT_CAUSE_CSV.exists() else None,
            data={
                "completed_stages": completed,
                "failed_stages": failed,
                "decision_boundary": (
                    "This component provides inspection focus only. "
                    "It does not schedule maintenance or make final maintenance decisions."
                ),
            },
        )

    except Exception as exc:
        return _failed_response("Root-cause analysis failed", exc)


@router.post("/explain", response_model=APIResponse)
def explain() -> APIResponse:
    """
    Generate explainability outputs:
    1. Sensor residual ranking
    2. Subsystem explanation
    3. Human-readable explanation reports
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::explain")
    try:
        stages = [
            ("sensor_residual_ranking", SensorResidualRanking().run),
            ("subsystem_explainer", SubsystemExplainer().run),
            ("explanation_generator", ExplanationGenerator().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Explainability reports generated."
                if status == "success"
                else "Explainability stopped safely."
            ),
            output_file=(
                str(Config.EXPLANATION_REPORTS_CSV)
                if Config.EXPLANATION_REPORTS_CSV.exists()
                else None
            ),
            data={
                "completed_stages": completed,
                "failed_stages": failed,
            },
        )

    except Exception as exc:
        return _failed_response("Explainability generation failed", exc)


@router.post("/confidence", response_model=APIResponse)
def confidence() -> APIResponse:
    """
    Generate model agreement, confidence, reliability, and uncertainty scores.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::confidence")
    try:
        stages = [
            ("model_agreement", ModelAgreementCalculator().run),
            ("confidence_estimation", ConfidenceEstimator().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Confidence and uncertainty calculation completed."
                if status == "success"
                else "Confidence calculation stopped safely."
            ),
            output_file=str(Config.CONFIDENCE_CSV) if Config.CONFIDENCE_CSV.exists() else None,
            data={
                "completed_stages": completed,
                "failed_stages": failed,
                "formula": (
                    "confidence = 0.35*model_agreement_score + "
                    "0.25*context_confidence + "
                    "0.25*anomaly_persistence_score + "
                    "0.15*data_quality_score"
                ),
            },
        )

    except Exception as exc:
        return _failed_response("Confidence estimation failed", exc)


@router.post("/feedback", response_model=APIResponse)
def feedback(request: FeedbackRequest) -> APIResponse:
    """
    Store operator feedback and update adaptive learning outputs.

    Feedback labels:
    - accepted_alert
    - rejected_false_alarm
    - missed_anomaly
    - uncertain
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::feedback")
    try:
        result = LearningUpdater().submit_feedback(
            unit_id=request.unit_id,
            cycle=request.cycle,
            context_id=request.context_id,
            alert_level=request.alert_level,
            final_anomaly_score=request.final_anomaly_score,
            root_cause_pattern=request.root_cause_pattern,
            feedback_label=request.feedback_label,
            operator_note=None,
        )
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Feedback update failed", exc)


@router.post("/dashboard", response_model=APIResponse)
def dashboard() -> APIResponse:
    """
    Generate dashboard_data.csv.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::dashboard")
    try:
        result = DashboardDataGenerator().run()
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Dashboard generation failed", exc)


@router.get("/dashboard/summary", response_model=APIResponse)
def dashboard_summary() -> APIResponse:
    """
    Return dashboard summary counts.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::dashboard_summary")
    try:
        result = DashboardAPI().get_summary()
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Dashboard summary failed", exc)


@router.get("/dashboard/latest-all", response_model=APIResponse)
def dashboard_latest_all() -> APIResponse:
    """
    Return latest health for all units.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::dashboard_latest_all")
    try:
        result = DashboardAPI().get_latest_all_units()
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Latest all-units dashboard query failed", exc)


@router.post("/dashboard/latest-unit", response_model=APIResponse)
def dashboard_latest_unit(request: UnitRequest) -> APIResponse:
    """
    Return latest health for one unit.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::dashboard_latest_unit")
    try:
        result = DashboardAPI().get_latest_unit_health(request.unit_id)
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Latest unit dashboard query failed", exc)


@router.post("/dashboard/health-trend", response_model=APIResponse)
def dashboard_health_trend(request: UnitRequest) -> APIResponse:
    """
    Return health trend for one unit.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::dashboard_health_trend")
    try:
        result = DashboardAPI().get_health_trend(request.unit_id)
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Health trend dashboard query failed", exc)


@router.post("/dashboard/anomalies", response_model=APIResponse)
def dashboard_anomalies(request: UnitRequest) -> APIResponse:
    """
    Return anomalies for one unit.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::dashboard_anomalies")
    try:
        result = DashboardAPI().get_anomalies(request.unit_id)
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Dashboard anomalies query failed", exc)


@router.post("/dashboard/explanation", response_model=APIResponse)
def dashboard_explanation(request: UnitCycleRequest) -> APIResponse:
    """
    Return explanation for one unit and cycle.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::dashboard_explanation")
    try:
        result = DashboardAPI().get_explanation(
            unit_id=request.unit_id,
            cycle=request.cycle,
        )
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Dashboard explanation query failed", exc)


@router.post("/dashboard/confidence", response_model=APIResponse)
def dashboard_confidence(request: UnitRequest) -> APIResponse:
    """
    Return confidence and uncertainty trend for one unit.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::dashboard_confidence")
    try:
        result = DashboardAPI().get_confidence_uncertainty(request.unit_id)
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Dashboard confidence query failed", exc)


@router.post("/evaluate", response_model=APIResponse)
def evaluate() -> APIResponse:
    """
    Run all evaluation modules:
    - digital twin
    - context
    - anomaly
    - health
    - reasoning
    - explainability
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::evaluate")
    try:
        stages = [
            ("evaluate_digital_twin", DigitalTwinEvaluator().run),
            ("evaluate_context", ContextEvaluator().run),
            ("evaluate_anomaly", AnomalyEvaluator().run),
            ("evaluate_health", HealthEvaluator().run),
            ("evaluate_reasoning", ReasoningEvaluator().run),
            ("evaluate_explainability", ExplainabilityEvaluator().run),
        ]

        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for stage_name, stage_function in stages:
            result = run_stage_safely(stage_name, stage_function)
            result_dict = _stage_result_to_dict(result)
            completed.append(result_dict)

            if result.status == "failed":
                failed.append(result_dict)
                break

        status = "success" if not failed else "partial_failure"

        return APIResponse(
            status=status,
            message=(
                "Evaluation completed."
                if status == "success"
                else "Evaluation stopped safely."
            ),
            output_file=str(Config.METRIC_DIR),
            data={
                "completed_stages": completed,
                "failed_stages": failed,
                "uses_y_targets": False,
            },
        )

    except Exception as exc:
        return _failed_response("Evaluation failed", exc)


@router.post("/full-pipeline", response_model=APIResponse)
def full_pipeline(include_shap: bool = False) -> APIResponse:
    """
    Run the full CA-EDT-AHMA pipeline safely.

    If one stage fails, previously generated files are not deleted.
    """
    print("[PROGRESS] Entering Backend/app/routers/Anomaly_Health_Monitering/Routes.py::full_pipeline")
    try:
        module = importlib.import_module(
            "app.pipeline.Anomaly_Health_Monitering.11_full_pipeline"
        )
        run_full_pipeline = getattr(module, "run_full_pipeline")
        result: Dict[str, Any] = run_full_pipeline(include_shap=include_shap)

        return APIResponse(
            status=str(result.get("status", "success")),
            message=str(result.get("message", "")),
            output_file=(
                str(result.get("dashboard_file"))
                if result.get("dashboard_file")
                else None
            ),
            records_count=int(result.get("completed_stage_count", 0)),
            data=result,
        )

    except Exception as exc:
        return _failed_response("Full pipeline failed", exc)
