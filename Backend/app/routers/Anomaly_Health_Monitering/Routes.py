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
- This route file is lightweight.
- Heavy service classes are imported lazily only when an endpoint runs.
- Numbered pipeline modules cannot be imported with normal Python syntax
  because filenames start with digits. This file uses importlib when needed.
- This component does not predict RUL.
- This component does not use Y_dev/Y_test.
- This component does not make final maintenance scheduling decisions.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/routers/Anomaly_Health_Monitering/Routes.py")

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import importlib
import os
import shutil
import sys

from fastapi import APIRouter, File, UploadFile


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
from app.schemas.Anomaly_Health_Monitering.anomaly_schema import FeedbackRequest
from app.schemas.Anomaly_Health_Monitering.common_schema import (
    APIResponse,
    UnitCycleRequest,
    UnitRequest,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


router = APIRouter(
    prefix="/anomaly-health-monitoring",
    tags=["Anomaly Health Monitoring"],
)


StageSpec = Tuple[str, Callable[[], Dict[str, object]]]


# ======================================================================================
# Response helpers
# ======================================================================================

def _api_response_from_dict(result: Dict[str, Any]) -> APIResponse:
    """
    Convert any service response dictionary into APIResponse.

    Args:
        result: Service result dictionary.

    Returns:
        APIResponse: Standard API response.
    """
    print("[PROGRESS] Entering Routes.py::_api_response_from_dict")

    standard_keys = {
        "status",
        "message",
        "output_file",
        "records_count",
        "metrics",
        "errors",
        "data",
    }

    data = result.get("data")

    if data is None:
        extra_data = {
            key: value
            for key, value in result.items()
            if key not in standard_keys and value is not None
        }
        data = extra_data if extra_data else None

    return APIResponse(
        status=str(result.get("status", "success")),
        message=str(result.get("message", "")),
        output_file=(
            str(result.get("output_file"))
            if result.get("output_file")
            else None
        ),
        records_count=(
            int(result["records_count"])
            if result.get("records_count") is not None
            else None
        ),
        metrics=result.get("metrics") if isinstance(result.get("metrics"), dict) else None,
        errors=result.get("errors") if isinstance(result.get("errors"), list) else None,
        data=data,
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
    print("[PROGRESS] Entering Routes.py::_failed_response")

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
    print("[PROGRESS] Entering Routes.py::_stage_result_to_dict")

    return {
        "status": stage_result.status,
        "message": stage_result.message,
        "output_file": stage_result.output_file,
        "records_count": stage_result.records_count,
        "elapsed_seconds": stage_result.elapsed_seconds,
    }


# ======================================================================================
# Lazy service helpers
# ======================================================================================

def _build_service_callable(
    module_path: str,
    class_name: str,
    method_name: str = "run",
) -> Callable[[], Dict[str, object]]:
    """
    Build a lazy service callable.

    This avoids importing and initializing heavy services at FastAPI startup.
    """

    def _call() -> Dict[str, object]:
        print(
            "[PROGRESS] Lazy loading service: "
            f"{module_path}.{class_name}.{method_name}"
        )

        module = importlib.import_module(module_path)
        service_class = getattr(module, class_name)
        instance = service_class()
        method = getattr(instance, method_name)

        return method()

    return _call


def _stage(
    stage_name: str,
    module_path: str,
    class_name: str,
    method_name: str = "run",
) -> StageSpec:
    """
    Create a stage spec.
    """
    return (
        stage_name,
        _build_service_callable(
            module_path=module_path,
            class_name=class_name,
            method_name=method_name,
        ),
    )


def _run_stage_group(
    stages: List[StageSpec],
    success_message: str,
    stopped_message: str,
    output_file: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> APIResponse:
    """
    Run a list of stages safely and return APIResponse.
    """
    print("[PROGRESS] Entering Routes.py::_run_stage_group")

    completed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for stage_name, stage_function in stages:
        print("=" * 100)
        print(f"[PROGRESS] Running API stage: {stage_name}")

        result: StageResult = run_stage_safely(stage_name, stage_function)
        result_dict = _stage_result_to_dict(result)

        completed.append(result_dict)

        print(f"[PROGRESS] API stage result: {result_dict}")

        if result.status == "failed":
            failed.append(result_dict)
            break

    status = "success" if not failed else "partial_failure"

    data: Dict[str, Any] = {
        "completed_stages": completed,
        "failed_stages": failed,
        "target_usage": {
            "uses_y_dev_y_test": False,
            "uses_rul_targets": False,
            "predicts_rul": False,
        },
        "decision_boundary": {
            "makes_maintenance_scheduling_decisions": False,
        },
    }

    if extra_data:
        data.update(extra_data)

    return APIResponse(
        status=status,
        message=success_message if status == "success" else stopped_message,
        output_file=output_file,
        data=data,
    )


def _run_single_service(
    module_path: str,
    class_name: str,
    method_name: str = "run",
) -> APIResponse:
    """
    Run one service lazily and convert response to APIResponse.
    """
    print("[PROGRESS] Entering Routes.py::_run_single_service")

    result = _build_service_callable(
        module_path=module_path,
        class_name=class_name,
        method_name=method_name,
    )()

    return _api_response_from_dict(result)


# ======================================================================================
# Upload endpoint
# ======================================================================================

@router.post("/upload", response_model=APIResponse)
async def upload_dataset(file: UploadFile = File(...)) -> APIResponse:
    """
    Upload the N-CMAPSS HDF5 file.

    The uploaded file is saved as:
    Backend/data/Anomaly_Health_Monitering/N-CMAPSS_DS01-005.h5
    """
    print("[PROGRESS] Entering Routes.py::upload_dataset")

    try:
        Config.create_directories()

        if not file.filename:
            return APIResponse(
                status="failed",
                message="Uploaded file has no filename.",
                errors=["Missing filename."],
            )

        filename = Path(file.filename).name

        if not filename.lower().endswith(".h5"):
            return APIResponse(
                status="failed",
                message="Only .h5 files are accepted.",
                errors=["Invalid file extension."],
            )

        output_path: Path = Config.H5_FILE_PATH
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if temp_path.exists():
            temp_path.unlink()

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        os.replace(temp_path, output_path)

        logger.info("Dataset uploaded to %s.", output_path)

        return APIResponse(
            status="success",
            message="Dataset uploaded successfully.",
            output_file=str(output_path),
            data={
                "stored_filename": output_path.name,
                "original_filename": filename,
            },
        )

    except Exception as exc:
        return _failed_response("Dataset upload failed", exc)


# ======================================================================================
# Pipeline stage endpoints
# ======================================================================================

@router.post("/preprocess", response_model=APIResponse)
def preprocess() -> APIResponse:
    """
    Run preprocessing:
    1. Load HDF5
    2. Clean data
    3. Engineer features
    4. Scale features with dev-only fitting
    """
    print("[PROGRESS] Entering Routes.py::preprocess")

    try:
        stages = [
            _stage(
                "data_loading",
                "app.services.Anomaly_Health_Monitering.Data_Preprocessing.data_loader",
                "DataLoader",
                "save_raw_data",
            ),
            _stage(
                "cleaning",
                "app.services.Anomaly_Health_Monitering.Data_Preprocessing.cleaner",
                "DataCleaner",
            ),
            _stage(
                "feature_engineering",
                "app.services.Anomaly_Health_Monitering.Data_Preprocessing.feature_engineering",
                "FeatureEngineer",
            ),
            _stage(
                "dev_only_scaling",
                "app.services.Anomaly_Health_Monitering.Data_Preprocessing.scaler",
                "FeatureScaler",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Preprocessing completed.",
            stopped_message="Preprocessing stopped safely. Previous outputs were not deleted.",
            output_file=str(Config.SCALED_CSV) if Config.SCALED_CSV.exists() else None,
            extra_data={
                "fit_rule": "Scaler fitted only on dev split and test transformed only.",
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
    print("[PROGRESS] Entering Routes.py::context_modeling")

    try:
        stages = [
            _stage(
                "operating_mode_detection",
                "app.services.Anomaly_Health_Monitering.context_modeling.operating_mode_detector",
                "OperatingModeDetector",
            ),
            _stage(
                "context_drift_detection",
                "app.services.Anomaly_Health_Monitering.context_modeling.context_drift",
                "ContextDriftDetector",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Context modeling completed.",
            stopped_message="Context modeling stopped safely.",
            output_file=str(Config.CONTEXT_CSV) if Config.CONTEXT_CSV.exists() else None,
            extra_data={
                "fit_rule": "K-Means and GMM fitted only on W_dev. Test is prediction/scoring only.",
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
    print("[PROGRESS] Entering Routes.py::train_digital_twin")

    try:
        stages = [
            _stage(
                "random_forest_twin",
                "app.services.Anomaly_Health_Monitering.digital_twin.random_forest_twin",
                "RandomForestTwin",
            ),
            _stage(
                "xgboost_twin",
                "app.services.Anomaly_Health_Monitering.digital_twin.xgboost_twin",
                "XGBoostTwin",
            ),
            _stage(
                "lightgbm_twin",
                "app.services.Anomaly_Health_Monitering.digital_twin.lightgbm_twin",
                "LightGBMTwin",
            ),
            _stage(
                "ensemble_twin",
                "app.services.Anomaly_Health_Monitering.digital_twin.ensemble_twin",
                "EnsembleDigitalTwin",
            ),
            _stage(
                "twin_comparator",
                "app.services.Anomaly_Health_Monitering.digital_twin.twin_comparator",
                "TwinComparator",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Digital twin training and inference completed.",
            stopped_message="Digital twin stage stopped safely.",
            output_file=(
                str(Config.ENSEMBLE_PREDICTIONS_CSV)
                if Config.ENSEMBLE_PREDICTIONS_CSV.exists()
                else None
            ),
            extra_data={
                "fit_rule": "RF, XGBoost, and LightGBM trained only on dev split.",
                "target": "Measured X_s sensor values only, not RUL.",
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
    print("[PROGRESS] Entering Routes.py::generate_residuals")

    try:
        return _run_single_service(
            module_path="app.services.Anomaly_Health_Monitering.digital_twin.residual_calculator",
            class_name="ResidualCalculator",
        )

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
    print("[PROGRESS] Entering Routes.py::detect_anomalies")

    try:
        stages = [
            _stage(
                "residual_anomaly_detector",
                "app.services.Anomaly_Health_Monitering.anomaly_detection.residual_anomaly_detector",
                "ResidualAnomalyDetector",
            ),
            _stage(
                "isolation_forest_detector",
                "app.services.Anomaly_Health_Monitering.anomaly_detection.isolation_forest_detector",
                "IsolationForestDetector",
            ),
            _stage(
                "mahalanobis_detector",
                "app.services.Anomaly_Health_Monitering.anomaly_detection.mahalanobis_detector",
                "MahalanobisDetector",
            ),
            _stage(
                "anomaly_fusion",
                "app.services.Anomaly_Health_Monitering.anomaly_detection.anomaly_fusion",
                "AnomalyFusion",
            ),
            _stage(
                "severity_classifier",
                "app.services.Anomaly_Health_Monitering.anomaly_detection.severity_classifier",
                "SeverityClassifier",
            ),
            _stage(
                "early_warning_score",
                "app.services.Anomaly_Health_Monitering.anomaly_detection.early_warning_score",
                "EarlyWarningScore",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Anomaly detection completed.",
            stopped_message="Anomaly detection stopped safely.",
            output_file=(
                str(Config.ANOMALY_FUSION_CSV)
                if Config.ANOMALY_FUSION_CSV.exists()
                else None
            ),
            extra_data={
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
    print("[PROGRESS] Entering Routes.py::generate_health_index")

    try:
        return _run_single_service(
            module_path="app.services.Anomaly_Health_Monitering.health_monitoring.health_index_calculator",
            class_name="HealthIndexCalculator",
        )

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
    print("[PROGRESS] Entering Routes.py::generate_health_score")

    try:
        stages = [
            _stage(
                "health_index",
                "app.services.Anomaly_Health_Monitering.health_monitoring.health_index_calculator",
                "HealthIndexCalculator",
            ),
            _stage(
                "health_state",
                "app.services.Anomaly_Health_Monitering.health_monitoring.health_state_classifier",
                "HealthStateClassifier",
            ),
            _stage(
                "health_trend",
                "app.services.Anomaly_Health_Monitering.health_monitoring.health_trend_tracker",
                "HealthTrendTracker",
            ),
            _stage(
                "health_alerts",
                "app.services.Anomaly_Health_Monitering.health_monitoring.health_alert_engine",
                "HealthAlertEngine",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Health score generation completed.",
            stopped_message="Health score generation stopped safely.",
            output_file=(
                str(Config.HEALTH_STATES_CSV)
                if Config.HEALTH_STATES_CSV.exists()
                else None
            ),
            extra_data={
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
    print("[PROGRESS] Entering Routes.py::classify_health_state")

    try:
        return _run_single_service(
            module_path="app.services.Anomaly_Health_Monitering.health_monitoring.health_state_classifier",
            class_name="HealthStateClassifier",
        )

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
    print("[PROGRESS] Entering Routes.py::root_cause_analysis")

    try:
        stages = [
            _stage(
                "sensor_dependency_graph",
                "app.services.Anomaly_Health_Monitering.reasoning.sensor_dependency_graph",
                "SensorDependencyGraph",
            ),
            _stage(
                "root_cause_analysis",
                "app.services.Anomaly_Health_Monitering.reasoning.root_cause_analyzer",
                "RootCauseAnalyzer",
            ),
            _stage(
                "root_cause_tracking",
                "app.services.Anomaly_Health_Monitering.reasoning.root_cause_tracker",
                "RootCauseTracker",
            ),
            _stage(
                "temporal_reasoning",
                "app.services.Anomaly_Health_Monitering.reasoning.temporal_reasoning",
                "TemporalReasoning",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Root-cause analysis completed.",
            stopped_message="Root-cause analysis stopped safely.",
            output_file=str(Config.ROOT_CAUSE_CSV) if Config.ROOT_CAUSE_CSV.exists() else None,
            extra_data={
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
    print("[PROGRESS] Entering Routes.py::explain")

    try:
        stages = [
            _stage(
                "sensor_residual_ranking",
                "app.services.Anomaly_Health_Monitering.explainability.sensor_residual_ranking",
                "SensorResidualRanking",
            ),
            _stage(
                "subsystem_explainer",
                "app.services.Anomaly_Health_Monitering.explainability.subsystem_explainer",
                "SubsystemExplainer",
            ),
            _stage(
                "explanation_generator",
                "app.services.Anomaly_Health_Monitering.explainability.explanation_generator",
                "ExplanationGenerator",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Explainability reports generated.",
            stopped_message="Explainability stopped safely.",
            output_file=(
                str(Config.EXPLANATION_REPORTS_CSV)
                if Config.EXPLANATION_REPORTS_CSV.exists()
                else None
            ),
            extra_data={
                "hard_causality_claim": False,
                "maintenance_decision": False,
            },
        )

    except Exception as exc:
        return _failed_response("Explainability generation failed", exc)


@router.post("/confidence", response_model=APIResponse)
def confidence() -> APIResponse:
    """
    Generate model agreement, confidence, reliability, and uncertainty scores.
    """
    print("[PROGRESS] Entering Routes.py::confidence")

    try:
        stages = [
            _stage(
                "model_agreement",
                "app.services.Anomaly_Health_Monitering.uncertainty.model_agreement",
                "ModelAgreementCalculator",
            ),
            _stage(
                "confidence_estimation",
                "app.services.Anomaly_Health_Monitering.uncertainty.confidence_estimator",
                "ConfidenceEstimator",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Confidence and uncertainty calculation completed.",
            stopped_message="Confidence calculation stopped safely.",
            output_file=str(Config.CONFIDENCE_CSV) if Config.CONFIDENCE_CSV.exists() else None,
            extra_data={
                "formula": (
                    "confidence = 0.35*model_agreement_score + "
                    "0.25*context_confidence + "
                    "0.25*anomaly_persistence_score + "
                    "0.15*data_quality_score"
                ),
                "model_agreement_normalization_fit_split": Config.DEV_SPLIT_NAME,
                "test_split_used_for_normalization": False,
            },
        )

    except Exception as exc:
        return _failed_response("Confidence estimation failed", exc)


# ======================================================================================
# Feedback endpoints
# ======================================================================================

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
    print("[PROGRESS] Entering Routes.py::feedback")

    try:
        module = importlib.import_module(
            "app.services.Anomaly_Health_Monitering.feedback.learning_updater"
        )
        learning_updater_class = getattr(module, "LearningUpdater")
        updater = learning_updater_class()

        result = updater.submit_feedback(
            unit_id=request.unit_id,
            cycle=request.cycle,
            context_id=request.context_id,
            alert_level=request.alert_level,
            final_anomaly_score=request.final_anomaly_score,
            root_cause_pattern=request.root_cause_pattern,
            feedback_label=request.feedback_label,
            operator_note=getattr(request, "operator_note", None),
        )

        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Feedback update failed", exc)


# ======================================================================================
# Dashboard endpoints
# ======================================================================================

@router.post("/dashboard", response_model=APIResponse)
def dashboard() -> APIResponse:
    """
    Generate dashboard_data.csv.
    """
    print("[PROGRESS] Entering Routes.py::dashboard")

    try:
        return _run_single_service(
            module_path="app.services.Anomaly_Health_Monitering.dashboard.dashboard_data_generator",
            class_name="DashboardDataGenerator",
        )

    except Exception as exc:
        return _failed_response("Dashboard generation failed", exc)


@router.get("/dashboard/summary", response_model=APIResponse)
def dashboard_summary() -> APIResponse:
    """
    Return dashboard summary counts.
    """
    print("[PROGRESS] Entering Routes.py::dashboard_summary")

    try:
        module = importlib.import_module(
            "app.services.Anomaly_Health_Monitering.dashboard.dashboard_api"
        )
        dashboard_api_class = getattr(module, "DashboardAPI")
        result = dashboard_api_class().get_summary()
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Dashboard summary failed", exc)


@router.get("/dashboard/latest-all", response_model=APIResponse)
def dashboard_latest_all() -> APIResponse:
    """
    Return latest health for all units.
    """
    print("[PROGRESS] Entering Routes.py::dashboard_latest_all")

    try:
        module = importlib.import_module(
            "app.services.Anomaly_Health_Monitering.dashboard.dashboard_api"
        )
        dashboard_api_class = getattr(module, "DashboardAPI")
        result = dashboard_api_class().get_latest_all_units()
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Latest all-units dashboard query failed", exc)


@router.post("/dashboard/latest-unit", response_model=APIResponse)
def dashboard_latest_unit(request: UnitRequest) -> APIResponse:
    """
    Return latest health for one unit.
    """
    print("[PROGRESS] Entering Routes.py::dashboard_latest_unit")

    try:
        module = importlib.import_module(
            "app.services.Anomaly_Health_Monitering.dashboard.dashboard_api"
        )
        dashboard_api_class = getattr(module, "DashboardAPI")
        result = dashboard_api_class().get_latest_unit_health(request.unit_id)
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Latest unit dashboard query failed", exc)


@router.post("/dashboard/health-trend", response_model=APIResponse)
def dashboard_health_trend(request: UnitRequest) -> APIResponse:
    """
    Return health trend for one unit.
    """
    print("[PROGRESS] Entering Routes.py::dashboard_health_trend")

    try:
        module = importlib.import_module(
            "app.services.Anomaly_Health_Monitering.dashboard.dashboard_api"
        )
        dashboard_api_class = getattr(module, "DashboardAPI")
        result = dashboard_api_class().get_health_trend(request.unit_id)
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Health trend dashboard query failed", exc)


@router.post("/dashboard/anomalies", response_model=APIResponse)
def dashboard_anomalies(request: UnitRequest) -> APIResponse:
    """
    Return anomalies for one unit.
    """
    print("[PROGRESS] Entering Routes.py::dashboard_anomalies")

    try:
        module = importlib.import_module(
            "app.services.Anomaly_Health_Monitering.dashboard.dashboard_api"
        )
        dashboard_api_class = getattr(module, "DashboardAPI")
        result = dashboard_api_class().get_anomalies(request.unit_id)
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Dashboard anomalies query failed", exc)


@router.post("/dashboard/explanation", response_model=APIResponse)
def dashboard_explanation(request: UnitCycleRequest) -> APIResponse:
    """
    Return explanation for one unit and cycle.
    """
    print("[PROGRESS] Entering Routes.py::dashboard_explanation")

    try:
        module = importlib.import_module(
            "app.services.Anomaly_Health_Monitering.dashboard.dashboard_api"
        )
        dashboard_api_class = getattr(module, "DashboardAPI")
        result = dashboard_api_class().get_explanation(
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
    print("[PROGRESS] Entering Routes.py::dashboard_confidence")

    try:
        module = importlib.import_module(
            "app.services.Anomaly_Health_Monitering.dashboard.dashboard_api"
        )
        dashboard_api_class = getattr(module, "DashboardAPI")
        result = dashboard_api_class().get_confidence_uncertainty(request.unit_id)
        return _api_response_from_dict(result)

    except Exception as exc:
        return _failed_response("Dashboard confidence query failed", exc)


# ======================================================================================
# Evaluation endpoint
# ======================================================================================

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
    print("[PROGRESS] Entering Routes.py::evaluate")

    try:
        stages = [
            _stage(
                "evaluate_digital_twin",
                "app.services.Anomaly_Health_Monitering.evaluation.evaluate_digital_twin",
                "DigitalTwinEvaluator",
            ),
            _stage(
                "evaluate_context",
                "app.services.Anomaly_Health_Monitering.evaluation.evaluate_context",
                "ContextEvaluator",
            ),
            _stage(
                "evaluate_anomaly",
                "app.services.Anomaly_Health_Monitering.evaluation.evaluate_anomaly",
                "AnomalyEvaluator",
            ),
            _stage(
                "evaluate_health",
                "app.services.Anomaly_Health_Monitering.evaluation.evaluate_health",
                "HealthEvaluator",
            ),
            _stage(
                "evaluate_reasoning",
                "app.services.Anomaly_Health_Monitering.evaluation.evaluate_reasoning",
                "ReasoningEvaluator",
            ),
            _stage(
                "evaluate_explainability",
                "app.services.Anomaly_Health_Monitering.evaluation.evaluate_explainability",
                "ExplainabilityEvaluator",
            ),
        ]

        return _run_stage_group(
            stages=stages,
            success_message="Evaluation completed.",
            stopped_message="Evaluation stopped safely.",
            output_file=str(Config.METRIC_DIR),
            extra_data={
                "uses_y_targets": False,
                "evaluation_mode": "label-free component evaluation unless external anomaly labels are provided.",
            },
        )

    except Exception as exc:
        return _failed_response("Evaluation failed", exc)


# ======================================================================================
# Full pipeline endpoint
# ======================================================================================

@router.post("/full-pipeline", response_model=APIResponse)
def full_pipeline(
    include_shap: bool = False,
    include_evaluation: bool = True,
    include_dashboard: bool = True,
    include_feedback: bool = True,
    include_context_drift: bool = True,
    include_twin_comparison: bool = True,
) -> APIResponse:
    """
    Run the full CA-EDT-AHMA pipeline safely.

    If one stage fails, previously generated files are not deleted.
    """
    print("[PROGRESS] Entering Routes.py::full_pipeline")

    try:
        module = importlib.import_module(
            "app.pipeline.Anomaly_Health_Monitering.11_full_pipeline"
        )
        run_full_pipeline = getattr(module, "run_full_pipeline")

        result: Dict[str, Any] = run_full_pipeline(
            include_shap=include_shap,
            include_evaluation=include_evaluation,
            include_dashboard=include_dashboard,
            include_feedback=include_feedback,
            include_context_drift=include_context_drift,
            include_twin_comparison=include_twin_comparison,
        )

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