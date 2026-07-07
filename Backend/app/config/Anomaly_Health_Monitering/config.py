"""
Central configuration for CA-EDT-AHMA.

CA-EDT-AHMA:
Context-Aware Ensemble Digital Twin for Explainable Health Monitoring
and Anomaly Reasoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


class Config:
    """
    Central configuration class.

    This class avoids hardcoded paths inside service files and keeps the
    complete project reproducible.
    """

    PROJECT_NAME: str = "CA-EDT-AHMA"
    SHORT_MODEL_NAME: str = "CA-EDT-AHMA"
    FULL_MODEL_NAME: str = (
        "Context-Aware Ensemble Digital Twin for Explainable Health Monitoring "
        "and Anomaly Reasoning"
    )

    BASE_DIR: Path = Path(__file__).resolve().parents[3]

    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "Anomaly_Health_Monitering"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "processed"
    OUTPUT_DIR: Path = BASE_DIR / "outputs" / "Anomaly_Health_Monitering"

    MODEL_DIR: Path = BASE_DIR / "models"
    CONTEXT_MODEL_DIR: Path = MODEL_DIR / "context"
    DIGITAL_TWIN_MODEL_DIR: Path = MODEL_DIR / "digital_twin"
    ANOMALY_MODEL_DIR: Path = MODEL_DIR / "anomaly"
    HEALTH_MODEL_DIR: Path = MODEL_DIR / "health"
    UNCERTAINTY_MODEL_DIR: Path = MODEL_DIR / "uncertainty"
    FEEDBACK_MODEL_DIR: Path = MODEL_DIR / "feedback"

    LOG_DIR: Path = BASE_DIR / "logs"
    REPORT_DIR: Path = BASE_DIR / "reports"
    METRIC_DIR: Path = BASE_DIR / "metrics"
    EXPERIMENT_DIR: Path = BASE_DIR / "experiments"
    MLRUNS_DIR: Path = BASE_DIR / "mlruns"

    H5_FILE_PATH: Path = RAW_DATA_DIR / "N-CMAPSS_DS01-005.h5"

    RAW_CSV: Path = PROCESSED_DATA_DIR / "raw_data.csv"
    CLEANED_CSV: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
    ENGINEERED_CSV: Path = PROCESSED_DATA_DIR / "engineered_features.csv"
    SCALED_CSV: Path = PROCESSED_DATA_DIR / "scaled_features.csv"

    CONTEXT_CSV: Path = OUTPUT_DIR / "context_clusters.csv"

    RF_PREDICTIONS_CSV: Path = OUTPUT_DIR / "rf_predictions.csv"
    XGB_PREDICTIONS_CSV: Path = OUTPUT_DIR / "xgb_predictions.csv"
    LGBM_PREDICTIONS_CSV: Path = OUTPUT_DIR / "lgbm_predictions.csv"
    ENSEMBLE_PREDICTIONS_CSV: Path = OUTPUT_DIR / "ensemble_predictions.csv"
    RESIDUALS_CSV: Path = OUTPUT_DIR / "residuals.csv"

    RESIDUAL_ANOMALY_CSV: Path = OUTPUT_DIR / "residual_anomaly_scores.csv"
    IFOREST_CSV: Path = OUTPUT_DIR / "isolation_forest_scores.csv"
    MAHALANOBIS_CSV: Path = OUTPUT_DIR / "mahalanobis_scores.csv"
    ANOMALY_FUSION_CSV: Path = OUTPUT_DIR / "anomaly_fusion.csv"

    HEALTH_INDEX_CSV: Path = OUTPUT_DIR / "health_index.csv"
    HEALTH_STATES_CSV: Path = OUTPUT_DIR / "health_states.csv"
    ROOT_CAUSE_CSV: Path = OUTPUT_DIR / "root_cause_analysis.csv"
    SHAP_CSV: Path = OUTPUT_DIR / "shap_explanations.csv"
    EXPLANATION_REPORTS_CSV: Path = OUTPUT_DIR / "explanation_reports.csv"
    MODEL_AGREEMENT_CSV: Path = OUTPUT_DIR / "model_agreement.csv"
    CONFIDENCE_CSV: Path = OUTPUT_DIR / "confidence_scores.csv"
    FEEDBACK_UPDATES_CSV: Path = OUTPUT_DIR / "feedback_updates.csv"
    DASHBOARD_CSV: Path = OUTPUT_DIR / "dashboard_data.csv"
    ALERT_MEMORY_CSV: Path = OUTPUT_DIR / "alert_memory.csv"

    SCALER_PATH: Path = MODEL_DIR / "feature_scaler.pkl"

    KMEANS_MODEL_PATH: Path = CONTEXT_MODEL_DIR / "kmeans_context.pkl"
    GMM_MODEL_PATH: Path = CONTEXT_MODEL_DIR / "gmm_context.pkl"

    RF_MODEL_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "random_forest_twin.pkl"
    XGB_MODEL_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "xgboost_twin.pkl"
    LGBM_MODEL_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "lightgbm_twin.pkl"
    ENSEMBLE_WEIGHTS_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "ensemble_weights.json"

    RESIDUAL_THRESHOLDS_PATH: Path = ANOMALY_MODEL_DIR / "residual_thresholds.json"
    IFOREST_MODEL_PATH: Path = ANOMALY_MODEL_DIR / "isolation_forest.pkl"
    MAHALANOBIS_PARAMS_PATH: Path = ANOMALY_MODEL_DIR / "mahalanobis_params.pkl"
    FUSION_WEIGHTS_PATH: Path = ANOMALY_MODEL_DIR / "fusion_weights.json"

    HEALTH_INDEX_CONFIG_PATH: Path = HEALTH_MODEL_DIR / "health_index_config.json"
    HEALTH_STATE_THRESHOLDS_PATH: Path = HEALTH_MODEL_DIR / "health_state_thresholds.json"

    CONFIDENCE_CONFIG_PATH: Path = UNCERTAINTY_MODEL_DIR / "confidence_config.json"

    ADAPTIVE_THRESHOLDS_PATH: Path = FEEDBACK_MODEL_DIR / "adaptive_thresholds.json"

    REQUIRED_H5_GROUPS: Tuple[str, ...] = (
        "A_dev",
        "A_test",
        "W_dev",
        "W_test",
        "X_s_dev",
        "X_s_test",
        "X_v_dev",
        "X_v_test",
    )

    OPTIONAL_H5_GROUPS: Tuple[str, ...] = (
        "T_dev",
        "T_test",
    )

    IGNORED_H5_GROUPS: Tuple[str, ...] = (
        "Y_dev",
        "Y_test",
    )

    VARIABLE_NAME_GROUPS: Dict[str, str] = {
        "A": "A_var",
        "W": "W_var",
        "X_s": "X_s_var",
        "X_v": "X_v_var",
        "T": "T_var",
    }

    DEV_SPLIT_NAME: str = "dev"
    TEST_SPLIT_NAME: str = "test"

    RANDOM_SEED: int = 42
    CONTEXT_CLUSTER_COUNT: int = 6
    GMM_COVARIANCE_TYPE: str = "full"
    ROLLING_WINDOW: int = 5

    API_TITLE: str = "CA-EDT-AHMA API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    KMEANS_PARAMS: Dict[str, int] = {
        "n_init": 10,
        "max_iter": 300,
    }

    GMM_PARAMS: Dict[str, int | str] = {
        "max_iter": 300,
        "init_params": "kmeans",
    }

    RF_PARAMS: Dict[str, int | float | bool | None] = {
        "n_estimators": 120,
        "max_depth": 18,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    XGB_PARAMS: Dict[str, int | float | str] = {
        "n_estimators": 180,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
    }

    LGBM_PARAMS: Dict[str, int | float | str] = {
        "n_estimators": 180,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    }

    RESIDUAL_PERCENTILES: Dict[str, float] = {
        "watch": 90.0,
        "warning": 95.0,
        "critical": 99.0,
    }

    FUSION_WEIGHTS: Dict[str, float] = {
        "residual": 0.50,
        "iforest": 0.30,
        "mahalanobis": 0.20,
    }

    HEALTH_WEIGHTS: Dict[str, float] = {
        "final_anomaly_score": 60.0,
        "residual_trend_score": 25.0,
        "anomaly_persistence_score": 15.0,
    }

    CONFIDENCE_WEIGHTS: Dict[str, float] = {
        "model_agreement_score": 0.35,
        "context_confidence": 0.25,
        "anomaly_persistence_score": 0.25,
        "data_quality_score": 0.15,
    }

    FEEDBACK_LABELS: Tuple[str, ...] = (
        "accepted_alert",
        "rejected_false_alarm",
        "missed_anomaly",
        "uncertain",
    )

    FINAL_DASHBOARD_COLUMNS: Tuple[str, ...] = (
        "unit_id",
        "cycle",
        "split",
        "kmeans_context_id",
        "gmm_context_id",
        "context_confidence",
        "health_index",
        "remaining_health_percentage",
        "health_state",
        "final_anomaly_score",
        "alert_level",
        "top_sensor_1",
        "top_sensor_2",
        "top_sensor_3",
        "contribution_1",
        "contribution_2",
        "contribution_3",
        "root_cause_pattern",
        "explanation_text",
        "model_agreement_score",
        "confidence_score",
        "uncertainty_score",
        "reliability_score",
        "feedback_status",
    )

    @classmethod
    def create_directories(cls) -> None:
        """
        Create the required project directories safely.

        This method never deletes existing directories or files.
        """
        directories: List[Path] = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.MODEL_DIR,
            cls.CONTEXT_MODEL_DIR,
            cls.DIGITAL_TWIN_MODEL_DIR,
            cls.ANOMALY_MODEL_DIR,
            cls.HEALTH_MODEL_DIR,
            cls.UNCERTAINTY_MODEL_DIR,
            cls.FEEDBACK_MODEL_DIR,
            cls.LOG_DIR,
            cls.REPORT_DIR,
            cls.METRIC_DIR,
            cls.EXPERIMENT_DIR,
            cls.MLRUNS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    Config.create_directories()
    print(f"{Config.PROJECT_NAME} directory structure initialized at: {Config.BASE_DIR}")
