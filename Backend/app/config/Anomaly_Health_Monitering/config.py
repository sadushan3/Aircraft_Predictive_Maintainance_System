"""
Central configuration for CA-EDT-AHMA.

CA-EDT-AHMA:
Context-Aware Ensemble Digital Twin for Explainable Health Monitoring
and Anomaly Reasoning.

Important:
TensorFlow is a library/framework, not the model name.

Main digital twin models:
1. Random Forest Regressor implemented using scikit-learn.
2. XGBoost Regressor implemented using XGBoost.
3. LightGBM Regressor implemented using LightGBM.
4. MLP Digital Twin Regressor implemented using TensorFlow/Keras.

Final active digital twin:
- 4-model weighted Ensemble Digital Twin.

Main anomaly model:
- Residual Autoencoder Anomaly Detector implemented using TensorFlow/Keras.
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

    # ==================================================================================
    # Base directories
    # ==================================================================================

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

    # ==================================================================================
    # Input data paths
    # ==================================================================================

    H5_FILE_PATH: Path = RAW_DATA_DIR / "N-CMAPSS_DS01-005.h5"

    RAW_CSV: Path = PROCESSED_DATA_DIR / "raw_data.csv"
    CLEANED_CSV: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
    ENGINEERED_CSV: Path = PROCESSED_DATA_DIR / "engineered_features.csv"
    SCALED_CSV: Path = PROCESSED_DATA_DIR / "scaled_features.csv"

    # ==================================================================================
    # Context modeling outputs
    # ==================================================================================

    CONTEXT_CSV: Path = OUTPUT_DIR / "context_clusters.csv"
    CONTEXT_DRIFT_CSV: Path = OUTPUT_DIR / "context_drift.csv"

    # ==================================================================================
    # Digital twin prediction outputs
    # ==================================================================================

    RF_PREDICTIONS_CSV: Path = OUTPUT_DIR / "rf_predictions.csv"
    XGB_PREDICTIONS_CSV: Path = OUTPUT_DIR / "xgb_predictions.csv"
    LGBM_PREDICTIONS_CSV: Path = OUTPUT_DIR / "lgbm_predictions.csv"
    MLP_TWIN_PREDICTIONS_CSV: Path = OUTPUT_DIR / "mlp_twin_predictions.csv"
    ENSEMBLE_PREDICTIONS_CSV: Path = OUTPUT_DIR / "ensemble_predictions.csv"

    # Backward-compatible TensorFlow prediction alias
    TF_PREDICTIONS_CSV: Path = MLP_TWIN_PREDICTIONS_CSV

    # ==================================================================================
    # Digital twin model paths
    # ==================================================================================

    RF_MODEL_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "random_forest_twin.pkl"
    RF_MODEL_METADATA_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "random_forest_twin_metadata.json"

    XGB_MODEL_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "xgboost_twin.pkl"
    XGB_MODEL_METADATA_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "xgboost_twin_metadata.json"

    LGBM_MODEL_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "lightgbm_twin.pkl"
    LGBM_MODEL_METADATA_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "lightgbm_twin_metadata.json"

    MLP_TWIN_MODEL_NAME: str = "mlp_digital_twin_regressor"
    MLP_TWIN_LIBRARY: str = "TensorFlow/Keras"
    MLP_TWIN_MODEL_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "mlp_digital_twin.keras"
    MLP_TWIN_METADATA_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "mlp_digital_twin_metadata.json"
    MLP_TWIN_METRICS_CSV: Path = METRIC_DIR / "mlp_digital_twin_metrics.csv"
    MLP_TWIN_PREDICTION_PREFIX: str = "tf_predicted_"

    # Backward-compatible TensorFlow aliases
    TF_MODEL_PATH: Path = MLP_TWIN_MODEL_PATH
    TF_MODEL_METADATA_PATH: Path = MLP_TWIN_METADATA_PATH
    TF_METRICS_CSV: Path = MLP_TWIN_METRICS_CSV
    TF_PREDICTION_PREFIX: str = MLP_TWIN_PREDICTION_PREFIX

    # ==================================================================================
    # 4-model Ensemble Digital Twin - ACTIVE digital twin
    # ==================================================================================

    ENSEMBLE_MODEL_NAME: str = "four_model_weighted_ensemble_digital_twin"
    ENSEMBLE_LIBRARY: str = "scikit-learn + XGBoost + LightGBM + TensorFlow/Keras"

    ENSEMBLE_MEMBERS: Tuple[str, ...] = (
        "random_forest",
        "xgboost",
        "lightgbm",
        "mlp_digital_twin",
    )

    ENSEMBLE_WEIGHTS_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "ensemble_weights.json"
    ENSEMBLE_METADATA_PATH: Path = DIGITAL_TWIN_MODEL_DIR / "ensemble_metadata.json"
    ENSEMBLE_METRICS_CSV: Path = METRIC_DIR / "ensemble_digital_twin_metrics.csv"
    ENSEMBLE_CHUNK_SIZE: int = 25_000

    ACTIVE_DIGITAL_TWIN_MODEL_NAME: str = ENSEMBLE_MODEL_NAME
    ACTIVE_DIGITAL_TWIN_LIBRARY: str = ENSEMBLE_LIBRARY
    ACTIVE_DIGITAL_TWIN_PREDICTIONS_CSV: Path = ENSEMBLE_PREDICTIONS_CSV
    ACTIVE_DIGITAL_TWIN_PREDICTION_PREFIX: str = "ensemble_predicted_"

    # ==================================================================================
    # Residual and anomaly outputs
    # ==================================================================================

    RESIDUALS_CSV: Path = OUTPUT_DIR / "residuals.csv"

    RESIDUAL_ANOMALY_CSV: Path = OUTPUT_DIR / "residual_anomaly_scores.csv"
    IFOREST_CSV: Path = OUTPUT_DIR / "isolation_forest_scores.csv"
    MAHALANOBIS_CSV: Path = OUTPUT_DIR / "mahalanobis_scores.csv"
    ANOMALY_FUSION_CSV: Path = OUTPUT_DIR / "anomaly_fusion.csv"

    # ==================================================================================
    # Residual Autoencoder implemented using TensorFlow/Keras - PRIMARY anomaly model
    # ==================================================================================

    RESIDUAL_AUTOENCODER_MODEL_NAME: str = "residual_autoencoder_anomaly_detector"
    RESIDUAL_AUTOENCODER_LIBRARY: str = "TensorFlow/Keras"

    RESIDUAL_AUTOENCODER_MODEL_PATH: Path = (
        ANOMALY_MODEL_DIR / "residual_autoencoder.keras"
    )
    RESIDUAL_AUTOENCODER_METADATA_PATH: Path = (
        ANOMALY_MODEL_DIR / "residual_autoencoder_metadata.json"
    )
    RESIDUAL_AUTOENCODER_SCORES_CSV: Path = OUTPUT_DIR / "residual_autoencoder_scores.csv"
    RESIDUAL_AUTOENCODER_METRICS_CSV: Path = METRIC_DIR / "residual_autoencoder_metrics.csv"

    # Backward-compatible aliases if code uses TF_ANOMALY_* names
    TF_ANOMALY_MODEL_PATH: Path = RESIDUAL_AUTOENCODER_MODEL_PATH
    TF_ANOMALY_METADATA_PATH: Path = RESIDUAL_AUTOENCODER_METADATA_PATH
    TF_ANOMALY_CSV: Path = RESIDUAL_AUTOENCODER_SCORES_CSV
    TF_ANOMALY_METRICS_CSV: Path = RESIDUAL_AUTOENCODER_METRICS_CSV

    ACTIVE_ANOMALY_MODEL_NAME: str = RESIDUAL_AUTOENCODER_MODEL_NAME
    ACTIVE_ANOMALY_LIBRARY: str = RESIDUAL_AUTOENCODER_LIBRARY
    ACTIVE_ANOMALY_SCORE_CSV: Path = RESIDUAL_AUTOENCODER_SCORES_CSV

    # ==================================================================================
    # Health, reasoning, explainability, uncertainty, feedback, dashboard outputs
    # ==================================================================================

    HEALTH_INDEX_CSV: Path = OUTPUT_DIR / "health_index.csv"
    HEALTH_STATES_CSV: Path = OUTPUT_DIR / "health_states.csv"

    ROOT_CAUSE_CSV: Path = OUTPUT_DIR / "root_cause_analysis.csv"
    ROOT_CAUSE_MEMORY_CSV: Path = OUTPUT_DIR / "root_cause_memory.csv"
    TEMPORAL_REASONING_CSV: Path = OUTPUT_DIR / "temporal_reasoning.csv"

    SHAP_CSV: Path = OUTPUT_DIR / "shap_explanations.csv"
    EXPLANATION_REPORTS_CSV: Path = OUTPUT_DIR / "explanation_reports.csv"

    MODEL_AGREEMENT_CSV: Path = OUTPUT_DIR / "model_agreement.csv"
    CONFIDENCE_CSV: Path = OUTPUT_DIR / "confidence_scores.csv"

    FEEDBACK_UPDATES_CSV: Path = OUTPUT_DIR / "feedback_updates.csv"
    ALERT_MEMORY_CSV: Path = OUTPUT_DIR / "alert_memory.csv"

    DASHBOARD_CSV: Path = OUTPUT_DIR / "dashboard_data.csv"

    # ==================================================================================
    # Saved preprocessing/model config paths
    # ==================================================================================

    SCALER_PATH: Path = MODEL_DIR / "feature_scaler.pkl"

    KMEANS_MODEL_PATH: Path = CONTEXT_MODEL_DIR / "kmeans_context.pkl"
    GMM_MODEL_PATH: Path = CONTEXT_MODEL_DIR / "gmm_context.pkl"

    RESIDUAL_THRESHOLDS_PATH: Path = ANOMALY_MODEL_DIR / "residual_thresholds.json"
    IFOREST_MODEL_PATH: Path = ANOMALY_MODEL_DIR / "isolation_forest.pkl"
    MAHALANOBIS_PARAMS_PATH: Path = ANOMALY_MODEL_DIR / "mahalanobis_params.pkl"
    FUSION_WEIGHTS_PATH: Path = ANOMALY_MODEL_DIR / "fusion_weights.json"

    HEALTH_INDEX_CONFIG_PATH: Path = HEALTH_MODEL_DIR / "health_index_config.json"
    HEALTH_STATE_THRESHOLDS_PATH: Path = HEALTH_MODEL_DIR / "health_state_thresholds.json"

    CONFIDENCE_CONFIG_PATH: Path = UNCERTAINTY_MODEL_DIR / "confidence_config.json"
    ADAPTIVE_THRESHOLDS_PATH: Path = FEEDBACK_MODEL_DIR / "adaptive_thresholds.json"

    # ==================================================================================
    # N-CMAPSS HDF5 groups
    # ==================================================================================

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

    # ==================================================================================
    # Split names
    # ==================================================================================

    DEV_SPLIT_NAME: str = "dev"
    TEST_SPLIT_NAME: str = "test"

    # ==================================================================================
    # Global constants
    # ==================================================================================

    RANDOM_SEED: int = 42
    CONTEXT_CLUSTER_COUNT: int = 6
    GMM_COVARIANCE_TYPE: str = "full"
    ROLLING_WINDOW: int = 5

    API_TITLE: str = "CA-EDT-AHMA API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # ==================================================================================
    # Context model parameters
    # ==================================================================================

    KMEANS_PARAMS: Dict[str, int] = {
        "n_init": 10,
        "max_iter": 300,
    }

    GMM_PARAMS: Dict[str, int | str] = {
        "max_iter": 300,
        "init_params": "kmeans",
    }

    # ==================================================================================
    # Random Forest full-dev memory-safe training parameters
    # ==================================================================================

    RF_TRAIN_CHUNK_SIZE: int = 50_000
    RF_PREDICTION_BATCH_SIZE: int = 50_000
    RF_TRAIN_N_JOBS: int = 2
    RF_VERBOSE: int = 2
    RF_REBUILD_MEMMAP: bool = True
    RF_CLEANUP_MEMMAP_AFTER_TRAINING: bool = False
    RF_TRAIN_MEMMAP_DIR: Path = DIGITAL_TWIN_MODEL_DIR / "rf_full_dev_memmap"

    RF_PARAMS: Dict[str, int | float | bool | None] = {
        "n_estimators": 120,
        "max_depth": 18,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_SEED,
        "n_jobs": RF_TRAIN_N_JOBS,
    }

    # ==================================================================================
    # XGBoost full-dev memory-safe training parameters
    # ==================================================================================

    XGB_TRAIN_CHUNK_SIZE: int = 50_000
    XGB_PREDICTION_BATCH_SIZE: int = 50_000
    XGB_TRAIN_N_JOBS: int = 2
    XGB_VERBOSITY: int = 1
    XGB_REBUILD_MEMMAP: bool = True
    XGB_CLEANUP_MEMMAP_AFTER_TRAINING: bool = False
    XGB_TRAIN_MEMMAP_DIR: Path = DIGITAL_TWIN_MODEL_DIR / "xgb_full_dev_memmap"

    XGB_PARAMS: Dict[str, int | float | str] = {
        "n_estimators": 180,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
        "n_jobs": XGB_TRAIN_N_JOBS,
        "verbosity": XGB_VERBOSITY,
    }

    # ==================================================================================
    # LightGBM full-dev memory-safe training parameters
    # ==================================================================================

    LGBM_TRAIN_CHUNK_SIZE: int = 50_000
    LGBM_PREDICTION_BATCH_SIZE: int = 50_000
    LGBM_TRAIN_N_JOBS: int = 2
    LGBM_VERBOSITY: int = -1
    LGBM_REBUILD_MEMMAP: bool = True
    LGBM_CLEANUP_MEMMAP_AFTER_TRAINING: bool = False
    LGBM_TRAIN_MEMMAP_DIR: Path = DIGITAL_TWIN_MODEL_DIR / "lgbm_full_dev_memmap"

    LGBM_PARAMS: Dict[str, int | float | str] = {
        "n_estimators": 180,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_SEED,
        "n_jobs": LGBM_TRAIN_N_JOBS,
        "verbose": LGBM_VERBOSITY,
        "verbosity": LGBM_VERBOSITY,
    }

    # ==================================================================================
    # MLP Digital Twin hyperparameters
    # ==================================================================================

    MLP_TWIN_TRAIN_CHUNK_SIZE: int = 50_000
    MLP_TWIN_PREDICTION_BATCH_SIZE: int = 50_000
    MLP_TWIN_BATCH_SIZE: int = 4096
    MLP_TWIN_EPOCHS: int = 20
    MLP_TWIN_LEARNING_RATE: float = 0.001
    MLP_TWIN_VALIDATION_FRACTION: float = 0.10
    MLP_TWIN_RANDOM_SEED: int = RANDOM_SEED

    MLP_TWIN_HIDDEN_UNITS: Tuple[int, ...] = (
        256,
        256,
        128,
        64,
    )

    MLP_TWIN_DROPOUT_RATE: float = 0.10
    MLP_TWIN_USE_BATCH_NORM: bool = True
    MLP_TWIN_EARLY_STOPPING_PATIENCE: int = 4
    MLP_TWIN_REDUCE_LR_PATIENCE: int = 2
    MLP_TWIN_MIN_LEARNING_RATE: float = 1e-6

    # Backward-compatible aliases for current tensorflow_twin.py
    TF_TRAIN_CHUNK_SIZE: int = MLP_TWIN_TRAIN_CHUNK_SIZE
    TF_PREDICTION_BATCH_SIZE: int = MLP_TWIN_PREDICTION_BATCH_SIZE
    TF_BATCH_SIZE: int = MLP_TWIN_BATCH_SIZE
    TF_EPOCHS: int = MLP_TWIN_EPOCHS
    TF_LEARNING_RATE: float = MLP_TWIN_LEARNING_RATE
    TF_VALIDATION_FRACTION: float = MLP_TWIN_VALIDATION_FRACTION
    TF_RANDOM_SEED: int = MLP_TWIN_RANDOM_SEED

    # ==================================================================================
    # Residual Autoencoder anomaly detector hyperparameters
    # ==================================================================================

    RESIDUAL_AUTOENCODER_TRAIN_CHUNK_SIZE: int = 50_000
    RESIDUAL_AUTOENCODER_PREDICTION_BATCH_SIZE: int = 50_000
    RESIDUAL_AUTOENCODER_BATCH_SIZE: int = 4096
    RESIDUAL_AUTOENCODER_EPOCHS: int = 20
    RESIDUAL_AUTOENCODER_LEARNING_RATE: float = 0.001
    RESIDUAL_AUTOENCODER_VALIDATION_FRACTION: float = 0.10

    RESIDUAL_AUTOENCODER_ENCODER_UNITS: Tuple[int, ...] = (
        128,
        64,
        32,
    )

    RESIDUAL_AUTOENCODER_LATENT_DIM: int = 16
    RESIDUAL_AUTOENCODER_DROPOUT_RATE: float = 0.10
    RESIDUAL_AUTOENCODER_THRESHOLD_PERCENTILE: float = 99.0
    RESIDUAL_AUTOENCODER_EARLY_STOPPING_PATIENCE: int = 4
    RESIDUAL_AUTOENCODER_REDUCE_LR_PATIENCE: int = 2
    RESIDUAL_AUTOENCODER_MIN_LEARNING_RATE: float = 1e-6

    # Backward-compatible aliases for TF anomaly code
    TF_ANOMALY_TRAIN_CHUNK_SIZE: int = RESIDUAL_AUTOENCODER_TRAIN_CHUNK_SIZE
    TF_ANOMALY_PREDICTION_BATCH_SIZE: int = RESIDUAL_AUTOENCODER_PREDICTION_BATCH_SIZE
    TF_ANOMALY_BATCH_SIZE: int = RESIDUAL_AUTOENCODER_BATCH_SIZE
    TF_ANOMALY_EPOCHS: int = RESIDUAL_AUTOENCODER_EPOCHS
    TF_ANOMALY_LEARNING_RATE: float = RESIDUAL_AUTOENCODER_LEARNING_RATE
    TF_ANOMALY_VALIDATION_FRACTION: float = RESIDUAL_AUTOENCODER_VALIDATION_FRACTION
    TF_ANOMALY_THRESHOLD_PERCENTILE: float = RESIDUAL_AUTOENCODER_THRESHOLD_PERCENTILE

    # ==================================================================================
    # Comparator/evaluation parameters
    # ==================================================================================

    TWIN_COMPARATOR_CHUNK_SIZE: int = 25_000

    # ==================================================================================
    # Residual/anomaly/health thresholds and weights
    # ==================================================================================

    RESIDUAL_PERCENTILES: Dict[str, float] = {
        "watch": 90.0,
        "warning": 95.0,
        "critical": 99.0,
    }

    # TensorFlow/Keras residual-autoencoder fusion weights
    FUSION_WEIGHTS: Dict[str, float] = {
        "residual": 0.40,
        "residual_autoencoder": 0.60,
    }

    # Kept for backup if classical anomaly models are still used
    CLASSICAL_FUSION_WEIGHTS: Dict[str, float] = {
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

    # ==================================================================================
    # Dashboard output columns
    # ==================================================================================

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

    # ==================================================================================
    # Directory creation
    # ==================================================================================

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
            cls.RF_TRAIN_MEMMAP_DIR,
            cls.XGB_TRAIN_MEMMAP_DIR,
            cls.LGBM_TRAIN_MEMMAP_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    Config.create_directories()

    print(f"{Config.PROJECT_NAME} directory structure initialized at: {Config.BASE_DIR}")

    print(f"RF train n_jobs: {Config.RF_TRAIN_N_JOBS}")
    print(f"RF memmap directory: {Config.RF_TRAIN_MEMMAP_DIR}")
    print(f"RF model path: {Config.RF_MODEL_PATH}")
    print(f"RF metadata path: {Config.RF_MODEL_METADATA_PATH}")
    print(f"RF predictions: {Config.RF_PREDICTIONS_CSV}")

    print(f"XGB train n_jobs: {Config.XGB_TRAIN_N_JOBS}")
    print(f"XGB memmap directory: {Config.XGB_TRAIN_MEMMAP_DIR}")
    print(f"XGB model path: {Config.XGB_MODEL_PATH}")
    print(f"XGB metadata path: {Config.XGB_MODEL_METADATA_PATH}")
    print(f"XGB predictions: {Config.XGB_PREDICTIONS_CSV}")

    print(f"LGBM train n_jobs: {Config.LGBM_TRAIN_N_JOBS}")
    print(f"LGBM memmap directory: {Config.LGBM_TRAIN_MEMMAP_DIR}")
    print(f"LGBM model path: {Config.LGBM_MODEL_PATH}")
    print(f"LGBM metadata path: {Config.LGBM_MODEL_METADATA_PATH}")
    print(f"LGBM predictions: {Config.LGBM_PREDICTIONS_CSV}")

    print(f"MLP model path: {Config.MLP_TWIN_MODEL_PATH}")
    print(f"MLP metadata path: {Config.MLP_TWIN_METADATA_PATH}")
    print(f"MLP predictions: {Config.MLP_TWIN_PREDICTIONS_CSV}")
    print(f"MLP prediction prefix: {Config.MLP_TWIN_PREDICTION_PREFIX}")

    print(f"Ensemble weights path: {Config.ENSEMBLE_WEIGHTS_PATH}")
    print(f"Ensemble metadata path: {Config.ENSEMBLE_METADATA_PATH}")
    print(f"Ensemble predictions: {Config.ENSEMBLE_PREDICTIONS_CSV}")

    print(f"Active digital twin model: {Config.ACTIVE_DIGITAL_TWIN_MODEL_NAME}")
    print(f"Active digital twin library: {Config.ACTIVE_DIGITAL_TWIN_LIBRARY}")
    print(f"Active digital twin predictions: {Config.ACTIVE_DIGITAL_TWIN_PREDICTIONS_CSV}")
    print(f"Active digital twin prefix: {Config.ACTIVE_DIGITAL_TWIN_PREDICTION_PREFIX}")

    print(f"Active anomaly model: {Config.ACTIVE_ANOMALY_MODEL_NAME}")
    print(f"Active anomaly library: {Config.ACTIVE_ANOMALY_LIBRARY}")
    print(f"Active anomaly scores: {Config.ACTIVE_ANOMALY_SCORE_CSV}")

    print(f"Twin comparator chunk size: {Config.TWIN_COMPARATOR_CHUNK_SIZE}")