"""
Digital twin training and inference pipeline for CA-EDT-AHMA.

Stages:
1. Random Forest twin
2. XGBoost twin
3. LightGBM twin
4. Ensemble digital twin
5. Twin comparison metrics

Important:
- RF, XGBoost, and LightGBM are trained only on dev split.
- Test split is used only for inference/evaluation.
- This pipeline predicts measured sensor behavior, not RUL.
- Y_dev/Y_test are ignored because they are RUL targets.
- This pipeline does not make maintenance scheduling decisions.
- Failed execution must not delete previous outputs.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "03_digital_twin.py"
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
from app.services.Anomaly_Health_Monitering.digital_twin.ensemble_twin import (
    EnsembleDigitalTwin,
)
from app.services.Anomaly_Health_Monitering.digital_twin.lightgbm_twin import (
    LightGBMTwin,
)
from app.services.Anomaly_Health_Monitering.digital_twin.random_forest_twin import (
    RandomForestTwin,
)
from app.services.Anomaly_Health_Monitering.digital_twin.twin_comparator import (
    TwinComparator,
)
from app.services.Anomaly_Health_Monitering.digital_twin.xgboost_twin import (
    XGBoostTwin,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


class DigitalTwinPipeline:
    """
    Complete digital twin pipeline.

    This wrapper delegates memory-heavy model training and inference to the
    individual digital twin services. Each service should remain memory-safe
    independently.
    """

    def __init__(self) -> None:
        """
        Initialize digital twin pipeline.
        """
        print("[PROGRESS] Entering DigitalTwinPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "03_digital_twin_summary.json"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Scaled CSV: {Config.SCALED_CSV}")
        print(f"[PROGRESS] Context CSV: {Config.CONTEXT_CSV}")
        print(f"[PROGRESS] RF predictions CSV: {Config.RF_PREDICTIONS_CSV}")
        print(f"[PROGRESS] XGB predictions CSV: {Config.XGB_PREDICTIONS_CSV}")
        print(f"[PROGRESS] LGBM predictions CSV: {Config.LGBM_PREDICTIONS_CSV}")
        print(f"[PROGRESS] Ensemble predictions CSV: {Config.ENSEMBLE_PREDICTIONS_CSV}")
        print(f"[PROGRESS] Dev split name: {Config.DEV_SPLIT_NAME}")
        print(f"[PROGRESS] Test split name: {Config.TEST_SPLIT_NAME}")

    # ==================================================================================
    # Stage wrappers
    # ==================================================================================

    def _run_random_forest_twin(self) -> Dict[str, object]:
        """
        Run Random Forest digital twin.

        Expected service responsibility:
        - Fit model using dev split only.
        - Predict dev/test measured sensor values.
        - Save RF model and prediction CSV.
        """
        print("[PROGRESS] Starting stage wrapper: random_forest_twin")
        return RandomForestTwin().run()

    def _run_xgboost_twin(self) -> Dict[str, object]:
        """
        Run XGBoost digital twin.

        Expected service responsibility:
        - Fit model using dev split only.
        - Predict dev/test measured sensor values.
        - Save XGBoost model and prediction CSV.
        """
        print("[PROGRESS] Starting stage wrapper: xgboost_twin")
        return XGBoostTwin().run()

    def _run_lightgbm_twin(self) -> Dict[str, object]:
        """
        Run LightGBM digital twin.

        Expected service responsibility:
        - Fit model using dev split only.
        - Predict dev/test measured sensor values.
        - Save LightGBM model and prediction CSV.
        """
        print("[PROGRESS] Starting stage wrapper: lightgbm_twin")
        return LightGBMTwin().run()

    def _run_ensemble_twin(self) -> Dict[str, object]:
        """
        Run ensemble digital twin.

        Expected service responsibility:
        - Combine RF, XGBoost, and LightGBM predictions.
        - Save ensemble prediction CSV.
        - Not retrain base models unless explicitly designed to do so.
        """
        print("[PROGRESS] Starting stage wrapper: ensemble_twin")
        return EnsembleDigitalTwin().run()

    def _run_twin_comparison(self) -> Dict[str, object]:
        """
        Run digital twin comparison metrics.

        Expected service responsibility:
        - Compare model predictions and residual behavior.
        - Use test split only for evaluation/reporting, not fitting.
        - Avoid using Y_dev/Y_test.
        """
        print("[PROGRESS] Starting stage wrapper: twin_comparison")
        return TwinComparator().run()

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self, include_comparison: bool = True) -> Dict[str, object]:
        """
        Run digital twin pipeline safely.

        Args:
            include_comparison: Whether to run twin comparison metrics.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering DigitalTwinPipeline.run")
        print(f"[PROGRESS] include_comparison={include_comparison}")

        try:
            stages: List[tuple[str, Callable[[], Dict[str, object]]]] = [
                ("random_forest_twin", self._run_random_forest_twin),
                ("xgboost_twin", self._run_xgboost_twin),
                ("lightgbm_twin", self._run_lightgbm_twin),
                ("ensemble_twin", self._run_ensemble_twin),
            ]

            if include_comparison:
                stages.append(("twin_comparison", self._run_twin_comparison))

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                print("=" * 100)
                print(f"[PROGRESS] Running digital twin stage: {stage_name}")

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    print(
                        "[PROGRESS] Digital twin pipeline stopped safely after failed stage. "
                        "Previous outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"

            final_outputs = {
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
                "rf_model": (
                    str(Config.RF_MODEL_PATH)
                    if Config.RF_MODEL_PATH.exists()
                    else None
                ),
                "xgb_model": (
                    str(Config.XGB_MODEL_PATH)
                    if Config.XGB_MODEL_PATH.exists()
                    else None
                ),
                "lgbm_model": (
                    str(Config.LGBM_MODEL_PATH)
                    if Config.LGBM_MODEL_PATH.exists()
                    else None
                ),
                "ensemble_weights": (
                    str(Config.ENSEMBLE_WEIGHTS_PATH)
                    if Config.ENSEMBLE_WEIGHTS_PATH.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Digital twin pipeline completed."
                    if status == "success"
                    else "Digital twin pipeline stopped safely. Previous outputs were not deleted."
                ),
                "include_comparison": bool(include_comparison),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": final_outputs["ensemble_predictions_csv"],
                "final_outputs": final_outputs,
                "training_scope": {
                    "fit_split": Config.DEV_SPLIT_NAME,
                    "test_usage": "predict_or_evaluate_only",
                    "input_features": "W operating conditions + X_v virtual sensors + context id",
                    "targets": "raw measured X_s sensor values only",
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Digital twins predict measured sensor behavior for residual-based "
                        "anomaly detection. Y_dev/Y_test are RUL targets and are ignored."
                    ),
                },
                "leakage_audit": {
                    "rf_fit_split": Config.DEV_SPLIT_NAME,
                    "xgb_fit_split": Config.DEV_SPLIT_NAME,
                    "lgbm_fit_split": Config.DEV_SPLIT_NAME,
                    "test_split_usage": "prediction_and_evaluation_only",
                    "does_not_use_y_dev_y_test": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "previous_outputs_deleted_on_failure": False,
                },
            }

            print(f"[PROGRESS] Writing digital twin summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info("Digital twin pipeline finished with status=%s.", status)

            return summary

        except Exception as exc:
            logger.exception("Digital twin pipeline failed.")
            raise RuntimeError("Digital twin pipeline failed.") from exc


def run_digital_twin_pipeline(include_comparison: bool = True) -> Dict[str, object]:
    """
    Execute digital twin pipeline.

    Args:
        include_comparison: Whether to include twin comparison.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering run_digital_twin_pipeline")

    pipeline = DigitalTwinPipeline()
    return pipeline.run(include_comparison=include_comparison)


if __name__ == "__main__":
    print("[PROGRESS] 03_digital_twin.py execution started")
    result = run_digital_twin_pipeline(include_comparison=True)
    print("[PROGRESS] 03_digital_twin.py execution finished")
    print(result)