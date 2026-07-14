"""
Residual analysis pipeline for CA-EDT-AHMA.

Stage:
1. Calculate actual X_s minus ensemble-predicted X_s

Output:
outputs/Anomaly_Health_Monitering/residuals.csv

Important:
- Residuals are calculated from measured X_s and digital twin predicted X_s.
- This pipeline does not use Y_dev/Y_test.
- This pipeline does not predict RUL.
- This pipeline does not make maintenance decisions.
- Failed execution must not delete previous outputs.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "04_residual_analysis.py"
)

from typing import Dict, List
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
from app.services.Anomaly_Health_Monitering.digital_twin.residual_calculator import (
    ResidualCalculator,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


class ResidualAnalysisPipeline:
    """
    Residual analysis pipeline.

    This wrapper delegates the memory-heavy residual calculation to
    ResidualCalculator. ResidualCalculator itself must remain chunk-safe.
    """

    def __init__(self) -> None:
        """
        Initialize residual analysis pipeline.
        """
        print("[PROGRESS] Entering ResidualAnalysisPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "04_residual_analysis_summary.json"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Scaled CSV: {Config.SCALED_CSV}")
        print(f"[PROGRESS] Ensemble predictions CSV: {Config.ENSEMBLE_PREDICTIONS_CSV}")
        print(f"[PROGRESS] Residuals CSV: {Config.RESIDUALS_CSV}")

    # ==================================================================================
    # Stage wrapper
    # ==================================================================================

    def _run_residual_calculation(self) -> Dict[str, object]:
        """
        Run residual calculation.

        Expected service responsibility:
        - Read actual measured X_s values.
        - Read ensemble-predicted X_s values.
        - Calculate residual = actual - predicted.
        - Write residuals.csv safely.
        - Avoid using Y_dev/Y_test.
        """
        print("[PROGRESS] Starting stage wrapper: residual_calculation")
        return ResidualCalculator().run()

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run residual analysis safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering ResidualAnalysisPipeline.run")

        try:
            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            print("=" * 100)
            print("[PROGRESS] Running residual analysis stage: residual_calculation")

            result: StageResult = run_stage_safely(
                "residual_calculation",
                self._run_residual_calculation,
            )

            result_dict = result.__dict__
            completed.append(result_dict)

            print(f"[PROGRESS] Stage result: {result_dict}")

            if result.status == "failed":
                failed.append(result_dict)
                print(
                    "[PROGRESS] Residual analysis failed safely. "
                    "Previous outputs were not deleted."
                )

            status = "success" if not failed else "partial_failure"

            final_outputs = {
                "residuals_csv": (
                    str(Config.RESIDUALS_CSV)
                    if Config.RESIDUALS_CSV.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Residual analysis completed."
                    if status == "success"
                    else "Residual analysis failed safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": final_outputs["residuals_csv"],
                "final_outputs": final_outputs,
                "residual_scope": {
                    "formula": "residual = actual_X_s - predicted_X_s",
                    "prediction_source": "ensemble digital twin predictions",
                    "target_type": "raw measured X_s sensor values",
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Residual analysis compares measured sensor values with "
                        "digital twin sensor predictions. Y_dev/Y_test are RUL "
                        "targets and are intentionally ignored."
                    ),
                },
                "leakage_audit": {
                    "does_not_use_y_dev_y_test": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "test_split_usage": "residual_scoring_only",
                    "previous_outputs_deleted_on_failure": False,
                },
            }

            print(f"[PROGRESS] Writing residual analysis summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info("Residual analysis pipeline finished with status=%s.", status)

            return summary

        except Exception as exc:
            logger.exception("Residual analysis pipeline failed.")
            raise RuntimeError("Residual analysis pipeline failed.") from exc


def run_residual_analysis_pipeline() -> Dict[str, object]:
    """
    Execute residual analysis pipeline.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering run_residual_analysis_pipeline")

    pipeline = ResidualAnalysisPipeline()
    return pipeline.run()


if __name__ == "__main__":
    print("[PROGRESS] 04_residual_analysis.py execution started")
    result = run_residual_analysis_pipeline()
    print("[PROGRESS] 04_residual_analysis.py execution finished")
    print(result)