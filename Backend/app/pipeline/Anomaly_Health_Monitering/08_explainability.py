"""
Explainability pipeline for CA-EDT-AHMA.

Stages:
1. Sensor residual ranking
2. Subsystem explanation
3. Human-readable explanation generation
4. Optional SHAP explanation

Important:
- This stage provides explanation support only.
- This stage does not predict RUL.
- This stage does not use Y_dev/Y_test.
- This stage does not make maintenance scheduling decisions.
- SHAP can be computationally expensive and is disabled by default.
- SHAP must use bounded samples / memory-safe logic inside SHAPExplainer.
- Failed execution must not delete previous outputs.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "08_explainability.py"
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
from app.services.Anomaly_Health_Monitering.explainability.explanation_generator import (
    ExplanationGenerator,
)
from app.services.Anomaly_Health_Monitering.explainability.sensor_residual_ranking import (
    SensorResidualRanking,
)
from app.services.Anomaly_Health_Monitering.explainability.shap_explainer import (
    SHAPExplainer,
)
from app.services.Anomaly_Health_Monitering.explainability.subsystem_explainer import (
    SubsystemExplainer,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


class ExplainabilityPipeline:
    """
    Complete explainability pipeline.

    This wrapper delegates memory-heavy explainability work to individual
    explainability services. Each service should remain memory-safe independently.
    """

    def __init__(self) -> None:
        """
        Initialize explainability pipeline.
        """
        print("[PROGRESS] Entering ExplainabilityPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "08_explainability_summary.json"

        self.sensor_ranking_csv = Config.OUTPUT_DIR / "sensor_residual_ranking.csv"
        self.subsystem_explanations_csv = Config.OUTPUT_DIR / "subsystem_explanations.csv"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Residuals CSV: {Config.RESIDUALS_CSV}")
        print(f"[PROGRESS] Root-cause CSV: {Config.ROOT_CAUSE_CSV}")
        print(f"[PROGRESS] Sensor residual ranking CSV: {self.sensor_ranking_csv}")
        print(f"[PROGRESS] Subsystem explanations CSV: {self.subsystem_explanations_csv}")
        print(f"[PROGRESS] Explanation reports CSV: {Config.EXPLANATION_REPORTS_CSV}")
        print(f"[PROGRESS] SHAP CSV: {Config.SHAP_CSV}")

    # ==================================================================================
    # Stage wrappers
    # ==================================================================================

    def _run_sensor_residual_ranking(self) -> Dict[str, object]:
        """
        Run sensor residual ranking.

        Expected service responsibility:
        - Read residuals.csv.
        - Rank top contributing sensors using absolute residual contribution.
        - Write sensor_residual_ranking.csv.
        - Avoid Y_dev/Y_test and RUL targets.
        """
        print("[PROGRESS] Starting stage wrapper: sensor_residual_ranking")
        return SensorResidualRanking().run()

    def _run_subsystem_explainer(self) -> Dict[str, object]:
        """
        Run subsystem explanation.

        Expected service responsibility:
        - Read root_cause_analysis.csv.
        - Map top sensors to broad subsystem-level explanation labels.
        - Write subsystem_explanations.csv.
        - Avoid hard physical causality claims.
        """
        print("[PROGRESS] Starting stage wrapper: subsystem_explainer")
        return SubsystemExplainer().run()

    def _run_explanation_generator(self) -> Dict[str, object]:
        """
        Run human-readable explanation generation.

        Expected service responsibility:
        - Combine anomaly, health, root-cause, subsystem, and temporal evidence.
        - Write explanation_reports.csv.
        - Provide explanation/inspection focus only.
        - Avoid maintenance scheduling decisions.
        """
        print("[PROGRESS] Starting stage wrapper: explanation_generator")
        return ExplanationGenerator().run()

    def _run_shap_explainer(self) -> Dict[str, object]:
        """
        Run optional SHAP explanation.

        Expected service responsibility:
        - Use bounded samples only.
        - Avoid full-row SHAP over 7.6M rows.
        - Avoid Random Forest full TreeExplainer memory explosion.
        - Use safe Random Forest fallback/permutation explanation if needed.
        - Write shap_explanations.csv.
        """
        print("[PROGRESS] Starting stage wrapper: shap_explainer")
        return SHAPExplainer().run()

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self, include_shap: bool = False) -> Dict[str, object]:
        """
        Run explainability pipeline safely.

        Args:
            include_shap: Whether to calculate SHAP explanations.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering ExplainabilityPipeline.run")
        print(f"[PROGRESS] include_shap={include_shap}")

        try:
            stages: List[tuple[str, Callable[[], Dict[str, object]]]] = [
                ("sensor_residual_ranking", self._run_sensor_residual_ranking),
                ("subsystem_explainer", self._run_subsystem_explainer),
                ("explanation_generator", self._run_explanation_generator),
            ]

            if include_shap:
                stages.append(("shap_explainer", self._run_shap_explainer))

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                print("=" * 100)
                print(f"[PROGRESS] Running explainability stage: {stage_name}")

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    print(
                        "[PROGRESS] Explainability stopped safely after failed stage. "
                        "Previous outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"

            final_outputs = {
                "sensor_residual_ranking_csv": (
                    str(self.sensor_ranking_csv)
                    if self.sensor_ranking_csv.exists()
                    else None
                ),
                "subsystem_explanations_csv": (
                    str(self.subsystem_explanations_csv)
                    if self.subsystem_explanations_csv.exists()
                    else None
                ),
                "explanation_reports_csv": (
                    str(Config.EXPLANATION_REPORTS_CSV)
                    if Config.EXPLANATION_REPORTS_CSV.exists()
                    else None
                ),
                "shap_explanations_csv": (
                    str(Config.SHAP_CSV)
                    if Config.SHAP_CSV.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Explainability pipeline completed."
                    if status == "success"
                    else "Explainability stopped safely. Previous outputs were not deleted."
                ),
                "include_shap": bool(include_shap),
                "shap_enabled": bool(include_shap),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": final_outputs["explanation_reports_csv"],
                "final_outputs": final_outputs,
                "pipeline_order": [
                    "sensor_residual_ranking",
                    "subsystem_explainer",
                    "explanation_generator",
                    "shap_explainer_optional",
                ],
                "explainability_scope": {
                    "sensor_residual_ranking": (
                        "Ranks measured sensors by absolute residual contribution."
                    ),
                    "subsystem_explainer": (
                        "Maps top sensors and patterns to broad subsystem-level explanations."
                    ),
                    "explanation_generator": (
                        "Creates human-readable anomaly, health, and inspection-focus explanations."
                    ),
                    "shap_explainer": (
                        "Optional global model explainability using bounded samples only."
                    ),
                },
                "shap_safety": {
                    "enabled_this_run": bool(include_shap),
                    "should_not_run_full_dataset_row_level_shap": True,
                    "random_forest_full_tree_explainer_allowed": False,
                    "recommended_rf_strategy": (
                        "Use bounded-sample permutation/fallback explanation for Random Forest."
                    ),
                    "recommended_xgb_lgbm_strategy": (
                        "Use bounded-sample Tree SHAP for XGBoost and LightGBM."
                    ),
                },
                "allowed_outputs": [
                    "top residual sensors",
                    "sensor contribution scores",
                    "subsystem labels",
                    "root-cause explanation text",
                    "inspection focus",
                    "global feature importance / SHAP summary",
                ],
                "decision_boundary": {
                    "makes_maintenance_scheduling_decisions": False,
                    "hard_physical_causality_claim": False,
                    "note": (
                        "This component provides explanation and inspection focus only. "
                        "Final maintenance scheduling decisions belong to the autonomous "
                        "maintenance supervisor component."
                    ),
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Explainability is based on residuals, root-cause analysis, "
                        "health/anomaly outputs, and digital twin model behavior. "
                        "Y_dev/Y_test are RUL targets and are intentionally ignored."
                    ),
                },
                "leakage_audit": {
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_rul_targets": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_claim_hard_causality": True,
                    "previous_outputs_deleted_on_failure": False,
                },
            }

            print(f"[PROGRESS] Writing explainability summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info("Explainability pipeline finished with status=%s.", status)

            return summary

        except Exception as exc:
            logger.exception("Explainability pipeline failed.")
            raise RuntimeError("Explainability pipeline failed.") from exc


def run_explainability_pipeline(include_shap: bool = False) -> Dict[str, object]:
    """
    Execute explainability pipeline.

    Args:
        include_shap: Whether to run SHAP.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering run_explainability_pipeline")

    pipeline = ExplainabilityPipeline()
    return pipeline.run(include_shap=include_shap)


if __name__ == "__main__":
    print("[PROGRESS] 08_explainability.py execution started")
    result = run_explainability_pipeline(include_shap=False)
    print("[PROGRESS] 08_explainability.py execution finished")
    print(result)