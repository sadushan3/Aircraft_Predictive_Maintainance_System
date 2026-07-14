"""
Health monitoring pipeline for CA-EDT-AHMA.

Stages:
1. Health index calculation
2. Health state classification
3. Health trend tracking
4. Health alert generation

Important:
- This stage does not predict RUL.
- This stage does not use Y_dev/Y_test.
- This stage does not make maintenance scheduling decisions.
- Failed execution must not delete previous outputs.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "06_health_monitoring.py"
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
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


class HealthMonitoringPipeline:
    """
    Complete health monitoring pipeline.

    This wrapper delegates memory-heavy health processing to the individual
    health monitoring services. Each service should remain memory-safe
    independently.
    """

    def __init__(self) -> None:
        """
        Initialize health monitoring pipeline.
        """
        print("[PROGRESS] Entering HealthMonitoringPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "06_health_monitoring_summary.json"

        self.health_trends_csv = Config.OUTPUT_DIR / "health_trends.csv"
        self.health_alerts_csv = Config.OUTPUT_DIR / "health_alerts.csv"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Anomaly fusion CSV: {Config.ANOMALY_FUSION_CSV}")
        print(f"[PROGRESS] Health index CSV: {Config.HEALTH_INDEX_CSV}")
        print(f"[PROGRESS] Health states CSV: {Config.HEALTH_STATES_CSV}")
        print(f"[PROGRESS] Health trends CSV: {self.health_trends_csv}")
        print(f"[PROGRESS] Health alerts CSV: {self.health_alerts_csv}")

    # ==================================================================================
    # Stage wrappers
    # ==================================================================================

    def _run_health_index_calculation(self) -> Dict[str, object]:
        """
        Run health index calculation.

        Expected service responsibility:
        - Read anomaly_fusion.csv.
        - Calculate health_index and remaining_health_percentage.
        - Write health_index.csv.
        - Avoid Y_dev/Y_test and RUL targets.
        """
        print("[PROGRESS] Starting stage wrapper: health_index_calculation")
        return HealthIndexCalculator().run()

    def _run_health_state_classification(self) -> Dict[str, object]:
        """
        Run health state classification.

        Expected service responsibility:
        - Read health_index.csv.
        - Classify Healthy / Degrading / Warning / Critical.
        - Write health_states.csv.
        """
        print("[PROGRESS] Starting stage wrapper: health_state_classification")
        return HealthStateClassifier().run()

    def _run_health_trend_tracking(self) -> Dict[str, object]:
        """
        Run health trend tracking.

        Expected service responsibility:
        - Read health_states.csv.
        - Calculate rolling health trend and deterioration behavior.
        - Write health_trends.csv.
        """
        print("[PROGRESS] Starting stage wrapper: health_trend_tracking")
        return HealthTrendTracker().run()

    def _run_health_alert_generation(self) -> Dict[str, object]:
        """
        Run health alert generation.

        Expected service responsibility:
        - Read health_states.csv.
        - Generate dashboard-ready health alert messages.
        - Write health_alerts.csv.
        - Provide inspection focus only, not maintenance scheduling decisions.
        """
        print("[PROGRESS] Starting stage wrapper: health_alert_generation")
        return HealthAlertEngine().run()

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run health monitoring safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering HealthMonitoringPipeline.run")

        try:
            stages: List[tuple[str, Callable[[], Dict[str, object]]]] = [
                ("health_index_calculation", self._run_health_index_calculation),
                ("health_state_classification", self._run_health_state_classification),
                ("health_trend_tracking", self._run_health_trend_tracking),
                ("health_alert_generation", self._run_health_alert_generation),
            ]

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                print("=" * 100)
                print(f"[PROGRESS] Running health monitoring stage: {stage_name}")

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    print(
                        "[PROGRESS] Health monitoring stopped safely after failed stage. "
                        "Previous outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"

            final_outputs = {
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
                "health_trends_csv": (
                    str(self.health_trends_csv)
                    if self.health_trends_csv.exists()
                    else None
                ),
                "health_alerts_csv": (
                    str(self.health_alerts_csv)
                    if self.health_alerts_csv.exists()
                    else None
                ),
                "health_index_config": (
                    str(Config.HEALTH_INDEX_CONFIG_PATH)
                    if Config.HEALTH_INDEX_CONFIG_PATH.exists()
                    else None
                ),
                "health_state_thresholds": (
                    str(Config.HEALTH_STATE_THRESHOLDS_PATH)
                    if Config.HEALTH_STATE_THRESHOLDS_PATH.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Health monitoring pipeline completed."
                    if status == "success"
                    else "Health monitoring stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": final_outputs["health_states_csv"],
                "final_outputs": final_outputs,
                "pipeline_order": [
                    "health_index_calculation",
                    "health_state_classification",
                    "health_trend_tracking",
                    "health_alert_generation",
                ],
                "health_scope": {
                    "health_index_output": "health_index",
                    "state_output": "health_state",
                    "trend_output": "health_trend_label",
                    "alert_output": "inspection_priority / health_alert_message",
                    "health_states": ["Healthy", "Degrading", "Warning", "Critical"],
                },
                "input_scope": {
                    "primary_input": "anomaly_fusion.csv",
                    "uses_final_anomaly_score": True,
                    "uses_alert_level": True,
                    "uses_anomaly_persistence": True,
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Health monitoring converts anomaly evidence into health index, "
                        "health state, trend, and alert intelligence. It does not use "
                        "Y_dev/Y_test because those are RUL targets."
                    ),
                },
                "decision_boundary": {
                    "makes_maintenance_scheduling_decisions": False,
                    "allowed_outputs": [
                        "health index",
                        "health state",
                        "health trend",
                        "inspection priority",
                        "health alert message",
                    ],
                    "note": (
                        "This component provides health intelligence and inspection focus only. "
                        "Final maintenance scheduling belongs to the autonomous maintenance supervisor."
                    ),
                },
                "leakage_audit": {
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_rul_targets": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "previous_outputs_deleted_on_failure": False,
                },
            }

            print(f"[PROGRESS] Writing health monitoring summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info("Health monitoring pipeline finished with status=%s.", status)

            return summary

        except Exception as exc:
            logger.exception("Health monitoring pipeline failed.")
            raise RuntimeError("Health monitoring pipeline failed.") from exc


def run_health_monitoring_pipeline() -> Dict[str, object]:
    """
    Execute health monitoring pipeline.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering run_health_monitoring_pipeline")

    pipeline = HealthMonitoringPipeline()
    return pipeline.run()


if __name__ == "__main__":
    print("[PROGRESS] 06_health_monitoring.py execution started")
    result = run_health_monitoring_pipeline()
    print("[PROGRESS] 06_health_monitoring.py execution finished")
    print(result)