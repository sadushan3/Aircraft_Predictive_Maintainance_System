"""
Reasoning pipeline for CA-EDT-AHMA.

Stages:
1. Sensor dependency graph
2. Root-cause analysis
3. Root-cause tracking
4. Temporal reasoning

Important:
- This stage gives explanation and inspection focus only.
- This stage does not make final maintenance scheduling decisions.
- This stage does not predict RUL.
- This stage does not use Y_dev/Y_test.
- Failed execution must not delete previous outputs.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/"
    "07_reasoning.py"
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
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely


logger = get_logger(__name__)


class ReasoningPipeline:
    """
    Complete reasoning pipeline.

    This wrapper delegates memory-heavy reasoning work to the individual
    reasoning services. Each service should remain memory-safe independently.
    """

    def __init__(self) -> None:
        """
        Initialize reasoning pipeline.
        """
        print("[PROGRESS] Entering ReasoningPipeline.__init__")

        Config.create_directories()

        self.summary_json = Config.REPORT_DIR / "07_reasoning_summary.json"

        self.sensor_dependency_graph_csv = getattr(
            Config,
            "SENSOR_DEPENDENCY_GRAPH_CSV",
            Config.OUTPUT_DIR / "sensor_dependency_graph.csv",
        )
        self.root_cause_memory_csv = Config.OUTPUT_DIR / "root_cause_memory.csv"
        self.temporal_reasoning_csv = Config.OUTPUT_DIR / "temporal_reasoning.csv"

        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Residuals CSV: {Config.RESIDUALS_CSV}")
        print(f"[PROGRESS] Health states CSV: {Config.HEALTH_STATES_CSV}")
        print(f"[PROGRESS] Root-cause CSV: {Config.ROOT_CAUSE_CSV}")
        print(f"[PROGRESS] Sensor dependency graph CSV: {self.sensor_dependency_graph_csv}")
        print(f"[PROGRESS] Root-cause memory CSV: {self.root_cause_memory_csv}")
        print(f"[PROGRESS] Temporal reasoning CSV: {self.temporal_reasoning_csv}")

    # ==================================================================================
    # Stage wrappers
    # ==================================================================================

    def _run_sensor_dependency_graph(self) -> Dict[str, object]:
        """
        Run sensor dependency graph stage.

        Expected service responsibility:
        - Build conservative sensor-family dependency graph.
        - Use residual header/family information only where possible.
        - Avoid hard causal claims.
        """
        print("[PROGRESS] Starting stage wrapper: sensor_dependency_graph")
        return SensorDependencyGraph().run()

    def _run_root_cause_analysis(self) -> Dict[str, object]:
        """
        Run root-cause analysis.

        Expected service responsibility:
        - Rank top contributing measured sensors using residual contribution.
        - Write root_cause_analysis.csv.
        - Provide likely pattern / inspection focus only.
        - Avoid final maintenance scheduling decisions.
        """
        print("[PROGRESS] Starting stage wrapper: root_cause_analysis")
        return RootCauseAnalyzer().run()

    def _run_root_cause_tracking(self) -> Dict[str, object]:
        """
        Run root-cause recurrence tracking.

        Expected service responsibility:
        - Track repeated root-cause patterns over recent cycles.
        - Use memory-safe/chunk-safe logic.
        - Write root_cause_memory.csv.
        """
        print("[PROGRESS] Starting stage wrapper: root_cause_tracking")
        return RootCauseTracker().run()

    def _run_temporal_reasoning(self) -> Dict[str, object]:
        """
        Run temporal reasoning.

        Expected service responsibility:
        - Analyze anomaly and health evolution over time.
        - Write temporal_reasoning.csv.
        - Provide temporal explanation only.
        """
        print("[PROGRESS] Starting stage wrapper: temporal_reasoning")
        return TemporalReasoning().run()

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run reasoning pipeline safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering ReasoningPipeline.run")

        try:
            stages: List[tuple[str, Callable[[], Dict[str, object]]]] = [
                ("sensor_dependency_graph", self._run_sensor_dependency_graph),
                ("root_cause_analysis", self._run_root_cause_analysis),
                ("root_cause_tracking", self._run_root_cause_tracking),
                ("temporal_reasoning", self._run_temporal_reasoning),
            ]

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                print("=" * 100)
                print(f"[PROGRESS] Running reasoning stage: {stage_name}")

                result: StageResult = run_stage_safely(stage_name, stage_function)
                result_dict = result.__dict__

                completed.append(result_dict)

                print(f"[PROGRESS] Stage result: {result_dict}")

                if result.status == "failed":
                    failed.append(result_dict)
                    print(
                        "[PROGRESS] Reasoning stopped safely after failed stage. "
                        "Previous outputs were not deleted."
                    )
                    break

            status = "success" if not failed else "partial_failure"

            final_outputs = {
                "sensor_dependency_graph_csv": (
                    str(self.sensor_dependency_graph_csv)
                    if self.sensor_dependency_graph_csv.exists()
                    else None
                ),
                "root_cause_csv": (
                    str(Config.ROOT_CAUSE_CSV)
                    if Config.ROOT_CAUSE_CSV.exists()
                    else None
                ),
                "root_cause_memory_csv": (
                    str(self.root_cause_memory_csv)
                    if self.root_cause_memory_csv.exists()
                    else None
                ),
                "temporal_reasoning_csv": (
                    str(self.temporal_reasoning_csv)
                    if self.temporal_reasoning_csv.exists()
                    else None
                ),
            }

            summary = {
                "status": status,
                "message": (
                    "Reasoning pipeline completed."
                    if status == "success"
                    else "Reasoning stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": final_outputs["root_cause_csv"],
                "final_outputs": final_outputs,
                "pipeline_order": [
                    "sensor_dependency_graph",
                    "root_cause_analysis",
                    "root_cause_tracking",
                    "temporal_reasoning",
                ],
                "reasoning_scope": {
                    "sensor_dependency_graph": (
                        "Conservative sensor-family dependency graph for explanation support."
                    ),
                    "root_cause_analysis": (
                        "Residual-based top sensor contribution and pattern labeling."
                    ),
                    "root_cause_tracking": (
                        "Recurring root-cause pattern tracking over recent cycles."
                    ),
                    "temporal_reasoning": (
                        "Temporal anomaly/health behavior explanation."
                    ),
                },
                "allowed_outputs": [
                    "top contributing sensors",
                    "sensor contribution scores",
                    "root-cause pattern label",
                    "inspection focus",
                    "recurrence status",
                    "temporal pattern",
                    "temporal reasoning text",
                ],
                "decision_boundary": {
                    "makes_maintenance_scheduling_decisions": False,
                    "hard_physical_causality_claim": False,
                    "note": (
                        "This component provides explanation, anomaly reasoning, and "
                        "inspection focus only. Final maintenance scheduling decisions "
                        "belong to the autonomous maintenance supervisor component."
                    ),
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_rul_targets": False,
                    "predicts_rul": False,
                    "note": (
                        "Reasoning is based on residuals, anomaly/health outputs, and "
                        "top contributing sensors. Y_dev/Y_test are RUL targets and are "
                        "intentionally ignored."
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

            print(f"[PROGRESS] Writing reasoning summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info("Reasoning pipeline finished with status=%s.", status)

            return summary

        except Exception as exc:
            logger.exception("Reasoning pipeline failed.")
            raise RuntimeError("Reasoning pipeline failed.") from exc


def run_reasoning_pipeline() -> Dict[str, object]:
    """
    Execute reasoning pipeline.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering run_reasoning_pipeline")

    pipeline = ReasoningPipeline()
    return pipeline.run()


if __name__ == "__main__":
    print("[PROGRESS] 07_reasoning.py execution started")
    result = run_reasoning_pipeline()
    print("[PROGRESS] 07_reasoning.py execution finished")
    print(result)