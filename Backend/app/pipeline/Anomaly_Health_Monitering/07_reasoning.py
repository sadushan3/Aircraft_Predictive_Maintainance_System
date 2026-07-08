"""
Reasoning pipeline for CA-EDT-AHMA.

Stages:
1. Sensor dependency graph
2. Root-cause analysis
3. Root-cause tracking
4. Temporal reasoning

Important:
This stage gives inspection focus only.
It does not make final maintenance decisions.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/07_reasoning.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.reasoning.root_cause_analyzer import RootCauseAnalyzer
from app.services.Anomaly_Health_Monitering.reasoning.root_cause_tracker import RootCauseTracker
from app.services.Anomaly_Health_Monitering.reasoning.sensor_dependency_graph import SensorDependencyGraph
from app.services.Anomaly_Health_Monitering.reasoning.temporal_reasoning import TemporalReasoning
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class ReasoningPipeline:
    """
    Complete reasoning pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize reasoning pipeline.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/07_reasoning.py::__init__")
        Config.create_directories()

    def run(self) -> Dict[str, object]:
        """
        Run reasoning pipeline safely.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/07_reasoning.py::run")
        try:
            stages = [
                ("sensor_dependency_graph", SensorDependencyGraph().run),
                ("root_cause_analysis", RootCauseAnalyzer().run),
                ("root_cause_tracking", RootCauseTracker().run),
                ("temporal_reasoning", TemporalReasoning().run),
            ]

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                result: StageResult = run_stage_safely(stage_name, stage_function)
                completed.append(result.__dict__)

                if result.status == "failed":
                    failed.append(result.__dict__)
                    break

            status = "success" if not failed else "partial_failure"

            summary = {
                "status": status,
                "message": (
                    "Reasoning pipeline completed."
                    if status == "success"
                    else "Reasoning stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": str(Config.ROOT_CAUSE_CSV) if Config.ROOT_CAUSE_CSV.exists() else None,
                "maintenance_decision_note": (
                    "This component provides inspection focus only. "
                    "Maintenance scheduling decisions belong to another component."
                ),
            }

            atomic_write_json(summary, Config.REPORT_DIR / "07_reasoning_summary.json")
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
    print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/07_reasoning.py::run_reasoning_pipeline")
    pipeline = ReasoningPipeline()
    return pipeline.run()


if __name__ == "__main__":
    result = run_reasoning_pipeline()
    print(result)