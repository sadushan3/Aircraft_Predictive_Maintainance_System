"""
Health score engine for CA-EDT-AHMA.

Role:
Run or validate the complete health monitoring stage:
1. Health index calculation
2. Health state classification
3. Health trend tracking
4. Health alert summary

Important:
- This is an orchestrator only.
- It does not train a model.
- It does not predict RUL.
- It does not use Y_dev/Y_test.
- It does not use T_dev/T_test.
- It does not delete existing outputs.
- By default, it skips stages whose expected output already exists.

Reads:
outputs/Anomaly_Health_Monitering/anomaly_fusion.csv

Writes/checks:
outputs/Anomaly_Health_Monitering/health_index.csv
outputs/Anomaly_Health_Monitering/health_states.csv
outputs/Anomaly_Health_Monitering/health_trends.csv
outputs/Anomaly_Health_Monitering/health_alerts.csv
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "health_monitoring/health_score_engine.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, Optional
import os
import sys


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
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


logger = get_logger(__name__)


class HealthScoreEngine:
    """
    Safe full health monitoring orchestrator.

    Default behavior:
    - If output exists and has rows, skip that stage.
    - If output is missing, run that stage.
    - If force_rebuild=True, rerun all stages.
    """

    def __init__(self, force_rebuild: Optional[bool] = None) -> None:
        """
        Initialize health score engine.

        Args:
            force_rebuild:
                If True, rerun all health stages.
                If False, skip existing outputs.
                If None, reads Config.HEALTH_SCORE_FORCE_REBUILD, default False.
        """
        print("[PROGRESS] Entering HealthScoreEngine.__init__")

        Config.create_directories()

        self.force_rebuild = bool(
            force_rebuild
            if force_rebuild is not None
            else getattr(Config, "HEALTH_SCORE_FORCE_REBUILD", False)
        )

        self.health_index_csv: Path = Config.HEALTH_INDEX_CSV
        self.health_states_csv: Path = Config.HEALTH_STATES_CSV

        self.health_trends_csv: Path = getattr(
            Config,
            "HEALTH_TRENDS_CSV",
            Config.OUTPUT_DIR / "health_trends.csv",
        )

        self.health_alerts_csv: Path = getattr(
            Config,
            "HEALTH_ALERTS_CSV",
            Config.OUTPUT_DIR / "health_alerts.csv",
        )

        self.summary_json: Path = getattr(
            Config,
            "HEALTH_SCORE_ENGINE_SUMMARY_JSON",
            Config.REPORT_DIR / "health_score_engine_summary.json",
        )

        print(f"[PROGRESS] Force rebuild: {self.force_rebuild}")
        print(f"[PROGRESS] Health index CSV: {self.health_index_csv}")
        print(f"[PROGRESS] Health states CSV: {self.health_states_csv}")
        print(f"[PROGRESS] Health trends CSV: {self.health_trends_csv}")
        print(f"[PROGRESS] Health alerts CSV: {self.health_alerts_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")

    # ==================================================================================
    # Helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows safely without loading the file.

        Args:
            path: CSV file path.

        Returns:
            Number of data rows excluding header.
        """
        if not path.exists():
            return 0

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        return max(int(row_count), 0)

    def _output_ready(self, path: Path, stage_name: str) -> bool:
        """
        Check whether an output exists and has rows.

        Args:
            path: Output path.
            stage_name: Stage name for logs.

        Returns:
            True if output can be skipped.
        """
        if self.force_rebuild:
            print(f"[PROGRESS] Force rebuild enabled. Will rerun {stage_name}.")
            return False

        if not path.exists():
            print(f"[PROGRESS] {stage_name} output missing. Will run stage.")
            return False

        row_count = self._count_csv_rows(path)

        if row_count <= 0:
            print(f"[PROGRESS] {stage_name} output has zero rows. Will rerun stage.")
            return False

        print(
            f"[PROGRESS] {stage_name} output already exists and has rows={row_count}. "
            "Skipping stage."
        )

        return True

    def _skip_response(self, stage_name: str, output_file: Path) -> Dict[str, object]:
        """
        Build skip response.
        """
        return {
            "status": "skipped",
            "message": f"{stage_name} output already exists.",
            "output_file": str(output_file),
            "records_count": self._count_csv_rows(output_file),
        }

    # ==================================================================================
    # Stage runners
    # ==================================================================================

    def run_health_index(self) -> Dict[str, object]:
        """
        Run or skip health index calculation.
        """
        stage_name = "health_index"

        if self._output_ready(self.health_index_csv, stage_name):
            return self._skip_response(stage_name, self.health_index_csv)

        return HealthIndexCalculator().run()

    def run_health_states(self) -> Dict[str, object]:
        """
        Run or skip health state classification.
        """
        stage_name = "health_states"

        if self._output_ready(self.health_states_csv, stage_name):
            return self._skip_response(stage_name, self.health_states_csv)

        return HealthStateClassifier().run()

    def run_health_trends(self) -> Dict[str, object]:
        """
        Run or skip health trend tracking.
        """
        stage_name = "health_trends"

        if self._output_ready(self.health_trends_csv, stage_name):
            return self._skip_response(stage_name, self.health_trends_csv)

        return HealthTrendTracker().run()

    def run_health_alerts(self) -> Dict[str, object]:
        """
        Run or skip health alert generation.
        """
        stage_name = "health_alerts"

        if self._output_ready(self.health_alerts_csv, stage_name):
            return self._skip_response(stage_name, self.health_alerts_csv)

        return HealthAlertEngine().run()

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run full health monitoring stage in correct dependency order.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering HealthScoreEngine.run")

        try:
            started = perf_counter()

            if not Config.ANOMALY_FUSION_CSV.exists():
                raise FileNotFoundError(
                    f"Required anomaly fusion CSV missing: {Config.ANOMALY_FUSION_CSV}"
                )

            health_index_result = self.run_health_index()
            health_state_result = self.run_health_states()
            trend_result = self.run_health_trends()
            alert_result = self.run_health_alerts()

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "message": "Complete health monitoring orchestration finished.",
                "force_rebuild": bool(self.force_rebuild),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "outputs": {
                    "health_index": str(self.health_index_csv),
                    "health_states": str(self.health_states_csv),
                    "health_trends": str(self.health_trends_csv),
                    "health_alerts": str(self.health_alerts_csv),
                },
                "stage_results": {
                    "health_index": health_index_result,
                    "health_states": health_state_result,
                    "health_trends": trend_result,
                    "health_alerts": alert_result,
                },
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                },
            }

            print(f"[PROGRESS] Writing health score engine summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            response = {
                "status": "success",
                "message": "Complete health monitoring stage finished.",
                "output_file": str(self.health_alerts_csv),
                "summary_file": str(self.summary_json),
                "records_count": alert_result.get(
                    "records_count",
                    health_state_result.get("records_count"),
                ),
                "data": {
                    "health_index": health_index_result,
                    "health_states": health_state_result,
                    "health_trends": trend_result,
                    "health_alerts": alert_result,
                },
            }

            print(f"[PROGRESS] Health score engine response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Health score engine failed: {exc}")
            logger.exception("Health score engine failed.")
            raise RuntimeError("Health score engine failed.") from exc


def run_health_score_engine() -> Dict[str, object]:
    """
    Execute full health monitoring orchestration.
    """
    print("[PROGRESS] Entering run_health_score_engine")

    engine = HealthScoreEngine()
    return engine.run()


if __name__ == "__main__":
    print("[PROGRESS] health_score_engine.py execution started")
    result = run_health_score_engine()
    print("[PROGRESS] health_score_engine.py execution finished successfully")
    print(result)