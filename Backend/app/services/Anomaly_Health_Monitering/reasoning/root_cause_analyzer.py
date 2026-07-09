"""
Root-cause analyzer for CA-EDT-AHMA.

Role:
Rank top contributing measured sensors using residual contribution.

Formula:
sensor_contribution = abs(sensor_residual) / sum(abs(all_sensor_residuals))

Output:
top_sensor_1
top_sensor_2
top_sensor_3
contribution_1
contribution_2
contribution_3
root_cause_pattern
inspection_focus

Important:
This module identifies likely contributing sensor patterns.
It does not make final maintenance decisions.

Reads:
data/outputs/residuals.csv
data/outputs/health_states.csv

Writes:
data/outputs/root_cause_analysis.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_analyzer.py")
from typing import Dict, List

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.reasoning.rule_engine import ReasoningRuleEngine
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv, read_csv_required
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_abs_residual_columns

logger = get_logger(__name__)


class RootCauseAnalyzer:
    """
    Residual-based root-cause analyzer.
    """

    def __init__(self) -> None:
        """
        Initialize root-cause analyzer.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_analyzer.py::__init__")
        Config.create_directories()
        self.rule_engine = ReasoningRuleEngine()

    def _top_sensor_contributions(self, row: pd.Series, abs_residual_columns: List[str]) -> Dict[str, object]:
        """
        Calculate top sensor contributions for one row.

        Args:
            row: Residual row.
            abs_residual_columns: Absolute residual columns.

        Returns:
            Dict[str, object]: Top sensor contribution record.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_analyzer.py::_top_sensor_contributions")
        try:
            abs_values = row[abs_residual_columns].astype(float)
            total_abs_residual = float(abs_values.sum())

            if total_abs_residual <= 1e-12:
                contributions = pd.Series(0.0, index=abs_residual_columns)
            else:
                contributions = abs_values / total_abs_residual

            top = contributions.sort_values(ascending=False).head(3)

            top_sensors = [
                column.replace("abs_residual_", "")
                for column in top.index.tolist()
            ]
            top_values = [float(value) for value in top.values.tolist()]

            while len(top_sensors) < 3:
                top_sensors.append("none")
                top_values.append(0.0)

            root_pattern = self.rule_engine.infer_root_cause_pattern(top_sensors)
            inspection_focus = self.rule_engine.recommend_inspection_focus(root_pattern)

            return {
                "top_sensor_1": top_sensors[0],
                "top_sensor_2": top_sensors[1],
                "top_sensor_3": top_sensors[2],
                "contribution_1": round(top_values[0], 6),
                "contribution_2": round(top_values[1], 6),
                "contribution_3": round(top_values[2], 6),
                "root_cause_pattern": root_pattern,
                "inspection_focus": inspection_focus,
            }

        except Exception as exc:
            logger.exception("Top sensor contribution calculation failed.")
            raise RuntimeError("Top sensor contribution calculation failed.") from exc

    def analyze(self) -> pd.DataFrame:
        """
        Run root-cause analysis.

        Returns:
            pd.DataFrame: Root-cause DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_analyzer.py::analyze")
        try:
            residual_df = read_csv_required(Config.RESIDUALS_CSV)
            health_df = read_csv_required(Config.HEALTH_STATES_CSV)

            merge_columns = ["unit_id", "cycle", "split"]

            df = residual_df.merge(
                health_df[
                    merge_columns
                    + [
                        "final_anomaly_score",
                        "alert_level",
                        "health_index",
                        "health_state",
                        "anomaly_persistence_score",
                    ]
                ],
                on=merge_columns,
                how="left",
            )

            abs_residual_columns = get_abs_residual_columns(df)

            if not abs_residual_columns:
                raise ValueError("No absolute residual columns found for root-cause analysis.")

            records: List[Dict[str, object]] = []

            for _, row in df.iterrows():
                contribution_record = self._top_sensor_contributions(row, abs_residual_columns)

                records.append(
                    {
                        "unit_id": row["unit_id"],
                        "cycle": row["cycle"],
                        "split": row["split"],
                        "final_anomaly_score": row.get("final_anomaly_score", 0.0),
                        "alert_level": row.get("alert_level", "Normal"),
                        "health_index": row.get("health_index", 100.0),
                        "health_state": row.get("health_state", "Healthy"),
                        **contribution_record,
                    }
                )

            result = pd.DataFrame(records)
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            logger.info("Root-cause analysis completed. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Root-cause analysis failed.")
            raise RuntimeError("Root-cause analysis failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run root-cause analyzer.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_analyzer.py::run")
        try:
            result = self.analyze()
            atomic_write_csv(result, Config.ROOT_CAUSE_CSV)

            return {
                "status": "success",
                "message": "Root-cause sensor attribution completed.",
                "output_file": str(Config.ROOT_CAUSE_CSV),
                "records_count": len(result),
            }

        except Exception as exc:
            logger.exception("Root-cause analyzer stage failed.")
            raise RuntimeError("Root-cause analyzer stage failed.") from exc


def run_root_cause_analysis() -> Dict[str, object]:
    """
    Execute root-cause analysis.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/root_cause_analyzer.py::run_root_cause_analysis")
    analyzer = RootCauseAnalyzer()
    return analyzer.run()


if __name__ == "__main__":
    result = run_root_cause_analysis()
    print(result)