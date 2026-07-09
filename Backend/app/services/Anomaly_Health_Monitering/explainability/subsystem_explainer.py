"""
Subsystem explainer for CA-EDT-AHMA.

Role:
Map top contributing sensors and residual patterns to broad subsystem-level
explanations for dashboard and human-readable reports.

Important:
Subsystem labels are explanation support, not confirmed physical causality.

Reads:
data/outputs/root_cause_analysis.csv

Writes:
data/outputs/subsystem_explanations.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/explainability/subsystem_explainer.py")
from pathlib import Path
from typing import Dict, List

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv, read_csv_required
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class SubsystemExplainer:
    """
    Generates subsystem-level explanations from top sensors.
    """

    def __init__(self) -> None:
        """
        Initialize subsystem explainer.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/subsystem_explainer.py::__init__")
        Config.create_directories()

    def sensor_to_subsystem(self, sensor_name: str) -> str:
        """
        Map sensor name to broad subsystem category.

        Args:
            sensor_name: Sensor name.

        Returns:
            str: Subsystem category.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/subsystem_explainer.py::sensor_to_subsystem")
        try:
            name = sensor_name.lower()

            if any(token in name for token in ["temp", "temperature", "t2", "t24", "t30", "t48"]):
                return "thermal_subsystem"
            if any(token in name for token in ["press", "pressure", "p2", "p15", "p30"]):
                return "pressure_subsystem"
            if any(token in name for token in ["flow", "fuel", "wf"]):
                return "flow_fuel_subsystem"
            if any(token in name for token in ["speed", "shaft", "n1", "n2"]):
                return "rotational_subsystem"
            if any(token in name for token in ["eff", "efficiency"]):
                return "efficiency_subsystem"

            return "general_sensor_subsystem"

        except Exception as exc:
            logger.exception("Sensor to subsystem mapping failed.")
            raise RuntimeError("Sensor to subsystem mapping failed.") from exc

    def explain(self, root_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate subsystem explanations.

        Args:
            root_df: Root-cause analysis DataFrame.

        Returns:
            pd.DataFrame: Subsystem explanation DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/subsystem_explainer.py::explain")
        try:
            required_columns = [
                "unit_id",
                "cycle",
                "split",
                "top_sensor_1",
                "top_sensor_2",
                "top_sensor_3",
                "root_cause_pattern",
                "inspection_focus",
            ]

            missing = [column for column in required_columns if column not in root_df.columns]
            if missing:
                raise KeyError(f"Missing required subsystem explanation columns: {missing}")

            result = root_df[required_columns].copy()

            result["subsystem_1"] = result["top_sensor_1"].apply(self.sensor_to_subsystem)
            result["subsystem_2"] = result["top_sensor_2"].apply(self.sensor_to_subsystem)
            result["subsystem_3"] = result["top_sensor_3"].apply(self.sensor_to_subsystem)

            result["primary_subsystem"] = result["subsystem_1"]

            result["subsystem_explanation"] = result.apply(
                lambda row: (
                    f"The leading residual contribution is associated with "
                    f"{row['primary_subsystem']}. Supporting contributors are "
                    f"{row['subsystem_2']} and {row['subsystem_3']}. "
                    f"The pattern label is {row['root_cause_pattern']}. "
                    f"{row['inspection_focus']}"
                ),
                axis=1,
            )

            logger.info("Subsystem explanations generated. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Subsystem explanation generation failed.")
            raise RuntimeError("Subsystem explanation generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run subsystem explanation generation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/subsystem_explainer.py::run")
        try:
            root_df = read_csv_required(Config.ROOT_CAUSE_CSV)
            subsystem_df = self.explain(root_df)

            output_path: Path = Config.OUTPUT_DIR / "subsystem_explanations.csv"
            atomic_write_csv(subsystem_df, output_path)

            return {
                "status": "success",
                "message": "Subsystem explanations generated.",
                "output_file": str(output_path),
                "records_count": len(subsystem_df),
            }

        except Exception as exc:
            logger.exception("Subsystem explainer stage failed.")
            raise RuntimeError("Subsystem explainer stage failed.") from exc


def run_subsystem_explainer() -> Dict[str, object]:
    """
    Execute subsystem explanation generation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/subsystem_explainer.py::run_subsystem_explainer")
    explainer = SubsystemExplainer()
    return explainer.run()


if __name__ == "__main__":
    result = run_subsystem_explainer()
    print(result)