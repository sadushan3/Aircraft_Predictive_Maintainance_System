"""
Sensor dependency graph for CA-EDT-AHMA.

Role:
Create a conservative, explainable sensor-family dependency graph used by the
reasoning stage.

Important:
- This graph is not a hard causal graph.
- It supports explanation and inspection focus only.
- It does not make maintenance decisions.
- It does not predict RUL.
- It does not use Y_dev/Y_test.
- It reads residuals.csv header only, not the full file.

Reads:
outputs/Anomaly_Health_Monitering/residuals.csv header only

Writes:
outputs/Anomaly_Health_Monitering/sensor_dependency_graph.csv
reports/sensor_dependency_graph_summary.json
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "reasoning/sensor_dependency_graph.py"
)

from pathlib import Path
from typing import Dict, List
import os
import re
import sys

import pandas as pd


if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    if BACKEND_ROOT not in sys.path:
        sys.path.append(BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv, atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class SensorDependencyGraph:
    """
    Build a static, domain-informed graph between measured sensor families.

    The graph is intentionally conservative:
    - explanation support only
    - inspection focus only
    - no hard physical causality claim
    - no maintenance decision
    """

    FAMILY_RULES: Dict[str, List[str]] = {
        "Temperature": [
            "t2",
            "t24",
            "t30",
            "t40",
            "t48",
            "t50",
            "temp",
            "temperature",
        ],
        "Pressure": [
            "p2",
            "p15",
            "p21",
            "p24",
            "p30",
            "ps30",
            "p40",
            "p45",
            "p50",
            "press",
            "pressure",
        ],
        "Rotational": [
            "nf",
            "nc",
            "n1",
            "n2",
            "speed",
            "shaft",
        ],
        "Fuel_Flow": [
            "wf",
            "fuel",
            "flow",
        ],
        "Efficiency": [
            "eff",
            "epr",
            "bpr",
        ],
    }

    DEFAULT_DEPENDENCIES: List[Dict[str, object]] = [
        {
            "source_family": "Fuel_Flow",
            "target_family": "Temperature",
            "dependency_type": "fuel_thermal_response_coupling",
            "dependency_strength": 0.75,
            "reasoning_note": "Fuel-flow changes can influence turbine temperature residual behavior.",
        },
        {
            "source_family": "Rotational",
            "target_family": "Pressure",
            "dependency_type": "shaft_pressure_coupling",
            "dependency_strength": 0.70,
            "reasoning_note": "Rotor-speed deviations can co-occur with compressor or turbine pressure residuals.",
        },
        {
            "source_family": "Pressure",
            "target_family": "Temperature",
            "dependency_type": "thermodynamic_coupling",
            "dependency_strength": 0.68,
            "reasoning_note": "Pressure and temperature residuals are coupled across gas-path operation.",
        },
        {
            "source_family": "Temperature",
            "target_family": "Efficiency",
            "dependency_type": "thermal_performance_indicator",
            "dependency_strength": 0.60,
            "reasoning_note": "Hot-section temperature residuals can support efficiency-loss reasoning.",
        },
        {
            "source_family": "Pressure",
            "target_family": "Efficiency",
            "dependency_type": "pressure_performance_indicator",
            "dependency_strength": 0.60,
            "reasoning_note": "Core pressure residuals can support performance-deviation reasoning.",
        },
        {
            "source_family": "Fuel_Flow",
            "target_family": "Pressure",
            "dependency_type": "fuel_pressure_response_coupling",
            "dependency_strength": 0.58,
            "reasoning_note": "Fuel-flow deviations may appear together with pressure response deviations.",
        },
        {
            "source_family": "Rotational",
            "target_family": "Temperature",
            "dependency_type": "shaft_thermal_response_coupling",
            "dependency_strength": 0.55,
            "reasoning_note": "Rotational-speed deviations may co-occur with thermal response deviations.",
        },
    ]

    def __init__(self) -> None:
        print("[PROGRESS] Entering SensorDependencyGraph.__init__")
        Config.create_directories()

        self.residual_csv: Path = Config.RESIDUALS_CSV

        self.output_csv: Path = getattr(
            Config,
            "SENSOR_DEPENDENCY_GRAPH_CSV",
            Config.OUTPUT_DIR / "sensor_dependency_graph.csv",
        )

        self.summary_json: Path = getattr(
            Config,
            "SENSOR_DEPENDENCY_GRAPH_SUMMARY_JSON",
            Config.REPORT_DIR / "sensor_dependency_graph_summary.json",
        )

        print(f"[PROGRESS] Residual CSV: {self.residual_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")

    def _read_residual_columns(self) -> List[str]:
        """
        Read only residuals.csv header and return raw absolute residual columns.

        Expected:
        abs_residual_Xs_T24
        abs_residual_Xs_T30
        ...
        """
        print("[PROGRESS] Entering SensorDependencyGraph._read_residual_columns")

        if not self.residual_csv.exists():
            raise FileNotFoundError(f"Residual CSV not found: {self.residual_csv}")

        columns = list(pd.read_csv(self.residual_csv, nrows=0).columns)

        abs_residual_columns = []

        for column in columns:
            if not column.startswith("abs_residual_"):
                continue

            lower = column.lower()

            excluded_tokens = [
                "rolling",
                "trend",
                "mean",
                "std",
                "delta",
                "slope",
                "lag",
                "temporal",
            ]

            if any(token in lower for token in excluded_tokens):
                continue

            abs_residual_columns.append(column)

        if not abs_residual_columns:
            raise ValueError(
                "No raw absolute residual columns found in residuals.csv header."
            )

        print(f"[PROGRESS] Raw absolute residual columns: {abs_residual_columns}")
        return abs_residual_columns

    def _sensor_name_from_column(self, column: str) -> str:
        """
        Convert abs_residual_Xs_T24 -> Xs_T24.
        """
        sensor = re.sub(r"^abs_residual_", "", str(column))
        sensor = re.sub(r"^residual_", "", sensor)
        return sensor

    def _family_for_sensor(self, sensor: str) -> str:
        """
        Map sensor name to a broad reasoning family.
        """
        normalized = (
            str(sensor)
            .lower()
            .replace("xs_", "")
            .replace("x_s_", "")
            .replace("xv_", "")
            .replace("x_v_", "")
        )

        for family, tokens in self.FAMILY_RULES.items():
            if any(token in normalized for token in tokens):
                return family

        return "Other"

    def _family_sensor_map(self, sensor_families: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Build family -> sensors map.
        """
        family_map: Dict[str, List[str]] = {}

        for sensor, family in sensor_families.items():
            family_map.setdefault(family, []).append(sensor)

        for family in family_map:
            family_map[family] = sorted(family_map[family])

        return family_map

    def build(self) -> pd.DataFrame:
        """
        Build sensor dependency graph as a DataFrame.
        """
        print("[PROGRESS] Entering SensorDependencyGraph.build")

        try:
            residual_columns = self._read_residual_columns()

            sensor_families: Dict[str, str] = {}

            for column in residual_columns:
                sensor = self._sensor_name_from_column(column)
                sensor_families[sensor] = self._family_for_sensor(sensor)

            family_map = self._family_sensor_map(sensor_families)
            available_families = set(sensor_families.values())

            rows: List[Dict[str, object]] = []

            for dependency in self.DEFAULT_DEPENDENCIES:
                source_family = str(dependency["source_family"])
                target_family = str(dependency["target_family"])

                source_sensors = family_map.get(source_family, [])
                target_sensors = family_map.get(target_family, [])

                source_available = source_family in available_families
                target_available = target_family in available_families

                rows.append(
                    {
                        "edge_type": "family_dependency",
                        "source_family": source_family,
                        "target_family": target_family,
                        "source_sensors": ", ".join(source_sensors),
                        "target_sensors": ", ".join(target_sensors),
                        "source_family_available": bool(source_available),
                        "target_family_available": bool(target_available),
                        "available_sensor_count": int(
                            len(source_sensors) + len(target_sensors)
                        ),
                        "dependency_type": str(dependency["dependency_type"]),
                        "dependency_strength": float(dependency["dependency_strength"]),
                        "reasoning_note": str(dependency["reasoning_note"]),
                        "hard_causal_claim": False,
                        "maintenance_decision": "Not generated by this component",
                        "component_role": "Reasoning support and inspection-focus explanation",
                    }
                )

            for sensor, family in sorted(sensor_families.items()):
                rows.append(
                    {
                        "edge_type": "sensor_family_membership",
                        "source_family": family,
                        "target_family": family,
                        "source_sensors": sensor,
                        "target_sensors": sensor,
                        "source_family_available": True,
                        "target_family_available": True,
                        "available_sensor_count": 1,
                        "dependency_type": "sensor_family_membership",
                        "dependency_strength": 1.0,
                        "reasoning_note": (
                            f"{sensor} is grouped under the {family} reasoning family."
                        ),
                        "hard_causal_claim": False,
                        "maintenance_decision": "Not generated by this component",
                        "component_role": "Reasoning support and inspection-focus explanation",
                    }
                )

            result = pd.DataFrame(rows)

            logger.info(
                "Sensor dependency graph built. rows=%s sensors=%s families=%s",
                len(result),
                len(sensor_families),
                len(family_map),
            )

            return result

        except Exception as exc:
            logger.exception("Sensor dependency graph build failed.")
            raise RuntimeError("Sensor dependency graph build failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Build and save sensor dependency graph.
        """
        print("[PROGRESS] Entering SensorDependencyGraph.run")

        try:
            result = self.build()

            atomic_write_csv(result, self.output_csv)

            family_counts = (
                result[result["edge_type"] == "sensor_family_membership"]["source_family"]
                .value_counts()
                .to_dict()
            )

            dependency_counts = (
                result[result["edge_type"] == "family_dependency"]["dependency_type"]
                .value_counts()
                .to_dict()
            )

            summary = {
                "status": "success",
                "message": "Sensor dependency graph completed.",
                "output_file": str(self.output_csv),
                "records_count": int(len(result)),
                "residual_csv": str(self.residual_csv),
                "uses_residual_header_only": True,
                "family_counts": family_counts,
                "dependency_counts": dependency_counts,
                "hard_causal_graph": False,
                "maintenance_decision": False,
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "reads_residual_header_only": True,
                },
            }

            atomic_write_json(summary, self.summary_json)

            print(f"[PROGRESS] Sensor dependency graph response: {summary}")
            return summary

        except Exception as exc:
            logger.exception("Sensor dependency graph stage failed.")
            raise RuntimeError("Sensor dependency graph stage failed.") from exc


def run_sensor_dependency_graph() -> Dict[str, object]:
    """
    Execute sensor dependency graph generation.
    """
    print("[PROGRESS] Entering run_sensor_dependency_graph")

    graph = SensorDependencyGraph()
    return graph.run()


if __name__ == "__main__":
    print("[PROGRESS] sensor_dependency_graph.py execution started")
    result = run_sensor_dependency_graph()
    print("[PROGRESS] sensor_dependency_graph.py execution finished successfully")
    print(result)