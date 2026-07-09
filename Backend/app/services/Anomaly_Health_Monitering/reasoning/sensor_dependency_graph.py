"""
Sensor dependency graph for CA-EDT-AHMA.

Role:
Represent conservative sensor relationship groups for reasoning support.

Important:
This graph is not a hard causal graph.
It supports explanation and inspection focus only.
Maintenance decisions belong to the autonomous maintenance supervisor agent.

Writes:
data/outputs/sensor_dependency_graph.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/reasoning/sensor_dependency_graph.py")
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
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
from app.utils.Anomaly_Health_Monitering.model_utils import get_xs_columns

logger = get_logger(__name__)


class SensorDependencyGraph:
    """
    Builds a conservative dependency graph among measured sensors.
    """

    def __init__(self) -> None:
        """
        Initialize sensor dependency graph builder.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/sensor_dependency_graph.py::__init__")
        Config.create_directories()
        self.graph = nx.Graph()

    def _sensor_category(self, sensor_name: str) -> str:
        """
        Infer broad sensor category from sensor name.

        Args:
            sensor_name: Sensor column name.

        Returns:
            str: Sensor category.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/sensor_dependency_graph.py::_sensor_category")
        name = sensor_name.lower()

        if any(token in name for token in ["temp", "temperature", "t2", "t24", "t30", "t48"]):
            return "thermal"
        if any(token in name for token in ["press", "pressure", "p2", "p15", "p30"]):
            return "pressure"
        if any(token in name for token in ["flow", "fuel", "wf"]):
            return "flow_fuel"
        if any(token in name for token in ["speed", "shaft", "n1", "n2"]):
            return "rotational"
        if any(token in name for token in ["eff", "efficiency"]):
            return "efficiency"
        return "general_sensor"

    def build_graph(self, sensor_columns: List[str]) -> nx.Graph:
        """
        Build sensor dependency graph.

        Args:
            sensor_columns: Measured sensor columns.

        Returns:
            nx.Graph: Sensor dependency graph.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/sensor_dependency_graph.py::build_graph")
        try:
            if not sensor_columns:
                raise ValueError("No sensor columns provided for dependency graph.")

            graph = nx.Graph()

            for sensor in sensor_columns:
                graph.add_node(
                    sensor,
                    category=self._sensor_category(sensor),
                )

            for i, sensor_a in enumerate(sensor_columns):
                for sensor_b in sensor_columns[i + 1:]:
                    category_a = self._sensor_category(sensor_a)
                    category_b = self._sensor_category(sensor_b)

                    if category_a == category_b:
                        graph.add_edge(
                            sensor_a,
                            sensor_b,
                            relation="same_category",
                            weight=0.8,
                        )
                    elif {category_a, category_b} == {"thermal", "pressure"}:
                        graph.add_edge(
                            sensor_a,
                            sensor_b,
                            relation="thermal_pressure_coupling",
                            weight=0.6,
                        )
                    elif {category_a, category_b} == {"flow_fuel", "thermal"}:
                        graph.add_edge(
                            sensor_a,
                            sensor_b,
                            relation="flow_thermal_coupling",
                            weight=0.5,
                        )
                    elif {category_a, category_b} == {"rotational", "pressure"}:
                        graph.add_edge(
                            sensor_a,
                            sensor_b,
                            relation="rotational_pressure_coupling",
                            weight=0.5,
                        )

            self.graph = graph
            logger.info(
                "Sensor dependency graph built. nodes=%s edges=%s",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            return graph

        except Exception as exc:
            logger.exception("Sensor dependency graph build failed.")
            raise RuntimeError("Sensor dependency graph build failed.") from exc

    def graph_to_dataframe(self, graph: nx.Graph) -> pd.DataFrame:
        """
        Convert graph edges to DataFrame.

        Args:
            graph: Sensor dependency graph.

        Returns:
            pd.DataFrame: Edge table.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/sensor_dependency_graph.py::graph_to_dataframe")
        try:
            records: List[Dict[str, object]] = []

            for sensor_a, sensor_b, attributes in graph.edges(data=True):
                records.append(
                    {
                        "sensor_a": sensor_a,
                        "sensor_b": sensor_b,
                        "relation": attributes.get("relation", "related"),
                        "weight": float(attributes.get("weight", 0.0)),
                        "category_a": graph.nodes[sensor_a].get("category", "unknown"),
                        "category_b": graph.nodes[sensor_b].get("category", "unknown"),
                    }
                )

            graph_df = pd.DataFrame(records)
            logger.info("Sensor dependency graph converted to DataFrame. rows=%s", len(graph_df))
            return graph_df

        except Exception as exc:
            logger.exception("Graph to DataFrame conversion failed.")
            raise RuntimeError("Graph to DataFrame conversion failed.") from exc

    def related_sensors(self, sensor_name: str, top_k: int = 5) -> List[Dict[str, object]]:
        """
        Return related sensors for a sensor.

        Args:
            sensor_name: Sensor name.
            top_k: Number of related sensors.

        Returns:
            List[Dict[str, object]]: Related sensors.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/sensor_dependency_graph.py::related_sensors")
        try:
            if self.graph.number_of_nodes() == 0:
                scaled_df = read_csv_required(Config.SCALED_CSV)
                self.build_graph(get_xs_columns(scaled_df))

            if sensor_name not in self.graph:
                return []

            neighbors: List[Tuple[str, float, str]] = []
            for neighbor in self.graph.neighbors(sensor_name):
                edge = self.graph.get_edge_data(sensor_name, neighbor, default={})
                neighbors.append(
                    (
                        neighbor,
                        float(edge.get("weight", 0.0)),
                        str(edge.get("relation", "related")),
                    )
                )

            neighbors = sorted(neighbors, key=lambda item: item[1], reverse=True)[:top_k]

            return [
                {
                    "sensor": sensor,
                    "relation": relation,
                    "weight": weight,
                }
                for sensor, weight, relation in neighbors
            ]

        except Exception as exc:
            logger.exception("Related sensor lookup failed.")
            raise RuntimeError("Related sensor lookup failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Build and save sensor dependency graph.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/sensor_dependency_graph.py::run")
        try:
            scaled_df = read_csv_required(Config.SCALED_CSV)
            sensor_columns = get_xs_columns(scaled_df)

            graph = self.build_graph(sensor_columns)
            graph_df = self.graph_to_dataframe(graph)

            output_path: Path = Config.OUTPUT_DIR / "sensor_dependency_graph.csv"
            atomic_write_csv(graph_df, output_path)

            return {
                "status": "success",
                "message": "Sensor dependency graph generated for reasoning support.",
                "output_file": str(output_path),
                "records_count": len(graph_df),
            }

        except Exception as exc:
            logger.exception("Sensor dependency graph stage failed.")
            raise RuntimeError("Sensor dependency graph stage failed.") from exc


def run_sensor_dependency_graph() -> Dict[str, object]:
    """
    Execute sensor dependency graph generation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/sensor_dependency_graph.py::run_sensor_dependency_graph")
    service = SensorDependencyGraph()
    return service.run()


if __name__ == "__main__":
    result = run_sensor_dependency_graph()
    print(result)