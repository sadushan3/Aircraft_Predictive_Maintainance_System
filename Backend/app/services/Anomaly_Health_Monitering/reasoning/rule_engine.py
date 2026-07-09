"""
Rule engine for CA-EDT-AHMA reasoning.

Role:
Convert top sensor residual patterns into conservative reasoning labels.

Important:
The rules support explanation and inspection focus.
They do not claim absolute physical causality.
They do not issue maintenance decisions.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/reasoning/rule_engine.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class ReasoningRuleEngine:
    """
    Rule engine for conservative anomaly pattern reasoning.
    """

    def __init__(self) -> None:
        """
        Initialize reasoning rule engine.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/rule_engine.py::__init__")
        Config.create_directories()

    def infer_root_cause_pattern(self, top_sensors: List[str]) -> str:
        """
        Infer likely anomaly pattern from top contributing sensors.

        Args:
            top_sensors: Top contributing sensor names.

        Returns:
            str: Root-cause pattern label.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/rule_engine.py::infer_root_cause_pattern")
        try:
            joined = " ".join(top_sensors).lower()

            has_temperature = any(
                token in joined
                for token in ["temp", "temperature", "t2", "t24", "t30", "t48"]
            )
            has_pressure = any(
                token in joined
                for token in ["press", "pressure", "p2", "p15", "p30"]
            )
            has_flow_or_fuel = any(
                token in joined
                for token in ["flow", "fuel", "wf"]
            )
            has_rotational = any(
                token in joined
                for token in ["speed", "shaft", "n1", "n2"]
            )
            has_efficiency = any(
                token in joined
                for token in ["eff", "efficiency"]
            )

            if has_temperature and has_pressure:
                return "temperature_pressure_instability"
            if has_flow_or_fuel and has_temperature:
                return "flow_thermal_response_deviation"
            if has_rotational and has_pressure:
                return "rotational_pressure_deviation"
            if has_temperature:
                return "thermal_response_deviation"
            if has_pressure:
                return "pressure_system_deviation"
            if has_flow_or_fuel:
                return "flow_or_fuel_related_deviation"
            if has_rotational:
                return "rotational_speed_deviation"
            if has_efficiency:
                return "efficiency_related_deviation"

            return "multisensor_residual_deviation"

        except Exception as exc:
            logger.exception("Root-cause pattern inference failed.")
            raise RuntimeError("Root-cause pattern inference failed.") from exc

    def recommend_inspection_focus(self, root_cause_pattern: str) -> str:
        """
        Recommend inspection focus without maintenance decision-making.

        Args:
            root_cause_pattern: Root-cause pattern.

        Returns:
            str: Inspection focus text.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/rule_engine.py::recommend_inspection_focus")
        try:
            mapping: Dict[str, str] = {
                "temperature_pressure_instability": (
                    "Inspect thermal-pressure coupling and related measured sensor channels."
                ),
                "flow_thermal_response_deviation": (
                    "Inspect flow/fuel signal consistency and thermal response behavior."
                ),
                "rotational_pressure_deviation": (
                    "Inspect rotational speed signal consistency and pressure response behavior."
                ),
                "thermal_response_deviation": (
                    "Inspect temperature response sensors and heat-related signal patterns."
                ),
                "pressure_system_deviation": (
                    "Inspect pressure sensor behavior and compression-flow consistency."
                ),
                "flow_or_fuel_related_deviation": (
                    "Inspect flow/fuel-related signal consistency."
                ),
                "rotational_speed_deviation": (
                    "Inspect rotational speed signal consistency."
                ),
                "efficiency_related_deviation": (
                    "Inspect efficiency-related virtual/measured signal consistency."
                ),
                "multisensor_residual_deviation": (
                    "Inspect the top contributing measured sensor channels."
                ),
            }

            return mapping.get(
                root_cause_pattern,
                "Inspect the top contributing measured sensor channels.",
            )

        except Exception as exc:
            logger.exception("Inspection focus recommendation failed.")
            raise RuntimeError("Inspection focus recommendation failed.") from exc

    def classify_operational_vs_fault_related(
        self,
        final_anomaly_score: float,
        context_confidence: float,
        anomaly_persistence_score: float,
    ) -> str:
        """
        Classify whether anomaly is likely operational-change-related or fault-related.

        Args:
            final_anomaly_score: Final anomaly score.
            context_confidence: GMM context confidence.
            anomaly_persistence_score: Persistence score.

        Returns:
            str: Pattern type.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/rule_engine.py::classify_operational_vs_fault_related")
        try:
            if final_anomaly_score < 0.40:
                return "normal_operating_variation"

            if context_confidence < 0.50 and anomaly_persistence_score < 0.40:
                return "possible_operating_context_shift"

            if final_anomaly_score >= 0.65 and anomaly_persistence_score >= 0.40:
                return "likely_fault_related_residual_pattern"

            if final_anomaly_score >= 0.40:
                return "watch_level_residual_pattern"

            return "normal_operating_variation"

        except Exception as exc:
            logger.exception("Operational vs fault-related classification failed.")
            raise RuntimeError("Operational vs fault-related classification failed.") from exc

    def build_reasoning_summary(
        self,
        alert_level: str,
        root_cause_pattern: str,
        final_anomaly_score: float,
        context_confidence: float,
        anomaly_persistence_score: float,
    ) -> str:
        """
        Build a concise reasoning summary.

        Args:
            alert_level: Alert level.
            root_cause_pattern: Root-cause pattern.
            final_anomaly_score: Final anomaly score.
            context_confidence: Context confidence.
            anomaly_persistence_score: Anomaly persistence score.

        Returns:
            str: Reasoning summary.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/rule_engine.py::build_reasoning_summary")
        try:
            pattern_type = self.classify_operational_vs_fault_related(
                final_anomaly_score=final_anomaly_score,
                context_confidence=context_confidence,
                anomaly_persistence_score=anomaly_persistence_score,
            )

            return (
                f"The alert level is {alert_level}. The residual evidence suggests "
                f"{root_cause_pattern}. The operating-vs-fault reasoning label is "
                f"{pattern_type}. This is an inspection-focus recommendation, not a "
                f"maintenance scheduling decision."
            )

        except Exception as exc:
            logger.exception("Reasoning summary generation failed.")
            raise RuntimeError("Reasoning summary generation failed.") from exc


def run_rule_engine_self_check() -> Dict[str, object]:
    """
    Run a self-check of the rule engine.

    Returns:
        Dict[str, object]: Self-check result.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/rule_engine.py::run_rule_engine_self_check")
    engine = ReasoningRuleEngine()
    pattern = engine.infer_root_cause_pattern(["Xs_T30", "Xs_P30", "Xs_N2"])
    focus = engine.recommend_inspection_focus(pattern)

    return {
        "status": "success",
        "message": "Rule engine self-check completed.",
        "data": {
            "pattern": pattern,
            "inspection_focus": focus,
        },
    }


if __name__ == "__main__":
    result = run_rule_engine_self_check()
    print(result)