"""
Rule engine for CA-EDT-AHMA reasoning.

Role:
Convert top sensor residual patterns into conservative reasoning labels.

Important:
- The rules support explanation and inspection focus.
- They do not claim absolute physical causality.
- They do not issue maintenance decisions.
- They do not predict RUL.
- They do not use Y_dev/Y_test.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "reasoning/rule_engine.py"
)

from typing import Dict, List
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
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class ReasoningRuleEngine:
    """
    Conservative rule engine for anomaly pattern reasoning.

    This engine maps top residual-contributing sensors into explainable
    inspection-focus labels.
    """

    def __init__(self) -> None:
        """
        Initialize reasoning rule engine.
        """
        Config.create_directories()

    # ==================================================================================
    # Sensor family detection
    # ==================================================================================

    def _normalize_sensor_names(self, top_sensors: List[str]) -> List[str]:
        """
        Normalize sensor names for rule matching.

        Args:
            top_sensors: Sensor names such as Xs_T24, Xs_Ps30, Xs_Nf.

        Returns:
            Normalized lowercase sensor strings.
        """
        normalized: List[str] = []

        for sensor in top_sensors:
            value = str(sensor).strip().lower()

            value = value.replace("abs_residual_", "")
            value = value.replace("residual_", "")
            value = value.replace("xs_", "")
            value = value.replace("x_s_", "")
            value = value.replace("xv_", "")
            value = value.replace("x_v_", "")

            normalized.append(value)

        return normalized

    def _has_any(self, sensors: List[str], tokens: List[str]) -> bool:
        """
        Check whether any normalized sensor contains any token.
        """
        for sensor in sensors:
            for token in tokens:
                if token in sensor:
                    return True

        return False

    def _sensor_families(self, top_sensors: List[str]) -> Dict[str, bool]:
        """
        Detect broad sensor families from top contributing sensors.

        N-CMAPSS measured sensors used in your component:
        - Temperature: T24, T30, T48, T50
        - Pressure: P15, P2, P21, P24, Ps30, P40, P50
        - Rotational/speed: Nf, Nc
        - Fuel/flow: Wf
        """
        sensors = self._normalize_sensor_names(top_sensors)

        has_temperature = self._has_any(
            sensors,
            [
                "temp",
                "temperature",
                "t2",
                "t24",
                "t30",
                "t40",
                "t48",
                "t50",
            ],
        )

        has_pressure = self._has_any(
            sensors,
            [
                "press",
                "pressure",
                "p2",
                "p15",
                "p21",
                "p24",
                "p30",
                "ps30",
                "p40",
                "p45",
                "p50",
            ],
        )

        has_flow_or_fuel = self._has_any(
            sensors,
            [
                "flow",
                "fuel",
                "wf",
                "w21",
                "w22",
                "w25",
                "w31",
                "w32",
                "w48",
                "w50",
            ],
        )

        has_rotational = self._has_any(
            sensors,
            [
                "speed",
                "shaft",
                "n1",
                "n2",
                "nf",
                "nc",
                "nrf",
                "nrc",
            ],
        )

        has_efficiency = self._has_any(
            sensors,
            [
                "eff",
                "efficiency",
                "epr",
                "bpr",
            ],
        )

        # Hot-section sensors are especially useful for HPT/LPT-style reasoning.
        has_hot_section_temperature = self._has_any(
            sensors,
            [
                "t48",
                "t50",
            ],
        )

        has_core_pressure = self._has_any(
            sensors,
            [
                "ps30",
                "p40",
                "p50",
                "p30",
                "p45",
            ],
        )

        return {
            "has_temperature": has_temperature,
            "has_pressure": has_pressure,
            "has_flow_or_fuel": has_flow_or_fuel,
            "has_rotational": has_rotational,
            "has_efficiency": has_efficiency,
            "has_hot_section_temperature": has_hot_section_temperature,
            "has_core_pressure": has_core_pressure,
        }

    # ==================================================================================
    # Root-cause reasoning
    # ==================================================================================

    def infer_root_cause_pattern(self, top_sensors: List[str]) -> str:
        """
        Infer likely anomaly pattern from top contributing sensors.

        Args:
            top_sensors: Top contributing sensor names.

        Returns:
            Conservative root-cause pattern label.
        """
        try:
            families = self._sensor_families(top_sensors)

            has_temperature = families["has_temperature"]
            has_pressure = families["has_pressure"]
            has_flow_or_fuel = families["has_flow_or_fuel"]
            has_rotational = families["has_rotational"]
            has_efficiency = families["has_efficiency"]
            has_hot_section_temperature = families["has_hot_section_temperature"]
            has_core_pressure = families["has_core_pressure"]

            if has_hot_section_temperature and has_core_pressure:
                return "hot_section_pressure_thermal_deviation"

            if has_temperature and has_pressure and has_rotational:
                return "thermal_pressure_rotational_coupled_deviation"

            if has_temperature and has_pressure:
                return "temperature_pressure_instability"

            if has_flow_or_fuel and has_temperature:
                return "flow_thermal_response_deviation"

            if has_flow_or_fuel and has_pressure:
                return "flow_pressure_response_deviation"

            if has_rotational and has_pressure:
                return "rotational_pressure_deviation"

            if has_rotational and has_temperature:
                return "rotational_thermal_response_deviation"

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
            Inspection focus text.
        """
        try:
            mapping: Dict[str, str] = {
                "hot_section_pressure_thermal_deviation": (
                    "Inspect hot-section temperature and pressure response consistency, "
                    "especially HPT/LPT-related signal behavior."
                ),
                "thermal_pressure_rotational_coupled_deviation": (
                    "Inspect coupled thermal, pressure, and rotational-speed signal behavior."
                ),
                "temperature_pressure_instability": (
                    "Inspect thermal-pressure coupling and related measured sensor channels."
                ),
                "flow_thermal_response_deviation": (
                    "Inspect fuel/flow signal consistency and thermal response behavior."
                ),
                "flow_pressure_response_deviation": (
                    "Inspect fuel/flow signal consistency and pressure response behavior."
                ),
                "rotational_pressure_deviation": (
                    "Inspect rotational-speed signal consistency and pressure response behavior."
                ),
                "rotational_thermal_response_deviation": (
                    "Inspect rotational-speed signal consistency and thermal response behavior."
                ),
                "thermal_response_deviation": (
                    "Inspect temperature response sensors and heat-related signal patterns."
                ),
                "pressure_system_deviation": (
                    "Inspect pressure sensor behavior and compression-flow consistency."
                ),
                "flow_or_fuel_related_deviation": (
                    "Inspect fuel-flow-related signal consistency."
                ),
                "rotational_speed_deviation": (
                    "Inspect fan/core rotational-speed signal consistency."
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

    # ==================================================================================
    # Operational-vs-fault reasoning
    # ==================================================================================

    def classify_operational_vs_fault_related(
        self,
        final_anomaly_score: float,
        context_confidence: float,
        anomaly_persistence_score: float,
    ) -> str:
        """
        Classify whether anomaly is likely operating-context-related or
        persistent fault/degradation related.

        Args:
            final_anomaly_score: Final fused anomaly score.
            context_confidence: GMM context confidence.
            anomaly_persistence_score: Recent anomaly persistence.

        Returns:
            Conservative reasoning label.
        """
        try:
            final_anomaly_score = float(final_anomaly_score)
            context_confidence = float(context_confidence)
            anomaly_persistence_score = float(anomaly_persistence_score)

            if final_anomaly_score < 0.40:
                return "normal_operating_variation"

            if context_confidence < 0.50 and anomaly_persistence_score < 0.40:
                return "possible_operating_context_shift"

            if final_anomaly_score >= 0.85 and anomaly_persistence_score >= 0.40:
                return "critical_persistent_fault_related_pattern"

            if final_anomaly_score >= 0.65 and anomaly_persistence_score >= 0.40:
                return "likely_fault_related_residual_pattern"

            if final_anomaly_score >= 0.40 and anomaly_persistence_score >= 0.40:
                return "persistent_watch_level_residual_pattern"

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
            Reasoning summary.
        """
        try:
            pattern_type = self.classify_operational_vs_fault_related(
                final_anomaly_score=final_anomaly_score,
                context_confidence=context_confidence,
                anomaly_persistence_score=anomaly_persistence_score,
            )

            inspection_focus = self.recommend_inspection_focus(root_cause_pattern)

            return (
                f"The alert level is {alert_level}. Residual attribution suggests "
                f"{root_cause_pattern}. The operating-vs-fault reasoning label is "
                f"{pattern_type}. Recommended inspection focus: {inspection_focus} "
                f"This is an explanation and inspection-focus recommendation only, "
                f"not a maintenance scheduling decision."
            )

        except Exception as exc:
            logger.exception("Reasoning summary generation failed.")
            raise RuntimeError("Reasoning summary generation failed.") from exc


def run_rule_engine_self_check() -> Dict[str, object]:
    """
    Run a self-check of the rule engine.

    Returns:
        Self-check result.
    """
    print("[PROGRESS] Entering run_rule_engine_self_check")

    engine = ReasoningRuleEngine()

    test_sensors = ["Xs_T48", "Xs_P40", "Xs_Nc"]

    pattern = engine.infer_root_cause_pattern(test_sensors)
    focus = engine.recommend_inspection_focus(pattern)
    summary = engine.build_reasoning_summary(
        alert_level="Warning",
        root_cause_pattern=pattern,
        final_anomaly_score=0.72,
        context_confidence=0.91,
        anomaly_persistence_score=0.55,
    )

    return {
        "status": "success",
        "message": "Rule engine self-check completed.",
        "data": {
            "test_sensors": test_sensors,
            "pattern": pattern,
            "inspection_focus": focus,
            "reasoning_summary": summary,
        },
    }


if __name__ == "__main__":
    result = run_rule_engine_self_check()
    print(result)