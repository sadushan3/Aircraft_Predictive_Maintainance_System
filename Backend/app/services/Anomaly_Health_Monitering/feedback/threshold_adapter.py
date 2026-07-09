"""
Feedback threshold adapter for CA-EDT-AHMA.

Role:
Adapt context-specific residual thresholds using operator feedback.

Rules:
- rejected_false_alarm: increase threshold slightly
- missed_anomaly: decrease threshold slightly
- accepted_alert: keep or slightly strengthen threshold
- uncertain: no change

Reads:
models/anomaly/residual_thresholds.json
data/outputs/feedback_updates.csv

Writes:
models/feedback/adaptive_thresholds.json
data/outputs/feedback_updates.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/feedback/threshold_adapter.py")
from typing import Dict

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_required,
    read_json_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class ThresholdAdapter:
    """
    Adaptive threshold updater based on feedback.
    """

    def __init__(self, adjustment_step: float = 0.02) -> None:
        """
        Initialize threshold adapter.

        Args:
            adjustment_step: Relative adjustment step.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/threshold_adapter.py::__init__")
        Config.create_directories()

        if adjustment_step <= 0 or adjustment_step >= 0.5:
            raise ValueError("adjustment_step must be between 0 and 0.5.")

        self.adjustment_step = adjustment_step

    def _load_base_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Load residual thresholds.

        Returns:
            Dict[str, Dict[str, float]]: Thresholds.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/threshold_adapter.py::_load_base_thresholds")
        try:
            if Config.ADAPTIVE_THRESHOLDS_PATH.exists():
                data = read_json_required(Config.ADAPTIVE_THRESHOLDS_PATH)
                thresholds = data.get("thresholds", {})
                if thresholds:
                    return thresholds

            data = read_json_required(Config.RESIDUAL_THRESHOLDS_PATH)
            thresholds = data.get("thresholds", {})

            if not thresholds:
                raise ValueError("No residual thresholds available for adaptation.")

            return thresholds

        except Exception as exc:
            logger.exception("Failed to load thresholds for adaptation.")
            raise RuntimeError("Failed to load thresholds for adaptation.") from exc

    def _adjust_context_thresholds(
        self,
        thresholds: Dict[str, Dict[str, float]],
        context_key: str,
        feedback_label: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Adjust thresholds for one context.

        Args:
            thresholds: Threshold dictionary.
            context_key: Context id as string.
            feedback_label: Feedback label.

        Returns:
            Dict[str, Dict[str, float]]: Updated thresholds.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/threshold_adapter.py::_adjust_context_thresholds")
        try:
            if context_key not in thresholds:
                context_key = "global"

            if context_key not in thresholds:
                raise KeyError("Global thresholds are missing.")

            multiplier = 1.0

            if feedback_label == "rejected_false_alarm":
                multiplier = 1.0 + self.adjustment_step
            elif feedback_label == "missed_anomaly":
                multiplier = 1.0 - self.adjustment_step
            elif feedback_label == "accepted_alert":
                multiplier = 1.0
            elif feedback_label == "uncertain":
                multiplier = 1.0
            else:
                raise ValueError(f"Unsupported feedback label: {feedback_label}")

            for level in ["watch", "warning", "critical"]:
                current_value = float(thresholds[context_key][level])
                thresholds[context_key][level] = max(current_value * multiplier, 1e-9)

            if thresholds[context_key]["watch"] > thresholds[context_key]["warning"]:
                thresholds[context_key]["warning"] = thresholds[context_key]["watch"] * 1.05

            if thresholds[context_key]["warning"] > thresholds[context_key]["critical"]:
                thresholds[context_key]["critical"] = thresholds[context_key]["warning"] * 1.05

            return thresholds

        except Exception as exc:
            logger.exception("Context threshold adjustment failed.")
            raise RuntimeError("Context threshold adjustment failed.") from exc

    def adapt(self) -> Dict[str, object]:
        """
        Adapt thresholds from all feedback records.

        Returns:
            Dict[str, object]: Adaptation summary.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/threshold_adapter.py::adapt")
        try:
            thresholds = self._load_base_thresholds()

            if not Config.FEEDBACK_UPDATES_CSV.exists():
                atomic_write_json(
                    {
                        "thresholds": thresholds,
                        "adjustment_step": self.adjustment_step,
                        "feedback_records_used": 0,
                    },
                    Config.ADAPTIVE_THRESHOLDS_PATH,
                )
                return {
                    "thresholds": thresholds,
                    "feedback_records_used": 0,
                }

            feedback_df = read_csv_required(Config.FEEDBACK_UPDATES_CSV)

            if feedback_df.empty:
                atomic_write_json(
                    {
                        "thresholds": thresholds,
                        "adjustment_step": self.adjustment_step,
                        "feedback_records_used": 0,
                    },
                    Config.ADAPTIVE_THRESHOLDS_PATH,
                )
                return {
                    "thresholds": thresholds,
                    "feedback_records_used": 0,
                }

            used_count = 0

            for _, row in feedback_df.iterrows():
                feedback_label = str(row.get("feedback_label", "uncertain"))

                if feedback_label not in Config.FEEDBACK_LABELS:
                    logger.warning("Skipping invalid feedback label: %s", feedback_label)
                    continue

                context_key = str(int(row.get("context_id", -1)))
                thresholds = self._adjust_context_thresholds(
                    thresholds=thresholds,
                    context_key=context_key,
                    feedback_label=feedback_label,
                )
                used_count += 1

            atomic_write_json(
                {
                    "thresholds": thresholds,
                    "adjustment_step": self.adjustment_step,
                    "feedback_records_used": used_count,
                    "rules": {
                        "rejected_false_alarm": "increase threshold slightly",
                        "missed_anomaly": "decrease threshold slightly",
                        "accepted_alert": "keep threshold",
                        "uncertain": "no change",
                    },
                },
                Config.ADAPTIVE_THRESHOLDS_PATH,
            )

            feedback_df["feedback_status"] = "used_for_threshold_adaptation"
            atomic_write_csv(feedback_df, Config.FEEDBACK_UPDATES_CSV)

            logger.info("Threshold adaptation completed. feedback_records_used=%s", used_count)

            return {
                "thresholds": thresholds,
                "feedback_records_used": used_count,
            }

        except Exception as exc:
            logger.exception("Threshold adaptation failed.")
            raise RuntimeError("Threshold adaptation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run threshold adaptation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/threshold_adapter.py::run")
        try:
            result = self.adapt()

            return {
                "status": "success",
                "message": "Adaptive thresholds updated from feedback.",
                "output_file": str(Config.ADAPTIVE_THRESHOLDS_PATH),
                "records_count": int(result["feedback_records_used"]),
                "data": result,
            }

        except Exception as exc:
            logger.exception("Threshold adapter stage failed.")
            raise RuntimeError("Threshold adapter stage failed.") from exc


def run_threshold_adapter() -> Dict[str, object]:
    """
    Execute threshold adaptation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/threshold_adapter.py::run_threshold_adapter")
    adapter = ThresholdAdapter()
    return adapter.run()


if __name__ == "__main__":
    result = run_threshold_adapter()
    print(result)