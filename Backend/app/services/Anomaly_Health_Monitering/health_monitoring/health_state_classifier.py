"""
Health state classifier for CA-EDT-AHMA.

Rules:
85-100 = Healthy
65-84  = Degrading
40-64  = Warning
0-39   = Critical

Reads:
data/outputs/health_index.csv

Writes:
data/outputs/health_states.csv

Saves:
models/health/health_state_thresholds.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_state_classifier.py")
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
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import classify_health_state

logger = get_logger(__name__)


class HealthStateClassifier:
    """
    Classifies health state from health index.
    """

    def __init__(self) -> None:
        """
        Initialize health state classifier.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_state_classifier.py::__init__")
        Config.create_directories()

    def classify(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify health state.

        Args:
            health_df: Health index DataFrame.

        Returns:
            pd.DataFrame: DataFrame with health_state.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_state_classifier.py::classify")
        try:
            result = health_df.copy()

            if "health_index" not in result.columns:
                raise KeyError("health_index is required for health state classification.")

            result["health_state"] = result["health_index"].apply(
                lambda value: classify_health_state(float(value))
            )

            result["health_state_explanation"] = result["health_state"].map(
                {
                    "Healthy": "Health index is high and anomaly severity is low.",
                    "Degrading": "Health index shows early degradation signs.",
                    "Warning": "Health index indicates significant degradation behavior.",
                    "Critical": "Health index indicates severe anomaly behavior.",
                }
            )

            logger.info("Health state classification completed. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Health state classification failed.")
            raise RuntimeError("Health state classification failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run health state classification.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_state_classifier.py::run")
        try:
            health_df = read_csv_required(Config.HEALTH_INDEX_CSV)
            result = self.classify(health_df)

            atomic_write_json(
                {
                    "Healthy": {"min": 85, "max": 100},
                    "Degrading": {"min": 65, "max": 84},
                    "Warning": {"min": 40, "max": 64},
                    "Critical": {"min": 0, "max": 39},
                },
                Config.HEALTH_STATE_THRESHOLDS_PATH,
            )

            atomic_write_csv(result, Config.HEALTH_STATES_CSV)

            return {
                "status": "success",
                "message": "Health state classification completed.",
                "output_file": str(Config.HEALTH_STATES_CSV),
                "records_count": len(result),
            }

        except Exception as exc:
            logger.exception("Health state classifier stage failed.")
            raise RuntimeError("Health state classifier stage failed.") from exc


def run_health_state_classification() -> Dict[str, object]:
    """
    Execute health state classification.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_state_classifier.py::run_health_state_classification")
    classifier = HealthStateClassifier()
    return classifier.run()


if __name__ == "__main__":
    result = run_health_state_classification()
    print(result)