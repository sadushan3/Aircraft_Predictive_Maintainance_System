"""
Health index calculator for CA-EDT-AHMA.

Role:
Convert anomaly severity into health index from 0 to 100.

Formula:
health_index =
100
- 60 * final_anomaly_score
- 25 * residual_trend_score
- 15 * anomaly_persistence_score

The output is clipped between 0 and 100.

Important:
This module does not predict RUL.
This module does not use Y_dev or Y_test.

Reads:
data/outputs/anomaly_fusion.csv

Writes:
data/outputs/health_index.csv

Saves:
models/health/health_index_config.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_index_calculator.py")
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

logger = get_logger(__name__)


class HealthIndexCalculator:
    """
    Calculates health index and remaining health percentage.
    """

    def __init__(self, trend_window: int = 5) -> None:
        """
        Initialize health index calculator.

        Args:
            trend_window: Rolling window for residual/anomaly trend.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_index_calculator.py::__init__")
        Config.create_directories()

        if trend_window <= 1:
            raise ValueError("trend_window must be greater than 1.")

        self.trend_window = trend_window

    def calculate(self, anomaly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate health index.

        Args:
            anomaly_df: Anomaly fusion DataFrame.

        Returns:
            pd.DataFrame: Health index DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_index_calculator.py::calculate")
        try:
            if "final_anomaly_score" not in anomaly_df.columns:
                raise KeyError("final_anomaly_score is required for health index calculation.")

            result = anomaly_df.copy()
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            result["residual_trend_score"] = (
                result.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(lambda series: series.rolling(self.trend_window, min_periods=1).mean())
                .clip(0.0, 1.0)
            )

            result["anomaly_persistence_score"] = (
                result.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(
                    lambda series: (
                        (series >= 0.40).astype(float)
                        .rolling(self.trend_window, min_periods=1)
                        .mean()
                    )
                )
                .clip(0.0, 1.0)
            )

            weights = Config.HEALTH_WEIGHTS

            result["health_index"] = (
                100.0
                - weights["final_anomaly_score"] * result["final_anomaly_score"]
                - weights["residual_trend_score"] * result["residual_trend_score"]
                - weights["anomaly_persistence_score"] * result["anomaly_persistence_score"]
            ).clip(0.0, 100.0)

            result["remaining_health_percentage"] = result["health_index"]

            health_columns = [
                "unit_id",
                "cycle",
                "split",
                "gmm_context_id",
                "final_anomaly_score",
                "alert_level",
                "residual_trend_score",
                "anomaly_persistence_score",
                "health_index",
                "remaining_health_percentage",
            ]

            for column in health_columns:
                if column not in result.columns:
                    result[column] = 0.0

            health_df = result[health_columns].copy()

            logger.info("Health index calculation completed. rows=%s", len(health_df))
            return health_df

        except Exception as exc:
            logger.exception("Health index calculation failed.")
            raise RuntimeError("Health index calculation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run health index calculation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_index_calculator.py::run")
        try:
            anomaly_df = read_csv_required(Config.ANOMALY_FUSION_CSV)
            health_df = self.calculate(anomaly_df)

            atomic_write_json(
                {
                    "formula": (
                        "health_index = 100 - 60*final_anomaly_score "
                        "- 25*residual_trend_score - 15*anomaly_persistence_score"
                    ),
                    "weights": Config.HEALTH_WEIGHTS,
                    "clip_range": [0, 100],
                    "rul_prediction": False,
                    "uses_y_targets": False,
                },
                Config.HEALTH_INDEX_CONFIG_PATH,
            )

            atomic_write_csv(health_df, Config.HEALTH_INDEX_CSV)

            return {
                "status": "success",
                "message": "Health index calculated without RUL prediction.",
                "output_file": str(Config.HEALTH_INDEX_CSV),
                "records_count": len(health_df),
            }

        except Exception as exc:
            logger.exception("Health index calculator stage failed.")
            raise RuntimeError("Health index calculator stage failed.") from exc


def run_health_index_calculation() -> Dict[str, object]:
    """
    Execute health index calculation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_index_calculator.py::run_health_index_calculation")
    calculator = HealthIndexCalculator()
    return calculator.run()


if __name__ == "__main__":
    result = run_health_index_calculation()
    print(result)