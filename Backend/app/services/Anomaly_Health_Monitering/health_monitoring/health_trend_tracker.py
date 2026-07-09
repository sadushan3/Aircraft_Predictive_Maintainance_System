"""
Health trend tracker for CA-EDT-AHMA.

Role:
Track health deterioration trend over time.

Reads:
data/outputs/health_states.csv

Writes:
data/outputs/health_trends.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_trend_tracker.py")
from pathlib import Path
from typing import Dict

import numpy as np
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


class HealthTrendTracker:
    """
    Tracks health trend by unit and split.
    """

    def __init__(self, window: int = 5) -> None:
        """
        Initialize health trend tracker.

        Args:
            window: Rolling window.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_trend_tracker.py::__init__")
        Config.create_directories()

        if window <= 1:
            raise ValueError("window must be greater than 1.")

        self.window = window

    def track(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Track health trends.

        Args:
            health_df: Health states DataFrame.

        Returns:
            pd.DataFrame: Health trend DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_trend_tracker.py::track")
        try:
            if "health_index" not in health_df.columns:
                raise KeyError("health_index is required for health trend tracking.")

            result = health_df.copy()
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            result["health_index_rolling_mean"] = (
                result.groupby(["split", "unit_id"])["health_index"]
                .transform(lambda series: series.rolling(self.window, min_periods=1).mean())
            )

            result["health_index_delta"] = (
                result.groupby(["split", "unit_id"])["health_index"]
                .transform(lambda series: series.diff().fillna(0.0))
            )

            result["health_deterioration_score"] = (
                result.groupby(["split", "unit_id"])["health_index_delta"]
                .transform(
                    lambda series: (
                        series.apply(lambda value: abs(value) if value < 0 else 0.0)
                        .rolling(self.window, min_periods=1)
                        .mean()
                    )
                )
            )

            max_deterioration = float(result["health_deterioration_score"].max())
            if max_deterioration > 1e-12:
                result["health_deterioration_score"] = (
                    result["health_deterioration_score"] / max_deterioration
                ).clip(0.0, 1.0)
            else:
                result["health_deterioration_score"] = 0.0

            result["health_trend_label"] = np.where(
                result["health_index_delta"] < -2.0,
                "Deteriorating",
                np.where(
                    result["health_index_delta"] > 2.0,
                    "Recovering",
                    "Stable",
                ),
            )

            output_columns = [
                "unit_id",
                "cycle",
                "split",
                "health_index",
                "health_state",
                "health_index_rolling_mean",
                "health_index_delta",
                "health_deterioration_score",
                "health_trend_label",
            ]

            for column in output_columns:
                if column not in result.columns:
                    result[column] = np.nan

            trend_df = result[output_columns].copy()

            logger.info("Health trend tracking completed. rows=%s", len(trend_df))
            return trend_df

        except Exception as exc:
            logger.exception("Health trend tracking failed.")
            raise RuntimeError("Health trend tracking failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run health trend tracking.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_trend_tracker.py::run")
        try:
            health_df = read_csv_required(Config.HEALTH_STATES_CSV)
            trend_df = self.track(health_df)

            output_path: Path = Config.OUTPUT_DIR / "health_trends.csv"
            atomic_write_csv(trend_df, output_path)

            return {
                "status": "success",
                "message": "Health trend tracking completed.",
                "output_file": str(output_path),
                "records_count": len(trend_df),
            }

        except Exception as exc:
            logger.exception("Health trend tracker stage failed.")
            raise RuntimeError("Health trend tracker stage failed.") from exc


def run_health_trend_tracking() -> Dict[str, object]:
    """
    Execute health trend tracking.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/health_monitoring/health_trend_tracker.py::run_health_trend_tracking")
    tracker = HealthTrendTracker()
    return tracker.run()


if __name__ == "__main__":
    result = run_health_trend_tracking()
    print(result)