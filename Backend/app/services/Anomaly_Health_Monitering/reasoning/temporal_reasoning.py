"""
Temporal reasoning for CA-EDT-AHMA.

Role:
Analyze how anomaly severity and health evolve over time.

Reads:
data/outputs/health_states.csv
data/outputs/root_cause_analysis.csv

Writes:
data/outputs/temporal_reasoning.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/reasoning/temporal_reasoning.py")
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


class TemporalReasoning:
    """
    Temporal anomaly reasoning engine.
    """

    def __init__(self, window: int = 5) -> None:
        """
        Initialize temporal reasoning engine.

        Args:
            window: Rolling window for temporal features.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/temporal_reasoning.py::__init__")
        Config.create_directories()

        if window <= 1:
            raise ValueError("window must be greater than 1.")

        self.window = window

    def reason(self) -> pd.DataFrame:
        """
        Generate temporal reasoning output.

        Returns:
            pd.DataFrame: Temporal reasoning DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/temporal_reasoning.py::reason")
        try:
            health_df = read_csv_required(Config.HEALTH_STATES_CSV)
            root_df = read_csv_required(Config.ROOT_CAUSE_CSV)

            merge_columns = ["unit_id", "cycle", "split"]

            df = health_df.merge(
                root_df[
                    merge_columns
                    + [
                        "root_cause_pattern",
                        "top_sensor_1",
                        "top_sensor_2",
                        "top_sensor_3",
                        "inspection_focus",
                    ]
                ],
                on=merge_columns,
                how="left",
            )

            df = df.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            if "final_anomaly_score" not in df.columns:
                raise KeyError("final_anomaly_score is required for temporal reasoning.")
            if "health_index" not in df.columns:
                raise KeyError("health_index is required for temporal reasoning.")

            df["anomaly_score_rolling_mean"] = (
                df.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(lambda series: series.rolling(self.window, min_periods=1).mean())
            )

            df["anomaly_score_delta"] = (
                df.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(lambda series: series.diff().fillna(0.0))
            )

            df["health_index_delta"] = (
                df.groupby(["split", "unit_id"])["health_index"]
                .transform(lambda series: series.diff().fillna(0.0))
            )

            df["anomaly_persistence_score_temporal"] = (
                df.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(
                    lambda series: (
                        (series >= 0.40).astype(float)
                        .rolling(self.window, min_periods=1)
                        .mean()
                    )
                )
                .clip(0.0, 1.0)
            )

            df["temporal_pattern"] = df.apply(
                lambda row: self._classify_temporal_pattern(
                    anomaly_delta=float(row["anomaly_score_delta"]),
                    health_delta=float(row["health_index_delta"]),
                    persistence=float(row["anomaly_persistence_score_temporal"]),
                ),
                axis=1,
            )

            df["temporal_reasoning_text"] = df.apply(
                lambda row: (
                    f"Temporal pattern is {row['temporal_pattern']}. "
                    f"Recent anomaly persistence is "
                    f"{float(row['anomaly_persistence_score_temporal']):.2f}. "
                    f"Health index change from previous cycle is "
                    f"{float(row['health_index_delta']):.2f}."
                ),
                axis=1,
            )

            output_columns = [
                "unit_id",
                "cycle",
                "split",
                "final_anomaly_score",
                "health_index",
                "anomaly_score_rolling_mean",
                "anomaly_score_delta",
                "health_index_delta",
                "anomaly_persistence_score_temporal",
                "temporal_pattern",
                "temporal_reasoning_text",
                "root_cause_pattern",
                "inspection_focus",
            ]

            for column in output_columns:
                if column not in df.columns:
                    df[column] = np.nan

            result = df[output_columns].copy()

            logger.info("Temporal reasoning completed. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Temporal reasoning failed.")
            raise RuntimeError("Temporal reasoning failed.") from exc

    def _classify_temporal_pattern(
        self,
        anomaly_delta: float,
        health_delta: float,
        persistence: float,
    ) -> str:
        """
        Classify temporal anomaly pattern.

        Args:
            anomaly_delta: Change in anomaly score.
            health_delta: Change in health index.
            persistence: Recent anomaly persistence.

        Returns:
            str: Temporal pattern.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/temporal_reasoning.py::_classify_temporal_pattern")
        if persistence >= 0.8 and health_delta < -1.0:
            return "Persistent_Deterioration"
        if anomaly_delta > 0.05 and health_delta < 0.0:
            return "Increasing_Anomaly"
        if persistence >= 0.4:
            return "Intermittent_Anomaly"
        if anomaly_delta < -0.05 and health_delta >= 0.0:
            return "Recovering_Behaviour"
        return "Stable_Behaviour"

    def run(self) -> Dict[str, object]:
        """
        Run temporal reasoning.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/temporal_reasoning.py::run")
        try:
            result = self.reason()

            output_path: Path = Config.OUTPUT_DIR / "temporal_reasoning.csv"
            atomic_write_csv(result, output_path)

            return {
                "status": "success",
                "message": "Temporal reasoning completed.",
                "output_file": str(output_path),
                "records_count": len(result),
            }

        except Exception as exc:
            logger.exception("Temporal reasoning stage failed.")
            raise RuntimeError("Temporal reasoning stage failed.") from exc


def run_temporal_reasoning() -> Dict[str, object]:
    """
    Execute temporal reasoning.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/reasoning/temporal_reasoning.py::run_temporal_reasoning")
    service = TemporalReasoning()
    return service.run()


if __name__ == "__main__":
    result = run_temporal_reasoning()
    print(result)