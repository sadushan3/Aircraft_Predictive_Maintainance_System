"""
Early warning score for CA-EDT-AHMA.

Role:
Calculate early warning behavior from recent anomaly score trend.

Reads:
data/outputs/anomaly_fusion.csv

Writes:
data/outputs/early_warning_scores.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/early_warning_score.py")
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


class EarlyWarningScore:
    """
    Early warning score calculator.
    """

    def __init__(self, rolling_window: int = 5) -> None:
        """
        Initialize early warning calculator.

        Args:
            rolling_window: Rolling window for trend calculation.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/early_warning_score.py::__init__")
        Config.create_directories()

        if rolling_window <= 1:
            raise ValueError("rolling_window must be greater than 1.")

        self.rolling_window = rolling_window

    def calculate(self, fusion_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate early warning scores.

        Args:
            fusion_df: Anomaly fusion DataFrame.

        Returns:
            pd.DataFrame: Early warning DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/early_warning_score.py::calculate")
        try:
            if "final_anomaly_score" not in fusion_df.columns:
                raise KeyError("final_anomaly_score is required for early warning calculation.")

            result = fusion_df[["unit_id", "cycle", "split", "final_anomaly_score", "alert_level"]].copy()
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            result["rolling_anomaly_mean"] = (
                result.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(lambda s: s.rolling(self.rolling_window, min_periods=1).mean())
            )

            result["rolling_anomaly_slope"] = (
                result.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(lambda s: s.diff().rolling(self.rolling_window, min_periods=1).mean())
                .fillna(0.0)
            )

            result["early_warning_score"] = (
                0.7 * result["rolling_anomaly_mean"]
                + 0.3 * np.clip(result["rolling_anomaly_slope"], 0.0, 1.0)
            ).clip(0.0, 1.0)

            result["early_warning_label"] = np.where(
                result["early_warning_score"] >= 0.65,
                "Increasing_Risk",
                np.where(
                    result["early_warning_score"] >= 0.40,
                    "Watch_Risk",
                    "Stable",
                ),
            )

            logger.info("Early warning score calculation completed. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Early warning score calculation failed.")
            raise RuntimeError("Early warning score calculation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run early warning score calculation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/early_warning_score.py::run")
        try:
            fusion_df = read_csv_required(Config.ANOMALY_FUSION_CSV)
            result = self.calculate(fusion_df)

            output_path: Path = Config.OUTPUT_DIR / "early_warning_scores.csv"
            atomic_write_csv(result, output_path)

            return {
                "status": "success",
                "message": "Early warning scores generated.",
                "output_file": str(output_path),
                "records_count": len(result),
            }

        except Exception as exc:
            logger.exception("Early warning score stage failed.")
            raise RuntimeError("Early warning score stage failed.") from exc


def run_early_warning_score() -> Dict[str, object]:
    """
    Execute early warning score stage.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/early_warning_score.py::run_early_warning_score")
    service = EarlyWarningScore()
    return service.run()


if __name__ == "__main__":
    result = run_early_warning_score()
    print(result)