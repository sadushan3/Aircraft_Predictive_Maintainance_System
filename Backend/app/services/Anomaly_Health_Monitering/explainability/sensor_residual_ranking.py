"""
Sensor residual ranking for CA-EDT-AHMA.

Role:
Identify top contributing measured sensors based on absolute residuals.

Formula:
sensor_contribution = abs(sensor_residual) / sum(abs(all_sensor_residuals))

Reads:
data/outputs/residuals.csv

Writes:
data/outputs/sensor_residual_ranking.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/explainability/sensor_residual_ranking.py")
from pathlib import Path
from typing import Dict, List

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
from app.utils.Anomaly_Health_Monitering.model_utils import get_abs_residual_columns

logger = get_logger(__name__)


class SensorResidualRanking:
    """
    Ranks sensor residual contributions.
    """

    def __init__(self) -> None:
        """
        Initialize sensor residual ranking service.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/sensor_residual_ranking.py::__init__")
        Config.create_directories()

    def rank(self, residual_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank top sensor residuals for each row.

        Args:
            residual_df: Residual DataFrame.

        Returns:
            pd.DataFrame: Ranking DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/sensor_residual_ranking.py::rank")
        try:
            abs_residual_columns = get_abs_residual_columns(residual_df)

            if not abs_residual_columns:
                raise ValueError("No absolute residual columns found for sensor ranking.")

            records: List[Dict[str, object]] = []

            for _, row in residual_df.iterrows():
                abs_values = row[abs_residual_columns].astype(float)
                total_abs_residual = float(abs_values.sum())

                if total_abs_residual <= 1e-12:
                    contributions = pd.Series(0.0, index=abs_residual_columns)
                else:
                    contributions = abs_values / total_abs_residual

                top = contributions.sort_values(ascending=False).head(5)

                record: Dict[str, object] = {
                    "unit_id": row["unit_id"],
                    "cycle": row["cycle"],
                    "split": row["split"],
                    "total_abs_residual": total_abs_residual,
                }

                for index in range(5):
                    sensor_key = f"rank_{index + 1}_sensor"
                    contribution_key = f"rank_{index + 1}_contribution"

                    if index < len(top):
                        sensor_name = top.index[index].replace("abs_residual_", "")
                        contribution = float(top.iloc[index])
                    else:
                        sensor_name = "none"
                        contribution = 0.0

                    record[sensor_key] = sensor_name
                    record[contribution_key] = contribution

                records.append(record)

            ranking_df = pd.DataFrame(records)
            logger.info("Sensor residual ranking completed. rows=%s", len(ranking_df))
            return ranking_df

        except Exception as exc:
            logger.exception("Sensor residual ranking failed.")
            raise RuntimeError("Sensor residual ranking failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run sensor residual ranking.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/sensor_residual_ranking.py::run")
        try:
            residual_df = read_csv_required(Config.RESIDUALS_CSV)
            ranking_df = self.rank(residual_df)

            output_path: Path = Config.OUTPUT_DIR / "sensor_residual_ranking.csv"
            atomic_write_csv(ranking_df, output_path)

            return {
                "status": "success",
                "message": "Sensor residual ranking generated.",
                "output_file": str(output_path),
                "records_count": len(ranking_df),
            }

        except Exception as exc:
            logger.exception("Sensor residual ranking stage failed.")
            raise RuntimeError("Sensor residual ranking stage failed.") from exc


def run_sensor_residual_ranking() -> Dict[str, object]:
    """
    Execute sensor residual ranking.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/sensor_residual_ranking.py::run_sensor_residual_ranking")
    ranking = SensorResidualRanking()
    return ranking.run()


if __name__ == "__main__":
    result = run_sensor_residual_ranking()
    print(result)