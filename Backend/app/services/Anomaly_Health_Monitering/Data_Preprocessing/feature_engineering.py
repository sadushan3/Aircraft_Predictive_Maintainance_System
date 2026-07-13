"""
Feature engineering for CA-EDT-AHMA.

Reads:
data/processed/cleaned_data.csv

Writes:
data/processed/engineered_features.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/feature_engineering.py")
from typing import Dict, List

import numpy as np
import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv, read_csv_required
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_xs_columns

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering service.
    """

    def __init__(self) -> None:
        """
        Initialize feature engineering service.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/feature_engineering.py::__init__")
        Config.create_directories()

    def _add_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trajectory time index per unit and split.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with time index.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/feature_engineering.py::_add_time_index")
        result = df.copy()
        result["trajectory_index"] = result.groupby(["split", "unit_id"]).cumcount()
        return result

    def _add_rolling_features(self, df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """
        Add rolling mean and rolling std for measured sensors.

        Args:
            df: Input DataFrame.
            sensor_columns: Measured sensor columns.

        Returns:
            pd.DataFrame: DataFrame with rolling features.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/feature_engineering.py::_add_rolling_features")
        result = df.copy()
        selected_sensors = sensor_columns[: min(8, len(sensor_columns))]

        for column in selected_sensors:
            rolling_mean_col = f"{column}_rolling_mean_{Config.ROLLING_WINDOW}"
            rolling_std_col = f"{column}_rolling_std_{Config.ROLLING_WINDOW}"

            result[rolling_mean_col] = (
                result.groupby(["split", "unit_id"])[column]
                .transform(
                    lambda series: series.rolling(
                        window=Config.ROLLING_WINDOW,
                        min_periods=1,
                    ).mean()
                )
            )

            result[rolling_std_col] = (
                result.groupby(["split", "unit_id"])[column]
                .transform(
                    lambda series: series.rolling(
                        window=Config.ROLLING_WINDOW,
                        min_periods=1,
                    ).std()
                )
                .fillna(0.0)
            )

        return result

    def _add_trend_features(self, df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """
        Add first-difference trend features for measured sensors.

        Args:
            df: Input DataFrame.
            sensor_columns: Measured sensor columns.

        Returns:
            pd.DataFrame: DataFrame with trend features.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/feature_engineering.py::_add_trend_features")
        result = df.copy()
        selected_sensors = sensor_columns[: min(8, len(sensor_columns))]

        for column in selected_sensors:
            trend_col = f"{column}_trend"
            result[trend_col] = (
                result.groupby(["split", "unit_id"])[column]
                .transform(lambda series: series.diff().fillna(0.0))
            )

        return result

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features.

        Args:
            df: Cleaned DataFrame.

        Returns:
            pd.DataFrame: Engineered DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/feature_engineering.py::engineer")
        try:
            result = df.copy()
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            sensor_columns = get_xs_columns(result)
            if not sensor_columns:
                raise ValueError("No measured sensor columns found for feature engineering.")

            result = self._add_time_index(result)
            result = self._add_rolling_features(result, sensor_columns)
            result = self._add_trend_features(result, sensor_columns)

            result = result.replace([np.inf, -np.inf], np.nan)
            numeric_columns = result.select_dtypes(include=[np.number]).columns.tolist()
            for column in numeric_columns:
                result[column] = result[column].fillna(0.0)

            logger.info("Feature engineering completed with shape %s.", result.shape)
            return result

        except Exception as exc:
            logger.exception("Feature engineering failed.")
            raise RuntimeError("Feature engineering failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run feature engineering stage.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/feature_engineering.py::run")
        try:
            cleaned_df = read_csv_required(Config.CLEANED_CSV)
            engineered_df = self.engineer(cleaned_df)
            atomic_write_csv(engineered_df, Config.ENGINEERED_CSV)

            return {
                "status": "success",
                "message": "Feature engineering completed.",
                "output_file": str(Config.ENGINEERED_CSV),
                "records_count": len(engineered_df),
            }

        except Exception as exc:
            logger.exception("Feature engineering stage failed.")
            raise RuntimeError("Feature engineering stage failed.") from exc


def run_feature_engineering() -> Dict[str, object]:
    """
    Execute feature engineering stage.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/feature_engineering.py::run_feature_engineering")
    engineer = FeatureEngineer()
    return engineer.run()


if __name__ == "__main__":
    result = run_feature_engineering()
    print(result)