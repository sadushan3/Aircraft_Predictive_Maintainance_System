"""
Data cleaner for CA-EDT-AHMA.

Reads:
data/processed/raw_data.csv

Writes:
data/processed/cleaned_data.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py")
from typing import Dict, List

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


class DataCleaner:
    """
    Data cleaning service.
    """

    def __init__(self) -> None:
        """
        Initialize cleaner.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py::__init__")
        Config.create_directories()

    def _fill_numeric_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill numeric missing values split-wise using medians.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Filled DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py::_fill_numeric_missing_values")
        result = df.copy()
        numeric_columns = result.select_dtypes(include=[np.number]).columns.tolist()

        for column in numeric_columns:
            if result[column].isna().any():
                result[column] = result.groupby("split")[column].transform(
                    lambda series: series.fillna(series.median())
                )
                result[column] = result[column].fillna(result[column].median())
                result[column] = result[column].fillna(0.0)

        return result

    def _fill_non_numeric_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill non-numeric missing values.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Filled DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py::_fill_non_numeric_missing_values")
        result = df.copy()
        non_numeric_columns = result.select_dtypes(exclude=[np.number]).columns.tolist()

        for column in non_numeric_columns:
            if result[column].isna().any():
                result[column] = result[column].fillna("unknown")

        return result

    def _remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove constant numeric columns except essential metadata.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame without constant columns.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py::_remove_constant_columns")
        result = df.copy()
        metadata_columns = {"unit_id", "cycle"}
        numeric_columns = result.select_dtypes(include=[np.number]).columns.tolist()

        constant_columns: List[str] = []
        for column in numeric_columns:
            if column in metadata_columns:
                continue
            if result[column].nunique(dropna=True) <= 1:
                constant_columns.append(column)

        if constant_columns:
            result = result.drop(columns=constant_columns)
            logger.info("Removed constant columns: %s", constant_columns)

        return result

    def _coerce_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure metadata columns have correct types.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Metadata-corrected DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py::_coerce_metadata")
        result = df.copy()

        result["unit_id"] = pd.to_numeric(result["unit_id"], errors="coerce").fillna(1).astype(int)
        result["cycle"] = pd.to_numeric(result["cycle"], errors="coerce").fillna(0).astype(int)
        result["split"] = result["split"].astype(str)

        return result

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a raw DataFrame.

        Args:
            df: Raw DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py::clean")
        try:
            result = df.copy()
            result = result.replace([np.inf, -np.inf], np.nan)
            result = self._coerce_metadata(result)
            result = self._fill_numeric_missing_values(result)
            result = self._fill_non_numeric_missing_values(result)
            result = self._remove_constant_columns(result)
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            logger.info("Data cleaning completed with shape %s.", result.shape)
            return result

        except Exception as exc:
            logger.exception("Data cleaning failed.")
            raise RuntimeError("Data cleaning failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run cleaning stage.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py::run")
        try:
            raw_df = read_csv_required(Config.RAW_CSV)
            cleaned_df = self.clean(raw_df)
            atomic_write_csv(cleaned_df, Config.CLEANED_CSV)

            return {
                "status": "success",
                "message": "Data cleaning completed.",
                "output_file": str(Config.CLEANED_CSV),
                "records_count": len(cleaned_df),
            }

        except Exception as exc:
            logger.exception("Cleaning stage failed.")
            raise RuntimeError("Cleaning stage failed.") from exc


def run_cleaning() -> Dict[str, object]:
    """
    Execute cleaning stage.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/cleaner.py::run_cleaning")
    cleaner = DataCleaner()
    return cleaner.run()


if __name__ == "__main__":
    result = run_cleaning()
    print(result)