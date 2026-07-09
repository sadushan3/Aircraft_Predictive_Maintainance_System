"""
Data loading service for CA-EDT-AHMA.

This module converts the NASA N-CMAPSS HDF5 file into raw_data.csv.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/data_loader.py")
from typing import Dict

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.h5_reader import NCMAPSSH5Reader
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    HDF5 to CSV data loader.
    """

    def __init__(self) -> None:
        """
        Initialize loader.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/data_loader.py::__init__")
        Config.create_directories()
        self.reader = NCMAPSSH5Reader(Config.H5_FILE_PATH)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load HDF5 data into a DataFrame.

        Returns:
            pd.DataFrame: Raw combined dev/test data.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/data_loader.py::load_raw_data")
        try:
            logger.info("Starting raw data loading from HDF5.")
            keys = self.reader.inspect_keys()

            ignored_found = [group for group in Config.IGNORED_H5_GROUPS if group in keys]
            if ignored_found:
                logger.info("Ignoring RUL target groups: %s", ignored_found)

            raw_df = self.reader.load_all()

            if "split" not in raw_df.columns:
                raise KeyError("Loaded data must contain split column.")

            if "unit_id" not in raw_df.columns:
                raise KeyError("Loaded data must contain unit_id column.")

            if "cycle" not in raw_df.columns:
                raise KeyError("Loaded data must contain cycle column.")

            logger.info("Raw data loaded successfully with shape %s.", raw_df.shape)
            return raw_df

        except Exception as exc:
            logger.exception("Raw data loading failed.")
            raise RuntimeError("Raw data loading failed.") from exc

    def save_raw_data(self) -> Dict[str, object]:
        """
        Load and save raw_data.csv.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/data_loader.py::save_raw_data")
        try:
            raw_df = self.load_raw_data()
            atomic_write_csv(raw_df, Config.RAW_CSV)

            return {
                "status": "success",
                "message": "Raw data loaded from HDF5. Y_dev and Y_test ignored.",
                "output_file": str(Config.RAW_CSV),
                "records_count": len(raw_df),
            }

        except Exception as exc:
            logger.exception("Saving raw data failed.")
            raise RuntimeError("Saving raw data failed.") from exc


def run_data_loading() -> Dict[str, object]:
    """
    Execute data loading stage.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/data_loader.py::run_data_loading")
    loader = DataLoader()
    return loader.save_raw_data()


if __name__ == "__main__":
    result = run_data_loading()
    print(result)