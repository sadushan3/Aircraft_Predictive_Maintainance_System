"""
Feature scaling for CA-EDT-AHMA.

Reads:
data/processed/engineered_features.csv

Writes:
data/processed/scaled_features.csv

Important:
Scaler is fitted only on the dev split.
The test split is transformed using the dev-fitted scaler.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/scaler.py")
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_save_joblib,
    atomic_write_csv,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_feature_columns

logger = get_logger(__name__)


class FeatureScaler:
    """
    Dev-only fitted feature scaler.
    """

    def __init__(self) -> None:
        """
        Initialize scaler service.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/scaler.py::__init__")
        Config.create_directories()

    def get_scalable_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get columns that should be scaled.

        Args:
            df: Engineered DataFrame.

        Returns:
            List[str]: Scalable feature columns.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/scaler.py::get_scalable_columns")
        try:
            candidate_columns = get_feature_columns(
                df,
                (
                    "W_",
                    "Xs_",
                    "X_s_",
                    "Xv_",
                    "X_v_",
                ),
            )

            engineered_columns = [
                column
                for column in df.columns
                if (
                    "_rolling_mean_" in column
                    or "_rolling_std_" in column
                    or column.endswith("_trend")
                    or column == "trajectory_index"
                )
            ]

            scalable_columns = list(dict.fromkeys(candidate_columns + engineered_columns))

            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            scalable_columns = [
                column
                for column in scalable_columns
                if column in numeric_columns and column not in {"unit_id", "cycle"}
            ]

            if not scalable_columns:
                raise ValueError("No scalable columns found.")

            logger.info("Selected %s scalable columns.", len(scalable_columns))
            return scalable_columns

        except Exception as exc:
            logger.exception("Failed to identify scalable columns.")
            raise RuntimeError("Failed to identify scalable columns.") from exc

    def fit_transform_dev_transform_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler on dev and transform both dev and test.

        Args:
            df: Engineered DataFrame.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/scaler.py::fit_transform_dev_transform_test")
        try:
            result = df.copy()
            scalable_columns = self.get_scalable_columns(result)

            dev_mask = result["split"] == Config.DEV_SPLIT_NAME

            if dev_mask.sum() == 0:
                raise ValueError("No dev rows found. Cannot fit scaler.")

            scaler = StandardScaler()
            scaler.fit(result.loc[dev_mask, scalable_columns])

            scaled_values = scaler.transform(result[scalable_columns])

            for index, column in enumerate(scalable_columns):
                result[column] = scaled_values[:, index]

            scaler_payload = {
                "scaler": scaler,
                "feature_columns": scalable_columns,
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "transform_only",
            }

            atomic_save_joblib(scaler_payload, Config.SCALER_PATH)

            logger.info(
                "Scaling completed. Fitted on dev rows=%s and transformed rows=%s.",
                int(dev_mask.sum()),
                len(result),
            )

            return result

        except Exception as exc:
            logger.exception("Scaling failed.")
            raise RuntimeError("Scaling failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run scaling stage.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/scaler.py::run")
        try:
            engineered_df = read_csv_required(Config.ENGINEERED_CSV)
            scaled_df = self.fit_transform_dev_transform_test(engineered_df)
            atomic_write_csv(scaled_df, Config.SCALED_CSV)

            return {
                "status": "success",
                "message": "Scaling completed with dev-only scaler fitting.",
                "output_file": str(Config.SCALED_CSV),
                "records_count": len(scaled_df),
            }

        except Exception as exc:
            logger.exception("Scaling stage failed.")
            raise RuntimeError("Scaling stage failed.") from exc


def run_scaling() -> Dict[str, object]:
    """
    Execute scaling stage.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/scaler.py::run_scaling")
    scaler = FeatureScaler()
    return scaler.run()


if __name__ == "__main__":
    result = run_scaling()
    print(result)