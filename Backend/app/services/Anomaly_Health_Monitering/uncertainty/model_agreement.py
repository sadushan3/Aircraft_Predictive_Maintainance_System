"""
Model agreement calculator for CA-EDT-AHMA.

Role:
Estimate uncertainty from disagreement among:
- Random Forest Digital Twin
- XGBoost Digital Twin
- LightGBM Digital Twin

Formula:
model_disagreement = standard deviation across RF, XGBoost, and LightGBM predictions
model_agreement_score = 1 - normalized_model_disagreement

Reads:
data/outputs/rf_predictions.csv
data/outputs/xgb_predictions.csv
data/outputs/lgbm_predictions.csv

Writes:
data/outputs/model_agreement.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/uncertainty/model_agreement.py")
import gc
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
from app.utils.Anomaly_Health_Monitering.model_utils import normalize_min_max, safe_clip_01

logger = get_logger(__name__)


class ModelAgreementCalculator:
    """
    Calculates model agreement and disagreement among the three digital twins.
    """

    def __init__(self) -> None:
        """
        Initialize model agreement calculator.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/model_agreement.py::__init__")
        Config.create_directories()

    def _infer_sensors(self, rf_df: pd.DataFrame) -> List[str]:
        """
        Infer measured sensor targets from Random Forest prediction columns.

        Args:
            rf_df: Random Forest predictions DataFrame.

        Returns:
            List[str]: Sensor names.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/model_agreement.py::_infer_sensors")
        try:
            sensors = [
                column.replace("rf_predicted_", "")
                for column in rf_df.columns
                if column.startswith("rf_predicted_")
            ]

            if not sensors:
                raise ValueError("No Random Forest prediction columns found.")

            logger.info("Inferred %s target sensors for model agreement.", len(sensors))
            return sensors

        except Exception as exc:
            logger.exception("Sensor inference for model agreement failed.")
            raise RuntimeError("Sensor inference for model agreement failed.") from exc

    def calculate(self) -> pd.DataFrame:
        """
        Calculate model disagreement and agreement scores.

        Returns:
            pd.DataFrame: Model agreement DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/model_agreement.py::calculate")
        try:
            rf_df = read_csv_required(Config.RF_PREDICTIONS_CSV)
            xgb_df = read_csv_required(Config.XGB_PREDICTIONS_CSV)
            lgbm_df = read_csv_required(Config.LGBM_PREDICTIONS_CSV)

            merge_columns = ["unit_id", "cycle", "split"]

            df = rf_df.merge(xgb_df, on=merge_columns, how="left")
            df = df.merge(lgbm_df, on=merge_columns, how="left")

            sensors = self._infer_sensors(rf_df)
            model_disagreement = np.zeros(len(df), dtype=float)

            for sensor in sensors:
                rf_col = f"rf_predicted_{sensor}"
                xgb_col = f"xgb_predicted_{sensor}"
                lgbm_col = f"lgbm_predicted_{sensor}"

                missing = [
                    column
                    for column in [rf_col, xgb_col, lgbm_col]
                    if column not in df.columns
                ]

                if missing:
                    raise KeyError(f"Missing prediction columns for agreement calculation: {missing}")

                prediction_stack = df[[rf_col, xgb_col, lgbm_col]].to_numpy(copy=False)
                sensor_std = np.std(prediction_stack, axis=1)
                model_disagreement += sensor_std
                del prediction_stack, sensor_std

            model_disagreement = model_disagreement / float(len(sensors))
            normalized_disagreement = normalize_min_max(model_disagreement)

            result = df[merge_columns].copy()
            result["model_disagreement"] = model_disagreement
            result["normalized_model_disagreement"] = safe_clip_01(normalized_disagreement)
            result["model_agreement_score"] = safe_clip_01(1.0 - normalized_disagreement)
            result["uncertainty_from_model_disagreement"] = safe_clip_01(normalized_disagreement)

            del df, rf_df, xgb_df, lgbm_df
            gc.collect()

            logger.info("Model agreement calculation completed. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Model agreement calculation failed.")
            raise RuntimeError("Model agreement calculation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run model agreement stage.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/model_agreement.py::run")
        try:
            agreement_df = self.calculate()
            atomic_write_csv(agreement_df, Config.MODEL_AGREEMENT_CSV)

            return {
                "status": "success",
                "message": "Model agreement calculated from RF, XGBoost, and LightGBM predictions.",
                "output_file": str(Config.MODEL_AGREEMENT_CSV),
                "records_count": len(agreement_df),
            }

        except Exception as exc:
            logger.exception("Model agreement stage failed.")
            raise RuntimeError("Model agreement stage failed.") from exc


def run_model_agreement() -> Dict[str, object]:
    """
    Execute model agreement calculation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/model_agreement.py::run_model_agreement")
    calculator = ModelAgreementCalculator()
    return calculator.run()


if __name__ == "__main__":
    result = run_model_agreement()
    print(result)
