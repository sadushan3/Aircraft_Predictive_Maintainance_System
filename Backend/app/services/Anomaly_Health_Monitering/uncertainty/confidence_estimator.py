"""
Confidence estimator for CA-EDT-AHMA.

Role:
Calculate confidence, reliability, and uncertainty.

Inputs:
- model_agreement_score
- context_confidence
- anomaly_persistence_score
- data_quality_score

Formula:
confidence =
0.35 * model_agreement_score +
0.25 * context_confidence +
0.25 * anomaly_persistence_score +
0.15 * data_quality_score

Outputs:
confidence_score
reliability_score
uncertainty_score

Reads:
data/outputs/model_agreement.csv
data/outputs/context_clusters.csv
data/outputs/health_states.csv
data/processed/scaled_features.csv

Writes:
data/outputs/confidence_scores.csv

Saves:
models/uncertainty/confidence_config.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/uncertainty/confidence_estimator.py")
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

from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.uncertainty.model_agreement import (
    ModelAgreementCalculator,
)
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class ConfidenceEstimator:
    """
    Estimates confidence, reliability, and uncertainty for alerts.
    """

    def __init__(self, data_quality_batch_size: int = 100_000) -> None:
        """
        Initialize confidence estimator.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/confidence_estimator.py::__init__")
        Config.create_directories()
        self.data_quality_batch_size = data_quality_batch_size

    def _calculate_data_quality_score(self, scaled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple data quality score per row.

        Args:
            scaled_df: Scaled features DataFrame.

        Returns:
            pd.DataFrame: Data quality DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/confidence_estimator.py::_calculate_data_quality_score")
        try:
            merge_columns = ["unit_id", "cycle", "split"]
            result = scaled_df[merge_columns].copy()

            numeric_columns = scaled_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [
                column
                for column in numeric_columns
                if column not in {"unit_id", "cycle"}
            ]

            if not numeric_columns:
                result["data_quality_score"] = 1.0
                return result

            missing_ratio = np.empty(len(scaled_df), dtype=float)
            infinite_ratio = np.empty(len(scaled_df), dtype=float)

            for start in range(0, len(scaled_df), self.data_quality_batch_size):
                end = min(start + self.data_quality_batch_size, len(scaled_df))
                batch = scaled_df.iloc[start:end][numeric_columns]
                missing_ratio[start:end] = batch.isna().mean(axis=1).to_numpy()
                infinite_ratio[start:end] = np.isinf(batch.to_numpy(copy=False)).mean(axis=1)
                del batch

            data_quality = 1.0 - missing_ratio - infinite_ratio
            result["data_quality_score"] = np.clip(data_quality, 0.0, 1.0)
            del missing_ratio, infinite_ratio, data_quality
            gc.collect()

            logger.info("Data quality scores calculated. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Data quality score calculation failed.")
            raise RuntimeError("Data quality score calculation failed.") from exc

    def estimate(self) -> pd.DataFrame:
        """
        Estimate confidence and uncertainty.

        Returns:
            pd.DataFrame: Confidence DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/confidence_estimator.py::estimate")
        try:
            if not Config.MODEL_AGREEMENT_CSV.exists():
                ModelAgreementCalculator().run()

            agreement_df = read_csv_required(Config.MODEL_AGREEMENT_CSV)
            context_df = read_csv_required(Config.CONTEXT_CSV)
            health_df = read_csv_required(Config.HEALTH_STATES_CSV)
            scaled_df = read_csv_required(Config.SCALED_CSV)

            data_quality_df = self._calculate_data_quality_score(scaled_df)

            merge_columns = ["unit_id", "cycle", "split"]

            df = agreement_df.merge(
                context_df[merge_columns + ["context_confidence"]],
                on=merge_columns,
                how="left",
            )

            df = df.merge(
                health_df[merge_columns + ["anomaly_persistence_score"]],
                on=merge_columns,
                how="left",
            )

            df = df.merge(
                data_quality_df,
                on=merge_columns,
                how="left",
            )

            fill_defaults = {
                "model_agreement_score": 0.0,
                "context_confidence": 0.0,
                "anomaly_persistence_score": 0.0,
                "data_quality_score": 1.0,
            }

            for column, default_value in fill_defaults.items():
                if column not in df.columns:
                    df[column] = default_value
                df[column] = df[column].fillna(default_value)

            weights = Config.CONFIDENCE_WEIGHTS

            df["confidence_score"] = (
                weights["model_agreement_score"] * df["model_agreement_score"]
                + weights["context_confidence"] * df["context_confidence"]
                + weights["anomaly_persistence_score"] * df["anomaly_persistence_score"]
                + weights["data_quality_score"] * df["data_quality_score"]
            ).clip(0.0, 1.0)

            df["uncertainty_score"] = (1.0 - df["confidence_score"]).clip(0.0, 1.0)

            df["reliability_score"] = (
                0.50 * df["confidence_score"]
                + 0.30 * df["model_agreement_score"]
                + 0.20 * df["data_quality_score"]
            ).clip(0.0, 1.0)

            output_columns = [
                "unit_id",
                "cycle",
                "split",
                "model_disagreement",
                "normalized_model_disagreement",
                "model_agreement_score",
                "uncertainty_from_model_disagreement",
                "context_confidence",
                "anomaly_persistence_score",
                "data_quality_score",
                "confidence_score",
                "uncertainty_score",
                "reliability_score",
            ]

            for column in output_columns:
                if column not in df.columns:
                    df[column] = np.nan

            confidence_df = df[output_columns].copy()
            del agreement_df, context_df, health_df, scaled_df, data_quality_df, df
            gc.collect()

            logger.info("Confidence estimation completed. rows=%s", len(confidence_df))
            return confidence_df

        except Exception as exc:
            logger.exception("Confidence estimation failed.")
            raise RuntimeError("Confidence estimation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run confidence estimation stage.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/confidence_estimator.py::run")
        try:
            confidence_df = self.estimate()

            atomic_write_json(
                {
                    "formula": (
                        "confidence = 0.35*model_agreement_score + "
                        "0.25*context_confidence + "
                        "0.25*anomaly_persistence_score + "
                        "0.15*data_quality_score"
                    ),
                    "weights": Config.CONFIDENCE_WEIGHTS,
                    "uncertainty_formula": "uncertainty_score = 1 - confidence_score",
                },
                Config.CONFIDENCE_CONFIG_PATH,
            )

            atomic_write_csv(confidence_df, Config.CONFIDENCE_CSV)

            return {
                "status": "success",
                "message": "Confidence, reliability, and uncertainty scores calculated.",
                "output_file": str(Config.CONFIDENCE_CSV),
                "records_count": len(confidence_df),
            }

        except Exception as exc:
            logger.exception("Confidence estimator stage failed.")
            raise RuntimeError("Confidence estimator stage failed.") from exc


def run_confidence_estimation() -> Dict[str, object]:
    """
    Execute confidence estimation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/uncertainty/confidence_estimator.py::run_confidence_estimation")
    estimator = ConfidenceEstimator()
    return estimator.run()


if __name__ == "__main__":
    result = run_confidence_estimation()
    print(result)
