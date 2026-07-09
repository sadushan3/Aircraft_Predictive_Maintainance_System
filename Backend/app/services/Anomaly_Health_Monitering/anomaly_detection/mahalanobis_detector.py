"""
Mahalanobis distance detector for CA-EDT-AHMA.

Correct research version:
Input features:
1. raw residual_* columns
2. residual temporal resfeat_* columns

Training:
Fit mean/covariance/precision on dev anomaly features only.

Inference:
Score dev and test.

Important:
- Does not fit on test.
- Does not use Y_dev/Y_test.
- Uses LedoitWolf covariance shrinkage for stability.
"""

from __future__ import annotations

from typing import Dict, List

import os as _os
import sys as _sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", ".."))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_save_joblib,
    atomic_write_csv,
    load_joblib_required,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import normalize_min_max

logger = get_logger(__name__)


class MahalanobisDetector:
    """
    Mahalanobis residual distance detector.
    """

    def __init__(self, scoring_batch_size: int = 100_000) -> None:
        """
        Initialize Mahalanobis detector.
        """
        Config.create_directories()
        self.scoring_batch_size = scoring_batch_size

    def get_anomaly_feature_columns(self, residual_df: pd.DataFrame) -> List[str]:
        """
        Select anomaly feature columns.

        Uses:
        - residual_* raw residual columns
        - resfeat_* residual temporal features

        Excludes:
        - abs_residual_* threshold columns
        - metadata columns

        Args:
            residual_df: Residual DataFrame.

        Returns:
            List[str]: Anomaly feature columns.
        """
        try:
            residual_columns = [
                column
                for column in residual_df.columns
                if column.startswith("residual_")
            ]

            resfeat_columns = [
                column
                for column in residual_df.columns
                if column.startswith("resfeat_")
            ]

            feature_columns = residual_columns + resfeat_columns

            numeric_columns = residual_df.select_dtypes(include=[np.number]).columns.tolist()

            feature_columns = [
                column
                for column in feature_columns
                if column in numeric_columns
            ]

            if not feature_columns:
                raise ValueError(
                    "No anomaly feature columns found. Expected residual_* or resfeat_* columns."
                )

            print(f"[PROGRESS] Mahalanobis raw residual feature count: {len(residual_columns)}")
            print(f"[PROGRESS] Mahalanobis residual temporal feature count: {len(resfeat_columns)}")
            print(f"[PROGRESS] Mahalanobis total anomaly feature count: {len(feature_columns)}")

            return feature_columns

        except Exception as exc:
            logger.exception("Failed to select Mahalanobis anomaly features.")
            raise RuntimeError("Failed to select Mahalanobis anomaly features.") from exc

    def fit(self, residual_df: pd.DataFrame) -> Dict[str, object]:
        """
        Fit Mahalanobis parameters using dev anomaly features only.

        Args:
            residual_df: Residual DataFrame.

        Returns:
            Dict[str, object]: Parameter payload.
        """
        try:
            print("[TRAINING] Starting Mahalanobis detector fitting")
            print("[TRAINING] Fit split: dev only")
            print("[TRAINING] Features: residual_* + resfeat_*")

            feature_columns = self.get_anomaly_feature_columns(residual_df)

            dev_mask = residual_df["split"] == Config.DEV_SPLIT_NAME

            if dev_mask.sum() == 0:
                raise ValueError("No dev rows found. Cannot fit Mahalanobis detector.")

            x_dev = residual_df.loc[dev_mask, feature_columns].copy()
            x_dev = x_dev.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            print(f"[TRAINING] Mahalanobis X_dev shape: {x_dev.shape}")

            scaler = StandardScaler()
            x_dev_scaled = scaler.fit_transform(x_dev)

            covariance_model = LedoitWolf()
            print("[TRAINING] LedoitWolf covariance fit started")
            covariance_model.fit(x_dev_scaled)
            print("[TRAINING] LedoitWolf covariance fit completed")

            mean_vector = covariance_model.location_
            precision_matrix = covariance_model.precision_

            dev_distances = self._calculate_distances(
                values=x_dev_scaled,
                mean_vector=mean_vector,
                precision_matrix=precision_matrix,
            )

            threshold = float(np.percentile(dev_distances, 99.0))

            print(
                "[TRAINING] Dev Mahalanobis distance summary: "
                f"min={float(np.min(dev_distances))}, "
                f"max={float(np.max(dev_distances))}, "
                f"mean={float(np.mean(dev_distances))}, "
                f"p99_threshold={threshold}"
            )

            payload: Dict[str, object] = {
                "mean_vector": mean_vector,
                "precision_matrix": precision_matrix,
                "threshold": threshold,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "feature_type": "raw_residual_plus_residual_temporal_features",
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
            }

            print(f"[TRAINING] Saving Mahalanobis params to: {Config.MAHALANOBIS_PARAMS_PATH}")
            atomic_save_joblib(payload, Config.MAHALANOBIS_PARAMS_PATH)

            logger.info(
                "Mahalanobis parameters fitted on dev anomaly features only. rows=%s features=%s",
                int(dev_mask.sum()),
                len(feature_columns),
            )

            return payload

        except Exception as exc:
            logger.exception("Mahalanobis fitting failed.")
            raise RuntimeError("Mahalanobis fitting failed.") from exc

    def _calculate_distances(
        self,
        values: np.ndarray,
        mean_vector: np.ndarray,
        precision_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Mahalanobis distances efficiently.

        Args:
            values: Feature matrix.
            mean_vector: Mean vector.
            precision_matrix: Precision matrix.

        Returns:
            np.ndarray: Distance values.
        """
        try:
            delta = values - mean_vector
            squared_distances = np.einsum(
                "ij,jk,ik->i",
                delta,
                precision_matrix,
                delta,
            )
            squared_distances = np.maximum(squared_distances, 0.0)
            distances = np.sqrt(squared_distances)

            return distances.astype(float)

        except Exception as exc:
            logger.exception("Mahalanobis distance calculation failed.")
            raise RuntimeError("Mahalanobis distance calculation failed.") from exc

    def score(self, residual_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score dev and test residuals using saved dev-fitted parameters.

        Args:
            residual_df: Residual DataFrame.

        Returns:
            pd.DataFrame: Mahalanobis score DataFrame.
        """
        try:
            print("[PROGRESS] Starting Mahalanobis scoring")

            payload = load_joblib_required(Config.MAHALANOBIS_PARAMS_PATH)

            mean_vector = payload["mean_vector"]
            precision_matrix = payload["precision_matrix"]
            threshold = float(payload["threshold"])
            scaler: StandardScaler = payload["scaler"]
            feature_columns: List[str] = payload["feature_columns"]

            missing = [
                column
                for column in feature_columns
                if column not in residual_df.columns
            ]

            if missing:
                raise KeyError(f"Missing anomaly feature columns for Mahalanobis detector: {missing}")

            x_all = residual_df[feature_columns].copy()
            x_all = x_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            print(f"[PROGRESS] Mahalanobis scoring rows: {len(x_all)}")
            print(f"[PROGRESS] Mahalanobis scoring feature count: {len(feature_columns)}")

            distances = np.empty(len(x_all), dtype=float)

            for start in range(0, len(x_all), self.scoring_batch_size):
                end = min(start + self.scoring_batch_size, len(x_all))
                x_batch_scaled = scaler.transform(x_all.iloc[start:end])
                distances[start:end] = self._calculate_distances(
                    values=x_batch_scaled,
                    mean_vector=mean_vector,
                    precision_matrix=precision_matrix,
                )

            normalized_scores = normalize_min_max(distances)
            labels = (distances >= threshold).astype(int)

            result = residual_df[["unit_id", "cycle", "split"]].copy()
            result["mahalanobis_distance"] = distances
            result["mahalanobis_score"] = normalized_scores
            result["mahalanobis_anomaly_label"] = labels
            result["mahalanobis_threshold"] = threshold
            result["mahalanobis_feature_count"] = len(feature_columns)

            label_counts = result["mahalanobis_anomaly_label"].value_counts().to_dict()
            print(f"[PROGRESS] Mahalanobis label distribution: {label_counts}")
            print(
                "[PROGRESS] Mahalanobis score summary: "
                f"min={result['mahalanobis_score'].min()}, "
                f"max={result['mahalanobis_score'].max()}, "
                f"mean={result['mahalanobis_score'].mean()}"
            )

            logger.info("Mahalanobis scoring completed. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Mahalanobis scoring failed.")
            raise RuntimeError("Mahalanobis scoring failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run Mahalanobis detector.

        Returns:
            Dict[str, object]: Stage response.
        """
        try:
            residual_df = read_csv_required(Config.RESIDUALS_CSV)

            self.fit(residual_df)
            score_df = self.score(residual_df)

            print(f"[PROGRESS] Writing Mahalanobis scores to: {Config.MAHALANOBIS_CSV}")
            atomic_write_csv(score_df, Config.MAHALANOBIS_CSV)

            return {
                "status": "success",
                "message": (
                    "Mahalanobis scores generated using dev-fitted parameters. "
                    "Features include raw residuals and residual temporal features."
                ),
                "output_file": str(Config.MAHALANOBIS_CSV),
                "records_count": len(score_df),
            }

        except Exception as exc:
            logger.exception("Mahalanobis detector stage failed.")
            raise RuntimeError("Mahalanobis detector stage failed.") from exc


def run_mahalanobis_detection() -> Dict[str, object]:
    """
    Execute Mahalanobis detection.

    Returns:
        Dict[str, object]: Stage response.
    """
    detector = MahalanobisDetector()
    return detector.run()


if __name__ == "__main__":
    result = run_mahalanobis_detection()
    print(result)
