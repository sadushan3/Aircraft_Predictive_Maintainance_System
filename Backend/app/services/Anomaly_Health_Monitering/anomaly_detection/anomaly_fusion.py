"""
Anomaly fusion for CA-EDT-AHMA.

Role:
Combine residual anomaly score, Isolation Forest score, and Mahalanobis score.

Formula:
final_anomaly_score =
0.50 * residual_score +
0.30 * iforest_score +
0.20 * mahalanobis_score

Reads:
data/outputs/residual_anomaly_scores.csv
data/outputs/isolation_forest_scores.csv
data/outputs/mahalanobis_scores.csv

Writes:
data/outputs/anomaly_fusion.csv

Saves:
models/anomaly/fusion_weights.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/anomaly_fusion.py")
from typing import Dict

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.anomaly_detection.early_warning_score import EarlyWarningScore
from app.services.Anomaly_Health_Monitering.anomaly_detection.isolation_forest_detector import (
    IsolationForestDetector,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.mahalanobis_detector import (
    MahalanobisDetector,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.residual_anomaly_detector import (
    ResidualAnomalyDetector,
)
from app.services.Anomaly_Health_Monitering.anomaly_detection.severity_classifier import SeverityClassifier
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import classify_alert

logger = get_logger(__name__)


class AnomalyFusion:
    """
    Final anomaly fusion engine.
    """

    def __init__(self) -> None:
        """
        Initialize anomaly fusion engine.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/anomaly_fusion.py::__init__")
        Config.create_directories()

    def fuse(self) -> pd.DataFrame:
        """
        Fuse anomaly scores.

        Returns:
            pd.DataFrame: Fused anomaly DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/anomaly_fusion.py::fuse")
        try:
            residual_df = read_csv_required(Config.RESIDUAL_ANOMALY_CSV)
            iforest_df = read_csv_required(Config.IFOREST_CSV)
            mahalanobis_df = read_csv_required(Config.MAHALANOBIS_CSV)

            merge_columns = ["unit_id", "cycle", "split"]

            df = residual_df.merge(iforest_df, on=merge_columns, how="left")
            df = df.merge(mahalanobis_df, on=merge_columns, how="left")

            weights = Config.FUSION_WEIGHTS

            df["final_anomaly_score"] = (
                weights["residual"] * df["residual_anomaly_score"]
                + weights["iforest"] * df["iforest_anomaly_score"]
                + weights["mahalanobis"] * df["mahalanobis_score"]
            ).clip(0.0, 1.0)

            df["alert_level"] = df["final_anomaly_score"].apply(
                lambda value: classify_alert(float(value))
            )

            atomic_write_json(
                {
                    "weights": weights,
                    "formula": (
                        "final_anomaly_score = 0.50*residual + "
                        "0.30*iforest + 0.20*mahalanobis"
                    ),
                },
                Config.FUSION_WEIGHTS_PATH,
            )

            logger.info("Anomaly fusion completed. rows=%s", len(df))
            return df

        except Exception as exc:
            logger.exception("Anomaly fusion failed.")
            raise RuntimeError("Anomaly fusion failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run anomaly fusion stage.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/anomaly_fusion.py::run")
        try:
            fusion_df = self.fuse()
            atomic_write_csv(fusion_df, Config.ANOMALY_FUSION_CSV)

            SeverityClassifier().classify(fusion_df)
            EarlyWarningScore().run()

            return {
                "status": "success",
                "message": "Anomaly fusion completed.",
                "output_file": str(Config.ANOMALY_FUSION_CSV),
                "records_count": len(fusion_df),
            }

        except Exception as exc:
            logger.exception("Anomaly fusion stage failed.")
            raise RuntimeError("Anomaly fusion stage failed.") from exc

    def run_all_detectors_and_fusion(self) -> Dict[str, object]:
        """
        Run residual, Isolation Forest, Mahalanobis, and fusion stages.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/anomaly_fusion.py::run_all_detectors_and_fusion")
        try:
            ResidualAnomalyDetector().run()
            IsolationForestDetector().run()
            MahalanobisDetector().run()
            result = self.run()
            return result

        except Exception as exc:
            logger.exception("Full anomaly detection stage failed.")
            raise RuntimeError("Full anomaly detection stage failed.") from exc


def run_anomaly_fusion() -> Dict[str, object]:
    """
    Execute anomaly fusion.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/anomaly_fusion.py::run_anomaly_fusion")
    service = AnomalyFusion()
    return service.run()


def run_full_anomaly_detection() -> Dict[str, object]:
    """
    Execute all anomaly detection stages.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/anomaly_fusion.py::run_full_anomaly_detection")
    service = AnomalyFusion()
    return service.run_all_detectors_and_fusion()


if __name__ == "__main__":
    result = run_full_anomaly_detection()
    print(result)