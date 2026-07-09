"""
Severity classifier for CA-EDT-AHMA.

Role:
Classify final anomaly score into:
Normal, Watch, Warning, Critical.

Reads:
data/outputs/anomaly_fusion.csv

Writes:
data/outputs/anomaly_fusion.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/severity_classifier.py")
from typing import Dict

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
from app.utils.Anomaly_Health_Monitering.model_utils import classify_alert

logger = get_logger(__name__)


class SeverityClassifier:
    """
    Final anomaly severity classifier.
    """

    def __init__(self) -> None:
        """
        Initialize severity classifier.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/severity_classifier.py::__init__")
        Config.create_directories()

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify anomaly severity.

        Args:
            df: DataFrame containing final_anomaly_score.

        Returns:
            pd.DataFrame: DataFrame with alert_level.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/severity_classifier.py::classify")
        try:
            result = df.copy()

            if "final_anomaly_score" not in result.columns:
                raise KeyError("final_anomaly_score is required for severity classification.")

            result["alert_level"] = result["final_anomaly_score"].apply(
                lambda score: classify_alert(float(score))
            )

            logger.info("Severity classification completed. rows=%s", len(result))
            return result

        except Exception as exc:
            logger.exception("Severity classification failed.")
            raise RuntimeError("Severity classification failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run severity classification.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/severity_classifier.py::run")
        try:
            fusion_df = read_csv_required(Config.ANOMALY_FUSION_CSV)
            result = self.classify(fusion_df)
            atomic_write_csv(result, Config.ANOMALY_FUSION_CSV)

            return {
                "status": "success",
                "message": "Severity classification completed.",
                "output_file": str(Config.ANOMALY_FUSION_CSV),
                "records_count": len(result),
            }

        except Exception as exc:
            logger.exception("Severity classifier stage failed.")
            raise RuntimeError("Severity classifier stage failed.") from exc


def run_severity_classification() -> Dict[str, object]:
    """
    Execute severity classification.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/anomaly_detection/severity_classifier.py::run_severity_classification")
    classifier = SeverityClassifier()
    return classifier.run()


if __name__ == "__main__":
    result = run_severity_classification()
    print(result)