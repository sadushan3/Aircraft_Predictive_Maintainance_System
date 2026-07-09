"""
Anomaly detection evaluation for CA-EDT-AHMA.

Metrics:
- Alert distribution
- False-alarm proxy rate
- Detection persistence
- Early warning score summary

If true anomaly labels are later provided, this file can calculate:
- Precision
- Recall
- F1
- ROC-AUC

Current implementation remains label-safe and does not use Y_dev/Y_test.

Reads:
data/outputs/anomaly_fusion.csv
data/outputs/early_warning_scores.csv, if available

Writes:
metrics/evaluate_anomaly.csv
reports/evaluate_anomaly_summary.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_anomaly.py")
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class AnomalyEvaluator:
    """
    Evaluates anomaly detection outputs without requiring RUL targets.
    """

    def __init__(self) -> None:
        """
        Initialize anomaly evaluator.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_anomaly.py::__init__")
        Config.create_directories()

    def _load_optional_early_warning(self) -> Optional[pd.DataFrame]:
        """
        Load early warning score CSV if available.

        Returns:
            Optional[pd.DataFrame]: Early warning DataFrame or None.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_anomaly.py::_load_optional_early_warning")
        try:
            path = Config.OUTPUT_DIR / "early_warning_scores.csv"
            if path.exists():
                return read_csv_required(path)
            return None

        except Exception as exc:
            logger.exception("Failed to load optional early warning scores.")
            raise RuntimeError("Failed to load optional early warning scores.") from exc

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate anomaly detection outputs.

        Returns:
            pd.DataFrame: Anomaly evaluation metrics.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_anomaly.py::evaluate")
        try:
            anomaly_df = read_csv_required(Config.ANOMALY_FUSION_CSV)
            early_warning_df = self._load_optional_early_warning()

            records = []

            for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                split_df = anomaly_df[anomaly_df["split"] == split]

                if split_df.empty:
                    continue

                alert_counts = split_df["alert_level"].value_counts(normalize=True).to_dict()

                anomaly_mask = split_df["alert_level"].isin(["Watch", "Warning", "Critical"])

                persistence = (
                    split_df.sort_values(["unit_id", "cycle"])
                    .groupby("unit_id")["final_anomaly_score"]
                    .transform(lambda series: (series >= 0.40).astype(float).rolling(5, min_periods=1).mean())
                )

                record = {
                    "split": split,
                    "normal_ratio": float(alert_counts.get("Normal", 0.0)),
                    "watch_ratio": float(alert_counts.get("Watch", 0.0)),
                    "warning_ratio": float(alert_counts.get("Warning", 0.0)),
                    "critical_ratio": float(alert_counts.get("Critical", 0.0)),
                    "anomaly_alert_ratio": float(anomaly_mask.mean()),
                    "average_final_anomaly_score": float(split_df["final_anomaly_score"].mean()),
                    "average_anomaly_persistence_proxy": float(persistence.mean()),
                    "false_alarm_proxy_rate": float(
                        (
                            (split_df["alert_level"].isin(["Watch", "Warning", "Critical"]))
                            & (split_df["final_anomaly_score"] < 0.40)
                        ).mean()
                    ),
                    "true_labels_available": False,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "roc_auc": None,
                }

                if early_warning_df is not None:
                    ew_split = early_warning_df[early_warning_df["split"] == split]
                    if not ew_split.empty and "early_warning_score" in ew_split.columns:
                        record["average_early_warning_score"] = float(ew_split["early_warning_score"].mean())
                    else:
                        record["average_early_warning_score"] = 0.0
                else:
                    record["average_early_warning_score"] = 0.0

                records.append(record)

            metrics_df = pd.DataFrame(records)

            logger.info("Anomaly evaluation completed. rows=%s", len(metrics_df))
            return metrics_df

        except Exception as exc:
            logger.exception("Anomaly evaluation failed.")
            raise RuntimeError("Anomaly evaluation failed.") from exc

    def summarize(self, metrics_df: pd.DataFrame) -> Dict[str, object]:
        """
        Summarize anomaly evaluation.

        Args:
            metrics_df: Metrics DataFrame.

        Returns:
            Dict[str, object]: Summary.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_anomaly.py::summarize")
        try:
            return {
                "status": "success",
                "average_anomaly_alert_ratio": float(metrics_df["anomaly_alert_ratio"].mean()),
                "average_final_anomaly_score": float(metrics_df["average_final_anomaly_score"].mean()),
                "true_labels_available": False,
                "note": (
                    "Precision, recall, F1, and ROC-AUC require external anomaly labels. "
                    "Y_dev/Y_test are RUL targets and are intentionally not used."
                ),
            }

        except Exception as exc:
            logger.exception("Anomaly summary generation failed.")
            raise RuntimeError("Anomaly summary generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run anomaly evaluation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_anomaly.py::run")
        try:
            metrics_df = self.evaluate()
            output_path: Path = Config.METRIC_DIR / "evaluate_anomaly.csv"
            atomic_write_csv(metrics_df, output_path)

            summary = self.summarize(metrics_df)
            atomic_write_json(summary, Config.REPORT_DIR / "evaluate_anomaly_summary.json")

            return {
                "status": "success",
                "message": "Anomaly detection evaluation completed.",
                "output_file": str(output_path),
                "records_count": len(metrics_df),
                "metrics": summary,
            }

        except Exception as exc:
            logger.exception("Anomaly evaluator stage failed.")
            raise RuntimeError("Anomaly evaluator stage failed.") from exc


def run_anomaly_evaluation() -> Dict[str, object]:
    """
    Execute anomaly evaluation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_anomaly.py::run_anomaly_evaluation")
    evaluator = AnomalyEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    result = run_anomaly_evaluation()
    print(result)