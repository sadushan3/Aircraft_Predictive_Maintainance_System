"""
Health monitoring evaluation for CA-EDT-AHMA.

Metrics:
- Health trend smoothness
- Health deterioration consistency
- Health state distribution
- Optional correlation with T columns if available

Important:
This module does not use Y_dev or Y_test.

Reads:
data/outputs/health_states.csv
data/processed/scaled_features.csv

Writes:
metrics/evaluate_health.csv
reports/evaluate_health_summary.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_health.py")
from pathlib import Path
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
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class HealthEvaluator:
    """
    Evaluates health monitoring outputs.
    """

    def __init__(self) -> None:
        """
        Initialize health evaluator.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_health.py::__init__")
        Config.create_directories()

    def _get_t_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get optional T health/degradation parameter columns.

        Args:
            df: DataFrame.

        Returns:
            List[str]: T columns.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_health.py::_get_t_columns")
        return [column for column in df.columns if column.startswith("T_")]

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate health monitoring outputs.

        Returns:
            pd.DataFrame: Health evaluation metrics.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_health.py::evaluate")
        try:
            health_df = read_csv_required(Config.HEALTH_STATES_CSV)
            scaled_df = read_csv_required(Config.SCALED_CSV)

            merge_columns = ["unit_id", "cycle", "split"]

            t_columns = self._get_t_columns(scaled_df)

            if t_columns:
                df = health_df.merge(
                    scaled_df[merge_columns + t_columns],
                    on=merge_columns,
                    how="left",
                )
            else:
                df = health_df.copy()

            records: List[Dict[str, object]] = []

            for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                split_df = df[df["split"] == split]

                if split_df.empty:
                    continue

                health_delta = (
                    split_df.sort_values(["unit_id", "cycle"])
                    .groupby("unit_id")["health_index"]
                    .diff()
                    .fillna(0.0)
                )

                smoothness = float(1.0 / (1.0 + np.mean(np.abs(health_delta))))
                deterioration_ratio = float((health_delta <= 0).mean())

                state_counts = split_df["health_state"].value_counts(normalize=True).to_dict()

                record: Dict[str, object] = {
                    "split": split,
                    "health_trend_smoothness": smoothness,
                    "health_deterioration_consistency": deterioration_ratio,
                    "healthy_ratio": float(state_counts.get("Healthy", 0.0)),
                    "degrading_ratio": float(state_counts.get("Degrading", 0.0)),
                    "warning_ratio": float(state_counts.get("Warning", 0.0)),
                    "critical_ratio": float(state_counts.get("Critical", 0.0)),
                    "optional_t_correlation_available": bool(t_columns),
                }

                for t_column in t_columns:
                    try:
                        corr = split_df["health_index"].corr(split_df[t_column])
                        record[f"correlation_health_index_{t_column}"] = (
                            0.0 if pd.isna(corr) else float(corr)
                        )
                    except Exception:
                        record[f"correlation_health_index_{t_column}"] = 0.0

                records.append(record)

            metrics_df = pd.DataFrame(records)

            logger.info("Health evaluation completed. rows=%s", len(metrics_df))
            return metrics_df

        except Exception as exc:
            logger.exception("Health evaluation failed.")
            raise RuntimeError("Health evaluation failed.") from exc

    def summarize(self, metrics_df: pd.DataFrame) -> Dict[str, object]:
        """
        Summarize health evaluation.

        Args:
            metrics_df: Metrics DataFrame.

        Returns:
            Dict[str, object]: Summary.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_health.py::summarize")
        try:
            return {
                "status": "success",
                "average_health_trend_smoothness": float(metrics_df["health_trend_smoothness"].mean()),
                "average_health_deterioration_consistency": float(
                    metrics_df["health_deterioration_consistency"].mean()
                ),
                "uses_rul_targets": False,
            }

        except Exception as exc:
            logger.exception("Health summary generation failed.")
            raise RuntimeError("Health summary generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run health evaluation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_health.py::run")
        try:
            metrics_df = self.evaluate()
            output_path: Path = Config.METRIC_DIR / "evaluate_health.csv"
            atomic_write_csv(metrics_df, output_path)

            summary = self.summarize(metrics_df)
            atomic_write_json(summary, Config.REPORT_DIR / "evaluate_health_summary.json")

            return {
                "status": "success",
                "message": "Health monitoring evaluation completed.",
                "output_file": str(output_path),
                "records_count": len(metrics_df),
                "metrics": summary,
            }

        except Exception as exc:
            logger.exception("Health evaluator stage failed.")
            raise RuntimeError("Health evaluator stage failed.") from exc


def run_health_evaluation() -> Dict[str, object]:
    """
    Execute health evaluation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_health.py::run_health_evaluation")
    evaluator = HealthEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    result = run_health_evaluation()
    print(result)