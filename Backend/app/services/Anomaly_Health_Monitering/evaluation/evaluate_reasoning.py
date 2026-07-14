"""
Reasoning evaluation for CA-EDT-AHMA.

Metrics:
- Top sensor stability
- Root-cause pattern distribution
- Recurrence ratio
- Inspection focus completeness

Reads:
data/outputs/root_cause_analysis.csv
data/outputs/root_cause_memory.csv, if available
data/outputs/temporal_reasoning.csv, if available

Writes:
metrics/evaluate_reasoning.csv
reports/evaluate_reasoning_summary.json
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_reasoning.py")
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class ReasoningEvaluator:
    """
    Evaluates reasoning outputs.
    """

    def __init__(self) -> None:
        """
        Initialize reasoning evaluator.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_reasoning.py::__init__")
        Config.create_directories()

    def _load_optional_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """
        Load optional CSV.

        Args:
            path: CSV path.

        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_reasoning.py::_load_optional_csv")
        try:
            if path.exists():
                return read_csv_required(path)
            return None

        except Exception as exc:
            logger.exception("Failed to load optional reasoning CSV: %s", path)
            raise RuntimeError(f"Failed to load optional reasoning CSV: {path}") from exc

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate reasoning outputs.

        Returns:
            pd.DataFrame: Reasoning metrics.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_reasoning.py::evaluate")
        try:
            root_df = read_csv_required(Config.ROOT_CAUSE_CSV)

            memory_df = self._load_optional_csv(Config.OUTPUT_DIR / "root_cause_memory.csv")
            temporal_df = self._load_optional_csv(Config.OUTPUT_DIR / "temporal_reasoning.csv")

            records = []

            for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                split_df = root_df[root_df["split"] == split]

                if split_df.empty:
                    continue

                top_sensor_stability = float(split_df["top_sensor_1"].value_counts(normalize=True).max())
                inspection_focus_completeness = float(split_df["inspection_focus"].notna().mean())

                root_pattern_distribution = split_df["root_cause_pattern"].value_counts(normalize=True).to_dict()

                recurrence_ratio = 0.0
                if memory_df is not None and not memory_df.empty:
                    memory_split = memory_df[memory_df["split"] == split]
                    if not memory_split.empty and "root_cause_recurrence_status" in memory_split.columns:
                        recurrence_ratio = float(
                            (memory_split["root_cause_recurrence_status"] == "Recurring").mean()
                        )

                temporal_available_ratio = 0.0
                if temporal_df is not None and not temporal_df.empty:
                    temporal_split = temporal_df[temporal_df["split"] == split]
                    if not temporal_split.empty:
                        temporal_available_ratio = 1.0

                records.append(
                    {
                        "split": split,
                        "top_sensor_stability": top_sensor_stability,
                        "inspection_focus_completeness": inspection_focus_completeness,
                        "recurrence_ratio": recurrence_ratio,
                        "temporal_reasoning_available": temporal_available_ratio,
                        "dominant_root_cause_pattern": (
                            max(root_pattern_distribution, key=root_pattern_distribution.get)
                            if root_pattern_distribution
                            else "unknown"
                        ),
                    }
                )

            metrics_df = pd.DataFrame(records)

            logger.info("Reasoning evaluation completed. rows=%s", len(metrics_df))
            return metrics_df

        except Exception as exc:
            logger.exception("Reasoning evaluation failed.")
            raise RuntimeError("Reasoning evaluation failed.") from exc

    def summarize(self, metrics_df: pd.DataFrame) -> Dict[str, object]:
        """
        Summarize reasoning evaluation.

        Args:
            metrics_df: Metrics DataFrame.

        Returns:
            Dict[str, object]: Summary.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_reasoning.py::summarize")
        try:
            return {
                "status": "success",
                "average_top_sensor_stability": float(metrics_df["top_sensor_stability"].mean()),
                "average_inspection_focus_completeness": float(
                    metrics_df["inspection_focus_completeness"].mean()
                ),
                "average_recurrence_ratio": float(metrics_df["recurrence_ratio"].mean()),
                "note": "Reasoning support is conservative and does not make maintenance decisions.",
            }

        except Exception as exc:
            logger.exception("Reasoning summary generation failed.")
            raise RuntimeError("Reasoning summary generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run reasoning evaluation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_reasoning.py::run")
        try:
            metrics_df = self.evaluate()
            output_path: Path = Config.METRIC_DIR / "evaluate_reasoning.csv"
            atomic_write_csv(metrics_df, output_path)

            summary = self.summarize(metrics_df)
            atomic_write_json(summary, Config.REPORT_DIR / "evaluate_reasoning_summary.json")

            return {
                "status": "success",
                "message": "Reasoning evaluation completed.",
                "output_file": str(output_path),
                "records_count": len(metrics_df),
                "metrics": summary,
            }

        except Exception as exc:
            logger.exception("Reasoning evaluator stage failed.")
            raise RuntimeError("Reasoning evaluator stage failed.") from exc


def run_reasoning_evaluation() -> Dict[str, object]:
    """
    Execute reasoning evaluation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_reasoning.py::run_reasoning_evaluation")
    evaluator = ReasoningEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    result = run_reasoning_evaluation()
    print(result)