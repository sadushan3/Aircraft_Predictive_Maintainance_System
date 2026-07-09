"""
Digital Twin Evaluation Wrapper for CA-EDT-AHMA.

This wrapper calls the memory-safe TwinComparator.

Final evaluated models:
- Random Forest
- XGBoost
- LightGBM
- MLP Digital Twin
- 4-model Ensemble
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..")
    )
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.services.Anomaly_Health_Monitering.digital_twin.twin_comparator import (
    TwinComparator,
)
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class DigitalTwinEvaluator:
    """
    Memory-safe digital twin evaluator wrapper.

    Uses TwinComparator instead of full DataFrame merges.
    """

    def __init__(self) -> None:
        Config.create_directories()

    def run(self) -> Dict[str, object]:
        try:
            comparator = TwinComparator()
            response = comparator.run()

            summary = {
                "status": "success",
                "message": "Digital twin evaluation completed using memory-safe TwinComparator.",
                "models_evaluated": [
                    "random_forest",
                    "xgboost",
                    "lightgbm",
                    "mlp_digital_twin",
                    "ensemble",
                ],
                "target_type": "raw_X_s_only",
                "metrics_file": response["output_file"],
                "records_count": response["records_count"],
            }

            summary_path: Path = Config.REPORT_DIR / "evaluate_digital_twin_summary.json"
            atomic_write_json(summary, summary_path)

            return {
                "status": "success",
                "message": "Digital twin evaluation completed for all 4 models and ensemble.",
                "output_file": response["output_file"],
                "summary_file": str(summary_path),
                "records_count": response["records_count"],
                "metrics": summary,
            }

        except Exception as exc:
            logger.exception("Digital twin evaluator stage failed.")
            raise RuntimeError("Digital twin evaluator stage failed.") from exc


def run_digital_twin_evaluation() -> Dict[str, object]:
    evaluator = DigitalTwinEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    result = run_digital_twin_evaluation()
    print(result)