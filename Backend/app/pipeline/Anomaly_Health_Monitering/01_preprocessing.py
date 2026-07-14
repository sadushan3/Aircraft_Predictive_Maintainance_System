"""
Preprocessing pipeline for CA-EDT-AHMA.

Stages:
1. HDF5 to raw_data.csv
2. cleaned_data.csv
3. engineered_features.csv
4. scaled_features.csv

Important:
Scaling fits only on dev split and transforms dev/test.
Y_dev and Y_test are ignored.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/pipeline/Anomaly_Health_Monitering/01_preprocessing.py")
from typing import Dict, List

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.cleaner import DataCleaner
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.data_loader import DataLoader
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.feature_engineering import FeatureEngineer
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.scaler import FeatureScaler
from app.services.Anomaly_Health_Monitering.Data_Preprocessing.sequence_generator import SequenceGenerator
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.utils import StageResult, run_stage_safely

logger = get_logger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize preprocessing pipeline.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/01_preprocessing.py::__init__")
        Config.create_directories()

    def run(self, include_sequences: bool = False) -> Dict[str, object]:
        """
        Run preprocessing pipeline safely.

        Args:
            include_sequences: Whether to also generate sequence index.

        Returns:
            Dict[str, object]: Pipeline result.
        """
        print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/01_preprocessing.py::run")
        try:
            stages = [
                ("data_loading", DataLoader().save_raw_data),
                ("cleaning", DataCleaner().run),
                ("feature_engineering", FeatureEngineer().run),
                ("dev_only_scaling", FeatureScaler().run),
            ]

            if include_sequences:
                stages.append(("sequence_generation", SequenceGenerator().run))

            completed: List[Dict[str, object]] = []
            failed: List[Dict[str, object]] = []

            for stage_name, stage_function in stages:
                result: StageResult = run_stage_safely(stage_name, stage_function)
                completed.append(result.__dict__)

                if result.status == "failed":
                    failed.append(result.__dict__)
                    break

            status = "success" if not failed else "partial_failure"

            summary = {
                "status": status,
                "message": (
                    "Preprocessing pipeline completed."
                    if status == "success"
                    else "Preprocessing stopped safely. Previous outputs were not deleted."
                ),
                "completed_stages": completed,
                "failed_stages": failed,
                "final_output_file": str(Config.SCALED_CSV) if Config.SCALED_CSV.exists() else None,
            }

            atomic_write_json(summary, Config.REPORT_DIR / "01_preprocessing_summary.json")
            logger.info("Preprocessing pipeline finished with status=%s.", status)
            return summary

        except Exception as exc:
            logger.exception("Preprocessing pipeline failed.")
            raise RuntimeError("Preprocessing pipeline failed.") from exc


def run_preprocessing_pipeline(include_sequences: bool = False) -> Dict[str, object]:
    """
    Execute preprocessing pipeline.

    Args:
        include_sequences: Whether to generate optional sequence index.

    Returns:
        Dict[str, object]: Pipeline result.
    """
    print("[PROGRESS] Entering Backend/app/pipeline/Anomaly_Health_Monitering/01_preprocessing.py::run_preprocessing_pipeline")
    pipeline = PreprocessingPipeline()
    return pipeline.run(include_sequences=include_sequences)


if __name__ == "__main__":
    result = run_preprocessing_pipeline(include_sequences=False)
    print(result)