"""
Sequence generator for trajectory-aware analysis.

This module does not change the main CSV flow. It provides optional
windowed sequences for temporal evaluation or future sequence-based models.

Reads:
data/processed/scaled_features.csv

Writes:
data/outputs/sequence_index.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/sequence_generator.py")
from pathlib import Path
from typing import Dict, List, Tuple

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
from app.utils.Anomaly_Health_Monitering.model_utils import get_w_columns, get_xs_columns, get_xv_columns

logger = get_logger(__name__)


class SequenceGenerator:
    """
    Generates rolling sequence windows by unit and split.
    """

    def __init__(self, window_size: int = 20, stride: int = 1) -> None:
        """
        Initialize sequence generator.

        Args:
            window_size: Number of time steps in each window.
            stride: Step size between windows.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/sequence_generator.py::__init__")
        Config.create_directories()
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.window_size = window_size
        self.stride = stride

    def get_sequence_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature columns for sequence generation.

        Args:
            df: Source DataFrame.

        Returns:
            List[str]: Sequence feature columns.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/sequence_generator.py::get_sequence_feature_columns")
        columns = get_w_columns(df) + get_xv_columns(df) + get_xs_columns(df)
        columns = list(dict.fromkeys(columns))

        if not columns:
            raise ValueError("No sequence feature columns found.")

        return columns

    def build_sequence_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a sequence index table without storing large arrays in CSV.

        Args:
            df: Scaled feature DataFrame.

        Returns:
            pd.DataFrame: Sequence index.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/sequence_generator.py::build_sequence_index")
        try:
            records: List[Dict[str, object]] = []
            sorted_df = df.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            for (split, unit_id), group in sorted_df.groupby(["split", "unit_id"]):
                group = group.sort_values("cycle").reset_index()
                n_rows = len(group)

                if n_rows < self.window_size:
                    continue

                for start in range(0, n_rows - self.window_size + 1, self.stride):
                    end = start + self.window_size - 1
                    records.append(
                        {
                            "split": split,
                            "unit_id": int(unit_id),
                            "sequence_start_cycle": int(group.loc[start, "cycle"]),
                            "sequence_end_cycle": int(group.loc[end, "cycle"]),
                            "start_row_index": int(group.loc[start, "index"]),
                            "end_row_index": int(group.loc[end, "index"]),
                            "window_size": self.window_size,
                            "stride": self.stride,
                        }
                    )

            sequence_index = pd.DataFrame(records)
            logger.info("Generated sequence index with %s windows.", len(sequence_index))
            return sequence_index

        except Exception as exc:
            logger.exception("Sequence index generation failed.")
            raise RuntimeError("Sequence index generation failed.") from exc

    def generate_numpy_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate in-memory numpy sequences and matching metadata.

        Args:
            df: Scaled feature DataFrame.

        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Sequence array and metadata.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/sequence_generator.py::generate_numpy_sequences")
        try:
            feature_columns = self.get_sequence_feature_columns(df)
            sorted_df = df.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            sequences: List[np.ndarray] = []
            metadata_records: List[Dict[str, object]] = []

            for (split, unit_id), group in sorted_df.groupby(["split", "unit_id"]):
                group = group.sort_values("cycle").reset_index(drop=True)
                values = group[feature_columns].values

                if len(group) < self.window_size:
                    continue

                for start in range(0, len(group) - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    sequence = values[start:end]
                    sequences.append(sequence)
                    metadata_records.append(
                        {
                            "split": split,
                            "unit_id": int(unit_id),
                            "sequence_start_cycle": int(group.loc[start, "cycle"]),
                            "sequence_end_cycle": int(group.loc[end - 1, "cycle"]),
                        }
                    )

            if sequences:
                sequence_array = np.stack(sequences)
            else:
                sequence_array = np.empty((0, self.window_size, len(feature_columns)))

            metadata = pd.DataFrame(metadata_records)
            logger.info("Generated numpy sequence array with shape %s.", sequence_array.shape)

            return sequence_array, metadata

        except Exception as exc:
            logger.exception("Numpy sequence generation failed.")
            raise RuntimeError("Numpy sequence generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run sequence index generation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/sequence_generator.py::run")
        try:
            scaled_df = read_csv_required(Config.SCALED_CSV)
            sequence_index = self.build_sequence_index(scaled_df)

            output_path: Path = Config.OUTPUT_DIR / "sequence_index.csv"
            atomic_write_csv(sequence_index, output_path)

            return {
                "status": "success",
                "message": "Sequence index generated.",
                "output_file": str(output_path),
                "records_count": len(sequence_index),
            }

        except Exception as exc:
            logger.exception("Sequence generation stage failed.")
            raise RuntimeError("Sequence generation stage failed.") from exc


def run_sequence_generation() -> Dict[str, object]:
    """
    Execute sequence generation stage.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/sequence_generator.py::run_sequence_generation")
    generator = SequenceGenerator()
    return generator.run()


if __name__ == "__main__":
    result = run_sequence_generation()
    print(result)