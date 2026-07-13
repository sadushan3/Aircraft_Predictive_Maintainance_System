"""
HDF5 reader for NASA N-CMAPSS dataset.

This module reads the required N-CMAPSS HDF5 groups:
A_dev, A_test, W_dev, W_test, X_s_dev, X_s_test, X_v_dev, X_v_test.

It optionally loads:
T_dev, T_test.

It intentionally ignores:
Y_dev, Y_test.

Reason:
This component is not responsible for RUL prediction.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py")
from pathlib import Path
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[4]))

from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class NCMAPSSH5Reader:
    """
    Reader for NASA N-CMAPSS HDF5 files.
    """

    def __init__(self, h5_path: Path = Config.H5_FILE_PATH) -> None:
        """
        Initialize HDF5 reader.

        Args:
            h5_path: Path to N-CMAPSS HDF5 file.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py::__init__")
        self.h5_path = h5_path

    def inspect_keys(self) -> List[str]:
        """
        Inspect all root keys inside the HDF5 file.

        Returns:
            List[str]: HDF5 root keys.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py::inspect_keys")
        try:
            if not self.h5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

            with h5py.File(self.h5_path, "r") as h5_file:
                keys = list(h5_file.keys())

            logger.info("HDF5 keys found: %s", keys)
            return keys

        except Exception as exc:
            logger.exception("Failed to inspect HDF5 keys.")
            raise RuntimeError("Failed to inspect HDF5 keys.") from exc

    def validate_required_groups(self) -> Dict[str, object]:
        """
        Validate required HDF5 groups.

        Returns:
            Dict[str, object]: Validation result.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py::validate_required_groups")
        try:
            keys = self.inspect_keys()
            missing = [group for group in Config.REQUIRED_H5_GROUPS if group not in keys]
            ignored_found = [group for group in Config.IGNORED_H5_GROUPS if group in keys]
            optional_found = [group for group in Config.OPTIONAL_H5_GROUPS if group in keys]

            if missing:
                raise KeyError(f"Missing required HDF5 groups: {missing}")

            logger.info("Required HDF5 groups validated successfully.")
            logger.info("Ignored Y groups found and ignored: %s", ignored_found)
            logger.info("Optional T groups found: %s", optional_found)

            return {
                "status": "success",
                "available_keys": keys,
                "missing_required": missing,
                "optional_found": optional_found,
                "ignored_found": ignored_found,
            }

        except Exception as exc:
            logger.exception("HDF5 group validation failed.")
            raise RuntimeError("HDF5 group validation failed.") from exc

    def _decode_variable_names(
        self,
        h5_file: h5py.File,
        variable_group: Optional[str],
        fallback_prefix: str,
        width: int,
    ) -> List[str]:
        """
        Decode variable names from HDF5 variable-name groups.

        Args:
            h5_file: Open HDF5 file.
            variable_group: Variable-name group.
            fallback_prefix: Fallback column prefix.
            width: Expected number of columns.

        Returns:
            List[str]: Decoded column names.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py::_decode_variable_names")
        try:
            if not variable_group or variable_group not in h5_file:
                return [f"{fallback_prefix}_{idx + 1}" for idx in range(width)]

            raw_names = np.array(h5_file[variable_group])
            decoded_names: List[str] = []

            for item in raw_names:
                if isinstance(item, bytes):
                    decoded_names.append(item.decode("utf-8").strip())
                elif isinstance(item, np.bytes_):
                    decoded_names.append(item.decode("utf-8").strip())
                elif isinstance(item, np.ndarray):
                    chars: List[str] = []
                    for char in item:
                        if isinstance(char, bytes):
                            chars.append(char.decode("utf-8"))
                        elif isinstance(char, np.bytes_):
                            chars.append(char.decode("utf-8"))
                        else:
                            chars.append(str(char))
                    decoded_names.append("".join(chars).strip())
                else:
                    decoded_names.append(str(item).strip())

            if len(decoded_names) != width:
                logger.warning(
                    "Variable group %s length mismatch. Using fallback names.",
                    variable_group,
                )
                return [f"{fallback_prefix}_{idx + 1}" for idx in range(width)]

            cleaned_names: List[str] = []
            for idx, name in enumerate(decoded_names):
                clean_name = name.replace(" ", "_").replace("-", "_").replace("/", "_")
                if not clean_name:
                    clean_name = f"{idx + 1}"

                if clean_name.startswith(f"{fallback_prefix}_"):
                    cleaned_names.append(clean_name)
                else:
                    cleaned_names.append(f"{fallback_prefix}_{clean_name}")

            return cleaned_names

        except Exception as exc:
            logger.exception("Variable-name decoding failed for group: %s", variable_group)
            raise RuntimeError(f"Variable-name decoding failed for group: {variable_group}") from exc

    def _load_group_as_dataframe(
        self,
        h5_file: h5py.File,
        group_name: str,
        variable_group: Optional[str],
        fallback_prefix: str,
    ) -> pd.DataFrame:
        """
        Load one HDF5 group as a DataFrame.

        Args:
            h5_file: Open HDF5 file.
            group_name: Group name to load.
            variable_group: Optional variable-name group.
            fallback_prefix: Column prefix.

        Returns:
            pd.DataFrame: Loaded group DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py::_load_group_as_dataframe")
        try:
            if group_name not in h5_file:
                raise KeyError(f"HDF5 group missing: {group_name}")

            data = np.array(h5_file[group_name])

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if data.ndim != 2:
                raise ValueError(f"Expected 2D data for group {group_name}, got shape {data.shape}")

            columns = self._decode_variable_names(
                h5_file=h5_file,
                variable_group=variable_group,
                fallback_prefix=fallback_prefix,
                width=data.shape[1],
            )

            df = pd.DataFrame(data, columns=columns)
            logger.info("Loaded group %s with shape %s.", group_name, df.shape)
            return df

        except Exception as exc:
            logger.exception("Failed to load HDF5 group: %s", group_name)
            raise RuntimeError(f"Failed to load HDF5 group: {group_name}") from exc

    def _add_unit_and_cycle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add unit_id and cycle columns using A metadata if available.

        Args:
            df: Combined DataFrame.

        Returns:
            pd.DataFrame: DataFrame with unit_id and cycle.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py::_add_unit_and_cycle")
        try:
            result = df.copy()
            lower_to_original = {column.lower(): column for column in result.columns}

            unit_candidates = [
                original
                for lower, original in lower_to_original.items()
                if (
                    "unit" in lower
                    or "engine" in lower
                    or lower.endswith("_id")
                    or lower.endswith("id")
                )
                and original not in {"unit_id"}
            ]

            cycle_candidates = [
                original
                for lower, original in lower_to_original.items()
                if (
                    "cycle" in lower
                    or "flight" in lower
                    or "time" in lower
                    or "hs" in lower
                )
                and original not in {"cycle"}
            ]

            if "unit_id" not in result.columns:
                if unit_candidates:
                    result["unit_id"] = pd.to_numeric(
                        result[unit_candidates[0]],
                        errors="coerce",
                    ).fillna(1).astype(int)
                else:
                    result["unit_id"] = 1

            if "cycle" not in result.columns:
                if cycle_candidates:
                    result["cycle"] = pd.to_numeric(
                        result[cycle_candidates[0]],
                        errors="coerce",
                    ).fillna(0).astype(int)
                else:
                    result["cycle"] = result.groupby("unit_id").cumcount() + 1

            result["unit_id"] = pd.to_numeric(result["unit_id"], errors="coerce").fillna(1).astype(int)
            result["cycle"] = pd.to_numeric(result["cycle"], errors="coerce").fillna(0).astype(int)

            return result

        except Exception as exc:
            logger.exception("Failed to add unit_id and cycle columns.")
            raise RuntimeError("Failed to add unit_id and cycle columns.") from exc

    def load_split(self, split: str) -> pd.DataFrame:
        """
        Load one split from HDF5.

        Args:
            split: dev or test.

        Returns:
            pd.DataFrame: Combined split data.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py::load_split")
        try:
            if split not in {Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME}:
                raise ValueError("split must be either 'dev' or 'test'.")

            suffix = "dev" if split == Config.DEV_SPLIT_NAME else "test"

            with h5py.File(self.h5_path, "r") as h5_file:
                a_df = self._load_group_as_dataframe(
                    h5_file=h5_file,
                    group_name=f"A_{suffix}",
                    variable_group="A_var",
                    fallback_prefix="A",
                )
                w_df = self._load_group_as_dataframe(
                    h5_file=h5_file,
                    group_name=f"W_{suffix}",
                    variable_group="W_var",
                    fallback_prefix="W",
                )
                xs_df = self._load_group_as_dataframe(
                    h5_file=h5_file,
                    group_name=f"X_s_{suffix}",
                    variable_group="X_s_var",
                    fallback_prefix="Xs",
                )
                xv_df = self._load_group_as_dataframe(
                    h5_file=h5_file,
                    group_name=f"X_v_{suffix}",
                    variable_group="X_v_var",
                    fallback_prefix="Xv",
                )

                frames = [a_df, w_df, xs_df, xv_df]

                t_group = f"T_{suffix}"
                if t_group in h5_file:
                    t_df = self._load_group_as_dataframe(
                        h5_file=h5_file,
                        group_name=t_group,
                        variable_group="T_var",
                        fallback_prefix="T",
                    )
                    frames.append(t_df)

                combined = pd.concat(frames, axis=1)
                combined["split"] = split
                combined = self._add_unit_and_cycle(combined)

            combined = combined.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
            logger.info("Loaded %s split with shape %s.", split, combined.shape)
            return combined

        except Exception as exc:
            logger.exception("Failed to load split: %s", split)
            raise RuntimeError(f"Failed to load split: {split}") from exc

    def load_all(self) -> pd.DataFrame:
        """
        Load dev and test splits.

        Returns:
            pd.DataFrame: Combined dev/test DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/Data_Preprocessing/h5_reader.py::load_all")
        try:
            self.validate_required_groups()

            dev_df = self.load_split(Config.DEV_SPLIT_NAME)
            test_df = self.load_split(Config.TEST_SPLIT_NAME)

            combined = pd.concat([dev_df, test_df], axis=0, ignore_index=True)
            combined = combined.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            logger.info("Loaded complete dataset with shape %s.", combined.shape)
            return combined

        except Exception as exc:
            logger.exception("Failed to load all HDF5 data.")
            raise RuntimeError("Failed to load all HDF5 data.") from exc


if __name__ == "__main__":
    Config.create_directories()
    reader = NCMAPSSH5Reader()
    print(reader.validate_required_groups())
    loaded_df = reader.load_all()
    print(loaded_df.head())
