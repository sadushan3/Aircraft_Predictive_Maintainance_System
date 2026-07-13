"""
Safe file utilities for CA-EDT-AHMA.

Important:
This module never deletes previous successful outputs.
All writes are atomic. If a new write fails, the older file remains unchanged.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/utils/Anomaly_Health_Monitering/file_utils.py")
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.insert(0, _backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)

CSV_WRITE_CHUNK_SIZE = 100_000


def ensure_parent(path: Path) -> None:
    """
    Ensure parent directory exists.

    Args:
        path: File path.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/file_utils.py::ensure_parent")
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Atomically write a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Target CSV path.

    Returns:
        Path: Written file path.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/file_utils.py::atomic_write_csv")
    ensure_parent(path)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    temp_path = Path(temp_name)

    try:
        df.to_csv(temp_path, index=False, chunksize=CSV_WRITE_CHUNK_SIZE)
        os.replace(temp_path, path)
        logger.info("CSV saved safely: %s | rows=%s | cols=%s", path, len(df), len(df.columns))
        return path
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        logger.exception("Failed to write CSV safely: %s", path)
        raise RuntimeError(f"Failed to write CSV safely: {path}") from exc


def atomic_write_json(data: Dict[str, Any], path: Path) -> Path:
    """
    Atomically write JSON data.

    Args:
        data: JSON-serializable dictionary.
        path: Target JSON path.

    Returns:
        Path: Written file path.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/file_utils.py::atomic_write_json")
    ensure_parent(path)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    temp_path = Path(temp_name)

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        os.replace(temp_path, path)
        logger.info("JSON saved safely: %s", path)
        return path
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        logger.exception("Failed to write JSON safely: %s", path)
        raise RuntimeError(f"Failed to write JSON safely: {path}") from exc


def read_json_required(path: Path) -> Dict[str, Any]:
    """
    Read a required JSON file.

    Args:
        path: JSON file path.

    Returns:
        Dict[str, Any]: Loaded JSON data.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/file_utils.py::read_json_required")
    try:
        if not path.exists():
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        logger.info("JSON loaded: %s", path)
        return data
    except Exception as exc:
        logger.exception("Failed to read JSON: %s", path)
        raise RuntimeError(f"Failed to read JSON: {path}") from exc


def atomic_save_joblib(obj: Any, path: Path) -> Path:
    """
    Atomically save a Python object using joblib.

    Args:
        obj: Python object.
        path: Target path.

    Returns:
        Path: Written file path.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/file_utils.py::atomic_save_joblib")
    ensure_parent(path)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    temp_path = Path(temp_name)

    try:
        joblib.dump(obj, temp_path)
        os.replace(temp_path, path)
        logger.info("Joblib object saved safely: %s", path)
        return path
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        logger.exception("Failed to save joblib object safely: %s", path)
        raise RuntimeError(f"Failed to save joblib object safely: {path}") from exc


def load_joblib_required(path: Path) -> Any:
    """
    Load a required joblib object.

    Args:
        path: Joblib path.

    Returns:
        Any: Loaded object.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/file_utils.py::load_joblib_required")
    try:
        if not path.exists():
            raise FileNotFoundError(f"Required joblib file not found: {path}")
        obj = joblib.load(path)
        logger.info("Joblib object loaded: %s", path)
        return obj
    except Exception as exc:
        logger.exception("Failed to load joblib object: %s", path)
        raise RuntimeError(f"Failed to load joblib object: {path}") from exc


def read_csv_required(path: Path) -> pd.DataFrame:
    """
    Read a required CSV file.

    Args:
        path: CSV path.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/file_utils.py::read_csv_required")
    try:
        if not path.exists():
            raise FileNotFoundError(f"Required CSV not found: {path}")
        df = pd.read_csv(path, low_memory=False, memory_map=True)
        logger.info("CSV loaded: %s | rows=%s | cols=%s", path, len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.exception("Failed to read CSV: %s", path)
        raise RuntimeError(f"Failed to read CSV: {path}") from exc


def initialize_output_files() -> None:
    """
    Initialize project directories safely.

    This does not remove or overwrite existing outputs.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/file_utils.py::initialize_output_files")
    Config.create_directories()
    logger.info("Project output directories initialized safely.")


if __name__ == "__main__":
    initialize_output_files()
    print("File utilities initialized successfully.")
