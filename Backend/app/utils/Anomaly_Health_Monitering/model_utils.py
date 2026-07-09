"""
Model utility functions for CA-EDT-AHMA.

Important:
get_raw_xs_columns() returns only physical measured X_s sensor columns.
It excludes engineered columns such as rolling mean, rolling std, trend,
lag, delta, and other derived features.

This keeps the Digital Twin research-correct:
Digital Twin target = raw X_s only.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import os as _os
import sys as _sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


def get_feature_columns(df: pd.DataFrame, prefixes: Tuple[str, ...]) -> List[str]:
    """
    Get feature columns by prefix.
    """
    columns = [column for column in df.columns if column.startswith(prefixes)]
    logger.info("Selected %s feature columns for prefixes %s.", len(columns), prefixes)
    return columns


def get_w_columns(df: pd.DataFrame) -> List[str]:
    """
    Get W operating-condition columns.
    """
    return get_feature_columns(df, ("W_",))


def get_xv_columns(df: pd.DataFrame) -> List[str]:
    """
    Get X_v virtual sensor feature columns.
    """
    return get_feature_columns(df, ("Xv_", "X_v_"))


def get_xs_columns(df: pd.DataFrame) -> List[str]:
    """
    Get all measured-sensor related X_s columns.

    This may include engineered X_s columns if they exist.
    Use get_raw_xs_columns() for Digital Twin target columns.
    """
    return get_feature_columns(df, ("Xs_", "X_s_"))


def get_raw_xs_columns(df: pd.DataFrame) -> List[str]:
    """
    Get only raw measured X_s sensor columns.

    Excludes:
    - rolling mean
    - rolling std
    - trend
    - lag
    - delta
    - difference
    - rate
    - engineered features

    Args:
        df: Source DataFrame.

    Returns:
        List[str]: Raw physical measured sensor columns only.
    """
    xs_columns = get_xs_columns(df)

    engineered_tokens = [
        "rolling",
        "trend",
        "lag",
        "delta",
        "diff",
        "difference",
        "rate",
        "slope",
        "mean_",
        "std_",
        "var_",
        "min_",
        "max_",
        "ewm",
    ]

    raw_columns: List[str] = []

    for column in xs_columns:
        lower_name = column.lower()

        if any(token in lower_name for token in engineered_tokens):
            continue

        raw_columns.append(column)

    if not raw_columns:
        raise ValueError(
            "No raw X_s sensor columns found. Check column names or preprocessing output."
        )

    logger.info(
        "Selected %s raw X_s target columns. Excluded %s engineered X_s columns.",
        len(raw_columns),
        len(xs_columns) - len(raw_columns),
    )

    return raw_columns


def get_residual_columns(df: pd.DataFrame) -> List[str]:
    """
    Get residual columns.
    """
    return [column for column in df.columns if column.startswith("residual_")]


def get_abs_residual_columns(df: pd.DataFrame) -> List[str]:
    """
    Get absolute residual columns.
    """
    return [column for column in df.columns if column.startswith("abs_residual_")]


def normalize_min_max(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Normalize numeric values between 0 and 1.
    """
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return arr

    min_value = np.nanmin(arr)
    max_value = np.nanmax(arr)
    denominator = max_value - min_value

    if denominator <= 1e-12:
        return np.zeros_like(arr, dtype=float)

    return (arr - min_value) / denominator


def safe_clip_01(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Clip values between 0 and 1.
    """
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def classify_alert(score: float) -> str:
    """
    Convert anomaly score to alert level.
    """
    if score >= 0.85:
        return "Critical"
    if score >= 0.65:
        return "Warning"
    if score >= 0.40:
        return "Watch"
    return "Normal"


def classify_health_state(health_index: float) -> str:
    """
    Convert health index to health state.
    """
    if health_index >= 85:
        return "Healthy"
    if health_index >= 65:
        return "Degrading"
    if health_index >= 40:
        return "Warning"
    return "Critical"


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers.
    """
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


if __name__ == "__main__":
    logger.info("Model utilities loaded.")
    demo = pd.DataFrame(
        columns=[
            "Xs_T24",
            "Xs_T30",
            "Xs_T24_rolling_mean_5",
            "Xs_T30_trend",
            "Xv_W48",
            "W_alt",
        ]
    )
    print("All Xs:", get_xs_columns(demo))
    print("Raw Xs:", get_raw_xs_columns(demo))
