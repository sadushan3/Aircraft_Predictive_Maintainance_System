"""
Reasoning schemas for root-cause and temporal anomaly reasoning.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/schemas/Anomaly_Health_Monitering/reasoning_schema.py")
from pydantic import BaseModel, Field

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class RootCauseRequest(BaseModel):
    """
    Request schema for root-cause reasoning.
    """

    unit_id: int = Field(..., ge=0)
    cycle: int = Field(..., ge=0)


class RootCauseRecord(BaseModel):
    """
    Root-cause output schema.
    """

    unit_id: int
    cycle: int
    split: str
    top_sensor_1: str
    top_sensor_2: str
    top_sensor_3: str
    contribution_1: float = Field(..., ge=0.0, le=1.0)
    contribution_2: float = Field(..., ge=0.0, le=1.0)
    contribution_3: float = Field(..., ge=0.0, le=1.0)
    root_cause_pattern: str
    inspection_focus: str


class TemporalReasoningRecord(BaseModel):
    """
    Temporal reasoning output schema.
    """

    unit_id: int
    cycle: int
    anomaly_persistence_score: float = Field(..., ge=0.0, le=1.0)
    residual_trend_score: float = Field(..., ge=0.0, le=1.0)
    temporal_pattern: str


if __name__ == "__main__":
    logger.info("Reasoning schemas loaded.")
    print(RootCauseRequest(unit_id=1, cycle=10).model_dump())