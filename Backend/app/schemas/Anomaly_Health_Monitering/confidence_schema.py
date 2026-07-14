"""
Confidence and uncertainty schemas for CA-EDT-AHMA.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/schemas/Anomaly_Health_Monitering/confidence_schema.py")
from pydantic import BaseModel, Field

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class ConfidenceRecord(BaseModel):
    """
    Confidence record schema.
    """

    unit_id: int
    cycle: int
    split: str
    model_agreement_score: float = Field(..., ge=0.0, le=1.0)
    context_confidence: float = Field(..., ge=0.0, le=1.0)
    anomaly_persistence_score: float = Field(..., ge=0.0, le=1.0)
    data_quality_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    reliability_score: float = Field(..., ge=0.0, le=1.0)


class ConfidenceRequest(BaseModel):
    """
    Request schema for confidence calculation.
    """

    recompute_model_agreement: bool = True


if __name__ == "__main__":
    logger.info("Confidence schemas loaded.")
    print(ConfidenceRequest(recompute_model_agreement=True).model_dump())