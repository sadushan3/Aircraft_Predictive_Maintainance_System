"""
Explainability schemas for CA-EDT-AHMA.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/schemas/Anomaly_Health_Monitering/explainability_schema.py")
from pydantic import BaseModel, Field

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class ExplanationRequest(BaseModel):
    """
    Request schema for explanation generation.
    """

    include_shap: bool = True
    include_residual_ranking: bool = True


class ExplanationRecord(BaseModel):
    """
    Explanation report schema.
    """

    unit_id: int
    cycle: int
    split: str
    gmm_context_id: int
    context_confidence: float = Field(..., ge=0.0, le=1.0)
    health_index: float = Field(..., ge=0.0, le=100.0)
    health_state: str
    final_anomaly_score: float = Field(..., ge=0.0, le=1.0)
    alert_level: str
    top_sensor_1: str
    top_sensor_2: str
    top_sensor_3: str
    root_cause_pattern: str
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation_text: str


if __name__ == "__main__":
    logger.info("Explainability schemas loaded.")
    print(ExplanationRequest(include_shap=True, include_residual_ranking=True).model_dump())