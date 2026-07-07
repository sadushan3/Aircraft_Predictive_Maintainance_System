"""
Common Pydantic schemas for CA-EDT-AHMA API responses and requests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class APIResponse(BaseModel):
    """
    Standard API response schema.
    """

    status: str = Field(..., description="success, failed, partial_failure, or not_found")
    message: str
    output_file: Optional[str] = None
    records_count: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    data: Optional[Any] = None


class UnitRequest(BaseModel):
    """
    Request schema for unit-level dashboard queries.
    """

    unit_id: int = Field(..., ge=0)


class UnitCycleRequest(BaseModel):
    """
    Request schema for cycle-level dashboard queries.
    """

    unit_id: int = Field(..., ge=0)
    cycle: int = Field(..., ge=0)


class StageRequest(BaseModel):
    """
    Generic request schema for running a pipeline stage.
    """

    force_recompute: bool = Field(
        default=False,
        description="If true, stage output is recomputed. Existing files are overwritten atomically.",
    )


if __name__ == "__main__":
    logger.info("Common schemas loaded.")
    print(APIResponse(status="success", message="Common schemas are valid.").model_dump())