from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/schemas/Anomaly_Health_Monitering/common_schema.py")
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    status: str = Field(..., description="success, failed, partial_failure, or not_found")
    message: str
    output_file: Optional[str] = None
    records_count: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    data: Optional[Any] = None


class UnitRequest(BaseModel):
    unit_id: int = Field(..., ge=0)


class UnitCycleRequest(BaseModel):
    unit_id: int = Field(..., ge=0)
    cycle: int = Field(..., ge=0)
