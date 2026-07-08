from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/schemas/Anomaly_Health_Monitering/anomaly_schema.py")
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Schema for operator feedback submissions."""

    unit_id: int = Field(..., ge=0)
    cycle: int = Field(..., ge=0)
    context_id: str | None = None
    alert_level: str | None = None
    final_anomaly_score: float | None = None
    root_cause_pattern: str | None = None
    feedback_label: str = Field(..., description="accepted_alert, rejected_false_alarm, missed_anomaly, or uncertain")
