"""
Feedback store for CA-EDT-AHMA.

Role:
Store operator feedback for anomaly alerts.

Feedback labels:
- accepted_alert
- rejected_false_alarm
- missed_anomaly
- uncertain

Reads/Writes:
data/outputs/feedback_updates.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/feedback/feedback_store.py")
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv, read_csv_required
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


class FeedbackStore:
    """
    Store and retrieve operator feedback.
    """

    def __init__(self) -> None:
        """
        Initialize feedback store.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/feedback_store.py::__init__")
        Config.create_directories()

    def _empty_feedback_dataframe(self) -> pd.DataFrame:
        """
        Create empty feedback DataFrame.

        Returns:
            pd.DataFrame: Empty feedback DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/feedback_store.py::_empty_feedback_dataframe")
        return pd.DataFrame(
            columns=[
                "feedback_id",
                "timestamp_utc",
                "unit_id",
                "cycle",
                "context_id",
                "alert_level",
                "final_anomaly_score",
                "root_cause_pattern",
                "feedback_label",
                "feedback_status",
                "operator_note",
            ]
        )

    def load_feedback(self) -> pd.DataFrame:
        """
        Load feedback CSV if it exists.

        Returns:
            pd.DataFrame: Feedback DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/feedback_store.py::load_feedback")
        try:
            if Config.FEEDBACK_UPDATES_CSV.exists():
                return read_csv_required(Config.FEEDBACK_UPDATES_CSV)

            empty_df = self._empty_feedback_dataframe()
            atomic_write_csv(empty_df, Config.FEEDBACK_UPDATES_CSV)
            return empty_df

        except Exception as exc:
            logger.exception("Failed to load feedback store.")
            raise RuntimeError("Failed to load feedback store.") from exc

    def validate_feedback_label(self, feedback_label: str) -> None:
        """
        Validate feedback label.

        Args:
            feedback_label: Operator feedback label.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/feedback_store.py::validate_feedback_label")
        if feedback_label not in Config.FEEDBACK_LABELS:
            raise ValueError(
                f"Invalid feedback_label={feedback_label}. "
                f"Allowed labels: {Config.FEEDBACK_LABELS}"
            )

    def store_feedback(
        self,
        unit_id: int,
        cycle: int,
        context_id: int,
        alert_level: str,
        final_anomaly_score: float,
        root_cause_pattern: str,
        feedback_label: str,
        operator_note: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Store one feedback record safely.

        Args:
            unit_id: Engine/unit id.
            cycle: Cycle.
            context_id: GMM context id.
            alert_level: Alert level.
            final_anomaly_score: Final anomaly score.
            root_cause_pattern: Root-cause pattern.
            feedback_label: Feedback label.
            operator_note: Optional operator note.

        Returns:
            pd.DataFrame: Updated feedback DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/feedback_store.py::store_feedback")
        try:
            self.validate_feedback_label(feedback_label)

            feedback_df = self.load_feedback()

            next_id = 1
            if not feedback_df.empty and "feedback_id" in feedback_df.columns:
                numeric_ids = pd.to_numeric(feedback_df["feedback_id"], errors="coerce").dropna()
                if not numeric_ids.empty:
                    next_id = int(numeric_ids.max()) + 1

            new_record = pd.DataFrame(
                [
                    {
                        "feedback_id": next_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "unit_id": int(unit_id),
                        "cycle": int(cycle),
                        "context_id": int(context_id),
                        "alert_level": str(alert_level),
                        "final_anomaly_score": float(final_anomaly_score),
                        "root_cause_pattern": str(root_cause_pattern),
                        "feedback_label": str(feedback_label),
                        "feedback_status": "stored",
                        "operator_note": operator_note or "",
                    }
                ]
            )

            updated_df = pd.concat([feedback_df, new_record], axis=0, ignore_index=True)
            atomic_write_csv(updated_df, Config.FEEDBACK_UPDATES_CSV)

            logger.info(
                "Feedback stored. unit_id=%s cycle=%s label=%s",
                unit_id,
                cycle,
                feedback_label,
            )

            return updated_df

        except Exception as exc:
            logger.exception("Failed to store feedback.")
            raise RuntimeError("Failed to store feedback.") from exc

    def run(self) -> Dict[str, object]:
        """
        Initialize feedback store.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/feedback_store.py::run")
        try:
            feedback_df = self.load_feedback()
            atomic_write_csv(feedback_df, Config.FEEDBACK_UPDATES_CSV)

            return {
                "status": "success",
                "message": "Feedback store initialized safely.",
                "output_file": str(Config.FEEDBACK_UPDATES_CSV),
                "records_count": len(feedback_df),
            }

        except Exception as exc:
            logger.exception("Feedback store initialization failed.")
            raise RuntimeError("Feedback store initialization failed.") from exc


def run_feedback_store() -> Dict[str, object]:
    """
    Execute feedback store initialization.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/feedback/feedback_store.py::run_feedback_store")
    store = FeedbackStore()
    return store.run()


if __name__ == "__main__":
    result = run_feedback_store()
    print(result)
