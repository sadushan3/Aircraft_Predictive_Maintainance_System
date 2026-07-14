"""
Dashboard service for CA-EDT-AHMA.

Role:
Read dashboard_data.csv and provide dashboard-ready query outputs.

Supports:
1. Latest unit health
2. Health trend by unit
3. Anomalies by unit
4. Root-cause explanation by unit and cycle
5. Confidence and uncertainty
6. Summary counts
7. Latest health for all units

Important:
- Does not load full dashboard_data.csv into RAM.
- Reads dashboard_data.csv in chunks.
- Does not train models.
- Does not predict RUL.
- Does not use Y_dev/Y_test.
- Does not make maintenance decisions.
- Does not auto-run heavy dashboard generation inside query methods.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "dashboard/dashboard_service.py"
)

from pathlib import Path
from typing import Dict, List, Optional, Any
import gc
import json
import math
import os
import sys

import numpy as np
import pandas as pd


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    if BACKEND_ROOT not in sys.path:
        sys.path.append(BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class DashboardService:
    """
    Memory-safe dashboard query service.

    All methods stream dashboard_data.csv in chunks instead of loading
    the full 7.6M-row dashboard file.
    """

    def __init__(self, chunk_size: int = 150_000) -> None:
        """
        Initialize dashboard service.

        Args:
            chunk_size: Rows read per chunk.
        """
        print("[PROGRESS] Entering DashboardService.__init__")

        Config.create_directories()

        self.dashboard_csv: Path = Config.DASHBOARD_CSV

        self.summary_json: Path = getattr(
            Config,
            "DASHBOARD_DATA_SUMMARY_JSON",
            Config.REPORT_DIR / "dashboard_data_summary.json",
        )

        self.chunk_size = int(
            getattr(Config, "DASHBOARD_SERVICE_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("DASHBOARD_SERVICE_CHUNK_SIZE must be positive.")

        self.max_records = int(
            getattr(Config, "DASHBOARD_SERVICE_MAX_RECORDS", 5000)
        )

        if self.max_records <= 0:
            self.max_records = 5000

        print(f"[PROGRESS] Dashboard CSV: {self.dashboard_csv}")
        print(f"[PROGRESS] Dashboard summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Max returned records: {self.max_records}")

    # ==================================================================================
    # Basic helpers
    # ==================================================================================

    def _ensure_dashboard_exists(self) -> None:
        """
        Ensure dashboard_data.csv exists.

        Heavy generation is intentionally not triggered here.
        """
        if not self.dashboard_csv.exists():
            raise FileNotFoundError(
                f"dashboard_data.csv not found: {self.dashboard_csv}. "
                "Run dashboard_data_generator.py first."
            )

    def _read_header_columns(self) -> List[str]:
        """
        Read dashboard CSV header only.
        """
        self._ensure_dashboard_exists()
        return list(pd.read_csv(self.dashboard_csv, nrows=0).columns)

    def _safe_usecols(self, requested_columns: List[str]) -> List[str]:
        """
        Return only requested columns that exist in dashboard_data.csv.

        Always requires unit_id/cycle/split when available.
        """
        available_columns = self._read_header_columns()
        available_set = set(available_columns)

        usecols = []

        for column in requested_columns:
            if column in available_set and column not in usecols:
                usecols.append(column)

        if not usecols:
            raise ValueError(
                f"None of the requested columns exist in dashboard_data.csv: {requested_columns}"
            )

        return usecols

    def _all_dashboard_columns(self) -> List[str]:
        """
        Return all dashboard columns from file header.
        """
        return self._read_header_columns()

    def _iter_dashboard_chunks(
        self,
        usecols: Optional[List[str]] = None,
    ):
        """
        Yield dashboard chunks.
        """
        self._ensure_dashboard_exists()

        kwargs: Dict[str, Any] = {
            "chunksize": self.chunk_size,
            "low_memory": True,
        }

        if usecols is not None:
            kwargs["usecols"] = usecols

        for chunk in pd.read_csv(self.dashboard_csv, **kwargs):
            yield chunk.reset_index(drop=True)

    def _sanitize_value(self, value: Any) -> Any:
        """
        Convert numpy/pandas values into JSON-safe Python values.
        """
        if pd.isna(value):
            return None

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            if not math.isfinite(float(value)):
                return None
            return float(value)

        if isinstance(value, (np.bool_,)):
            return bool(value)

        return value

    def _records_from_df(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        """
        Convert DataFrame to JSON-safe records.
        """
        records = df.to_dict(orient="records")

        clean_records: List[Dict[str, object]] = []

        for record in records:
            clean_records.append(
                {key: self._sanitize_value(value) for key, value in record.items()}
            )

        return clean_records

    def _failed_response(
        self,
        message: str,
        data: Any = None,
        records_count: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Standard failure response.
        """
        response: Dict[str, object] = {
            "status": "failed",
            "message": message,
            "data": data,
        }

        if records_count is not None:
            response["records_count"] = records_count

        return response

    def _add_anomaly_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add direct anomaly fields if they are missing.
        """
        result = df.copy()

        if "alert_level" in result.columns:
            if "anomaly_detected" not in result.columns:
                result["anomaly_detected"] = result["alert_level"].astype(str).ne("Normal")

            if "anomaly_status" not in result.columns:
                result["anomaly_status"] = np.where(
                    result["anomaly_detected"],
                    "Anomaly_Detected",
                    "No_Anomaly",
                )

        return result

    # ==================================================================================
    # Query methods
    # ==================================================================================

    def latest_unit_health(self, unit_id: int) -> Dict[str, object]:
        """
        Return latest health row for one unit.
        """
        print("[PROGRESS] Entering DashboardService.latest_unit_health")

        try:
            usecols = self._all_dashboard_columns()

            latest_row: Optional[pd.Series] = None
            latest_cycle: Optional[int] = None
            matched_rows = 0

            for chunk in self._iter_dashboard_chunks(usecols=usecols):
                if "unit_id" not in chunk.columns or "cycle" not in chunk.columns:
                    raise KeyError("dashboard_data.csv must contain unit_id and cycle.")

                unit_chunk = chunk[chunk["unit_id"] == unit_id]

                if unit_chunk.empty:
                    del chunk
                    gc.collect()
                    continue

                matched_rows += int(len(unit_chunk))

                unit_chunk = unit_chunk.sort_values("cycle")
                candidate = unit_chunk.tail(1).iloc[0]
                candidate_cycle = int(candidate["cycle"])

                if latest_cycle is None or candidate_cycle > latest_cycle:
                    latest_cycle = candidate_cycle
                    latest_row = candidate.copy()

                del chunk
                del unit_chunk
                gc.collect()

            if latest_row is None:
                return {
                    "status": "not_found",
                    "message": f"No dashboard data found for unit_id={unit_id}.",
                    "data": None,
                }

            latest_df = pd.DataFrame([latest_row])
            latest_df = self._add_anomaly_fields(latest_df)

            return {
                "status": "success",
                "message": "Latest unit health returned.",
                "matched_rows": int(matched_rows),
                "data": self._records_from_df(latest_df)[0],
            }

        except Exception as exc:
            logger.exception("Latest unit health query failed.")
            return self._failed_response(str(exc), data=None)

    def health_trend_by_unit(self, unit_id: int) -> Dict[str, object]:
        """
        Return health trend for one unit.
        """
        print("[PROGRESS] Entering DashboardService.health_trend_by_unit")

        try:
            requested_columns = [
                "unit_id",
                "cycle",
                "split",
                "health_index",
                "remaining_health_percentage",
                "health_state",
                "final_anomaly_score",
                "alert_level",
                "anomaly_detected",
                "anomaly_status",
                "confidence_score",
                "uncertainty_score",
                "reliability_score",
            ]

            usecols = self._safe_usecols(requested_columns)

            frames: List[pd.DataFrame] = []
            total_matching = 0

            for chunk in self._iter_dashboard_chunks(usecols=usecols):
                unit_chunk = chunk[chunk["unit_id"] == unit_id]

                if not unit_chunk.empty:
                    total_matching += int(len(unit_chunk))

                    if sum(len(frame) for frame in frames) < self.max_records:
                        remaining = self.max_records - sum(len(frame) for frame in frames)
                        frames.append(unit_chunk.head(remaining).copy())

                del chunk
                del unit_chunk
                gc.collect()

            if frames:
                result_df = pd.concat(frames, axis=0, ignore_index=True)
                result_df = result_df.sort_values("cycle").reset_index(drop=True)
                result_df = self._add_anomaly_fields(result_df)
            else:
                result_df = pd.DataFrame(columns=usecols)

            return {
                "status": "success",
                "message": "Health trend returned.",
                "records_count": int(len(result_df)),
                "total_matching_records": int(total_matching),
                "truncated": bool(total_matching > len(result_df)),
                "max_records": int(self.max_records),
                "data": self._records_from_df(result_df),
            }

        except Exception as exc:
            logger.exception("Health trend query failed.")
            return self._failed_response(str(exc), data=[], records_count=0)

    def anomalies_by_unit(self, unit_id: int) -> Dict[str, object]:
        """
        Return non-normal alert records for one unit.
        """
        print("[PROGRESS] Entering DashboardService.anomalies_by_unit")

        try:
            requested_columns = [
                "unit_id",
                "cycle",
                "split",
                "final_anomaly_score",
                "alert_level",
                "anomaly_detected",
                "anomaly_status",
                "health_index",
                "health_state",
                "confidence_score",
                "uncertainty_score",
                "reliability_score",
                "root_cause_pattern",
                "inspection_focus",
                "top_sensor_1",
                "top_sensor_2",
                "top_sensor_3",
                "contribution_1",
                "contribution_2",
                "contribution_3",
                "explanation_text",
                "feedback_status",
            ]

            usecols = self._safe_usecols(requested_columns)

            if "alert_level" not in usecols:
                raise KeyError("alert_level is required for anomaly query.")

            frames: List[pd.DataFrame] = []
            total_matching = 0

            for chunk in self._iter_dashboard_chunks(usecols=usecols):
                mask = (
                    (chunk["unit_id"] == unit_id)
                    & (chunk["alert_level"].astype(str).isin(["Watch", "Warning", "Critical"]))
                )

                anomaly_chunk = chunk[mask]

                if not anomaly_chunk.empty:
                    total_matching += int(len(anomaly_chunk))

                    if sum(len(frame) for frame in frames) < self.max_records:
                        remaining = self.max_records - sum(len(frame) for frame in frames)
                        frames.append(anomaly_chunk.head(remaining).copy())

                del chunk
                del anomaly_chunk
                gc.collect()

            if frames:
                result_df = pd.concat(frames, axis=0, ignore_index=True)
                result_df = result_df.sort_values("cycle").reset_index(drop=True)
                result_df = self._add_anomaly_fields(result_df)
            else:
                result_df = pd.DataFrame(columns=usecols)

            return {
                "status": "success",
                "message": "Anomalies returned.",
                "records_count": int(len(result_df)),
                "total_matching_records": int(total_matching),
                "truncated": bool(total_matching > len(result_df)),
                "max_records": int(self.max_records),
                "data": self._records_from_df(result_df),
            }

        except Exception as exc:
            logger.exception("Anomalies by unit query failed.")
            return self._failed_response(str(exc), data=[], records_count=0)

    def root_cause_explanation(self, unit_id: int, cycle: int) -> Dict[str, object]:
        """
        Return root-cause explanation for one unit and cycle.
        """
        print("[PROGRESS] Entering DashboardService.root_cause_explanation")

        try:
            requested_columns = [
                "unit_id",
                "cycle",
                "split",
                "alert_level",
                "anomaly_detected",
                "anomaly_status",
                "final_anomaly_score",
                "health_index",
                "health_state",
                "top_sensor_1",
                "top_sensor_2",
                "top_sensor_3",
                "contribution_1",
                "contribution_2",
                "contribution_3",
                "root_cause_pattern",
                "inspection_focus",
                "explanation_text",
                "confidence_score",
                "uncertainty_score",
                "reliability_score",
            ]

            usecols = self._safe_usecols(requested_columns)

            for chunk in self._iter_dashboard_chunks(usecols=usecols):
                selected = chunk[
                    (chunk["unit_id"] == unit_id)
                    & (chunk["cycle"] == cycle)
                ]

                if not selected.empty:
                    selected = self._add_anomaly_fields(selected.head(1).copy())
                    row = self._records_from_df(selected)[0]

                    return {
                        "status": "success",
                        "message": "Root-cause explanation returned.",
                        "data": row,
                    }

                del chunk
                del selected
                gc.collect()

            return {
                "status": "not_found",
                "message": f"No explanation found for unit_id={unit_id}, cycle={cycle}.",
                "data": None,
            }

        except Exception as exc:
            logger.exception("Root-cause explanation query failed.")
            return self._failed_response(str(exc), data=None)

    def confidence_uncertainty_by_unit(self, unit_id: int) -> Dict[str, object]:
        """
        Return confidence and uncertainty trend for one unit.
        """
        print("[PROGRESS] Entering DashboardService.confidence_uncertainty_by_unit")

        try:
            requested_columns = [
                "unit_id",
                "cycle",
                "split",
                "model_agreement_score",
                "context_confidence",
                "confidence_score",
                "uncertainty_score",
                "reliability_score",
                "confidence_label",
                "uncertainty_label",
            ]

            usecols = self._safe_usecols(requested_columns)

            frames: List[pd.DataFrame] = []
            total_matching = 0

            for chunk in self._iter_dashboard_chunks(usecols=usecols):
                unit_chunk = chunk[chunk["unit_id"] == unit_id]

                if not unit_chunk.empty:
                    total_matching += int(len(unit_chunk))

                    if sum(len(frame) for frame in frames) < self.max_records:
                        remaining = self.max_records - sum(len(frame) for frame in frames)
                        frames.append(unit_chunk.head(remaining).copy())

                del chunk
                del unit_chunk
                gc.collect()

            if frames:
                result_df = pd.concat(frames, axis=0, ignore_index=True)
                result_df = result_df.sort_values("cycle").reset_index(drop=True)
            else:
                result_df = pd.DataFrame(columns=usecols)

            return {
                "status": "success",
                "message": "Confidence and uncertainty returned.",
                "records_count": int(len(result_df)),
                "total_matching_records": int(total_matching),
                "truncated": bool(total_matching > len(result_df)),
                "max_records": int(self.max_records),
                "data": self._records_from_df(result_df),
            }

        except Exception as exc:
            logger.exception("Confidence/uncertainty query failed.")
            return self._failed_response(str(exc), data=[], records_count=0)

    def summary_counts(self) -> Dict[str, object]:
        """
        Return dashboard summary counts using chunk scan.

        This avoids loading full dashboard_data.csv.
        """
        print("[PROGRESS] Entering DashboardService.summary_counts")

        try:
            requested_columns = [
                "unit_id",
                "alert_level",
                "health_state",
                "health_index",
                "confidence_score",
                "uncertainty_score",
                "reliability_score",
            ]

            usecols = self._safe_usecols(requested_columns)

            alert_counts: Dict[str, int] = {}
            health_counts: Dict[str, int] = {}
            unique_units = set()

            total_records = 0
            health_sum = 0.0
            confidence_sum = 0.0
            uncertainty_sum = 0.0
            reliability_sum = 0.0

            for chunk in self._iter_dashboard_chunks(usecols=usecols):
                total_records += int(len(chunk))

                if "unit_id" in chunk.columns:
                    unique_units.update(chunk["unit_id"].dropna().astype(int).unique().tolist())

                if "alert_level" in chunk.columns:
                    counts = chunk["alert_level"].astype(str).value_counts().to_dict()
                    for label, count in counts.items():
                        alert_counts[label] = alert_counts.get(label, 0) + int(count)

                if "health_state" in chunk.columns:
                    counts = chunk["health_state"].astype(str).value_counts().to_dict()
                    for label, count in counts.items():
                        health_counts[label] = health_counts.get(label, 0) + int(count)

                if "health_index" in chunk.columns:
                    health_sum += float(pd.to_numeric(chunk["health_index"], errors="coerce").fillna(0.0).sum())

                if "confidence_score" in chunk.columns:
                    confidence_sum += float(pd.to_numeric(chunk["confidence_score"], errors="coerce").fillna(0.0).sum())

                if "uncertainty_score" in chunk.columns:
                    uncertainty_sum += float(pd.to_numeric(chunk["uncertainty_score"], errors="coerce").fillna(0.0).sum())

                if "reliability_score" in chunk.columns:
                    reliability_sum += float(pd.to_numeric(chunk["reliability_score"], errors="coerce").fillna(0.0).sum())

                del chunk
                gc.collect()

            normal = int(alert_counts.get("Normal", 0))
            watch = int(alert_counts.get("Watch", 0))
            warning = int(alert_counts.get("Warning", 0))
            critical = int(alert_counts.get("Critical", 0))

            anomaly_total = watch + warning + critical

            data = {
                "alert_counts": {
                    "normal": normal,
                    "watch": watch,
                    "warning": warning,
                    "critical": critical,
                },
                "anomaly_summary": {
                    "anomaly_records": int(anomaly_total),
                    "normal_records": int(normal),
                    "anomaly_detected": bool(anomaly_total > 0),
                    "anomaly_ratio": float(anomaly_total / max(total_records, 1)),
                },
                "health_state_counts": {
                    "healthy": int(health_counts.get("Healthy", 0)),
                    "degrading": int(health_counts.get("Degrading", 0)),
                    "warning": int(health_counts.get("Warning", 0)),
                    "critical": int(health_counts.get("Critical", 0)),
                },
                "total_records": int(total_records),
                "unique_units": int(len(unique_units)),
                "average_health_index": float(health_sum / max(total_records, 1)),
                "average_confidence_score": float(confidence_sum / max(total_records, 1)),
                "average_uncertainty_score": float(uncertainty_sum / max(total_records, 1)),
                "average_reliability_score": float(reliability_sum / max(total_records, 1)),
            }

            return {
                "status": "success",
                "message": "Dashboard summary counts returned.",
                "data": data,
            }

        except Exception as exc:
            logger.exception("Dashboard summary query failed.")
            return self._failed_response(str(exc), data={})

    def latest_all_units(self) -> Dict[str, object]:
        """
        Return latest health row for every unit using streaming dictionary.
        """
        print("[PROGRESS] Entering DashboardService.latest_all_units")

        try:
            usecols = self._all_dashboard_columns()

            latest_by_unit: Dict[int, Dict[str, object]] = {}
            latest_cycle_by_unit: Dict[int, int] = {}

            for chunk in self._iter_dashboard_chunks(usecols=usecols):
                if "unit_id" not in chunk.columns or "cycle" not in chunk.columns:
                    raise KeyError("dashboard_data.csv must contain unit_id and cycle.")

                chunk = self._add_anomaly_fields(chunk)

                for _, row in chunk.iterrows():
                    try:
                        unit = int(row["unit_id"])
                        cycle = int(row["cycle"])

                        if unit not in latest_cycle_by_unit or cycle > latest_cycle_by_unit[unit]:
                            latest_cycle_by_unit[unit] = cycle
                            latest_by_unit[unit] = {
                                key: self._sanitize_value(value)
                                for key, value in row.to_dict().items()
                            }

                    except Exception:
                        continue

                del chunk
                gc.collect()

            data = [
                latest_by_unit[unit_id]
                for unit_id in sorted(latest_by_unit.keys())
            ]

            return {
                "status": "success",
                "message": "Latest health for all units returned.",
                "records_count": int(len(data)),
                "data": data,
            }

        except Exception as exc:
            logger.exception("Latest all units query failed.")
            return self._failed_response(str(exc), data=[], records_count=0)


def run_dashboard_service_self_check() -> Dict[str, object]:
    """
    Run dashboard service self-check.
    """
    print("[PROGRESS] Entering run_dashboard_service_self_check")

    service = DashboardService()
    return service.summary_counts()


if __name__ == "__main__":
    print("[PROGRESS] dashboard_service.py execution started")
    result = run_dashboard_service_self_check()
    print("[PROGRESS] dashboard_service.py execution finished")
    print(result)