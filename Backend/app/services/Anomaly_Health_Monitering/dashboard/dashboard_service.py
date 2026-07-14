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

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Deque, Tuple
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

    def _file_metadata(self, path: Path) -> Dict[str, object]:
        """Return JSON-safe metadata for an existing artifact."""
        stat = path.stat()
        return {
            "name": path.name,
            "extension": path.suffix.lower().lstrip("."),
            "size_bytes": int(stat.st_size),
            "updated_at": datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).isoformat(),
        }

    def _read_json_artifact(
        self,
        path: Path,
        max_bytes: int = 1_000_000,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Read one bounded JSON artifact.

        Returns the parsed payload and an optional reason when content is not
        available. The size guard prevents a dashboard request from loading an
        unexpectedly large file into memory.
        """
        if not path.exists() or not path.is_file():
            return None, "not_found"

        if path.stat().st_size > max_bytes:
            return None, "content_too_large"

        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle), None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unable to read JSON artifact %s: %s", path, exc)
            return None, "invalid_json"

    def _report_entry(
        self,
        path: Path,
        include_content: bool = True,
        max_bytes: int = 1_000_000,
    ) -> Dict[str, object]:
        """Build one report catalog entry from a real JSON report file."""
        entry = self._file_metadata(path)
        content, content_error = self._read_json_artifact(path, max_bytes=max_bytes)

        if isinstance(content, dict):
            entry["status"] = str(content.get("status", "available"))
            if content.get("message") is not None:
                entry["message"] = str(content["message"])
            if content.get("records_count") is not None:
                try:
                    entry["records_count"] = int(content["records_count"])
                except (TypeError, ValueError):
                    pass
        else:
            entry["status"] = "available" if content_error is None else content_error

        if include_content:
            entry["content"] = content
            entry["content_status"] = "loaded" if content_error is None else content_error

        return entry

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
        Return dashboard summary counts.

        The generated dashboard summary JSON is preferred so normal page loads
        do not rescan the multi-gigabyte dashboard CSV. The existing chunk scan
        remains a fallback when no valid cached summary exists.
        """
        print("[PROGRESS] Entering DashboardService.summary_counts")

        try:
            cached_summary, cached_error = self._read_json_artifact(self.summary_json)
            if cached_error is None and isinstance(cached_summary, dict):
                raw_alert_counts = cached_summary.get("alert_counts")
                raw_health_counts = cached_summary.get("health_state_counts")
                raw_averages = cached_summary.get("averages")
                raw_anomaly_summary = cached_summary.get("anomaly_summary")
                raw_total_records = cached_summary.get(
                    "records_count",
                    cached_summary.get("expected_rows"),
                )

                if (
                    isinstance(raw_alert_counts, dict)
                    and isinstance(raw_health_counts, dict)
                    and isinstance(raw_averages, dict)
                    and isinstance(raw_anomaly_summary, dict)
                    and raw_total_records is not None
                ):
                    alert_counts_by_label = {
                        str(key).lower(): int(value)
                        for key, value in raw_alert_counts.items()
                    }
                    health_counts_by_label = {
                        str(key).lower(): int(value)
                        for key, value in raw_health_counts.items()
                    }
                    total_records = int(raw_total_records)

                    data = {
                        "alert_counts": {
                            "normal": int(alert_counts_by_label.get("normal", 0)),
                            "watch": int(alert_counts_by_label.get("watch", 0)),
                            "warning": int(alert_counts_by_label.get("warning", 0)),
                            "critical": int(alert_counts_by_label.get("critical", 0)),
                        },
                        "anomaly_summary": raw_anomaly_summary,
                        "health_state_counts": {
                            "healthy": int(health_counts_by_label.get("healthy", 0)),
                            "degrading": int(health_counts_by_label.get("degrading", 0)),
                            "warning": int(health_counts_by_label.get("warning", 0)),
                            "critical": int(health_counts_by_label.get("critical", 0)),
                        },
                        "total_records": total_records,
                        "unique_units": (
                            int(cached_summary["unique_units"])
                            if cached_summary.get("unique_units") is not None
                            else None
                        ),
                        "average_health_index": float(
                            raw_averages["average_health_index"]
                        ) if raw_averages.get("average_health_index") is not None else None,
                        "average_confidence_score": (
                            float(raw_averages["average_confidence_score"])
                            if raw_averages.get("average_confidence_score") is not None
                            else None
                        ),
                        "average_uncertainty_score": (
                            float(raw_averages["average_uncertainty_score"])
                            if raw_averages.get("average_uncertainty_score") is not None
                            else None
                        ),
                        "average_reliability_score": (
                            float(raw_averages["average_reliability_score"])
                            if raw_averages.get("average_reliability_score") is not None
                            else None
                        ),
                        "source": {
                            "type": "generated_summary_cache",
                            **self._file_metadata(self.summary_json),
                        },
                    }

                    return {
                        "status": "success",
                        "message": (
                            "Dashboard summary counts returned from the generated "
                            "summary cache; unique_units is null when not recorded "
                            "by that artifact."
                        ),
                        "data": data,
                    }

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
                    health_sum += float(
                        pd.to_numeric(chunk["health_index"], errors="coerce")
                        .fillna(0.0)
                        .sum()
                    )

                if "confidence_score" in chunk.columns:
                    confidence_sum += float(
                        pd.to_numeric(chunk["confidence_score"], errors="coerce")
                        .fillna(0.0)
                        .sum()
                    )

                if "uncertainty_score" in chunk.columns:
                    uncertainty_sum += float(
                        pd.to_numeric(chunk["uncertainty_score"], errors="coerce")
                        .fillna(0.0)
                        .sum()
                    )

                if "reliability_score" in chunk.columns:
                    reliability_sum += float(
                        pd.to_numeric(chunk["reliability_score"], errors="coerce")
                        .fillna(0.0)
                        .sum()
                    )

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

    def reports_catalog(self) -> Dict[str, object]:
        """
        Return bounded report content and metadata for generated artifacts.

        JSON report payloads are parsed from ``Config.REPORT_DIR``. Output and
        metric files are represented by metadata only; large CSV files are
        never opened by this method.
        """
        print("[PROGRESS] Entering DashboardService.reports_catalog")

        try:
            max_report_files = 100
            max_metadata_files = 250
            max_report_bytes = 1_000_000
            max_catalog_content_bytes = 5_000_000

            report_paths = sorted(
                (
                    path
                    for path in Config.REPORT_DIR.glob("*.json")
                    if path.is_file()
                ),
                key=lambda path: path.name.lower(),
            )

            reports: List[Dict[str, object]] = []
            indexed_record_counts: Dict[str, int] = {}
            content_bytes_used = 0

            for path in report_paths[:max_report_files]:
                size_bytes = int(path.stat().st_size)
                include_content = (
                    size_bytes <= max_report_bytes
                    and content_bytes_used + size_bytes <= max_catalog_content_bytes
                )
                entry = self._report_entry(
                    path,
                    include_content=include_content,
                    max_bytes=max_report_bytes,
                )

                if include_content:
                    content_bytes_used += size_bytes
                else:
                    entry["content"] = None
                    entry["content_status"] = "catalog_size_limit"

                content = entry.get("content")
                if isinstance(content, dict):
                    output_file = content.get("output_file")
                    records_count = content.get("records_count")
                    if output_file and records_count is not None:
                        try:
                            indexed_record_counts[Path(str(output_file)).name] = int(records_count)
                        except (TypeError, ValueError):
                            pass

                reports.append(entry)

            def metadata_entries(
                directory: Path,
            ) -> Tuple[List[Dict[str, object]], int]:
                if not directory.exists():
                    return [], 0

                paths = sorted(
                    (
                        path
                        for path in directory.iterdir()
                        if path.is_file()
                        and path.suffix.lower() in {".csv", ".json", ".parquet"}
                    ),
                    key=lambda path: path.name.lower(),
                )
                entries: List[Dict[str, object]] = []
                for path in paths[:max_metadata_files]:
                    entry = self._file_metadata(path)
                    if path.name in indexed_record_counts:
                        entry["records_count"] = indexed_record_counts[path.name]
                    entries.append(entry)
                return entries, len(paths)

            output_files, total_output_files = metadata_entries(Config.OUTPUT_DIR)
            metric_files, total_metric_files = metadata_entries(Config.METRIC_DIR)

            total_files = len(report_paths) + total_output_files + total_metric_files
            return {
                "status": "success",
                "message": "Generated report and artifact catalog returned.",
                "records_count": int(total_files),
                "data": {
                    "reports": reports,
                    "output_files": output_files,
                    "metric_files": metric_files,
                    "report_count": int(len(reports)),
                    "output_file_count": int(len(output_files)),
                    "metric_file_count": int(len(metric_files)),
                    "truncated": bool(
                        len(report_paths) > max_report_files
                        or total_output_files > max_metadata_files
                        or total_metric_files > max_metadata_files
                    ),
                },
            }

        except Exception as exc:
            logger.exception("Report catalog query failed.")
            return self._failed_response(
                str(exc),
                data={"reports": [], "output_files": [], "metric_files": []},
                records_count=0,
            )

    def overview_summary(self) -> Dict[str, object]:
        """Return fast overview aggregates from the generated dashboard report."""
        print("[PROGRESS] Entering DashboardService.overview_summary")

        try:
            if not self.summary_json.exists() or not self.summary_json.is_file():
                return {
                    "status": "not_found",
                    "message": "No dashboard summary report is available.",
                    "data": None,
                }

            content, content_error = self._read_json_artifact(self.summary_json)
            if content_error is not None or not isinstance(content, dict):
                return self._failed_response(
                    f"Dashboard summary report could not be loaded: {content_error or 'invalid_content'}",
                    data=None,
                )

            raw_alert_counts = content.get("alert_counts", {})
            alert_counts = (
                {str(key).lower(): int(value) for key, value in raw_alert_counts.items()}
                if isinstance(raw_alert_counts, dict)
                else {}
            )
            raw_health_counts = content.get("health_state_counts", {})
            health_state_counts = (
                {str(key).lower(): int(value) for key, value in raw_health_counts.items()}
                if isinstance(raw_health_counts, dict)
                else {}
            )

            aggregates: Dict[str, object] = {
                "alert_counts": alert_counts,
                "health_state_counts": health_state_counts,
            }
            direct_fields = [
                "records_count",
                "expected_rows",
                "anomaly_summary",
                "averages",
                "split_counts",
                "feedback_counts",
            ]
            for field in direct_fields:
                if field in content:
                    aggregates[field] = content[field]

            report_records = content.get("records_count")
            return {
                "status": "success",
                "message": "Dashboard overview aggregates returned from the generated summary.",
                "records_count": (
                    int(report_records)
                    if report_records is not None
                    else None
                ),
                "data": {
                    "aggregates": aggregates,
                    "source": self._file_metadata(self.summary_json),
                    "summary": content,
                },
            }

        except Exception as exc:
            logger.exception("Dashboard overview query failed.")
            return self._failed_response(str(exc), data=None)

    def anomalies_all(self, limit: int = 500) -> Dict[str, object]:
        """Return a bounded fleet-wide sample from persisted alert memory."""
        print("[PROGRESS] Entering DashboardService.anomalies_all")

        try:
            safe_limit = max(1, min(int(limit), self.max_records))
            path = Config.ALERT_MEMORY_CSV
            if not path.exists() or not path.is_file():
                return {
                    "status": "not_found",
                    "message": "No alert-memory artifact is available.",
                    "records_count": 0,
                    "data": [],
                }

            available_columns = list(pd.read_csv(path, nrows=0).columns)
            requested_columns = [
                "unit_id",
                "cycle",
                "split",
                "context_id",
                "alert_level",
                "final_anomaly_score",
                "root_cause_pattern",
                "top_sensor_1",
                "top_sensor_2",
                "top_sensor_3",
                "feedback_status",
            ]
            usecols = [
                column
                for column in requested_columns
                if column in available_columns
            ]
            if not usecols:
                raise KeyError("alert_memory.csv contains none of the dashboard alert columns.")

            sample_df = pd.read_csv(
                path,
                usecols=usecols,
                nrows=safe_limit,
                low_memory=False,
            )
            data = self._records_from_df(sample_df)

            total_records: Optional[int] = None
            alert_summary_path = Config.REPORT_DIR / "alert_memory_summary.json"
            summary, _ = self._read_json_artifact(alert_summary_path)
            if isinstance(summary, dict) and summary.get("records_count") is not None:
                try:
                    total_records = int(summary["records_count"])
                except (TypeError, ValueError):
                    total_records = None

            response: Dict[str, object] = {
                "status": "success",
                "message": "Fleet-wide alert-memory sample returned.",
                "records_count": int(len(data)),
                "metrics": {
                    "max_records": int(safe_limit),
                    "sampling": "first_rows_in_persisted_alert_memory",
                },
                "data": data,
            }
            if total_records is not None:
                metrics = response["metrics"]
                if isinstance(metrics, dict):
                    metrics["total_matching_records"] = total_records
                    metrics["truncated"] = bool(total_records > len(data))
            return response

        except pd.errors.EmptyDataError:
            return {
                "status": "success",
                "message": "Alert memory is empty.",
                "records_count": 0,
                "data": [],
            }
        except Exception as exc:
            logger.exception("Fleet-wide anomaly query failed.")
            return self._failed_response(str(exc), data=[], records_count=0)

    def pipeline_status(self) -> Dict[str, object]:
        """
        Return broad pipeline stage status derived from real artifacts.

        A stage is reported as ``success`` only when a successful report or its
        primary output exists. A failed report takes precedence when it is the
        newest report for that stage. No in-memory or fabricated run state is
        used.
        """
        print("[PROGRESS] Entering DashboardService.pipeline_status")

        try:
            report_dir = Config.REPORT_DIR
            stages = [
                {
                    "id": "preprocessing",
                    "name": "Data preprocessing",
                    "primary": Config.SCALED_CSV,
                    "artifacts": [Config.RAW_CSV, Config.CLEANED_CSV, Config.ENGINEERED_CSV, Config.SCALED_CSV],
                    "reports": [],
                },
                {
                    "id": "context_modeling",
                    "name": "Context modeling",
                    "primary": Config.CONTEXT_CSV,
                    "artifacts": [Config.CONTEXT_CSV, Config.CONTEXT_DRIFT_CSV],
                    "reports": [report_dir / "evaluate_context_summary.json"],
                },
                {
                    "id": "digital_twin",
                    "name": "Digital twin",
                    "primary": Config.ENSEMBLE_PREDICTIONS_CSV,
                    "artifacts": [
                        Config.RF_PREDICTIONS_CSV,
                        Config.XGB_PREDICTIONS_CSV,
                        Config.LGBM_PREDICTIONS_CSV,
                        Config.ENSEMBLE_PREDICTIONS_CSV,
                    ],
                    "reports": [report_dir / "evaluate_digital_twin_summary.json"],
                },
                {
                    "id": "residual_analysis",
                    "name": "Residual analysis",
                    "primary": Config.RESIDUALS_CSV,
                    "artifacts": [Config.RESIDUALS_CSV],
                    "reports": [report_dir / "residual_validation_summary.json"],
                },
                {
                    "id": "anomaly_detection",
                    "name": "Anomaly detection",
                    "primary": Config.ANOMALY_FUSION_CSV,
                    "artifacts": [
                        Config.RESIDUAL_ANOMALY_CSV,
                        Config.IFOREST_CSV,
                        Config.MAHALANOBIS_CSV,
                        Config.ANOMALY_FUSION_CSV,
                    ],
                    "reports": [
                        report_dir / "anomaly_fusion_summary.json",
                        report_dir / "evaluate_anomaly_summary.json",
                    ],
                },
                {
                    "id": "health_monitoring",
                    "name": "Health monitoring",
                    "primary": Config.HEALTH_ALERTS_CSV,
                    "artifacts": [
                        Config.HEALTH_INDEX_CSV,
                        Config.HEALTH_STATES_CSV,
                        Config.HEALTH_TRENDS_CSV,
                        Config.HEALTH_ALERTS_CSV,
                    ],
                    "reports": [
                        Config.HEALTH_SCORE_ENGINE_SUMMARY_JSON,
                        report_dir / "evaluate_health_summary.json",
                    ],
                },
                {
                    "id": "reasoning",
                    "name": "Anomaly reasoning",
                    "primary": Config.ROOT_CAUSE_CSV,
                    "artifacts": [
                        Config.ROOT_CAUSE_CSV,
                        Config.ROOT_CAUSE_MEMORY_CSV,
                        Config.TEMPORAL_REASONING_CSV,
                        Config.SENSOR_DEPENDENCY_GRAPH_CSV,
                    ],
                    "reports": [
                        Config.ROOT_CAUSE_SUMMARY_JSON,
                        Config.TEMPORAL_REASONING_SUMMARY_JSON,
                    ],
                },
                {
                    "id": "explainability",
                    "name": "Explainability",
                    "primary": Config.EXPLANATION_REPORTS_CSV,
                    "artifacts": [Config.SHAP_CSV, Config.EXPLANATION_REPORTS_CSV],
                    "reports": [
                        report_dir / "shap_summary.json",
                        report_dir / "explanation_reports_summary.json",
                    ],
                },
                {
                    "id": "uncertainty",
                    "name": "Confidence and uncertainty",
                    "primary": Config.CONFIDENCE_CSV,
                    "artifacts": [Config.MODEL_AGREEMENT_CSV, Config.CONFIDENCE_CSV],
                    "reports": [
                        report_dir / "model_agreement_summary.json",
                        report_dir / "confidence_scores_summary.json",
                    ],
                },
                {
                    "id": "feedback_learning",
                    "name": "Feedback learning",
                    "primary": Config.FEEDBACK_UPDATES_CSV,
                    "artifacts": [
                        Config.FEEDBACK_UPDATES_CSV,
                        Config.ALERT_MEMORY_CSV,
                        Config.ADAPTIVE_THRESHOLDS_PATH,
                    ],
                    "reports": [
                        report_dir / "learning_updater_summary.json",
                        report_dir / "alert_memory_summary.json",
                    ],
                },
                {
                    "id": "dashboard",
                    "name": "Dashboard data",
                    "primary": Config.DASHBOARD_CSV,
                    "artifacts": [Config.DASHBOARD_CSV],
                    "reports": [self.summary_json],
                },
            ]

            stage_results: List[Dict[str, object]] = []

            for sequence, spec in enumerate(stages, start=1):
                artifact_paths = [path for path in spec["artifacts"] if path.exists() and path.is_file()]
                report_paths = [path for path in spec["reports"] if path.exists() and path.is_file()]

                source_entries = [self._file_metadata(path) for path in artifact_paths]
                source_entries.extend(self._file_metadata(path) for path in report_paths)
                source_entries.sort(key=lambda item: str(item["updated_at"]), reverse=True)

                newest_report: Optional[Path] = None
                if report_paths:
                    newest_report = max(report_paths, key=lambda path: path.stat().st_mtime)

                report_content: Optional[Any] = None
                if newest_report is not None:
                    report_content, _ = self._read_json_artifact(newest_report)

                report_status = (
                    str(report_content.get("status", "")).lower()
                    if isinstance(report_content, dict)
                    else ""
                )

                primary_exists = bool(spec["primary"].exists() and spec["primary"].is_file())
                if report_status in {"failed", "failure", "partial_failure", "error"}:
                    status = "failed"
                    status_source = f"report:{newest_report.name}" if newest_report else "report"
                elif report_status in {"success", "completed", "complete"}:
                    status = "success"
                    status_source = f"report:{newest_report.name}" if newest_report else "report"
                elif primary_exists:
                    status = "success"
                    status_source = f"artifact:{spec['primary'].name}"
                else:
                    status = "not_run"
                    status_source = "no_artifact"

                records_count: Optional[int] = None
                duration_seconds: Optional[float] = None
                message: Optional[str] = None
                if isinstance(report_content, dict):
                    if report_content.get("records_count") is not None:
                        try:
                            records_count = int(report_content["records_count"])
                        except (TypeError, ValueError):
                            records_count = None
                    if report_content.get("duration_seconds") is not None:
                        try:
                            duration_seconds = float(report_content["duration_seconds"])
                        except (TypeError, ValueError):
                            duration_seconds = None
                    if report_content.get("message") is not None:
                        message = str(report_content["message"])

                result: Dict[str, object] = {
                    "sequence": sequence,
                    "id": spec["id"],
                    "name": spec["name"],
                    "status": status,
                    "status_source": status_source,
                    "primary_output": spec["primary"].name,
                    "primary_output_exists": primary_exists,
                    "last_updated_at": source_entries[0]["updated_at"] if source_entries else None,
                    "sources": source_entries,
                }
                if records_count is not None:
                    result["records_count"] = records_count
                if duration_seconds is not None:
                    result["duration_seconds"] = duration_seconds
                if message is not None:
                    result["message"] = message

                stage_results.append(result)

            success_count = sum(stage["status"] == "success" for stage in stage_results)
            failed_count = sum(stage["status"] == "failed" for stage in stage_results)
            not_run_count = sum(stage["status"] == "not_run" for stage in stage_results)

            overall_status = (
                "failed"
                if failed_count > 0
                else "success"
                if success_count == len(stage_results)
                else "partial"
                if success_count > 0
                else "not_run"
            )

            return {
                "status": "success",
                "message": "Pipeline artifact status returned.",
                "records_count": int(len(stage_results)),
                "data": {
                    "overall_status": overall_status,
                    "success_count": int(success_count),
                    "failed_count": int(failed_count),
                    "not_run_count": int(not_run_count),
                    "stages": stage_results,
                },
            }

        except Exception as exc:
            logger.exception("Pipeline status query failed.")
            return self._failed_response(str(exc), data={"stages": []}, records_count=0)

    def feedback_history(self, limit: int = 500) -> Dict[str, object]:
        """Return the newest persisted operator feedback rows, bounded by limit."""
        print("[PROGRESS] Entering DashboardService.feedback_history")

        try:
            safe_limit = max(1, min(int(limit), self.max_records))
            feedback_path = Config.FEEDBACK_UPDATES_CSV
            rows: Deque[Dict[str, object]] = deque(maxlen=safe_limit)
            total_records = 0

            if feedback_path.exists() and feedback_path.is_file():
                for chunk in pd.read_csv(
                    feedback_path,
                    chunksize=min(self.chunk_size, 25_000),
                    low_memory=False,
                ):
                    total_records += int(len(chunk))
                    for record in self._records_from_df(chunk):
                        rows.append(record)
                    del chunk
                    gc.collect()

            feedback_records = list(rows)
            if feedback_records and "timestamp_utc" in feedback_records[0]:
                feedback_records.sort(
                    key=lambda record: str(record.get("timestamp_utc") or ""),
                    reverse=True,
                )

            recent_alerts: List[Dict[str, object]] = []
            alert_path = Config.ALERT_MEMORY_CSV
            if alert_path.exists() and alert_path.is_file():
                requested_alert_columns = [
                    "unit_id",
                    "cycle",
                    "split",
                    "context_id",
                    "alert_level",
                    "final_anomaly_score",
                    "root_cause_pattern",
                    "top_sensor_1",
                    "top_sensor_2",
                    "top_sensor_3",
                    "feedback_status",
                ]
                available_alert_columns = list(pd.read_csv(alert_path, nrows=0).columns)
                usecols = [
                    column
                    for column in requested_alert_columns
                    if column in available_alert_columns
                ]
                if usecols:
                    alert_sample = pd.read_csv(
                        alert_path,
                        usecols=usecols,
                        nrows=min(100, safe_limit),
                        low_memory=False,
                    )
                    recent_alerts = self._records_from_df(alert_sample)

            return {
                "status": "success",
                "message": "Feedback history returned.",
                "records_count": int(len(feedback_records)),
                "total_matching_records": int(total_records),
                "truncated": bool(total_records > len(feedback_records)),
                "max_records": int(safe_limit),
                "data": {
                    "feedback": feedback_records,
                    "feedback_total_records": int(total_records),
                    "feedback_truncated": bool(total_records > len(feedback_records)),
                    "feedback_limit": int(safe_limit),
                    "recent_alerts": recent_alerts,
                    "recent_alerts_sampling": "first_rows_in_persisted_alert_memory",
                },
            }

        except pd.errors.EmptyDataError:
            return {
                "status": "success",
                "message": "Feedback history is empty.",
                "records_count": 0,
                "data": {"feedback": [], "recent_alerts": []},
            }
        except Exception as exc:
            logger.exception("Feedback history query failed.")
            return self._failed_response(
                str(exc),
                data={"feedback": [], "recent_alerts": []},
                records_count=0,
            )

    def adaptive_thresholds(self) -> Dict[str, object]:
        """Return the persisted adaptive threshold configuration."""
        print("[PROGRESS] Entering DashboardService.adaptive_thresholds")

        try:
            path = Config.ADAPTIVE_THRESHOLDS_PATH
            if not path.exists() or not path.is_file():
                return {
                    "status": "not_found",
                    "message": "No adaptive threshold artifact is available.",
                    "data": None,
                }

            content, content_error = self._read_json_artifact(path)
            if content_error is not None:
                return self._failed_response(
                    f"Adaptive threshold artifact could not be loaded: {content_error}",
                    data=None,
                )

            return {
                "status": "success",
                "message": "Adaptive thresholds returned.",
                "data": {
                    "metadata": self._file_metadata(path),
                    "content": content,
                },
            }

        except Exception as exc:
            logger.exception("Adaptive threshold query failed.")
            return self._failed_response(str(exc), data=None)

    def reasoning_summary(self) -> Dict[str, object]:
        """Return bounded aggregate reasoning reports that currently exist."""
        print("[PROGRESS] Entering DashboardService.reasoning_summary")

        try:
            report_paths = [
                Config.ROOT_CAUSE_SUMMARY_JSON,
                Config.ROOT_CAUSE_MEMORY_SUMMARY_JSON,
                Config.TEMPORAL_REASONING_SUMMARY_JSON,
                Config.SENSOR_DEPENDENCY_GRAPH_SUMMARY_JSON,
            ]
            reports = [
                self._report_entry(path, include_content=True)
                for path in report_paths
                if path.exists() and path.is_file()
            ]

            if not reports:
                return {
                    "status": "not_found",
                    "message": "No reasoning summary reports are available.",
                    "records_count": 0,
                    "data": {"reports": []},
                }

            return {
                "status": "success",
                "message": "Reasoning summary reports returned.",
                "records_count": int(len(reports)),
                "data": {"reports": reports},
            }

        except Exception as exc:
            logger.exception("Reasoning summary query failed.")
            return self._failed_response(str(exc), data={"reports": []}, records_count=0)

    def explainability_summary(self) -> Dict[str, object]:
        """Return real, bounded SHAP rows and explainability report summaries."""
        print("[PROGRESS] Entering DashboardService.explainability_summary")

        try:
            report_paths = [
                Config.REPORT_DIR / "shap_summary.json",
                Config.REPORT_DIR / "subsystem_explanations_summary.json",
                Config.REPORT_DIR / "sensor_residual_ranking_summary.json",
                Config.REPORT_DIR / "explanation_reports_summary.json",
            ]
            reports = [
                self._report_entry(path, include_content=True)
                for path in report_paths
                if path.exists() and path.is_file()
            ]

            shap_rows: List[Dict[str, object]] = []
            shap_metadata: Optional[Dict[str, object]] = None
            shap_rows_truncated = False
            if Config.SHAP_CSV.exists() and Config.SHAP_CSV.is_file():
                shap_metadata = self._file_metadata(Config.SHAP_CSV)
                shap_limit = min(500, self.max_records)
                try:
                    shap_df = pd.read_csv(
                        Config.SHAP_CSV,
                        nrows=shap_limit + 1,
                        low_memory=False,
                    )
                    shap_rows_truncated = bool(len(shap_df) > shap_limit)
                    shap_rows = self._records_from_df(shap_df.head(shap_limit))
                except pd.errors.EmptyDataError:
                    shap_rows = []

            if not reports and not shap_rows:
                return {
                    "status": "not_found",
                    "message": "No explainability artifacts are available.",
                    "records_count": 0,
                    "data": {"reports": [], "shap_rows": []},
                }

            return {
                "status": "success",
                "message": "Explainability summary returned.",
                "records_count": int(len(shap_rows)),
                "data": {
                    "reports": reports,
                    "shap_rows": shap_rows,
                    "shap_file": shap_metadata,
                    "shap_rows_truncated": shap_rows_truncated,
                },
            }

        except Exception as exc:
            logger.exception("Explainability summary query failed.")
            return self._failed_response(
                str(exc),
                data={"reports": [], "shap_rows": []},
                records_count=0,
            )

    def analytics_summary(self) -> Dict[str, object]:
        """
        Return lightweight analytics from existing JSON summaries only.

        The full dashboard and analysis CSV files are intentionally not scanned.
        Distributions, split statistics, min/max values, and averages are
        returned exactly when their source report contains them.
        """
        print("[PROGRESS] Entering DashboardService.analytics_summary")

        try:
            report_paths = [
                self.summary_json,
                Config.REPORT_DIR / "evaluate_anomaly_summary.json",
                Config.REPORT_DIR / "evaluate_health_summary.json",
                Config.REPORT_DIR / "evaluate_digital_twin_summary.json",
                Config.ROOT_CAUSE_SUMMARY_JSON,
                Config.TEMPORAL_REASONING_SUMMARY_JSON,
                Config.REPORT_DIR / "confidence_scores_summary.json",
                Config.REPORT_DIR / "model_agreement_summary.json",
                Config.REPORT_DIR / "shap_summary.json",
            ]

            summaries: Dict[str, object] = {}
            sources: List[Dict[str, object]] = []
            for path in report_paths:
                if not path.exists() or not path.is_file():
                    continue

                content, content_error = self._read_json_artifact(path)
                if content_error is not None:
                    continue

                summaries[path.stem] = content
                sources.append(self._file_metadata(path))

            if not summaries:
                return {
                    "status": "not_found",
                    "message": "No analytics summary reports are available.",
                    "records_count": 0,
                    "data": {"summaries": {}, "sources": []},
                }

            return {
                "status": "success",
                "message": "Analytics summaries returned from report artifacts.",
                "records_count": int(len(summaries)),
                "data": {
                    "summaries": summaries,
                    "sources": sources,
                },
            }

        except Exception as exc:
            logger.exception("Analytics summary query failed.")
            return self._failed_response(
                str(exc),
                data={"summaries": {}, "sources": []},
                records_count=0,
            )


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
