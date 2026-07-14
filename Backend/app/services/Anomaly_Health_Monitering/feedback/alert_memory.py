"""
Alert memory for CA-EDT-AHMA.

Role:
Store anomaly alert memory for later comparison and feedback learning.

Reads:
outputs/Anomaly_Health_Monitering/anomaly_fusion.csv
outputs/Anomaly_Health_Monitering/root_cause_analysis.csv
outputs/Anomaly_Health_Monitering/context_clusters.csv, optional fallback
outputs/Anomaly_Health_Monitering/feedback_updates.csv, if available

Writes:
outputs/Anomaly_Health_Monitering/alert_memory.csv
reports/alert_memory_summary.json

Important:
- Stores only anomaly alert rows: Watch, Warning, Critical.
- Does not store Normal rows.
- Does not train models.
- Does not predict RUL.
- Does not use Y_dev/Y_test.
- Does not make maintenance decisions.
- Uses aligned chunk reading instead of full-memory merges.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "feedback/alert_memory.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple, Optional
from itertools import repeat
import gc
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
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class AlertMemory:
    """
    Memory-safe alert memory manager.

    The final memory stores anomaly alert rows only:
        Watch / Warning / Critical
    """

    ALERT_LEVELS = {"Watch", "Warning", "Critical"}

    OUTPUT_COLUMNS = [
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

    def __init__(self, chunk_size: int = 200_000) -> None:
        """
        Initialize alert memory.

        Args:
            chunk_size: Rows processed per chunk.
        """
        print("[PROGRESS] Entering AlertMemory.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "ALERT_MEMORY_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("ALERT_MEMORY_CHUNK_SIZE must be positive.")

        self.anomaly_csv: Path = Config.ANOMALY_FUSION_CSV
        self.root_csv: Path = Config.ROOT_CAUSE_CSV
        self.context_csv: Path = Config.CONTEXT_CSV
        self.feedback_csv: Path = Config.FEEDBACK_UPDATES_CSV
        self.output_csv: Path = Config.ALERT_MEMORY_CSV

        self.summary_json: Path = getattr(
            Config,
            "ALERT_MEMORY_SUMMARY_JSON",
            Config.REPORT_DIR / "alert_memory_summary.json",
        )

        print(f"[PROGRESS] Anomaly CSV: {self.anomaly_csv}")
        print(f"[PROGRESS] Root-cause CSV: {self.root_csv}")
        print(f"[PROGRESS] Context CSV: {self.context_csv}")
        print(f"[PROGRESS] Feedback CSV: {self.feedback_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")

    # ==================================================================================
    # File helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path, required: bool = True) -> int:
        """
        Count CSV rows safely.
        """
        print(f"[PROGRESS] Counting rows safely: {path}")

        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required CSV not found: {path}")
            return 0

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)

        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_columns(self, path: Path, required: bool = True) -> List[str]:
        """
        Read CSV header only.
        """
        print(f"[PROGRESS] Reading header columns from: {path}")

        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required CSV not found: {path}")
            return []

        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_columns(
        self,
        columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate required columns.
        """
        missing = [column for column in required_columns if column not in columns]

        if missing:
            raise KeyError(f"Missing required columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _verify_key_alignment(
        self,
        base_chunk: pd.DataFrame,
        other_chunk: pd.DataFrame,
        label: str,
    ) -> None:
        """
        Verify unit_id/cycle/split alignment.
        """
        key_columns = ["unit_id", "cycle", "split"]

        if len(base_chunk) != len(other_chunk):
            raise ValueError(
                f"Chunk row mismatch for {label}: "
                f"base={len(base_chunk)}, other={len(other_chunk)}"
            )

        base_keys = base_chunk[key_columns].reset_index(drop=True)
        other_keys = other_chunk[key_columns].reset_index(drop=True)

        if not base_keys.equals(other_keys):
            raise ValueError(
                f"Row-key alignment failed for {label}. "
                "Regenerate upstream outputs using the same ordered base row flow."
            )

    # ==================================================================================
    # Empty memory / loading
    # ==================================================================================

    def _empty_alert_memory(self) -> pd.DataFrame:
        """
        Create empty alert memory DataFrame.
        """
        print("[PROGRESS] Entering AlertMemory._empty_alert_memory")
        return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

    def load_memory(self) -> pd.DataFrame:
        """
        Load alert memory.

        Note:
            alert_memory.csv stores anomaly rows only, so it should be much smaller
            than full dashboard/anomaly files.
        """
        print("[PROGRESS] Entering AlertMemory.load_memory")

        try:
            if self.output_csv.exists():
                return read_csv_required(self.output_csv)

            empty_df = self._empty_alert_memory()
            atomic_write_csv(empty_df, self.output_csv)
            return empty_df

        except Exception as exc:
            logger.exception("Failed to load alert memory.")
            raise RuntimeError("Failed to load alert memory.") from exc

    # ==================================================================================
    # Usecols builders
    # ==================================================================================

    def _build_anomaly_usecols(self, columns: List[str]) -> List[str]:
        """
        Build anomaly_fusion.csv usecols.
        """
        required = [
            "unit_id",
            "cycle",
            "split",
            "alert_level",
            "final_anomaly_score",
        ]

        self._validate_columns(columns, required, "anomaly_fusion.csv")

        optional = [
            "gmm_context_id",
            "context_id",
        ]

        usecols = list(required)

        for column in optional:
            if column in columns and column not in usecols:
                usecols.append(column)

        return usecols

    def _build_root_usecols(self, columns: List[str]) -> List[str]:
        """
        Build root_cause_analysis.csv usecols.
        """
        required = [
            "unit_id",
            "cycle",
            "split",
        ]

        self._validate_columns(columns, required, "root_cause_analysis.csv")

        optional = [
            "gmm_context_id",
            "context_id",
            "root_cause_pattern",
            "top_sensor_1",
            "top_sensor_2",
            "top_sensor_3",
        ]

        usecols = list(required)

        for column in optional:
            if column in columns and column not in usecols:
                usecols.append(column)

        return usecols

    def _build_context_usecols(self, columns: List[str]) -> List[str]:
        """
        Build context_clusters.csv usecols.
        """
        required = [
            "unit_id",
            "cycle",
            "split",
            "gmm_context_id",
        ]

        self._validate_columns(columns, required, "context_clusters.csv")

        return required

    # ==================================================================================
    # Feedback map
    # ==================================================================================

    def _load_feedback_status_map(self) -> Dict[Tuple[int, int, int], str]:
        """
        Load feedback updates into dictionary.

        Key:
            (unit_id, cycle, context_id)

        Value:
            feedback_label / feedback_status
        """
        print("[PROGRESS] Entering AlertMemory._load_feedback_status_map")

        if not self.feedback_csv.exists():
            print("[PROGRESS] feedback_updates.csv not found. Using no_feedback defaults.")
            return {}

        columns = self._read_header_columns(self.feedback_csv, required=False)

        if not columns:
            return {}

        required_base = ["unit_id", "cycle"]

        missing_base = [column for column in required_base if column not in columns]

        if missing_base:
            print(f"[WARNING] Feedback CSV missing base columns: {missing_base}")
            return {}

        if "context_id" in columns:
            context_column = "context_id"
        elif "gmm_context_id" in columns:
            context_column = "gmm_context_id"
        else:
            print("[WARNING] Feedback CSV missing context_id/gmm_context_id.")
            return {}

        if "feedback_label" in columns:
            status_column = "feedback_label"
        elif "feedback_status" in columns:
            status_column = "feedback_status"
        else:
            print("[WARNING] Feedback CSV missing feedback_label/feedback_status.")
            return {}

        usecols = ["unit_id", "cycle", context_column, status_column]

        has_feedback_id = "feedback_id" in columns

        if has_feedback_id:
            usecols.append("feedback_id")

        feedback_map: Dict[Tuple[int, int, int], str] = {}
        feedback_order: Dict[Tuple[int, int, int], float] = {}

        chunk_index = 0

        for chunk in pd.read_csv(
            self.feedback_csv,
            usecols=usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        ):
            chunk_index += 1
            print(f"[PROGRESS] Feedback chunk #{chunk_index} rows={len(chunk)}")

            for row_index, row in chunk.iterrows():
                try:
                    key = (
                        int(row["unit_id"]),
                        int(row["cycle"]),
                        int(row[context_column]),
                    )

                    status = str(row[status_column])

                    if has_feedback_id:
                        order_value = float(row["feedback_id"])
                    else:
                        order_value = float(row_index)

                    if key not in feedback_map or order_value >= feedback_order.get(key, -1.0):
                        feedback_map[key] = status
                        feedback_order[key] = order_value

                except Exception:
                    continue

            del chunk
            gc.collect()

        print(f"[PROGRESS] Feedback records loaded: {len(feedback_map)}")
        return feedback_map

    # ==================================================================================
    # Chunk building
    # ==================================================================================

    def _resolve_context_id(
        self,
        anomaly_chunk: pd.DataFrame,
        root_chunk: pd.DataFrame,
        context_chunk: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        Resolve context_id for alert memory.

        Priority:
        1. anomaly_fusion.gmm_context_id
        2. anomaly_fusion.context_id
        3. root_cause_analysis.gmm_context_id
        4. root_cause_analysis.context_id
        5. context_clusters.gmm_context_id
        6. -1
        """
        if "gmm_context_id" in anomaly_chunk.columns:
            return anomaly_chunk["gmm_context_id"].fillna(-1).astype(int)

        if "context_id" in anomaly_chunk.columns:
            return anomaly_chunk["context_id"].fillna(-1).astype(int)

        if "gmm_context_id" in root_chunk.columns:
            return root_chunk["gmm_context_id"].fillna(-1).astype(int)

        if "context_id" in root_chunk.columns:
            return root_chunk["context_id"].fillna(-1).astype(int)

        if context_chunk is not None and "gmm_context_id" in context_chunk.columns:
            return context_chunk["gmm_context_id"].fillna(-1).astype(int)

        return pd.Series([-1] * len(anomaly_chunk), index=anomaly_chunk.index)

    def _build_memory_chunk(
        self,
        anomaly_chunk: pd.DataFrame,
        root_chunk: pd.DataFrame,
        context_chunk: Optional[pd.DataFrame],
        feedback_map: Dict[Tuple[int, int, int], str],
    ) -> pd.DataFrame:
        """
        Build alert memory rows for one aligned chunk.
        """
        context_id = self._resolve_context_id(
            anomaly_chunk=anomaly_chunk,
            root_chunk=root_chunk,
            context_chunk=context_chunk,
        )

        alert_mask = anomaly_chunk["alert_level"].astype(str).isin(self.ALERT_LEVELS)

        if not alert_mask.any():
            return self._empty_alert_memory()

        selected_anomaly = anomaly_chunk.loc[alert_mask].reset_index(drop=True)
        selected_root = root_chunk.loc[alert_mask].reset_index(drop=True)
        selected_context_id = context_id.loc[alert_mask].reset_index(drop=True)

        memory_df = pd.DataFrame(
            {
                "unit_id": selected_anomaly["unit_id"].astype(int),
                "cycle": selected_anomaly["cycle"].astype(int),
                "split": selected_anomaly["split"].astype(str),
                "context_id": selected_context_id.astype(int),
                "alert_level": selected_anomaly["alert_level"].astype(str),
                "final_anomaly_score": pd.to_numeric(
                    selected_anomaly["final_anomaly_score"],
                    errors="coerce",
                ).fillna(0.0),
                "root_cause_pattern": (
                    selected_root["root_cause_pattern"].astype(str)
                    if "root_cause_pattern" in selected_root.columns
                    else "unknown"
                ),
                "top_sensor_1": (
                    selected_root["top_sensor_1"].astype(str)
                    if "top_sensor_1" in selected_root.columns
                    else "unknown"
                ),
                "top_sensor_2": (
                    selected_root["top_sensor_2"].astype(str)
                    if "top_sensor_2" in selected_root.columns
                    else "unknown"
                ),
                "top_sensor_3": (
                    selected_root["top_sensor_3"].astype(str)
                    if "top_sensor_3" in selected_root.columns
                    else "unknown"
                ),
            }
        )

        if feedback_map:
            statuses = []

            for unit_id, cycle, context in zip(
                memory_df["unit_id"].to_numpy(),
                memory_df["cycle"].to_numpy(),
                memory_df["context_id"].to_numpy(),
            ):
                statuses.append(
                    feedback_map.get(
                        (int(unit_id), int(cycle), int(context)),
                        "no_feedback",
                    )
                )

            memory_df["feedback_status"] = statuses

        else:
            memory_df["feedback_status"] = "no_feedback"

        for column in self.OUTPUT_COLUMNS:
            if column not in memory_df.columns:
                memory_df[column] = "unknown"

        return memory_df[self.OUTPUT_COLUMNS].copy()

    # ==================================================================================
    # Main build / update
    # ==================================================================================

    def build_memory_from_current_alerts(self) -> Dict[str, object]:
        """
        Build alert memory from current anomaly and root-cause outputs.

        Returns:
            Dict summary.
        """
        print("[PROGRESS] Entering AlertMemory.build_memory_from_current_alerts")

        started = perf_counter()

        anomaly_rows = self._count_csv_rows(self.anomaly_csv, required=True)
        root_rows = self._count_csv_rows(self.root_csv, required=True)

        if anomaly_rows <= 0:
            raise ValueError("anomaly_fusion.csv contains zero rows.")

        if anomaly_rows != root_rows:
            raise ValueError(
                "Alert memory input row-count mismatch: "
                f"anomaly={anomaly_rows}, root={root_rows}"
            )

        anomaly_columns = self._read_header_columns(self.anomaly_csv, required=True)
        root_columns = self._read_header_columns(self.root_csv, required=True)

        anomaly_usecols = self._build_anomaly_usecols(anomaly_columns)
        root_usecols = self._build_root_usecols(root_columns)

        context_needed = (
            "gmm_context_id" not in anomaly_usecols
            and "context_id" not in anomaly_usecols
            and "gmm_context_id" not in root_usecols
            and "context_id" not in root_usecols
            and self.context_csv.exists()
        )

        context_usecols: Optional[List[str]] = None

        if context_needed:
            context_rows = self._count_csv_rows(self.context_csv, required=True)

            if context_rows != anomaly_rows:
                raise ValueError(
                    "Context row-count mismatch for alert memory fallback: "
                    f"context={context_rows}, anomaly={anomaly_rows}"
                )

            context_columns = self._read_header_columns(self.context_csv, required=True)
            context_usecols = self._build_context_usecols(context_columns)

        feedback_map = self._load_feedback_status_map()

        temp_output_path = self.output_csv.with_suffix(self.output_csv.suffix + ".tmp")

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        if temp_output_path.exists():
            print("[PROGRESS] Removing old temporary alert memory CSV")
            temp_output_path.unlink()

        anomaly_iter = pd.read_csv(
            self.anomaly_csv,
            usecols=anomaly_usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        )

        root_iter = pd.read_csv(
            self.root_csv,
            usecols=root_usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        )

        if context_needed and context_usecols is not None:
            context_iter = pd.read_csv(
                self.context_csv,
                usecols=context_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )
        else:
            context_iter = repeat(None)

        first_batch = True
        total_rows_processed = 0
        total_alert_rows_written = 0
        chunk_index = 0

        alert_counts: Dict[str, int] = {}
        split_counts: Dict[str, int] = {}
        feedback_counts: Dict[str, int] = {}

        print("[PROGRESS] Starting memory-safe alert memory build")

        for anomaly_chunk, root_chunk, context_chunk in zip(
            anomaly_iter,
            root_iter,
            context_iter,
        ):
            chunk_index += 1

            anomaly_chunk = anomaly_chunk.reset_index(drop=True)
            root_chunk = root_chunk.reset_index(drop=True)

            if context_chunk is not None:
                context_chunk = context_chunk.reset_index(drop=True)

            print("=" * 100)
            print(f"[PROGRESS] Alert memory chunk #{chunk_index}")
            print(f"[PROGRESS] Chunk rows: {len(anomaly_chunk)}")

            self._verify_key_alignment(
                anomaly_chunk,
                root_chunk,
                "root_cause_analysis.csv",
            )

            if context_chunk is not None:
                self._verify_key_alignment(
                    anomaly_chunk,
                    context_chunk,
                    "context_clusters.csv",
                )

            memory_chunk = self._build_memory_chunk(
                anomaly_chunk=anomaly_chunk,
                root_chunk=root_chunk,
                context_chunk=context_chunk,
                feedback_map=feedback_map,
            )

            total_rows_processed += int(len(anomaly_chunk))

            if not memory_chunk.empty:
                memory_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_alert_rows_written += int(len(memory_chunk))

                for column, target_dict in [
                    ("alert_level", alert_counts),
                    ("split", split_counts),
                    ("feedback_status", feedback_counts),
                ]:
                    unique_values, unique_counts = np.unique(
                        memory_chunk[column].astype(str).to_numpy(dtype=object),
                        return_counts=True,
                    )

                    for value, count in zip(unique_values, unique_counts):
                        target_dict[str(value)] = target_dict.get(str(value), 0) + int(count)

            print(f"[PROGRESS] Total source rows processed: {total_rows_processed}")
            print(f"[PROGRESS] Total alert memory rows written: {total_alert_rows_written}")

            del anomaly_chunk
            del root_chunk
            del context_chunk
            del memory_chunk
            gc.collect()

        if first_batch:
            print("[PROGRESS] No anomaly alerts found. Writing empty alert memory CSV.")
            empty_df = self._empty_alert_memory()
            empty_df.to_csv(temp_output_path, index=False)

        print("=" * 100)
        print("[PROGRESS] All alert memory chunks completed")
        print(f"[PROGRESS] Source rows processed: {total_rows_processed}")
        print(f"[PROGRESS] Expected source rows: {anomaly_rows}")
        print(f"[PROGRESS] Alert memory rows written: {total_alert_rows_written}")

        if total_rows_processed != anomaly_rows:
            raise ValueError(
                "Alert memory source row count mismatch. "
                f"processed={total_rows_processed}, expected={anomaly_rows}. "
                "Final alert_memory.csv will not be replaced."
            )

        os.replace(temp_output_path, self.output_csv)

        duration = perf_counter() - started

        summary = {
            "status": "success",
            "message": "Alert memory built from current anomaly alerts.",
            "output_file": str(self.output_csv),
            "source_records_processed": int(total_rows_processed),
            "records_count": int(total_alert_rows_written),
            "stores_only_alert_rows": True,
            "stored_alert_levels": sorted(list(self.ALERT_LEVELS)),
            "alert_counts": alert_counts,
            "split_counts": split_counts,
            "feedback_counts": feedback_counts,
            "feedback_records_loaded": int(len(feedback_map)),
            "context_fallback_used": bool(context_needed),
            "chunk_size": int(self.chunk_size),
            "duration_seconds": float(duration),
            "duration_minutes": float(duration / 60.0),
            "leakage_audit": {
                "does_not_train_model": True,
                "does_not_predict_rul": True,
                "does_not_make_maintenance_decisions": True,
                "does_not_use_y_dev_y_test": True,
                "does_not_use_t_dev_t_test": True,
                "uses_anomaly_fusion": True,
                "uses_root_cause_analysis": True,
                "uses_context_only_for_context_id_fallback": bool(context_needed),
                "uses_feedback_if_available": bool(feedback_map),
            },
        }

        print(f"[PROGRESS] Writing alert memory summary to: {self.summary_json}")
        atomic_write_json(summary, self.summary_json)

        return summary

    def update_feedback_status(self) -> Dict[str, object]:
        """
        Re-apply feedback statuses to an existing alert_memory.csv.

        This is memory-safe and chunked.
        """
        print("[PROGRESS] Entering AlertMemory.update_feedback_status")

        if not self.output_csv.exists():
            raise FileNotFoundError(
                f"alert_memory.csv not found: {self.output_csv}. "
                "Run build_memory_from_current_alerts first."
            )

        feedback_map = self._load_feedback_status_map()

        if not feedback_map:
            return {
                "status": "success",
                "message": "No feedback updates found. Alert memory unchanged.",
                "output_file": str(self.output_csv),
                "feedback_records_loaded": 0,
            }

        rows = self._count_csv_rows(self.output_csv, required=True)
        columns = self._read_header_columns(self.output_csv, required=True)

        self._validate_columns(
            columns,
            ["unit_id", "cycle", "context_id", "feedback_status"],
            "alert_memory.csv",
        )

        temp_output_path = self.output_csv.with_suffix(self.output_csv.suffix + ".tmp")

        if temp_output_path.exists():
            temp_output_path.unlink()

        first_batch = True
        total_rows_written = 0

        for chunk in pd.read_csv(
            self.output_csv,
            chunksize=self.chunk_size,
            low_memory=True,
        ):
            chunk = chunk.reset_index(drop=True)

            statuses = []

            for unit_id, cycle, context_id in zip(
                chunk["unit_id"].to_numpy(),
                chunk["cycle"].to_numpy(),
                chunk["context_id"].to_numpy(),
            ):
                statuses.append(
                    feedback_map.get(
                        (int(unit_id), int(cycle), int(context_id)),
                        "no_feedback",
                    )
                )

            chunk["feedback_status"] = statuses

            chunk.to_csv(
                temp_output_path,
                mode="w" if first_batch else "a",
                header=first_batch,
                index=False,
            )

            first_batch = False
            total_rows_written += int(len(chunk))

            del chunk
            gc.collect()

        if total_rows_written != rows:
            raise ValueError(
                "Feedback update row count mismatch. "
                f"written={total_rows_written}, expected={rows}."
            )

        os.replace(temp_output_path, self.output_csv)

        return {
            "status": "success",
            "message": "Alert memory feedback statuses updated.",
            "output_file": str(self.output_csv),
            "records_count": int(total_rows_written),
            "feedback_records_loaded": int(len(feedback_map)),
        }

    def run(self) -> Dict[str, object]:
        """
        Run alert memory update.
        """
        print("[PROGRESS] Entering AlertMemory.run")

        try:
            summary = self.build_memory_from_current_alerts()

            response = {
                "status": "success",
                "message": "Alert memory updated safely.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(summary["records_count"]),
                "source_records_processed": int(summary["source_records_processed"]),
            }

            print(f"[PROGRESS] Alert memory response: {response}")

            logger.info(
                "Alert memory completed. alert_rows=%s source_rows=%s",
                summary["records_count"],
                summary["source_records_processed"],
            )

            return response

        except Exception as exc:
            print(f"[ERROR] Alert memory stage failed: {exc}")
            logger.exception("Alert memory stage failed.")
            raise RuntimeError("Alert memory stage failed.") from exc


def run_alert_memory() -> Dict[str, object]:
    """
    Execute alert memory update.
    """
    print("[PROGRESS] Entering run_alert_memory")

    memory = AlertMemory()
    return memory.run()


if __name__ == "__main__":
    print("[PROGRESS] alert_memory.py execution started")
    result = run_alert_memory()
    print("[PROGRESS] alert_memory.py execution finished successfully")
    print(result)