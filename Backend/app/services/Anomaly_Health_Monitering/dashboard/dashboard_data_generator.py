"""
Dashboard data generator for CA-EDT-AHMA.

Role:
Generate final dashboard_data.csv with all required dashboard columns.

Reads:
outputs/Anomaly_Health_Monitering/context_clusters.csv
outputs/Anomaly_Health_Monitering/health_states.csv
outputs/Anomaly_Health_Monitering/root_cause_analysis.csv
outputs/Anomaly_Health_Monitering/explanation_reports.csv
outputs/Anomaly_Health_Monitering/confidence_scores.csv
outputs/Anomaly_Health_Monitering/feedback_updates.csv, optional

Writes:
outputs/Anomaly_Health_Monitering/dashboard_data.csv
reports/dashboard_data_summary.json

Important:
- Does not load full CSV files into RAM.
- Uses aligned chunk processing.
- Does not use Y_dev/Y_test.
- Does not predict RUL.
- Does not make maintenance decisions.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "dashboard/dashboard_data_generator.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple
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
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class DashboardDataGenerator:
    """
    Generates final dashboard-ready CSV using memory-safe aligned chunks.
    """

    DEFAULT_DASHBOARD_COLUMNS: List[str] = [
        "unit_id",
        "cycle",
        "split",
        "kmeans_context_id",
        "gmm_context_id",
        "context_confidence",
        "health_index",
        "remaining_health_percentage",
        "health_state",
        "final_anomaly_score",
        "alert_level",
        "anomaly_detected",
        "anomaly_status",
        "top_sensor_1",
        "top_sensor_2",
        "top_sensor_3",
        "contribution_1",
        "contribution_2",
        "contribution_3",
        "root_cause_pattern",
        "inspection_focus",
        "explanation_text",
        "model_agreement_score",
        "confidence_score",
        "uncertainty_score",
        "reliability_score",
        "feedback_status",
    ]

    def __init__(self, chunk_size: int = 150_000) -> None:
        """
        Initialize dashboard data generator.

        Args:
            chunk_size: Number of rows per processing chunk.
        """
        print("[PROGRESS] Entering DashboardDataGenerator.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "DASHBOARD_GENERATION_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("DASHBOARD_GENERATION_CHUNK_SIZE must be positive.")

        self.context_csv: Path = Config.CONTEXT_CSV
        self.health_csv: Path = Config.HEALTH_STATES_CSV
        self.root_csv: Path = Config.ROOT_CAUSE_CSV
        self.explanation_csv: Path = Config.EXPLANATION_REPORTS_CSV
        self.confidence_csv: Path = Config.CONFIDENCE_CSV
        self.feedback_csv: Path = Config.FEEDBACK_UPDATES_CSV
        self.output_csv: Path = Config.DASHBOARD_CSV

        self.summary_json: Path = getattr(
            Config,
            "DASHBOARD_DATA_SUMMARY_JSON",
            Config.REPORT_DIR / "dashboard_data_summary.json",
        )

        self.final_columns: List[str] = list(
            getattr(Config, "FINAL_DASHBOARD_COLUMNS", self.DEFAULT_DASHBOARD_COLUMNS)
        )

        print(f"[PROGRESS] Context CSV: {self.context_csv}")
        print(f"[PROGRESS] Health CSV: {self.health_csv}")
        print(f"[PROGRESS] Root-cause CSV: {self.root_csv}")
        print(f"[PROGRESS] Explanation CSV: {self.explanation_csv}")
        print(f"[PROGRESS] Confidence CSV: {self.confidence_csv}")
        print(f"[PROGRESS] Feedback CSV: {self.feedback_csv}")
        print(f"[PROGRESS] Dashboard output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")

    # ==================================================================================
    # File helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path, required: bool = True) -> int:
        """
        Count CSV data rows without loading the file.
        """
        print(f"[PROGRESS] Counting rows safely: {path}")

        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required dashboard input not found: {path}")
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
                raise FileNotFoundError(f"Required dashboard input not found: {path}")
            return []

        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_required_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate required columns.
        """
        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            raise KeyError(f"Missing required dashboard columns in {label}: {missing}")

        print(f"[PROGRESS] Required dashboard columns validated for {label}")

    def _safe_usecols(
        self,
        path: Path,
        required_columns: List[str],
        optional_columns: List[str],
        label: str,
    ) -> List[str]:
        """
        Build usecols with required + existing optional columns.
        """
        available = self._read_header_columns(path, required=True)
        self._validate_required_columns(available, required_columns, label)

        usecols: List[str] = []

        for column in required_columns + optional_columns:
            if column in available and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] {label} usecols: {usecols}")
        return usecols

    def _verify_alignment(
        self,
        base_chunk: pd.DataFrame,
        other_chunk: pd.DataFrame,
        merge_columns: List[str],
        label: str,
    ) -> None:
        """
        Verify that two chunks contain the same ordered row keys.
        """
        if len(base_chunk) != len(other_chunk):
            raise ValueError(
                f"Dashboard chunk size mismatch for {label}: "
                f"{len(base_chunk)} != {len(other_chunk)}"
            )

        base_keys = base_chunk[merge_columns].reset_index(drop=True)
        other_keys = other_chunk[merge_columns].reset_index(drop=True)

        if not base_keys.equals(other_keys):
            raise ValueError(
                f"Dashboard row alignment failed for {label}. "
                "Regenerate upstream outputs from the same ordered input."
            )

    # ==================================================================================
    # Feedback helpers
    # ==================================================================================

    def _load_feedback_map(self) -> Dict[Tuple[int, int, int], str]:
        """
        Load optional feedback updates into a dictionary.

        Key:
            (unit_id, cycle, context_id)

        Value:
            feedback_label / feedback_status
        """
        print("[PROGRESS] Entering DashboardDataGenerator._load_feedback_map")

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

        print(f"[PROGRESS] Feedback records loaded into dashboard map: {len(feedback_map)}")
        return feedback_map

    def _apply_feedback_status(
        self,
        dashboard_chunk: pd.DataFrame,
        feedback_map: Dict[Tuple[int, int, int], str],
    ) -> pd.DataFrame:
        """
        Apply feedback status to one dashboard chunk.
        """
        result = dashboard_chunk.copy()

        result["feedback_status"] = "no_feedback"

        if not feedback_map:
            return result

        if "gmm_context_id" in result.columns:
            context_values = result["gmm_context_id"].fillna(-1).astype(int).to_numpy()
        elif "context_id" in result.columns:
            context_values = result["context_id"].fillna(-1).astype(int).to_numpy()
        else:
            context_values = np.full(len(result), -1, dtype=np.int64)

        statuses = []

        for unit_id, cycle, context_id in zip(
            result["unit_id"].to_numpy(),
            result["cycle"].to_numpy(),
            context_values,
        ):
            statuses.append(
                feedback_map.get(
                    (int(unit_id), int(cycle), int(context_id)),
                    "no_feedback",
                )
            )

        result["feedback_status"] = statuses

        return result

    # ==================================================================================
    # Dashboard chunk builder
    # ==================================================================================

    def _default_values(self) -> Dict[str, object]:
        """
        Default values for missing dashboard columns.
        """
        return {
            "kmeans_context_id": -1,
            "gmm_context_id": -1,
            "context_confidence": 0.0,
            "health_index": 100.0,
            "remaining_health_percentage": 100.0,
            "health_state": "Healthy",
            "health_state_explanation": "Health state not available.",
            "final_anomaly_score": 0.0,
            "alert_level": "Normal",
            "anomaly_detected": False,
            "anomaly_status": "No_Anomaly",
            "residual_trend_score": 0.0,
            "anomaly_persistence_score": 0.0,
            "top_sensor_1": "none",
            "top_sensor_2": "none",
            "top_sensor_3": "none",
            "contribution_1": 0.0,
            "contribution_2": 0.0,
            "contribution_3": 0.0,
            "root_cause_pattern": "normal_or_no_pattern",
            "inspection_focus": "No inspection focus available.",
            "explanation_text": "No anomaly explanation available.",
            "model_agreement_score": 0.0,
            "context_confidence_score": 0.0,
            "confidence_score": 0.0,
            "uncertainty_score": 1.0,
            "reliability_score": 0.0,
            "feedback_status": "no_feedback",
        }

    def _build_dashboard_chunk(
        self,
        health: pd.DataFrame,
        context: pd.DataFrame,
        root: pd.DataFrame,
        explanation: pd.DataFrame,
        confidence: pd.DataFrame,
        feedback_map: Dict[Tuple[int, int, int], str],
    ) -> pd.DataFrame:
        """
        Build one dashboard chunk from aligned upstream chunks.
        """
        merge_columns = ["unit_id", "cycle", "split"]

        dashboard_chunk = health.reset_index(drop=True).copy()

        for source in [context, root, explanation, confidence]:
            source = source.reset_index(drop=True)

            for column in source.columns:
                if column not in merge_columns:
                    dashboard_chunk[column] = source[column].to_numpy()

        defaults = self._default_values()

        for column, default_value in defaults.items():
            if column not in dashboard_chunk.columns:
                dashboard_chunk[column] = default_value

            dashboard_chunk[column] = dashboard_chunk[column].fillna(default_value)

        dashboard_chunk["anomaly_detected"] = (
            dashboard_chunk["alert_level"].astype(str).ne("Normal")
        )

        dashboard_chunk["anomaly_status"] = np.where(
            dashboard_chunk["anomaly_detected"],
            "Anomaly_Detected",
            "No_Anomaly",
        )

        dashboard_chunk = self._apply_feedback_status(
            dashboard_chunk=dashboard_chunk,
            feedback_map=feedback_map,
        )

        for column in self.final_columns:
            if column not in dashboard_chunk.columns:
                dashboard_chunk[column] = defaults.get(column, np.nan)

        dashboard_chunk = dashboard_chunk[self.final_columns].copy()

        return dashboard_chunk

    # ==================================================================================
    # Main generation
    # ==================================================================================

    def generate(self) -> Dict[str, object]:
        """
        Generate dashboard_data.csv.

        Important:
            This method is chunk-only. It does not return a full DataFrame.
        """
        print("[PROGRESS] Entering DashboardDataGenerator.generate")

        started = perf_counter()

        try:
            merge_columns = ["unit_id", "cycle", "split"]

            inputs = {
                "context": self.context_csv,
                "health": self.health_csv,
                "root_cause": self.root_csv,
                "explanation": self.explanation_csv,
                "confidence": self.confidence_csv,
            }

            row_counts = {
                label: self._count_csv_rows(path, required=True)
                for label, path in inputs.items()
            }

            if len(set(row_counts.values())) != 1:
                raise ValueError(f"Dashboard input row counts do not match: {row_counts}")

            expected_rows = next(iter(row_counts.values()))

            if expected_rows <= 0:
                raise ValueError("Dashboard input files contain no data rows.")

            print(f"[PROGRESS] Dashboard expected rows: {expected_rows}")

            health_usecols = self._safe_usecols(
                path=self.health_csv,
                required_columns=merge_columns,
                optional_columns=[
                    "health_index",
                    "remaining_health_percentage",
                    "health_state",
                    "health_state_explanation",
                    "final_anomaly_score",
                    "alert_level",
                    "residual_trend_score",
                    "anomaly_persistence_score",
                ],
                label="health_states.csv",
            )

            context_usecols = self._safe_usecols(
                path=self.context_csv,
                required_columns=merge_columns,
                optional_columns=[
                    "kmeans_context_id",
                    "gmm_context_id",
                    "context_probability",
                    "context_confidence",
                ],
                label="context_clusters.csv",
            )

            root_usecols = self._safe_usecols(
                path=self.root_csv,
                required_columns=merge_columns,
                optional_columns=[
                    "top_sensor_1",
                    "top_sensor_2",
                    "top_sensor_3",
                    "contribution_1",
                    "contribution_2",
                    "contribution_3",
                    "root_cause_pattern",
                    "inspection_focus",
                ],
                label="root_cause_analysis.csv",
            )

            explanation_usecols = self._safe_usecols(
                path=self.explanation_csv,
                required_columns=merge_columns,
                optional_columns=[
                    "explanation_text",
                    "reasoning_summary",
                    "recommendation_text",
                ],
                label="explanation_reports.csv",
            )

            confidence_usecols = self._safe_usecols(
                path=self.confidence_csv,
                required_columns=merge_columns,
                optional_columns=[
                    "model_disagreement",
                    "normalized_model_disagreement",
                    "model_agreement_score",
                    "uncertainty_from_model_disagreement",
                    "context_confidence",
                    "anomaly_persistence_score",
                    "data_quality_score",
                    "confidence_score",
                    "uncertainty_score",
                    "reliability_score",
                ],
                label="confidence_scores.csv",
            )

            feedback_map = self._load_feedback_map()

            health_iter = pd.read_csv(
                self.health_csv,
                usecols=health_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            context_iter = pd.read_csv(
                self.context_csv,
                usecols=context_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            root_iter = pd.read_csv(
                self.root_csv,
                usecols=root_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            explanation_iter = pd.read_csv(
                self.explanation_csv,
                usecols=explanation_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            confidence_iter = pd.read_csv(
                self.confidence_csv,
                usecols=confidence_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            temp_path = self.output_csv.with_suffix(self.output_csv.suffix + ".tmp")

            if temp_path.exists():
                print("[PROGRESS] Removing old dashboard temporary file")
                temp_path.unlink()

            first_chunk = True
            rows_written = 0
            chunk_index = 0

            alert_counts: Dict[str, int] = {}
            health_state_counts: Dict[str, int] = {}
            split_counts: Dict[str, int] = {}
            feedback_counts: Dict[str, int] = {}

            health_sum = 0.0
            confidence_sum = 0.0
            uncertainty_sum = 0.0
            reliability_sum = 0.0

            print("[PROGRESS] Starting memory-safe dashboard generation")

            try:
                for health, context, root, explanation, confidence in zip(
                    health_iter,
                    context_iter,
                    root_iter,
                    explanation_iter,
                    confidence_iter,
                ):
                    chunk_index += 1

                    health = health.reset_index(drop=True)
                    context = context.reset_index(drop=True)
                    root = root.reset_index(drop=True)
                    explanation = explanation.reset_index(drop=True)
                    confidence = confidence.reset_index(drop=True)

                    print("=" * 100)
                    print(f"[PROGRESS] Dashboard chunk #{chunk_index}")
                    print(f"[PROGRESS] Chunk rows: {len(health)}")

                    self._verify_alignment(health, context, merge_columns, "context")
                    self._verify_alignment(health, root, merge_columns, "root_cause")
                    self._verify_alignment(health, explanation, merge_columns, "explanation")
                    self._verify_alignment(health, confidence, merge_columns, "confidence")

                    dashboard_chunk = self._build_dashboard_chunk(
                        health=health,
                        context=context,
                        root=root,
                        explanation=explanation,
                        confidence=confidence,
                        feedback_map=feedback_map,
                    )

                    dashboard_chunk.to_csv(
                        temp_path,
                        mode="w" if first_chunk else "a",
                        header=first_chunk,
                        index=False,
                    )

                    first_chunk = False
                    rows_written += int(len(dashboard_chunk))

                    for column, target_dict in [
                        ("alert_level", alert_counts),
                        ("health_state", health_state_counts),
                        ("split", split_counts),
                        ("feedback_status", feedback_counts),
                    ]:
                        if column in dashboard_chunk.columns:
                            counts = dashboard_chunk[column].astype(str).value_counts().to_dict()
                            for key, value in counts.items():
                                target_dict[str(key)] = target_dict.get(str(key), 0) + int(value)

                    if "health_index" in dashboard_chunk.columns:
                        health_sum += float(
                            pd.to_numeric(
                                dashboard_chunk["health_index"],
                                errors="coerce",
                            ).fillna(0.0).sum()
                        )

                    if "confidence_score" in dashboard_chunk.columns:
                        confidence_sum += float(
                            pd.to_numeric(
                                dashboard_chunk["confidence_score"],
                                errors="coerce",
                            ).fillna(0.0).sum()
                        )

                    if "uncertainty_score" in dashboard_chunk.columns:
                        uncertainty_sum += float(
                            pd.to_numeric(
                                dashboard_chunk["uncertainty_score"],
                                errors="coerce",
                            ).fillna(0.0).sum()
                        )

                    if "reliability_score" in dashboard_chunk.columns:
                        reliability_sum += float(
                            pd.to_numeric(
                                dashboard_chunk["reliability_score"],
                                errors="coerce",
                            ).fillna(0.0).sum()
                        )

                    print(f"[PROGRESS] Dashboard rows written so far: {rows_written}")

                    del health
                    del context
                    del root
                    del explanation
                    del confidence
                    del dashboard_chunk
                    gc.collect()

                if rows_written != expected_rows:
                    raise ValueError(
                        "Dashboard rows written do not match inputs: "
                        f"{rows_written} != {expected_rows}"
                    )

                os.replace(temp_path, self.output_csv)

            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise

            duration = perf_counter() - started

            normal_records = int(alert_counts.get("Normal", 0))
            anomaly_records = int(rows_written - normal_records)

            summary = {
                "status": "success",
                "message": "Final dashboard_data.csv generated.",
                "output_file": str(self.output_csv),
                "records_count": int(rows_written),
                "expected_rows": int(expected_rows),
                "chunk_size": int(self.chunk_size),
                "chunks_processed": int(chunk_index),
                "row_counts": row_counts,
                "alert_counts": alert_counts,
                "health_state_counts": health_state_counts,
                "split_counts": split_counts,
                "feedback_counts": feedback_counts,
                "anomaly_summary": {
                    "normal_records": normal_records,
                    "anomaly_records": anomaly_records,
                    "anomaly_ratio": float(anomaly_records / max(rows_written, 1)),
                    "anomaly_detected": bool(anomaly_records > 0),
                },
                "averages": {
                    "average_health_index": float(health_sum / max(rows_written, 1)),
                    "average_confidence_score": float(confidence_sum / max(rows_written, 1)),
                    "average_uncertainty_score": float(uncertainty_sum / max(rows_written, 1)),
                    "average_reliability_score": float(reliability_sum / max(rows_written, 1)),
                },
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "uses_aligned_chunk_processing": True,
                    "full_dataframe_merge_used": False,
                    "full_csv_read_used": False,
                },
            }

            print(f"[PROGRESS] Writing dashboard summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info(
                "Dashboard data generated. rows=%s duration_seconds=%.2f",
                rows_written,
                duration,
            )

            return summary

        except Exception as exc:
            logger.exception("Dashboard data generation failed.")
            raise RuntimeError("Dashboard data generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run dashboard data generation.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering DashboardDataGenerator.run")

        try:
            summary = self.generate()

            response = {
                "status": "success",
                "message": "Final dashboard_data.csv generated.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(summary["records_count"]),
            }

            print(f"[PROGRESS] Dashboard data generator response: {response}")

            return response

        except Exception as exc:
            logger.exception("Dashboard data generator stage failed.")
            raise RuntimeError("Dashboard data generator stage failed.") from exc


def run_dashboard_data_generation() -> Dict[str, object]:
    """
    Execute dashboard data generation.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering run_dashboard_data_generation")

    generator = DashboardDataGenerator()
    return generator.run()


if __name__ == "__main__":
    print("[PROGRESS] dashboard_data_generator.py execution started")
    result = run_dashboard_data_generation()
    print("[PROGRESS] dashboard_data_generator.py execution finished successfully")
    print(result)