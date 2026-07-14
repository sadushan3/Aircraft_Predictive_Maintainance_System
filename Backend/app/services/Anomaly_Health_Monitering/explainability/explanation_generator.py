"""
Human-readable explanation generator for CA-EDT-AHMA.

Role:
Generate dashboard-ready natural language explanations.

Each explanation includes:
- Context ID
- Alert level
- Health index
- Health state
- Top contributing sensors
- Sensor contribution percentages
- Root-cause pattern
- Real confidence / uncertainty / reliability values when available
- Inspection focus

Reads:
outputs/Anomaly_Health_Monitering/health_states.csv
outputs/Anomaly_Health_Monitering/root_cause_analysis.csv
outputs/Anomaly_Health_Monitering/context_clusters.csv
outputs/Anomaly_Health_Monitering/confidence_scores.csv, if available

Writes:
outputs/Anomaly_Health_Monitering/explanation_reports.csv
reports/explanation_reports_summary.json

Memory-safe:
- Does not load full files into RAM.
- Reads aligned chunks.
- Does not rerun heavy explainability stages automatically.
- Writes to temporary CSV first.
- Replaces final CSV only after successful completion.

Important:
- Does not train a model.
- Does not predict RUL.
- Does not make maintenance decisions.
- Does not use Y_dev/Y_test.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "explainability/explanation_generator.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Optional
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


class ExplanationGenerator:
    """
    Memory-safe human-readable explanation generator.
    """

    OUTPUT_COLUMNS: List[str] = [
        "unit_id",
        "cycle",
        "split",
        "kmeans_context_id",
        "gmm_context_id",
        "context_confidence",
        "final_anomaly_score",
        "alert_level",
        "health_index",
        "health_state",
        "remaining_health_percentage",
        "anomaly_persistence_score",
        "residual_trend_score",
        "top_sensor_1",
        "top_sensor_2",
        "top_sensor_3",
        "contribution_1",
        "contribution_2",
        "contribution_3",
        "root_cause_pattern",
        "inspection_focus",
        "total_abs_residual",
        "top3_contribution_sum",
        "dominant_detector",
        "model_agreement_score",
        "confidence_score",
        "uncertainty_score",
        "reliability_score",
        "explanation_text",
        "maintenance_decision",
        "component_role",
    ]

    def __init__(self, chunk_size: int = 100_000) -> None:
        """
        Initialize explanation generator.

        Args:
            chunk_size: Rows processed per chunk.
        """
        print("[PROGRESS] Entering ExplanationGenerator.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "EXPLANATION_GENERATOR_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("EXPLANATION_GENERATOR_CHUNK_SIZE must be positive.")

        self.health_csv: Path = Config.HEALTH_STATES_CSV
        self.root_csv: Path = Config.ROOT_CAUSE_CSV
        self.context_csv: Path = Config.CONTEXT_CSV
        self.confidence_csv: Path = Config.CONFIDENCE_CSV

        self.output_csv: Path = Config.EXPLANATION_REPORTS_CSV

        self.summary_json: Path = getattr(
            Config,
            "EXPLANATION_REPORTS_SUMMARY_JSON",
            Config.REPORT_DIR / "explanation_reports_summary.json",
        )

        print(f"[PROGRESS] Health CSV: {self.health_csv}")
        print(f"[PROGRESS] Root-cause CSV: {self.root_csv}")
        print(f"[PROGRESS] Context CSV: {self.context_csv}")
        print(f"[PROGRESS] Confidence CSV: {self.confidence_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")

    # ==================================================================================
    # File helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path, required: bool = True) -> int:
        """
        Count CSV rows without loading full file.

        Args:
            path: CSV path.
            required: Whether missing file should raise.

        Returns:
            int: Data row count.
        """
        print(f"[PROGRESS] Counting rows safely: {path}")

        if not path.exists():
            if required:
                raise FileNotFoundError(f"CSV file not found: {path}")
            return 0

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)
        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_columns(self, path: Path, required: bool = True) -> List[str]:
        """
        Read CSV header only.

        Args:
            path: CSV path.
            required: Whether missing file should raise.

        Returns:
            List[str]: Header columns.
        """
        print(f"[PROGRESS] Reading header columns from: {path}")

        if not path.exists():
            if required:
                raise FileNotFoundError(f"CSV file not found: {path}")
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
        Verify aligned row keys between files.
        """
        merge_columns = ["unit_id", "cycle", "split"]

        if len(base_chunk) != len(other_chunk):
            raise ValueError(
                f"Chunk row count mismatch for {label}: "
                f"base={len(base_chunk)}, other={len(other_chunk)}"
            )

        base_keys = base_chunk[merge_columns].reset_index(drop=True)
        other_keys = other_chunk[merge_columns].reset_index(drop=True)

        if not base_keys.equals(other_keys):
            raise ValueError(
                f"Row-key alignment failed for {label}. "
                "Regenerate outputs using the same base row order."
            )

    # ==================================================================================
    # Usecols
    # ==================================================================================

    def _build_health_usecols(self, columns: List[str]) -> List[str]:
        required = [
            "unit_id",
            "cycle",
            "split",
            "final_anomaly_score",
            "alert_level",
            "health_index",
            "health_state",
        ]

        self._validate_columns(columns, required, "health_states.csv")

        optional = [
            "gmm_context_id",
            "remaining_health_percentage",
            "anomaly_persistence_score",
            "residual_trend_score",
        ]

        usecols = list(required)

        for column in optional:
            if column in columns and column not in usecols:
                usecols.append(column)

        return usecols

    def _build_root_usecols(self, columns: List[str]) -> List[str]:
        required = [
            "unit_id",
            "cycle",
            "split",
            "top_sensor_1",
            "top_sensor_2",
            "top_sensor_3",
            "contribution_1",
            "contribution_2",
            "contribution_3",
            "root_cause_pattern",
            "inspection_focus",
        ]

        self._validate_columns(columns, required, "root_cause_analysis.csv")

        optional = [
            "total_abs_residual",
            "top3_contribution_sum",
            "dominant_detector",
        ]

        usecols = list(required)

        for column in optional:
            if column in columns and column not in usecols:
                usecols.append(column)

        return usecols

    def _build_context_usecols(self, columns: List[str]) -> List[str]:
        required = ["unit_id", "cycle", "split"]

        self._validate_columns(columns, required, "context_clusters.csv")

        optional = [
            "kmeans_context_id",
            "gmm_context_id",
            "context_confidence",
        ]

        usecols = list(required)

        for column in optional:
            if column in columns and column not in usecols:
                usecols.append(column)

        return usecols

    def _build_confidence_usecols(self, columns: List[str]) -> List[str]:
        required = ["unit_id", "cycle", "split"]

        self._validate_columns(columns, required, "confidence_scores.csv")

        optional = [
            "model_agreement_score",
            "confidence_score",
            "uncertainty_score",
            "reliability_score",
        ]

        usecols = list(required)

        for column in optional:
            if column in columns and column not in usecols:
                usecols.append(column)

        return usecols

    # ==================================================================================
    # Text helpers
    # ==================================================================================

    def _series_or_default(
        self,
        df: pd.DataFrame,
        column: str,
        default_value: object,
    ) -> pd.Series:
        """
        Return existing column filled with default, or default-valued series.
        """
        if column in df.columns:
            return df[column].fillna(default_value)

        return pd.Series([default_value] * len(df), index=df.index)

    def _safe_float(self, value: object, default: float = 0.0) -> float:
        """
        Safely convert value to float.
        """
        try:
            if pd.isna(value):
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _safe_int(self, value: object, default: int = -1) -> int:
        """
        Safely convert value to int.
        """
        try:
            if pd.isna(value):
                return int(default)
            return int(float(value))
        except Exception:
            return int(default)

    # ==================================================================================
    # Text generation
    # ==================================================================================

    def _build_texts(
        self,
        health_chunk: pd.DataFrame,
        root_chunk: pd.DataFrame,
        context_chunk: pd.DataFrame,
        confidence_chunk: Optional[pd.DataFrame],
    ) -> List[str]:
        """
        Build human-readable explanation text for one chunk.
        """
        n = len(health_chunk)

        gmm_context = self._series_or_default(context_chunk, "gmm_context_id", -1)
        kmeans_context = self._series_or_default(context_chunk, "kmeans_context_id", -1)
        context_confidence = self._series_or_default(
            context_chunk,
            "context_confidence",
            0.0,
        )

        if confidence_chunk is not None:
            model_agreement = self._series_or_default(
                confidence_chunk,
                "model_agreement_score",
                0.0,
            )
            confidence_score = self._series_or_default(
                confidence_chunk,
                "confidence_score",
                0.0,
            )
            uncertainty_score = self._series_or_default(
                confidence_chunk,
                "uncertainty_score",
                1.0,
            )
            reliability_score = self._series_or_default(
                confidence_chunk,
                "reliability_score",
                0.0,
            )
        else:
            model_agreement = pd.Series([0.0] * n, index=health_chunk.index)
            confidence_score = pd.Series([0.0] * n, index=health_chunk.index)
            uncertainty_score = pd.Series([1.0] * n, index=health_chunk.index)
            reliability_score = pd.Series([0.0] * n, index=health_chunk.index)

        alert_level = self._series_or_default(health_chunk, "alert_level", "Normal")
        health_index = self._series_or_default(health_chunk, "health_index", 100.0)
        health_state = self._series_or_default(health_chunk, "health_state", "Healthy")
        final_anomaly_score = self._series_or_default(
            health_chunk,
            "final_anomaly_score",
            0.0,
        )

        top_sensor_1 = self._series_or_default(root_chunk, "top_sensor_1", "unknown")
        top_sensor_2 = self._series_or_default(root_chunk, "top_sensor_2", "unknown")
        top_sensor_3 = self._series_or_default(root_chunk, "top_sensor_3", "unknown")

        contribution_1 = self._series_or_default(root_chunk, "contribution_1", 0.0)
        contribution_2 = self._series_or_default(root_chunk, "contribution_2", 0.0)
        contribution_3 = self._series_or_default(root_chunk, "contribution_3", 0.0)

        root_cause_pattern = self._series_or_default(
            root_chunk,
            "root_cause_pattern",
            "unknown_residual_pattern",
        )
        inspection_focus = self._series_or_default(
            root_chunk,
            "inspection_focus",
            "Inspect the top contributing measured sensor channels.",
        )

        texts: List[str] = []

        for values in zip(
            alert_level,
            gmm_context,
            kmeans_context,
            context_confidence,
            final_anomaly_score,
            health_index,
            health_state,
            top_sensor_1,
            top_sensor_2,
            top_sensor_3,
            contribution_1,
            contribution_2,
            contribution_3,
            root_cause_pattern,
            inspection_focus,
            model_agreement,
            confidence_score,
            uncertainty_score,
            reliability_score,
        ):
            (
                alert,
                gmm_id,
                kmeans_id,
                ctx_conf,
                anomaly_score,
                h_index,
                h_state,
                sensor_1,
                sensor_2,
                sensor_3,
                contrib_1,
                contrib_2,
                contrib_3,
                pattern,
                focus,
                agreement,
                conf,
                uncert,
                reliability,
            ) = values

            agreement_value = self._safe_float(agreement, 0.0)
            confidence_value = self._safe_float(conf, 0.0)
            uncertainty_value = self._safe_float(uncert, 1.0)
            reliability_value = self._safe_float(reliability, 0.0)
            context_conf_value = self._safe_float(ctx_conf, 0.0)

            text = (
                f"The engine shows a {alert} alert under GMM operating context "
                f"{self._safe_int(gmm_id, -1)}. "
                f"The baseline K-Means context is "
                f"{self._safe_int(kmeans_id, -1)}. "
                f"The ensemble digital twin estimated expected measured sensor behavior "
                f"using operating conditions, virtual sensors, and context information, "
                f"then compared it with actual measured sensors. "
                f"The final anomaly score is {self._safe_float(anomaly_score, 0.0):.3f}. "
                f"The health index is {self._safe_float(h_index, 100.0):.1f}/100 "
                f"and the health state is {h_state}. "
                f"The main contributing sensors are "
                f"{sensor_1} ({self._safe_float(contrib_1, 0.0) * 100.0:.1f}%), "
                f"{sensor_2} ({self._safe_float(contrib_2, 0.0) * 100.0:.1f}%), and "
                f"{sensor_3} ({self._safe_float(contrib_3, 0.0) * 100.0:.1f}%). "
                f"The residual pattern is classified as {pattern}. "
                f"Recommended inspection focus: {focus} "
                f"Context confidence is {context_conf_value * 100.0:.1f}%, "
                f"model agreement is {agreement_value * 100.0:.1f}%, "
                f"confidence is {confidence_value * 100.0:.1f}%, "
                f"uncertainty is {uncertainty_value * 100.0:.1f}%, and "
                f"reliability is {reliability_value * 100.0:.1f}%. "
                f"This explanation supports inspection focus only and does not make "
                f"maintenance scheduling decisions."
            )

            texts.append(text)

        return texts

    # ==================================================================================
    # Result chunk builder
    # ==================================================================================

    def _build_result_chunk(
        self,
        health_chunk: pd.DataFrame,
        root_chunk: pd.DataFrame,
        context_chunk: pd.DataFrame,
        confidence_chunk: Optional[pd.DataFrame],
        explanation_texts: List[str],
    ) -> pd.DataFrame:
        """
        Build output chunk with all dashboard-useful explanation fields.
        """
        result_chunk = health_chunk[["unit_id", "cycle", "split"]].copy()

        context_defaults = {
            "kmeans_context_id": -1,
            "gmm_context_id": -1,
            "context_confidence": 0.0,
        }

        for column, default_value in context_defaults.items():
            if column in context_chunk.columns:
                result_chunk[column] = context_chunk[column].values
            else:
                result_chunk[column] = default_value

        health_defaults = {
            "final_anomaly_score": 0.0,
            "alert_level": "Normal",
            "health_index": 100.0,
            "health_state": "Healthy",
            "remaining_health_percentage": 100.0,
            "anomaly_persistence_score": 0.0,
            "residual_trend_score": 0.0,
        }

        for column, default_value in health_defaults.items():
            if column in health_chunk.columns:
                result_chunk[column] = health_chunk[column].values
            else:
                result_chunk[column] = default_value

        root_defaults = {
            "top_sensor_1": "unknown",
            "top_sensor_2": "unknown",
            "top_sensor_3": "unknown",
            "contribution_1": 0.0,
            "contribution_2": 0.0,
            "contribution_3": 0.0,
            "root_cause_pattern": "unknown_residual_pattern",
            "inspection_focus": "Inspect the top contributing measured sensor channels.",
            "total_abs_residual": 0.0,
            "top3_contribution_sum": 0.0,
            "dominant_detector": "unknown",
        }

        for column, default_value in root_defaults.items():
            if column in root_chunk.columns:
                result_chunk[column] = root_chunk[column].values
            else:
                result_chunk[column] = default_value

        confidence_defaults = {
            "model_agreement_score": 0.0,
            "confidence_score": 0.0,
            "uncertainty_score": 1.0,
            "reliability_score": 0.0,
        }

        if confidence_chunk is not None:
            for column, default_value in confidence_defaults.items():
                if column in confidence_chunk.columns:
                    result_chunk[column] = confidence_chunk[column].values
                else:
                    result_chunk[column] = default_value
        else:
            for column, default_value in confidence_defaults.items():
                result_chunk[column] = default_value

        for column in result_chunk.columns:
            if column in {
                "final_anomaly_score",
                "health_index",
                "remaining_health_percentage",
                "anomaly_persistence_score",
                "residual_trend_score",
                "contribution_1",
                "contribution_2",
                "contribution_3",
                "total_abs_residual",
                "top3_contribution_sum",
                "context_confidence",
                "model_agreement_score",
                "confidence_score",
                "uncertainty_score",
                "reliability_score",
            }:
                result_chunk[column] = pd.to_numeric(
                    result_chunk[column],
                    errors="coerce",
                ).fillna(0.0)

        result_chunk["explanation_text"] = explanation_texts
        result_chunk["maintenance_decision"] = "Not generated by this component"
        result_chunk["component_role"] = (
            "Human-readable explanation and inspection-focus support"
        )

        for column in self.OUTPUT_COLUMNS:
            if column not in result_chunk.columns:
                result_chunk[column] = np.nan

        result_chunk = result_chunk[self.OUTPUT_COLUMNS].copy()

        return result_chunk

    # ==================================================================================
    # Main generation
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Generate explanation reports in memory-safe chunks.
        """
        print("[PROGRESS] Entering ExplanationGenerator.run")

        temp_output_path: Optional[Path] = None

        try:
            started = perf_counter()

            expected_rows = self._count_csv_rows(self.health_csv)

            if expected_rows <= 0:
                raise ValueError("health_states.csv contains zero rows.")

            root_rows = self._count_csv_rows(self.root_csv)
            context_rows = self._count_csv_rows(self.context_csv)

            if root_rows != expected_rows:
                raise ValueError(
                    f"root_cause_analysis.csv row mismatch: "
                    f"{root_rows} != {expected_rows}"
                )

            if context_rows != expected_rows:
                raise ValueError(
                    f"context_clusters.csv row mismatch: "
                    f"{context_rows} != {expected_rows}"
                )

            health_columns = self._read_header_columns(self.health_csv)
            root_columns = self._read_header_columns(self.root_csv)
            context_columns = self._read_header_columns(self.context_csv)

            health_usecols = self._build_health_usecols(health_columns)
            root_usecols = self._build_root_usecols(root_columns)
            context_usecols = self._build_context_usecols(context_columns)

            confidence_available = False
            confidence_usecols: List[str] = []

            if self.confidence_csv.exists():
                confidence_rows = self._count_csv_rows(self.confidence_csv)

                if confidence_rows == expected_rows:
                    confidence_columns = self._read_header_columns(self.confidence_csv)
                    confidence_usecols = self._build_confidence_usecols(
                        confidence_columns
                    )
                    confidence_available = True
                else:
                    print(
                        "[WARNING] confidence_scores.csv row count mismatch. "
                        "Confidence fields will use defaults."
                    )
            else:
                print(
                    "[WARNING] confidence_scores.csv not found. "
                    "Confidence fields will use defaults."
                )

            print(f"[PROGRESS] Confidence available: {confidence_available}")
            print(f"[PROGRESS] Health usecols: {health_usecols}")
            print(f"[PROGRESS] Root usecols: {root_usecols}")
            print(f"[PROGRESS] Context usecols: {context_usecols}")
            print(f"[PROGRESS] Confidence usecols: {confidence_usecols}")

            health_iter = pd.read_csv(
                self.health_csv,
                usecols=health_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            root_iter = pd.read_csv(
                self.root_csv,
                usecols=root_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            context_iter = pd.read_csv(
                self.context_csv,
                usecols=context_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            confidence_iter: Optional[Iterator[pd.DataFrame]] = None

            if confidence_available:
                confidence_iter = pd.read_csv(
                    self.confidence_csv,
                    usecols=confidence_usecols,
                    chunksize=self.chunk_size,
                    low_memory=True,
                )

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary explanation reports CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            alert_counts: Dict[str, int] = {}
            health_state_counts: Dict[str, int] = {}
            pattern_counts: Dict[str, int] = {}

            confidence_sum = 0.0
            uncertainty_sum = 0.0
            reliability_sum = 0.0
            model_agreement_sum = 0.0

            print("[PROGRESS] Starting memory-safe explanation generation")

            for health_chunk, root_chunk, context_chunk in zip(
                health_iter,
                root_iter,
                context_iter,
            ):
                chunk_index += 1

                health_chunk = health_chunk.reset_index(drop=True)
                root_chunk = root_chunk.reset_index(drop=True)
                context_chunk = context_chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Explanation generation chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(health_chunk)}")

                self._verify_key_alignment(
                    health_chunk,
                    root_chunk,
                    "root_cause_analysis.csv",
                )
                self._verify_key_alignment(
                    health_chunk,
                    context_chunk,
                    "context_clusters.csv",
                )

                confidence_chunk = None

                if confidence_iter is not None:
                    try:
                        confidence_chunk = next(confidence_iter).reset_index(drop=True)
                    except StopIteration as exc:
                        raise ValueError(
                            "confidence_scores.csv ended before health_states.csv."
                        ) from exc

                    self._verify_key_alignment(
                        health_chunk,
                        confidence_chunk,
                        "confidence_scores.csv",
                    )

                explanation_texts = self._build_texts(
                    health_chunk=health_chunk,
                    root_chunk=root_chunk,
                    context_chunk=context_chunk,
                    confidence_chunk=confidence_chunk,
                )

                result_chunk = self._build_result_chunk(
                    health_chunk=health_chunk,
                    root_chunk=root_chunk,
                    context_chunk=context_chunk,
                    confidence_chunk=confidence_chunk,
                    explanation_texts=explanation_texts,
                )

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                for source_column, target_dict in [
                    ("alert_level", alert_counts),
                    ("health_state", health_state_counts),
                ]:
                    if source_column in result_chunk.columns:
                        counts = result_chunk[source_column].astype(str).value_counts().to_dict()
                        for value, count in counts.items():
                            target_dict[str(value)] = target_dict.get(str(value), 0) + int(count)

                if "root_cause_pattern" in result_chunk.columns:
                    counts = result_chunk["root_cause_pattern"].astype(str).value_counts().to_dict()
                    for pattern, count in counts.items():
                        pattern_counts[str(pattern)] = (
                            pattern_counts.get(str(pattern), 0) + int(count)
                        )

                model_agreement_sum += float(
                    pd.to_numeric(
                        result_chunk["model_agreement_score"],
                        errors="coerce",
                    ).fillna(0.0).sum()
                )
                confidence_sum += float(
                    pd.to_numeric(
                        result_chunk["confidence_score"],
                        errors="coerce",
                    ).fillna(0.0).sum()
                )
                uncertainty_sum += float(
                    pd.to_numeric(
                        result_chunk["uncertainty_score"],
                        errors="coerce",
                    ).fillna(0.0).sum()
                )
                reliability_sum += float(
                    pd.to_numeric(
                        result_chunk["reliability_score"],
                        errors="coerce",
                    ).fillna(0.0).sum()
                )

                print(f"[PROGRESS] Total explanation rows written: {total_rows_written}")
                print(f"[PROGRESS] Running alert counts: {alert_counts}")

                del health_chunk
                del root_chunk
                del context_chunk
                del confidence_chunk
                del result_chunk
                del explanation_texts
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All explanation chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Explanation report row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final explanation_reports.csv will not be replaced."
                )

            if confidence_iter is not None:
                try:
                    extra_chunk = next(confidence_iter)
                    if len(extra_chunk) > 0:
                        raise ValueError(
                            "confidence_scores.csv has extra rows after "
                            "health_states.csv ended."
                        )
                except StopIteration:
                    pass

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            average_model_agreement = model_agreement_sum / max(total_rows_written, 1)
            average_confidence = confidence_sum / max(total_rows_written, 1)
            average_uncertainty = uncertainty_sum / max(total_rows_written, 1)
            average_reliability = reliability_sum / max(total_rows_written, 1)

            summary = {
                "status": "success",
                "message": "Human-readable explanation reports generated.",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "confidence_available": bool(confidence_available),
                "alert_counts": alert_counts,
                "health_state_counts": health_state_counts,
                "root_cause_pattern_counts": pattern_counts,
                "averages": {
                    "average_model_agreement_score": float(average_model_agreement),
                    "average_confidence_score": float(average_confidence),
                    "average_uncertainty_score": float(average_uncertainty),
                    "average_reliability_score": float(average_reliability),
                },
                "chunk_size": int(self.chunk_size),
                "chunks_processed": int(chunk_index),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "text_consistency_fix": {
                    "confidence_text_uses_real_confidence_scores": bool(confidence_available),
                    "confidence_score_column_written": True,
                    "uncertainty_score_column_written": True,
                    "reliability_score_column_written": True,
                    "model_agreement_score_column_written": True,
                },
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_health_states": True,
                    "uses_root_cause_analysis": True,
                    "uses_context_clusters": True,
                    "uses_confidence_scores_if_available": bool(confidence_available),
                    "full_dataframe_merge_used": False,
                    "full_csv_read_used": False,
                    "aligned_chunk_processing": True,
                },
            }

            print(f"[PROGRESS] Writing explanation summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            response = {
                "status": "success",
                "message": "Human-readable explanation reports generated.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(total_rows_written),
            }

            print(f"[PROGRESS] Explanation generator response: {response}")
            return response

        except Exception as exc:
            if temp_output_path is not None and temp_output_path.exists():
                print("[PROGRESS] Removing failed temporary explanation reports CSV")
                temp_output_path.unlink()

            print(f"[ERROR] Explanation generator failed: {exc}")
            logger.exception("Explanation generator failed.")
            raise RuntimeError("Explanation generator failed.") from exc


def run_explanation_generator() -> Dict[str, object]:
    """
    Execute explanation generation.
    """
    print("[PROGRESS] Entering run_explanation_generator")

    generator = ExplanationGenerator()
    return generator.run()


if __name__ == "__main__":
    print("[PROGRESS] explanation_generator.py execution started")
    result = run_explanation_generator()
    print("[PROGRESS] explanation_generator.py execution finished successfully")
    print(result)