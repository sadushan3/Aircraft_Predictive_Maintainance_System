"""
Confidence estimator for CA-EDT-AHMA.

Role:
Calculate confidence, reliability, and uncertainty.

Inputs:
- model_agreement_score
- context_confidence
- anomaly_persistence_score
- data_quality_score

Formula:
confidence =
0.35 * model_agreement_score +
0.25 * context_confidence +
0.25 * anomaly_persistence_score +
0.15 * data_quality_score

Outputs:
confidence_score
reliability_score
uncertainty_score
confidence_label
uncertainty_label

Reads:
outputs/Anomaly_Health_Monitering/model_agreement.csv
outputs/Anomaly_Health_Monitering/context_clusters.csv
outputs/Anomaly_Health_Monitering/health_states.csv
processed/scaled_features.csv

Writes:
outputs/Anomaly_Health_Monitering/confidence_scores.csv
models/uncertainty/confidence_config.json
reports/confidence_scores_summary.json

Important:
- This module does not train a model.
- This module does not predict RUL.
- This module does not make maintenance decisions.
- This module does not use Y_dev/Y_test.
- T columns are excluded from data-quality scoring to avoid target/degradation leakage.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "uncertainty/confidence_estimator.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional
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


class ConfidenceEstimator:
    """
    Memory-safe confidence, reliability, and uncertainty estimator.
    """

    def __init__(self, chunk_size: int = 200_000) -> None:
        """
        Initialize confidence estimator.

        Args:
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering ConfidenceEstimator.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "CONFIDENCE_ESTIMATOR_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("CONFIDENCE_ESTIMATOR_CHUNK_SIZE must be positive.")

        self.agreement_csv: Path = Config.MODEL_AGREEMENT_CSV
        self.context_csv: Path = Config.CONTEXT_CSV
        self.health_csv: Path = Config.HEALTH_STATES_CSV
        self.scaled_csv: Path = Config.SCALED_CSV

        self.output_csv: Path = Config.CONFIDENCE_CSV
        self.config_json: Path = Config.CONFIDENCE_CONFIG_PATH

        self.summary_json: Path = getattr(
            Config,
            "CONFIDENCE_SUMMARY_JSON",
            Config.REPORT_DIR / "confidence_scores_summary.json",
        )

        print(f"[PROGRESS] Agreement CSV: {self.agreement_csv}")
        print(f"[PROGRESS] Context CSV: {self.context_csv}")
        print(f"[PROGRESS] Health CSV: {self.health_csv}")
        print(f"[PROGRESS] Scaled CSV: {self.scaled_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Config JSON: {self.config_json}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")

    # ==================================================================================
    # File helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows without loading full file.
        """
        print(f"[PROGRESS] Counting rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)

        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV header only.
        """
        print(f"[PROGRESS] Reading header columns from: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

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
        Verify row-key alignment against base chunk.
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
                "Regenerate outputs using the same row order."
            )

    # ==================================================================================
    # Usecols
    # ==================================================================================

    def _build_agreement_usecols(self, columns: List[str]) -> List[str]:
        required = [
            "unit_id",
            "cycle",
            "split",
            "model_disagreement",
            "normalized_model_disagreement",
            "model_agreement_score",
            "uncertainty_from_model_disagreement",
        ]

        self._validate_columns(columns, required, "model_agreement.csv")

        optional = [
            "agreement_normalization_threshold",
            "agreement_normalization_percentile",
            "agreement_fit_split",
            "agreement_sensor_count",
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

        if "context_confidence" not in usecols:
            print("[WARNING] context_confidence missing; default 0.0 will be used.")

        return usecols

    def _build_health_usecols(self, columns: List[str]) -> List[str]:
        required = [
            "unit_id",
            "cycle",
            "split",
        ]

        self._validate_columns(columns, required, "health_states.csv")

        optional = [
            "anomaly_persistence_score",
            "residual_trend_score",
            "final_anomaly_score",
            "alert_level",
            "health_index",
            "health_state",
        ]

        usecols = list(required)

        for column in optional:
            if column in columns and column not in usecols:
                usecols.append(column)

        if "anomaly_persistence_score" not in usecols:
            print(
                "[WARNING] anomaly_persistence_score missing; default 0.0 will be used."
            )

        return usecols

    def _build_scaled_usecols(self, columns: List[str]) -> List[str]:
        """
        Build scaled_features.csv usecols for data-quality scoring.

        Excludes T/Y/RUL-like columns to avoid degradation/RUL leakage.
        """
        required = ["unit_id", "cycle", "split"]

        self._validate_columns(columns, required, "scaled_features.csv")

        excluded_prefixes = ("T_", "Y_", "rul_", "RUL_")
        excluded_exact = {
            "RUL",
            "rul",
            "target",
            "label",
            "health_label",
            "failure_label",
        }

        usecols = list(required)

        for column in columns:
            if column in usecols:
                continue

            if column in excluded_exact:
                continue

            if column.startswith(excluded_prefixes):
                continue

            lower = column.lower()

            if "rul" in lower or "target" in lower or "label" in lower:
                continue

            usecols.append(column)

        print(f"[PROGRESS] Scaled data-quality usecols count: {len(usecols)}")

        return usecols

    # ==================================================================================
    # Data quality
    # ==================================================================================

    def _calculate_data_quality_score(self, scaled_chunk: pd.DataFrame) -> np.ndarray:
        """
        Calculate per-row data quality score for one scaled feature chunk.

        data_quality_score = 1 - missing_ratio - infinite_ratio

        T/Y/RUL-like columns are already excluded from usecols.
        """
        exclude_columns = {"unit_id", "cycle", "split"}

        numeric_columns = [
            column
            for column in scaled_chunk.select_dtypes(include=[np.number]).columns.tolist()
            if column not in exclude_columns
        ]

        if not numeric_columns:
            return np.ones(len(scaled_chunk), dtype=np.float32)

        numeric_values = scaled_chunk[numeric_columns].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        missing_ratio = np.isnan(numeric_values).mean(axis=1)
        infinite_ratio = np.isinf(numeric_values).mean(axis=1)

        data_quality_score = 1.0 - missing_ratio - infinite_ratio

        data_quality_score = np.clip(
            data_quality_score,
            0.0,
            1.0,
        ).astype(np.float32)

        return data_quality_score

    # ==================================================================================
    # Score helpers
    # ==================================================================================

    def _weights(self) -> Dict[str, float]:
        """
        Get confidence weights from Config with safe defaults.
        """
        default_weights = {
            "model_agreement_score": 0.35,
            "context_confidence": 0.25,
            "anomaly_persistence_score": 0.25,
            "data_quality_score": 0.15,
        }

        config_weights = getattr(Config, "CONFIDENCE_WEIGHTS", default_weights)

        weights = {}

        for key, default_value in default_weights.items():
            weights[key] = float(config_weights.get(key, default_value))

        total = sum(weights.values())

        if total <= 1e-12:
            raise ValueError("CONFIDENCE_WEIGHTS sum must be positive.")

        if abs(total - 1.0) > 1e-6:
            print(
                f"[WARNING] CONFIDENCE_WEIGHTS sum is {total}. "
                "Weights will be normalized to sum to 1."
            )

            weights = {key: value / total for key, value in weights.items()}

        return weights

    def _series_or_default(
        self,
        df: pd.DataFrame,
        column: str,
        default_value: float,
    ) -> pd.Series:
        """
        Return numeric series or default-valued series.
        """
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(default_value)

        return pd.Series([default_value] * len(df), index=df.index)

    def _label_confidence(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert confidence score to label.
        """
        return np.select(
            [
                scores >= 0.80,
                scores >= 0.60,
                scores >= 0.40,
            ],
            [
                "High_Confidence",
                "Moderate_Confidence",
                "Low_Confidence",
            ],
            default="Very_Low_Confidence",
        )

    def _label_uncertainty(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert uncertainty score to label.
        """
        return np.select(
            [
                scores >= 0.60,
                scores >= 0.40,
                scores >= 0.20,
            ],
            [
                "High_Uncertainty",
                "Moderate_Uncertainty",
                "Low_Uncertainty",
            ],
            default="Very_Low_Uncertainty",
        )

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run memory-safe confidence estimation.
        """
        print("[PROGRESS] Entering ConfidenceEstimator.run")

        try:
            started = perf_counter()

            if not self.agreement_csv.exists():
                raise FileNotFoundError(
                    f"Model agreement CSV not found: {self.agreement_csv}. "
                    "Run model_agreement.py first."
                )

            agreement_rows = self._count_csv_rows(self.agreement_csv)
            context_rows = self._count_csv_rows(self.context_csv)
            health_rows = self._count_csv_rows(self.health_csv)
            scaled_rows = self._count_csv_rows(self.scaled_csv)

            if agreement_rows <= 0:
                raise ValueError("model_agreement.csv contains zero rows.")

            if not (
                agreement_rows == context_rows == health_rows == scaled_rows
            ):
                raise ValueError(
                    "Confidence input row-count mismatch: "
                    f"agreement={agreement_rows}, context={context_rows}, "
                    f"health={health_rows}, scaled={scaled_rows}"
                )

            agreement_columns = self._read_header_columns(self.agreement_csv)
            context_columns = self._read_header_columns(self.context_csv)
            health_columns = self._read_header_columns(self.health_csv)
            scaled_columns = self._read_header_columns(self.scaled_csv)

            agreement_usecols = self._build_agreement_usecols(agreement_columns)
            context_usecols = self._build_context_usecols(context_columns)
            health_usecols = self._build_health_usecols(health_columns)
            scaled_usecols = self._build_scaled_usecols(scaled_columns)

            weights = self._weights()

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary confidence CSV")
                temp_output_path.unlink()

            agreement_iter = pd.read_csv(
                self.agreement_csv,
                usecols=agreement_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            context_iter = pd.read_csv(
                self.context_csv,
                usecols=context_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            health_iter = pd.read_csv(
                self.health_csv,
                usecols=health_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            scaled_iter = pd.read_csv(
                self.scaled_csv,
                usecols=scaled_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            confidence_sum = 0.0
            uncertainty_sum = 0.0
            reliability_sum = 0.0
            data_quality_sum = 0.0

            confidence_label_counts: Dict[str, int] = {}
            uncertainty_label_counts: Dict[str, int] = {}
            split_summary_accumulator: Dict[str, Dict[str, float]] = {}

            print("[PROGRESS] Starting memory-safe confidence estimation")

            for agreement_chunk, context_chunk, health_chunk, scaled_chunk in zip(
                agreement_iter,
                context_iter,
                health_iter,
                scaled_iter,
            ):
                chunk_index += 1

                agreement_chunk = agreement_chunk.reset_index(drop=True)
                context_chunk = context_chunk.reset_index(drop=True)
                health_chunk = health_chunk.reset_index(drop=True)
                scaled_chunk = scaled_chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Confidence estimation chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(agreement_chunk)}")

                self._verify_key_alignment(
                    agreement_chunk,
                    context_chunk,
                    "context_clusters.csv",
                )
                self._verify_key_alignment(
                    agreement_chunk,
                    health_chunk,
                    "health_states.csv",
                )
                self._verify_key_alignment(
                    agreement_chunk,
                    scaled_chunk,
                    "scaled_features.csv",
                )

                model_agreement_score = self._series_or_default(
                    agreement_chunk,
                    "model_agreement_score",
                    0.0,
                ).clip(0.0, 1.0)

                context_confidence = self._series_or_default(
                    context_chunk,
                    "context_confidence",
                    0.0,
                ).clip(0.0, 1.0)

                anomaly_persistence_score = self._series_or_default(
                    health_chunk,
                    "anomaly_persistence_score",
                    0.0,
                ).clip(0.0, 1.0)

                data_quality_score = self._calculate_data_quality_score(scaled_chunk)

                confidence_score = (
                    weights["model_agreement_score"] * model_agreement_score.to_numpy(dtype=np.float32)
                    + weights["context_confidence"] * context_confidence.to_numpy(dtype=np.float32)
                    + weights["anomaly_persistence_score"] * anomaly_persistence_score.to_numpy(dtype=np.float32)
                    + weights["data_quality_score"] * data_quality_score
                )

                confidence_score = np.clip(
                    confidence_score,
                    0.0,
                    1.0,
                ).astype(np.float32)

                uncertainty_score = np.clip(
                    1.0 - confidence_score,
                    0.0,
                    1.0,
                ).astype(np.float32)

                reliability_score = (
                    0.50 * confidence_score
                    + 0.30 * model_agreement_score.to_numpy(dtype=np.float32)
                    + 0.20 * data_quality_score
                )

                reliability_score = np.clip(
                    reliability_score,
                    0.0,
                    1.0,
                ).astype(np.float32)

                result_chunk = agreement_chunk[["unit_id", "cycle", "split"]].copy()

                for column in [
                    "kmeans_context_id",
                    "gmm_context_id",
                    "context_confidence",
                ]:
                    if column in context_chunk.columns:
                        result_chunk[column] = context_chunk[column].values

                agreement_passthrough = [
                    "model_disagreement",
                    "normalized_model_disagreement",
                    "model_agreement_score",
                    "uncertainty_from_model_disagreement",
                    "agreement_normalization_threshold",
                    "agreement_normalization_percentile",
                    "agreement_fit_split",
                    "agreement_sensor_count",
                ]

                for column in agreement_passthrough:
                    if column in agreement_chunk.columns:
                        result_chunk[column] = agreement_chunk[column].values

                health_passthrough = [
                    "final_anomaly_score",
                    "alert_level",
                    "health_index",
                    "health_state",
                    "anomaly_persistence_score",
                    "residual_trend_score",
                ]

                for column in health_passthrough:
                    if column in health_chunk.columns:
                        result_chunk[column] = health_chunk[column].values

                result_chunk["data_quality_score"] = data_quality_score
                result_chunk["confidence_score"] = confidence_score
                result_chunk["uncertainty_score"] = uncertainty_score
                result_chunk["reliability_score"] = reliability_score
                result_chunk["confidence_label"] = self._label_confidence(confidence_score)
                result_chunk["uncertainty_label"] = self._label_uncertainty(uncertainty_score)
                result_chunk["component_role"] = "Confidence, reliability, and uncertainty scoring"
                result_chunk["maintenance_decision"] = "Not generated by this component"

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                confidence_sum += float(np.sum(confidence_score, dtype=np.float64))
                uncertainty_sum += float(np.sum(uncertainty_score, dtype=np.float64))
                reliability_sum += float(np.sum(reliability_score, dtype=np.float64))
                data_quality_sum += float(np.sum(data_quality_score, dtype=np.float64))

                confidence_unique, confidence_counts = np.unique(
                    result_chunk["confidence_label"].astype(str).to_numpy(dtype=object),
                    return_counts=True,
                )

                for label, count in zip(confidence_unique, confidence_counts):
                    confidence_label_counts[str(label)] = (
                        confidence_label_counts.get(str(label), 0) + int(count)
                    )

                uncertainty_unique, uncertainty_counts = np.unique(
                    result_chunk["uncertainty_label"].astype(str).to_numpy(dtype=object),
                    return_counts=True,
                )

                for label, count in zip(uncertainty_unique, uncertainty_counts):
                    uncertainty_label_counts[str(label)] = (
                        uncertainty_label_counts.get(str(label), 0) + int(count)
                    )

                for split in result_chunk["split"].astype(str).unique():
                    split_mask = result_chunk["split"].astype(str) == split
                    split_count = int(split_mask.sum())

                    if split not in split_summary_accumulator:
                        split_summary_accumulator[split] = {
                            "rows": 0.0,
                            "confidence_sum": 0.0,
                            "uncertainty_sum": 0.0,
                            "reliability_sum": 0.0,
                            "data_quality_sum": 0.0,
                        }

                    split_summary_accumulator[split]["rows"] += split_count
                    split_summary_accumulator[split]["confidence_sum"] += float(
                        result_chunk.loc[split_mask, "confidence_score"].sum()
                    )
                    split_summary_accumulator[split]["uncertainty_sum"] += float(
                        result_chunk.loc[split_mask, "uncertainty_score"].sum()
                    )
                    split_summary_accumulator[split]["reliability_sum"] += float(
                        result_chunk.loc[split_mask, "reliability_score"].sum()
                    )
                    split_summary_accumulator[split]["data_quality_sum"] += float(
                        result_chunk.loc[split_mask, "data_quality_score"].sum()
                    )

                print(f"[PROGRESS] Total confidence rows written: {total_rows_written}")
                print(f"[PROGRESS] Running confidence labels: {confidence_label_counts}")

                del agreement_chunk
                del context_chunk
                del health_chunk
                del scaled_chunk
                del result_chunk
                del model_agreement_score
                del context_confidence
                del anomaly_persistence_score
                del data_quality_score
                del confidence_score
                del uncertainty_score
                del reliability_score
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All confidence chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {agreement_rows}")

            if total_rows_written != agreement_rows:
                raise ValueError(
                    "Confidence score row count mismatch. "
                    f"written={total_rows_written}, expected={agreement_rows}. "
                    "Final confidence_scores.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            split_summary = {}

            for split, values in split_summary_accumulator.items():
                rows = max(values["rows"], 1.0)

                split_summary[split] = {
                    "rows": int(values["rows"]),
                    "average_confidence_score": float(values["confidence_sum"] / rows),
                    "average_uncertainty_score": float(values["uncertainty_sum"] / rows),
                    "average_reliability_score": float(values["reliability_sum"] / rows),
                    "average_data_quality_score": float(values["data_quality_sum"] / rows),
                }

            config_payload = {
                "formula": (
                    "confidence = "
                    "w_model*model_agreement_score + "
                    "w_context*context_confidence + "
                    "w_persistence*anomaly_persistence_score + "
                    "w_quality*data_quality_score"
                ),
                "weights": weights,
                "uncertainty_formula": "uncertainty_score = 1 - confidence_score",
                "reliability_formula": (
                    "reliability_score = 0.50*confidence_score + "
                    "0.30*model_agreement_score + 0.20*data_quality_score"
                ),
                "confidence_labels": {
                    "High_Confidence": "confidence_score >= 0.80",
                    "Moderate_Confidence": "0.60 <= confidence_score < 0.80",
                    "Low_Confidence": "0.40 <= confidence_score < 0.60",
                    "Very_Low_Confidence": "confidence_score < 0.40",
                },
                "uncertainty_labels": {
                    "High_Uncertainty": "uncertainty_score >= 0.60",
                    "Moderate_Uncertainty": "0.40 <= uncertainty_score < 0.60",
                    "Low_Uncertainty": "0.20 <= uncertainty_score < 0.40",
                    "Very_Low_Uncertainty": "uncertainty_score < 0.20",
                },
                "data_quality": {
                    "method": "1 - missing_ratio - infinite_ratio",
                    "excluded_columns": "T_*, Y_*, RUL/rul/target/label columns",
                },
                "target_usage": {
                    "uses_y_dev_y_test": False,
                    "uses_t_dev_t_test": False,
                    "predicts_rul": False,
                    "makes_maintenance_decisions": False,
                },
            }

            summary = {
                "status": "success",
                "message": "Confidence, reliability, and uncertainty scores calculated.",
                "output_file": str(self.output_csv),
                "config_file": str(self.config_json),
                "records_count": int(total_rows_written),
                "weights": weights,
                "average_confidence_score": float(
                    confidence_sum / max(total_rows_written, 1)
                ),
                "average_uncertainty_score": float(
                    uncertainty_sum / max(total_rows_written, 1)
                ),
                "average_reliability_score": float(
                    reliability_sum / max(total_rows_written, 1)
                ),
                "average_data_quality_score": float(
                    data_quality_sum / max(total_rows_written, 1)
                ),
                "confidence_label_counts": confidence_label_counts,
                "uncertainty_label_counts": uncertainty_label_counts,
                "split_summary": split_summary,
                "chunk_size": int(self.chunk_size),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_model_agreement": True,
                    "uses_context_confidence": True,
                    "uses_health_anomaly_persistence": True,
                    "uses_scaled_features_for_data_quality_only": True,
                },
            }

            print(f"[PROGRESS] Writing confidence config to: {self.config_json}")
            atomic_write_json(config_payload, self.config_json)

            print(f"[PROGRESS] Writing confidence summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            response = {
                "status": "success",
                "message": "Confidence, reliability, and uncertainty scores calculated.",
                "output_file": str(self.output_csv),
                "config_file": str(self.config_json),
                "summary_file": str(self.summary_json),
                "records_count": int(total_rows_written),
            }

            print(f"[PROGRESS] Confidence estimator response: {response}")

            logger.info(
                "Confidence estimation completed. rows=%s",
                total_rows_written,
            )

            return response

        except Exception as exc:
            print(f"[ERROR] Confidence estimator stage failed: {exc}")
            logger.exception("Confidence estimator stage failed.")
            raise RuntimeError("Confidence estimator stage failed.") from exc


def run_confidence_estimation() -> Dict[str, object]:
    """
    Execute confidence estimation.
    """
    print("[PROGRESS] Entering run_confidence_estimation")

    estimator = ConfidenceEstimator()
    return estimator.run()


if __name__ == "__main__":
    print("[PROGRESS] confidence_estimator.py execution started")
    result = run_confidence_estimation()
    print("[PROGRESS] confidence_estimator.py execution finished successfully")
    print(result)