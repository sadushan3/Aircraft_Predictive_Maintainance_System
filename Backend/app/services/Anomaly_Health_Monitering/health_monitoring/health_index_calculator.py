"""
Health index calculator for CA-EDT-AHMA.

Role:
Convert fused anomaly behavior into health index from 0 to 100.

Formula:
health_index =
100
- 60 * final_anomaly_score
- 25 * residual_trend_score
- 15 * anomaly_persistence_score

The output is clipped between 0 and 100.

Important:
- This module does not predict RUL.
- This module does not use Y_dev or Y_test.
- This module does not use T_dev or T_test.
- It uses fused anomaly outputs only.

Reads:
outputs/Anomaly_Health_Monitering/anomaly_fusion.csv

Writes:
outputs/Anomaly_Health_Monitering/health_index.csv

Saves:
models/health/health_index_config.json

Memory-safe:
- Does not load full anomaly_fusion.csv into RAM.
- Reads anomaly_fusion.csv in chunks.
- Maintains rolling state across chunks per split/unit_id.
- Writes to temporary CSV first.
- Replaces final health_index.csv only after successful completion.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "health_index/health_index.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
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


class HealthIndexCalculator:
    """
    Memory-safe health index calculator.

    Converts fused anomaly severity into interpretable health percentage.
    """

    def __init__(self, trend_window: int | None = None, chunk_size: int = 25_000) -> None:
        """
        Initialize health index calculator.

        Args:
            trend_window: Rolling window for anomaly trend and persistence.
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering HealthIndexCalculator.__init__")

        Config.create_directories()

        self.trend_window = int(
            trend_window
            if trend_window is not None
            else getattr(Config, "HEALTH_TREND_WINDOW", getattr(Config, "ROLLING_WINDOW", 5))
        )

        self.chunk_size = int(
            getattr(Config, "HEALTH_INDEX_CHUNK_SIZE", chunk_size)
        )

        if self.trend_window <= 1:
            raise ValueError("HEALTH_TREND_WINDOW must be greater than 1.")

        if self.chunk_size <= 0:
            raise ValueError("HEALTH_INDEX_CHUNK_SIZE must be positive.")

        self.anomaly_threshold = float(
            getattr(Config, "HEALTH_ANOMALY_PERSISTENCE_THRESHOLD", 0.40)
        )

        if not (0.0 <= self.anomaly_threshold <= 1.0):
            raise ValueError("HEALTH_ANOMALY_PERSISTENCE_THRESHOLD must be between 0 and 1.")

        self.input_csv: Path = Config.ANOMALY_FUSION_CSV
        self.output_csv: Path = Config.HEALTH_INDEX_CSV
        self.config_json: Path = Config.HEALTH_INDEX_CONFIG_PATH
        self.summary_json: Path = getattr(
            Config,
            "HEALTH_INDEX_SUMMARY_JSON",
            Config.REPORT_DIR / "health_index_summary.json",
        )

        self.health_weights = dict(Config.HEALTH_WEIGHTS)

        required_weight_keys = [
            "final_anomaly_score",
            "residual_trend_score",
            "anomaly_persistence_score",
        ]

        missing_weights = [
            key for key in required_weight_keys if key not in self.health_weights
        ]

        if missing_weights:
            raise KeyError(f"Missing health weight keys: {missing_weights}")

        print(f"[PROGRESS] Input CSV: {self.input_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Config JSON: {self.config_json}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Trend window: {self.trend_window}")
        print(f"[PROGRESS] Persistence threshold: {self.anomaly_threshold}")
        print(f"[PROGRESS] Health weights: {self.health_weights}")

    # ==================================================================================
    # Helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows without loading the full file.
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
        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate required columns exist.
        """
        missing = [
            column
            for column in required_columns
            if column not in available_columns
        ]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _build_usecols(self, columns: List[str]) -> List[str]:
        """
        Build usecols from required and optional anomaly-fusion columns.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "final_anomaly_score",
            "alert_level",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="anomaly_fusion.csv",
        )

        optional_columns = [
            "gmm_context_id",
            "severity_rank",
            "severity_description",
            "detector_agreement_count",
            "detector_agreement_ratio",
            "dominant_detector",
            "residual_anomaly_score",
            "iforest_anomaly_score",
            "mahalanobis_score",
            "lstm_autoencoder_score",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Health index usecols: {usecols}")
        return usecols

    def _calculate_health_for_chunk(
        self,
        chunk: pd.DataFrame,
        state: Dict[Tuple[object, object], Dict[str, object]],
    ) -> pd.DataFrame:
        """
        Calculate rolling trend, persistence, and health index for one chunk.

        Maintains rolling state per (split, unit_id).
        """
        result = chunk.copy()

        result["final_anomaly_score"] = (
            result["final_anomaly_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .clip(0.0, 1.0)
        )

        result["residual_trend_score"] = np.zeros(len(result), dtype=np.float32)
        result["anomaly_persistence_score"] = np.zeros(len(result), dtype=np.float32)

        for group_key, group_index in result.groupby(["split", "unit_id"], sort=False).groups.items():
            scores = result.loc[group_index, "final_anomaly_score"].to_numpy(
                dtype=np.float32,
                copy=False,
            )

            anomaly_flags = (scores >= self.anomaly_threshold).astype(np.float32)

            group_state = state.get(
                group_key,
                {
                    "score_window": [],
                    "flag_window": [],
                },
            )

            previous_score_window = np.asarray(
                group_state["score_window"],
                dtype=np.float32,
            )

            previous_flag_window = np.asarray(
                group_state["flag_window"],
                dtype=np.float32,
            )

            combined_scores = np.concatenate([previous_score_window, scores])
            combined_flags = np.concatenate([previous_flag_window, anomaly_flags])

            trend_values = (
                pd.Series(combined_scores)
                .rolling(window=self.trend_window, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )[-len(scores):]

            persistence_values = (
                pd.Series(combined_flags)
                .rolling(window=self.trend_window, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )[-len(scores):]

            trend_values = np.clip(trend_values, 0.0, 1.0)
            persistence_values = np.clip(persistence_values, 0.0, 1.0)

            result.loc[group_index, "residual_trend_score"] = trend_values
            result.loc[group_index, "anomaly_persistence_score"] = persistence_values

            keep_count = max(self.trend_window - 1, 1)

            state[group_key] = {
                "score_window": combined_scores[-keep_count:].tolist(),
                "flag_window": combined_flags[-keep_count:].tolist(),
            }

            del scores
            del anomaly_flags
            del previous_score_window
            del previous_flag_window
            del combined_scores
            del combined_flags
            del trend_values
            del persistence_values

        weights = self.health_weights

        result["health_index"] = (
            100.0
            - float(weights["final_anomaly_score"]) * result["final_anomaly_score"]
            - float(weights["residual_trend_score"]) * result["residual_trend_score"]
            - float(weights["anomaly_persistence_score"]) * result["anomaly_persistence_score"]
        )

        result["health_index"] = (
            result["health_index"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .clip(0.0, 100.0)
        )

        result["remaining_health_percentage"] = result["health_index"]

        return result

    # ==================================================================================
    # Main calculation
    # ==================================================================================

    def calculate_file(self) -> int:
        """
        Calculate health index chunk-by-chunk.

        Returns:
            Number of rows written.
        """
        print("[PROGRESS] Entering HealthIndexCalculator.calculate_file")

        try:
            started = perf_counter()

            if not self.input_csv.exists():
                raise FileNotFoundError(f"Anomaly fusion CSV not found: {self.input_csv}")

            expected_rows = self._count_csv_rows(self.input_csv)

            if expected_rows <= 0:
                raise ValueError("anomaly_fusion.csv contains zero rows.")

            columns = self._read_header_columns(self.input_csv)
            usecols = self._build_usecols(columns)

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary health index CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            state: Dict[Tuple[object, object], Dict[str, object]] = {}

            health_sum = 0.0
            health_min = np.inf
            health_max = -np.inf
            trend_sum = 0.0
            persistence_sum = 0.0
            anomaly_score_sum = 0.0

            split_stats: Dict[str, Dict[str, float]] = {
                Config.DEV_SPLIT_NAME: {
                    "rows": 0,
                    "health_sum": 0.0,
                    "anomaly_score_sum": 0.0,
                    "trend_sum": 0.0,
                    "persistence_sum": 0.0,
                    "min_health": np.inf,
                    "max_health": -np.inf,
                },
                Config.TEST_SPLIT_NAME: {
                    "rows": 0,
                    "health_sum": 0.0,
                    "anomaly_score_sum": 0.0,
                    "trend_sum": 0.0,
                    "persistence_sum": 0.0,
                    "min_health": np.inf,
                    "max_health": -np.inf,
                },
            }

            print("[PROGRESS] Starting memory-safe health index calculation")

            for chunk in pd.read_csv(
                self.input_csv,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                chunk = chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Health index chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                result_chunk = self._calculate_health_for_chunk(
                    chunk=chunk,
                    state=state,
                )

                output_columns = [
                    "unit_id",
                    "cycle",
                    "split",
                ]

                if "gmm_context_id" in result_chunk.columns:
                    output_columns.append("gmm_context_id")

                output_columns.extend(
                    [
                        "final_anomaly_score",
                        "alert_level",
                        "residual_trend_score",
                        "anomaly_persistence_score",
                        "health_index",
                        "remaining_health_percentage",
                    ]
                )

                optional_output_columns = [
                    "severity_rank",
                    "severity_description",
                    "detector_agreement_count",
                    "detector_agreement_ratio",
                    "dominant_detector",
                    "residual_anomaly_score",
                    "iforest_anomaly_score",
                    "mahalanobis_score",
                    "lstm_autoencoder_score",
                ]

                for column in optional_output_columns:
                    if column in result_chunk.columns and column not in output_columns:
                        output_columns.append(column)

                result_chunk = result_chunk[output_columns]

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                health_values = result_chunk["health_index"].to_numpy(dtype=np.float32)
                anomaly_scores = result_chunk["final_anomaly_score"].to_numpy(dtype=np.float32)
                trend_scores = result_chunk["residual_trend_score"].to_numpy(dtype=np.float32)
                persistence_scores = result_chunk["anomaly_persistence_score"].to_numpy(dtype=np.float32)

                health_sum += float(np.sum(health_values, dtype=np.float64))
                health_min = min(health_min, float(np.min(health_values)))
                health_max = max(health_max, float(np.max(health_values)))
                anomaly_score_sum += float(np.sum(anomaly_scores, dtype=np.float64))
                trend_sum += float(np.sum(trend_scores, dtype=np.float64))
                persistence_sum += float(np.sum(persistence_scores, dtype=np.float64))

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_health = result_chunk.loc[split_mask, "health_index"].to_numpy(dtype=np.float32)
                    split_anomaly = result_chunk.loc[split_mask, "final_anomaly_score"].to_numpy(dtype=np.float32)
                    split_trend = result_chunk.loc[split_mask, "residual_trend_score"].to_numpy(dtype=np.float32)
                    split_persistence = result_chunk.loc[split_mask, "anomaly_persistence_score"].to_numpy(dtype=np.float32)

                    split_stats[split]["rows"] += int(len(split_health))
                    split_stats[split]["health_sum"] += float(np.sum(split_health, dtype=np.float64))
                    split_stats[split]["anomaly_score_sum"] += float(np.sum(split_anomaly, dtype=np.float64))
                    split_stats[split]["trend_sum"] += float(np.sum(split_trend, dtype=np.float64))
                    split_stats[split]["persistence_sum"] += float(np.sum(split_persistence, dtype=np.float64))
                    split_stats[split]["min_health"] = min(
                        float(split_stats[split]["min_health"]),
                        float(np.min(split_health)),
                    )
                    split_stats[split]["max_health"] = max(
                        float(split_stats[split]["max_health"]),
                        float(np.max(split_health)),
                    )

                print(f"[PROGRESS] Total health rows written: {total_rows_written}")
                print(
                    "[PROGRESS] Running average health index: "
                    f"{health_sum / max(total_rows_written, 1):.4f}"
                )

                del chunk
                del result_chunk
                del health_values
                del anomaly_scores
                del trend_scores
                del persistence_scores
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All health index chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Health index row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final health_index.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            split_summary: Dict[str, Dict[str, float]] = {}

            for split, stats in split_stats.items():
                rows = int(stats["rows"])

                if rows <= 0:
                    continue

                split_summary[split] = {
                    "rows": rows,
                    "average_health_index": float(stats["health_sum"] / rows),
                    "average_final_anomaly_score": float(stats["anomaly_score_sum"] / rows),
                    "average_residual_trend_score": float(stats["trend_sum"] / rows),
                    "average_anomaly_persistence_score": float(stats["persistence_sum"] / rows),
                    "min_health_index": float(stats["min_health"]),
                    "max_health_index": float(stats["max_health"]),
                }

            config_payload = {
                "formula": (
                    "health_index = 100 - 60*final_anomaly_score "
                    "- 25*residual_trend_score - 15*anomaly_persistence_score"
                ),
                "weights": self.health_weights,
                "clip_range": [0, 100],
                "trend_window": int(self.trend_window),
                "anomaly_persistence_threshold": float(self.anomaly_threshold),
                "rul_prediction": False,
                "uses_y_targets": False,
                "uses_t_targets": False,
                "input_file": str(self.input_csv),
                "output_file": str(self.output_csv),
            }

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "average_health_index": float(health_sum / max(total_rows_written, 1)),
                "min_health_index": float(health_min),
                "max_health_index": float(health_max),
                "average_final_anomaly_score": float(
                    anomaly_score_sum / max(total_rows_written, 1)
                ),
                "average_residual_trend_score": float(
                    trend_sum / max(total_rows_written, 1)
                ),
                "average_anomaly_persistence_score": float(
                    persistence_sum / max(total_rows_written, 1)
                ),
                "split_summary": split_summary,
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_predict_rul": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_anomaly_fusion_only": True,
                },
            }

            print(f"[PROGRESS] Writing health index config JSON to: {self.config_json}")
            atomic_write_json(config_payload, self.config_json)

            print(f"[PROGRESS] Writing health index summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Health index calculation completed successfully")
            print(f"[PROGRESS] Summary: {summary}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Health index calculation completed. rows=%s avg_health=%s",
                total_rows_written,
                summary["average_health_index"],
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Health index calculation failed: {exc}")
            logger.exception("Health index calculation failed.")
            raise RuntimeError("Health index calculation failed.") from exc

    def calculate(self, anomaly_df: pd.DataFrame) -> pd.DataFrame:
        """
        In-memory helper for small DataFrames only.

        Kept for compatibility. Production path is calculate_file().
        """
        print("[PROGRESS] Entering HealthIndexCalculator.calculate")

        try:
            if "final_anomaly_score" not in anomaly_df.columns:
                raise KeyError("final_anomaly_score is required for health index calculation.")

            result = anomaly_df.copy()
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            result["residual_trend_score"] = (
                result.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(
                    lambda series: series.rolling(
                        self.trend_window,
                        min_periods=1,
                    ).mean()
                )
                .clip(0.0, 1.0)
            )

            result["anomaly_persistence_score"] = (
                result.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(
                    lambda series: (
                        (series >= self.anomaly_threshold)
                        .astype(float)
                        .rolling(self.trend_window, min_periods=1)
                        .mean()
                    )
                )
                .clip(0.0, 1.0)
            )

            weights = self.health_weights

            result["health_index"] = (
                100.0
                - float(weights["final_anomaly_score"]) * result["final_anomaly_score"]
                - float(weights["residual_trend_score"]) * result["residual_trend_score"]
                - float(weights["anomaly_persistence_score"]) * result["anomaly_persistence_score"]
            ).clip(0.0, 100.0)

            result["remaining_health_percentage"] = result["health_index"]

            return result

        except Exception as exc:
            logger.exception("Health index calculation failed.")
            raise RuntimeError("Health index calculation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run health index calculation.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering HealthIndexCalculator.run")

        try:
            records_count = self.calculate_file()

            response = {
                "status": "success",
                "message": "Health index calculated without RUL prediction.",
                "output_file": str(self.output_csv),
                "config_file": str(self.config_json),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Health index calculator response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Health index calculator stage failed: {exc}")
            logger.exception("Health index calculator stage failed.")
            raise RuntimeError("Health index calculator stage failed.") from exc


def run_health_index_calculation() -> Dict[str, object]:
    """
    Execute health index calculation.
    """
    print("[PROGRESS] Entering run_health_index_calculation")

    calculator = HealthIndexCalculator()
    return calculator.run()


if __name__ == "__main__":
    print("[PROGRESS] health_index.py execution started")
    result = run_health_index_calculation()
    print("[PROGRESS] health_index.py execution finished successfully")
    print(result)