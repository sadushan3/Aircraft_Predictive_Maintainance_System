"""
Early warning score for CA-EDT-AHMA.

Role:
Calculate early warning behavior from recent fused anomaly score trend.

Reads:
outputs/Anomaly_Health_Monitering/anomaly_fusion.csv

Writes:
outputs/Anomaly_Health_Monitering/early_warning_scores.csv

Memory-safe:
- Does not load full anomaly_fusion.csv into RAM.
- Reads in chunks.
- Maintains rolling state across chunks per split/unit_id.
- Writes to temporary CSV first.
- Replaces final CSV only after successful completion.

Formula:
early_warning_score =
    0.70 * rolling_anomaly_mean
  + 0.30 * positive_rolling_anomaly_slope

Labels:
- Stable          : early_warning_score < 0.40
- Watch_Risk      : 0.40 <= score < 0.65
- Increasing_Risk : score >= 0.65
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "anomaly_detection/early_warning_score.py"
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
        sys.path.insert(0, BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class EarlyWarningScore:
    """
    Memory-safe early warning score calculator.

    Uses fused anomaly score trend per split/unit_id.
    """

    def __init__(self, rolling_window: int | None = None, chunk_size: int = 25_000) -> None:
        """
        Initialize early warning calculator.

        Args:
            rolling_window: Rolling window for trend calculation.
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering EarlyWarningScore.__init__")

        Config.create_directories()

        self.rolling_window = int(
            rolling_window
            if rolling_window is not None
            else getattr(Config, "EARLY_WARNING_ROLLING_WINDOW", getattr(Config, "ROLLING_WINDOW", 5))
        )

        self.chunk_size = int(
            getattr(Config, "EARLY_WARNING_CHUNK_SIZE", chunk_size)
        )

        self.watch_threshold = float(
            getattr(Config, "EARLY_WARNING_WATCH_THRESHOLD", 0.40)
        )

        self.increasing_threshold = float(
            getattr(Config, "EARLY_WARNING_INCREASING_THRESHOLD", 0.65)
        )

        self.mean_weight = float(
            getattr(Config, "EARLY_WARNING_MEAN_WEIGHT", 0.70)
        )

        self.slope_weight = float(
            getattr(Config, "EARLY_WARNING_SLOPE_WEIGHT", 0.30)
        )

        if self.rolling_window <= 1:
            raise ValueError("EARLY_WARNING_ROLLING_WINDOW must be greater than 1.")

        if self.chunk_size <= 0:
            raise ValueError("EARLY_WARNING_CHUNK_SIZE must be positive.")

        if not (0.0 <= self.watch_threshold <= self.increasing_threshold <= 1.0):
            raise ValueError(
                "Early warning thresholds must satisfy: "
                "0 <= watch <= increasing <= 1."
            )

        weight_sum = self.mean_weight + self.slope_weight
        if weight_sum <= 0:
            raise ValueError("Early warning weights must sum to a positive value.")

        self.mean_weight = self.mean_weight / weight_sum
        self.slope_weight = self.slope_weight / weight_sum

        self.input_csv: Path = Config.ANOMALY_FUSION_CSV
        self.output_csv: Path = getattr(
            Config,
            "EARLY_WARNING_CSV",
            Config.OUTPUT_DIR / "early_warning_scores.csv",
        )
        self.summary_json: Path = getattr(
            Config,
            "EARLY_WARNING_SUMMARY_JSON",
            Config.REPORT_DIR / "early_warning_summary.json",
        )

        print(f"[PROGRESS] Input CSV: {self.input_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Rolling window: {self.rolling_window}")
        print(f"[PROGRESS] Watch threshold: {self.watch_threshold}")
        print(f"[PROGRESS] Increasing threshold: {self.increasing_threshold}")
        print(f"[PROGRESS] Mean weight: {self.mean_weight}")
        print(f"[PROGRESS] Slope weight: {self.slope_weight}")

    # ==================================================================================
    # Helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV rows without loading the full file.

        Args:
            path: CSV path.

        Returns:
            Number of data rows excluding header.
        """
        print(f"[PROGRESS] Counting CSV rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)

        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV header columns only.
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
        Build read_csv usecols list.

        Keeps required columns plus useful optional explanation fields.
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

        print(f"[PROGRESS] Early warning usecols: {usecols}")
        return usecols

    def _label_array(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert early warning score to label.
        """
        labels = np.full(len(scores), "Stable", dtype=object)

        labels[scores >= self.watch_threshold] = "Watch_Risk"
        labels[scores >= self.increasing_threshold] = "Increasing_Risk"

        return labels

    def _risk_rank_array(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert early warning label to numeric rank.
        """
        ranks = np.zeros(len(labels), dtype=np.int8)

        ranks[labels == "Watch_Risk"] = 1
        ranks[labels == "Increasing_Risk"] = 2

        return ranks

    def _calculate_temporal_features_for_chunk(
        self,
        chunk: pd.DataFrame,
        state: Dict[Tuple[object, object], Dict[str, object]],
    ) -> pd.DataFrame:
        """
        Calculate rolling mean, slope, early warning score for current chunk.

        Maintains state across chunks per (split, unit_id).

        Args:
            chunk: Current anomaly fusion chunk.
            state: Rolling state dictionary.

        Returns:
            Result chunk with early warning fields.
        """
        result = chunk.copy()

        result["final_anomaly_score"] = (
            result["final_anomaly_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .clip(0.0, 1.0)
        )

        result["rolling_anomaly_mean"] = np.zeros(len(result), dtype=np.float32)
        result["rolling_anomaly_slope"] = np.zeros(len(result), dtype=np.float32)
        result["positive_rolling_anomaly_slope"] = np.zeros(len(result), dtype=np.float32)
        result["early_warning_score"] = np.zeros(len(result), dtype=np.float32)

        for group_key, group_index in result.groupby(["split", "unit_id"], sort=False).groups.items():
            scores = result.loc[group_index, "final_anomaly_score"].to_numpy(
                dtype=np.float32,
                copy=False,
            )

            group_state = state.get(
                group_key,
                {
                    "score_window": [],
                    "slope_window": [],
                    "last_score": None,
                    "last_cycle": None,
                },
            )

            previous_score_window = np.asarray(
                group_state["score_window"],
                dtype=np.float32,
            )

            previous_slope_window = np.asarray(
                group_state["slope_window"],
                dtype=np.float32,
            )

            combined_scores = np.concatenate([previous_score_window, scores])

            rolling_mean = (
                pd.Series(combined_scores)
                .rolling(window=self.rolling_window, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )[-len(scores):]

            slopes = np.zeros(len(scores), dtype=np.float32)

            if len(scores) > 0:
                last_score = group_state["last_score"]

                if last_score is None:
                    slopes[0] = 0.0
                else:
                    slopes[0] = scores[0] - float(last_score)

                if len(scores) > 1:
                    slopes[1:] = np.diff(scores)

            combined_slopes = np.concatenate([previous_slope_window, slopes])

            rolling_slope = (
                pd.Series(combined_slopes)
                .rolling(window=self.rolling_window, min_periods=1)
                .mean()
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )[-len(scores):]

            positive_rolling_slope = np.clip(rolling_slope, 0.0, 1.0)

            early_warning_score = (
                self.mean_weight * rolling_mean
                + self.slope_weight * positive_rolling_slope
            )

            early_warning_score = np.clip(
                early_warning_score,
                0.0,
                1.0,
            ).astype(np.float32, copy=False)

            result.loc[group_index, "rolling_anomaly_mean"] = rolling_mean
            result.loc[group_index, "rolling_anomaly_slope"] = rolling_slope
            result.loc[group_index, "positive_rolling_anomaly_slope"] = positive_rolling_slope
            result.loc[group_index, "early_warning_score"] = early_warning_score

            keep_count = max(self.rolling_window - 1, 1)

            state[group_key] = {
                "score_window": combined_scores[-keep_count:].tolist(),
                "slope_window": combined_slopes[-keep_count:].tolist(),
                "last_score": float(scores[-1]) if len(scores) > 0 else group_state["last_score"],
                "last_cycle": result.loc[group_index, "cycle"].iloc[-1]
                if len(group_index) > 0
                else group_state["last_cycle"],
            }

            del scores
            del previous_score_window
            del previous_slope_window
            del combined_scores
            del slopes
            del combined_slopes
            del rolling_mean
            del rolling_slope
            del positive_rolling_slope
            del early_warning_score

        ew_scores = result["early_warning_score"].to_numpy(dtype=np.float32, copy=False)
        labels = self._label_array(ew_scores)

        result["early_warning_label"] = labels
        result["early_warning_rank"] = self._risk_rank_array(labels)
        result["early_warning_window"] = int(self.rolling_window)
        result["early_warning_watch_threshold"] = float(self.watch_threshold)
        result["early_warning_increasing_threshold"] = float(self.increasing_threshold)

        return result

    # ==================================================================================
    # Main calculation
    # ==================================================================================

    def calculate_file(self) -> int:
        """
        Calculate early warning scores chunk-by-chunk.

        Returns:
            Number of rows written.
        """
        print("[PROGRESS] Entering EarlyWarningScore.calculate_file")

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
                print("[PROGRESS] Removing old temporary early warning CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            state: Dict[Tuple[object, object], Dict[str, object]] = {}

            label_counts = {
                "Stable": 0,
                "Watch_Risk": 0,
                "Increasing_Risk": 0,
            }

            split_label_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {
                    "Stable": 0,
                    "Watch_Risk": 0,
                    "Increasing_Risk": 0,
                },
                Config.TEST_SPLIT_NAME: {
                    "Stable": 0,
                    "Watch_Risk": 0,
                    "Increasing_Risk": 0,
                },
            }

            score_sum = 0.0
            rolling_mean_sum = 0.0
            rolling_slope_sum = 0.0

            print("[PROGRESS] Starting memory-safe early warning calculation")

            for chunk in pd.read_csv(
                self.input_csv,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                chunk = chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Early warning chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                result_chunk = self._calculate_temporal_features_for_chunk(
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
                        "rolling_anomaly_mean",
                        "rolling_anomaly_slope",
                        "positive_rolling_anomaly_slope",
                        "early_warning_score",
                        "early_warning_label",
                        "early_warning_rank",
                        "early_warning_window",
                        "early_warning_watch_threshold",
                        "early_warning_increasing_threshold",
                    ]
                )

                optional_output_columns = [
                    "severity_rank",
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

                labels = result_chunk["early_warning_label"].to_numpy(dtype=object)

                unique_labels, unique_counts = np.unique(labels, return_counts=True)

                for label, count in zip(unique_labels, unique_counts):
                    label_counts[str(label)] = label_counts.get(str(label), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_labels = labels[split_mask.to_numpy()]
                    split_unique, split_counts = np.unique(split_labels, return_counts=True)

                    for label, count in zip(split_unique, split_counts):
                        split_label_counts[split][str(label)] = (
                            split_label_counts[split].get(str(label), 0) + int(count)
                        )

                score_sum += float(
                    np.sum(result_chunk["early_warning_score"].to_numpy(dtype=np.float32), dtype=np.float64)
                )
                rolling_mean_sum += float(
                    np.sum(result_chunk["rolling_anomaly_mean"].to_numpy(dtype=np.float32), dtype=np.float64)
                )
                rolling_slope_sum += float(
                    np.sum(result_chunk["positive_rolling_anomaly_slope"].to_numpy(dtype=np.float32), dtype=np.float64)
                )

                print(f"[PROGRESS] Total early warning rows written: {total_rows_written}")
                print(f"[PROGRESS] Running early warning label counts: {label_counts}")

                del chunk
                del result_chunk
                del labels
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All early warning chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Early warning output row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final early_warning_scores.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "rolling_window": int(self.rolling_window),
                "weights": {
                    "rolling_anomaly_mean": float(self.mean_weight),
                    "positive_rolling_anomaly_slope": float(self.slope_weight),
                },
                "thresholds": {
                    "watch_risk": float(self.watch_threshold),
                    "increasing_risk": float(self.increasing_threshold),
                },
                "label_counts": label_counts,
                "split_label_counts": split_label_counts,
                "early_warning_score_mean": float(score_sum / max(total_rows_written, 1)),
                "rolling_anomaly_mean_average": float(
                    rolling_mean_sum / max(total_rows_written, 1)
                ),
                "positive_rolling_anomaly_slope_average": float(
                    rolling_slope_sum / max(total_rows_written, 1)
                ),
                "chunk_size": int(self.chunk_size),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_refit_anomaly_detectors": True,
                    "uses_fused_anomaly_score_only": True,
                    "does_not_use_y_targets": True,
                    "does_not_use_t_degradation_as_input": True,
                },
            }

            print(f"[PROGRESS] Writing early warning summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Early warning calculation completed successfully")
            print(f"[PROGRESS] Label counts: {label_counts}")
            print(f"[PROGRESS] Split label counts: {split_label_counts}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Early warning score calculation completed. rows=%s labels=%s",
                total_rows_written,
                label_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Early warning score calculation failed: {exc}")
            logger.exception("Early warning score calculation failed.")
            raise RuntimeError("Early warning score calculation failed.") from exc

    def calculate(self, fusion_df: pd.DataFrame) -> pd.DataFrame:
        """
        In-memory helper for small DataFrames only.

        Kept for compatibility with old code. Production path is calculate_file().
        """
        print("[PROGRESS] Entering EarlyWarningScore.calculate")

        try:
            if "final_anomaly_score" not in fusion_df.columns:
                raise KeyError(
                    "final_anomaly_score is required for early warning calculation."
                )

            result = fusion_df.copy()
            result = result.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            result["rolling_anomaly_mean"] = (
                result.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(
                    lambda series: series.rolling(
                        self.rolling_window,
                        min_periods=1,
                    ).mean()
                )
            )

            result["rolling_anomaly_slope"] = (
                result.groupby(["split", "unit_id"])["final_anomaly_score"]
                .transform(
                    lambda series: series.diff()
                    .rolling(self.rolling_window, min_periods=1)
                    .mean()
                )
                .fillna(0.0)
            )

            result["positive_rolling_anomaly_slope"] = np.clip(
                result["rolling_anomaly_slope"],
                0.0,
                1.0,
            )

            result["early_warning_score"] = (
                self.mean_weight * result["rolling_anomaly_mean"]
                + self.slope_weight * result["positive_rolling_anomaly_slope"]
            ).clip(0.0, 1.0)

            labels = self._label_array(
                result["early_warning_score"].to_numpy(dtype=np.float32)
            )

            result["early_warning_label"] = labels
            result["early_warning_rank"] = self._risk_rank_array(labels)
            result["early_warning_window"] = int(self.rolling_window)

            return result

        except Exception as exc:
            logger.exception("Early warning score calculation failed.")
            raise RuntimeError("Early warning score calculation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run early warning score calculation.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering EarlyWarningScore.run")

        try:
            records_count = self.calculate_file()

            response = {
                "status": "success",
                "message": "Early warning scores generated.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Early warning score response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Early warning score stage failed: {exc}")
            logger.exception("Early warning score stage failed.")
            raise RuntimeError("Early warning score stage failed.") from exc


def run_early_warning_score() -> Dict[str, object]:
    """
    Execute early warning score stage.
    """
    print("[PROGRESS] Entering run_early_warning_score")

    service = EarlyWarningScore()
    return service.run()


if __name__ == "__main__":
    print("[PROGRESS] early_warning_score.py execution started")
    result = run_early_warning_score()
    print("[PROGRESS] early_warning_score.py execution finished successfully")
    print(result)