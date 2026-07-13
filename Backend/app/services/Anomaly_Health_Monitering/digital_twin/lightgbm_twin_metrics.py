"""
LightGBM Digital Twin Evaluator for CA-EDT-AHMA.

Purpose:
Evaluate LightGBM digital twin predictions against actual raw X_s sensors.

Research-correct validation:
- Actual values come from scaled_features.csv raw X_s columns.
- Predictions come from lgbm_predictions.csv.
- Evaluates dev and test separately.
- Does not fit/train anything.
- Does not use Y or T.
- Uses raw X_s sensors only.
- Memory-safe chunked evaluation.

Output:
metrics/lightgbm_twin_metrics.csv

Rows:
- per split, per sensor
- plus ALL_RAW_XS summary rows for dev and test
"""

from __future__ import annotations

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
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_raw_xs_columns


logger = get_logger(__name__)


class LightGBMTwinEvaluator:
    """
    Memory-safe evaluator for LightGBM digital twin predictions.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize evaluator.

        Args:
            chunk_size: CSV rows processed per chunk.
        """
        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "LGBM_EVALUATION_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("LGBM evaluation chunk_size must be positive.")

        self.predictions_csv: Path = getattr(
            Config,
            "LGBM_PREDICTIONS_CSV",
            Config.OUTPUT_DIR / "lgbm_predictions.csv",
        )

        self.metrics_csv: Path = getattr(
            Config,
            "LGBM_METRICS_CSV",
            Config.METRIC_DIR / "lightgbm_twin_metrics.csv",
        )

        print("[PROGRESS] LightGBMTwinEvaluator initialized")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Scaled CSV: {Config.SCALED_CSV}")
        print(f"[PROGRESS] LightGBM predictions CSV: {self.predictions_csv}")
        print(f"[PROGRESS] Metrics CSV: {self.metrics_csv}")

    # ==================================================================================
    # Header helpers
    # ==================================================================================

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read CSV header only as empty DataFrame.

        Args:
            path: CSV path.

        Returns:
            Empty DataFrame with columns.
        """
        print(f"[PROGRESS] Reading header from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV column names only.

        Args:
            path: CSV path.

        Returns:
            Column list.
        """
        print(f"[PROGRESS] Reading columns from: {path}")
        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate required columns exist.

        Args:
            available_columns: Existing columns.
            required_columns: Required columns.
            label: Human-readable label.
        """
        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _verify_key_alignment(
        self,
        scaled_chunk: pd.DataFrame,
        prediction_chunk: pd.DataFrame,
        merge_columns: List[str],
    ) -> None:
        """
        Verify scaled and prediction chunks are row-aligned.

        Args:
            scaled_chunk: Chunk from scaled_features.csv.
            prediction_chunk: Chunk from lgbm_predictions.csv.
            merge_columns: Row identity columns.
        """
        if len(scaled_chunk) != len(prediction_chunk):
            raise ValueError(
                "Chunk row count mismatch: "
                f"scaled={len(scaled_chunk)}, prediction={len(prediction_chunk)}"
            )

        scaled_keys = scaled_chunk[merge_columns].reset_index(drop=True)
        prediction_keys = prediction_chunk[merge_columns].reset_index(drop=True)

        if not scaled_keys.equals(prediction_keys):
            raise ValueError(
                "scaled_features.csv and lgbm_predictions.csv are not row-aligned. "
                "Regenerate LightGBM predictions using the same scaled_features.csv order."
            )

    # ==================================================================================
    # Metric calculation
    # ==================================================================================

    def _initialize_accumulators(
        self,
        splits: List[str],
        sensors: List[str],
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Initialize metric accumulators.

        Args:
            splits: Split names.
            sensors: Raw X_s sensors.

        Returns:
            Accumulator dictionary.
        """
        accumulators: Dict[Tuple[str, str], Dict[str, float]] = {}

        for split in splits:
            for sensor in sensors:
                accumulators[(split, sensor)] = {
                    "count": 0.0,
                    "sum_abs_error": 0.0,
                    "sum_squared_error": 0.0,
                    "sum_y": 0.0,
                    "sum_y_squared": 0.0,
                }

        return accumulators

    def _accumulate_metrics(
        self,
        accumulators: Dict[Tuple[str, str], Dict[str, float]],
        split: str,
        sensor: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        Accumulate error statistics.

        Args:
            accumulators: Metric accumulators.
            split: Split name.
            sensor: Sensor name.
            y_true: Actual values.
            y_pred: Predicted values.
        """
        error = y_true - y_pred
        key = (split, sensor)

        accumulators[key]["count"] += float(error.size)
        accumulators[key]["sum_abs_error"] += float(
            np.abs(error).sum(dtype=np.float64)
        )
        accumulators[key]["sum_squared_error"] += float(
            np.square(error, dtype=np.float64).sum()
        )
        accumulators[key]["sum_y"] += float(y_true.sum(dtype=np.float64))
        accumulators[key]["sum_y_squared"] += float(
            np.square(y_true, dtype=np.float64).sum()
        )

    def _build_metrics_df(
        self,
        accumulators: Dict[Tuple[str, str], Dict[str, float]],
        splits: List[str],
        sensors: List[str],
    ) -> pd.DataFrame:
        """
        Convert accumulators into metrics DataFrame.

        Args:
            accumulators: Metric accumulators.
            splits: Split names.
            sensors: Raw X_s sensors.

        Returns:
            Metrics DataFrame.
        """
        records: List[Dict[str, object]] = []

        for split in splits:
            for sensor in sensors:
                stats = accumulators[(split, sensor)]
                count = stats["count"]

                if count <= 0:
                    print(f"[WARNING] No rows accumulated for split={split}, sensor={sensor}")
                    continue

                mae = stats["sum_abs_error"] / count
                rmse = float(np.sqrt(stats["sum_squared_error"] / count))

                sst = stats["sum_y_squared"] - (
                    (stats["sum_y"] ** 2) / max(count, 1.0)
                )

                if abs(sst) < 1e-12:
                    r2 = 0.0
                else:
                    r2 = 1.0 - (stats["sum_squared_error"] / sst)

                records.append(
                    {
                        "split": split,
                        "sensor": sensor,
                        "model": "lightgbm",
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "r2": float(r2),
                        "r2_percent": float(r2 * 100.0),
                        "count": int(count),
                        "target_type": "raw_X_s_only",
                    }
                )

        metrics_df = pd.DataFrame(records)

        overall_records: List[Dict[str, object]] = []

        for split in splits:
            split_rows = metrics_df[metrics_df["split"] == split]

            if split_rows.empty:
                continue

            overall_records.append(
                {
                    "split": split,
                    "sensor": "ALL_RAW_XS",
                    "model": "lightgbm",
                    "mae": float(split_rows["mae"].mean()),
                    "rmse": float(split_rows["rmse"].mean()),
                    "r2": float(split_rows["r2"].mean()),
                    "r2_percent": float(split_rows["r2_percent"].mean()),
                    "count": int(split_rows["count"].sum()),
                    "target_type": "raw_X_s_only",
                }
            )

        if overall_records:
            metrics_df = pd.concat(
                [pd.DataFrame(overall_records), metrics_df],
                ignore_index=True,
            )

        return metrics_df

    # ==================================================================================
    # Main evaluation
    # ==================================================================================

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate LightGBM predictions.

        Returns:
            Metrics DataFrame.
        """
        print("[PROGRESS] Entering LightGBMTwinEvaluator.evaluate")

        try:
            started = perf_counter()

            if not Config.SCALED_CSV.exists():
                raise FileNotFoundError(f"Scaled CSV not found: {Config.SCALED_CSV}")

            if not self.predictions_csv.exists():
                raise FileNotFoundError(
                    f"LightGBM predictions CSV not found: {self.predictions_csv}"
                )

            merge_columns = ["unit_id", "cycle", "split"]
            splits = [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]

            scaled_header_df = self._read_header_df(Config.SCALED_CSV)
            scaled_columns = list(scaled_header_df.columns)
            prediction_columns_available = self._read_header_columns(self.predictions_csv)

            sensors = get_raw_xs_columns(scaled_header_df)

            if not sensors:
                raise ValueError("No raw X_s sensors found in scaled CSV.")

            prediction_columns = [
                f"lgbm_predicted_{sensor}" for sensor in sensors
            ]

            self._validate_columns(
                available_columns=scaled_columns,
                required_columns=merge_columns + sensors,
                label="scaled_features.csv",
            )

            self._validate_columns(
                available_columns=prediction_columns_available,
                required_columns=merge_columns + prediction_columns,
                label="lgbm_predictions.csv",
            )

            print(f"[PROGRESS] Raw X_s sensors count: {len(sensors)}")
            print(f"[PROGRESS] Raw X_s sensors: {sensors}")
            print(f"[PROGRESS] LightGBM prediction columns count: {len(prediction_columns)}")

            accumulators = self._initialize_accumulators(
                splits=splits,
                sensors=sensors,
            )

            scaled_usecols = merge_columns + sensors
            prediction_usecols = merge_columns + prediction_columns

            scaled_iter = pd.read_csv(
                Config.SCALED_CSV,
                usecols=scaled_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            prediction_iter = pd.read_csv(
                self.predictions_csv,
                usecols=prediction_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            total_rows_seen = 0
            chunk_index = 0

            for scaled_chunk, prediction_chunk in zip(scaled_iter, prediction_iter):
                chunk_index += 1
                chunk_rows = len(scaled_chunk)
                total_rows_seen += chunk_rows

                print("=" * 100)
                print(f"[PROGRESS] Evaluating LightGBM chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {chunk_rows}")
                print(f"[PROGRESS] Total rows scanned: {total_rows_seen}")

                self._verify_key_alignment(
                    scaled_chunk=scaled_chunk,
                    prediction_chunk=prediction_chunk,
                    merge_columns=merge_columns,
                )

                for split in splits:
                    split_mask = scaled_chunk["split"] == split
                    split_rows = int(split_mask.sum())

                    print(f"[PROGRESS] Split={split}, rows in chunk={split_rows}")

                    if split_rows == 0:
                        continue

                    for sensor in sensors:
                        actual_column = sensor
                        prediction_column = f"lgbm_predicted_{sensor}"

                        y_true = scaled_chunk.loc[
                            split_mask,
                            actual_column,
                        ].to_numpy(dtype=np.float32, copy=False)

                        y_pred = prediction_chunk.loc[
                            split_mask,
                            prediction_column,
                        ].to_numpy(dtype=np.float32, copy=False)

                        self._accumulate_metrics(
                            accumulators=accumulators,
                            split=split,
                            sensor=sensor,
                            y_true=y_true,
                            y_pred=y_pred,
                        )

                        del y_true
                        del y_pred

                del scaled_chunk
                del prediction_chunk
                gc.collect()

            metrics_df = self._build_metrics_df(
                accumulators=accumulators,
                splits=splits,
                sensors=sensors,
            )

            duration = perf_counter() - started

            print("[PROGRESS] LightGBM evaluation completed")
            print(f"[PROGRESS] Metrics rows: {len(metrics_df)}")
            print(f"[PROGRESS] Total rows scanned: {total_rows_seen}")
            print(f"[PROGRESS] Evaluation duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Evaluation duration minutes: {duration / 60.0:.2f}")

            print("[PROGRESS] Overall LightGBM performance rows:")
            overall_df = metrics_df[metrics_df["sensor"] == "ALL_RAW_XS"]
            print(overall_df.to_string(index=False))

            logger.info("LightGBM evaluation completed. rows=%s", len(metrics_df))

            return metrics_df

        except Exception as exc:
            print(f"[ERROR] LightGBM evaluation failed: {exc}")
            logger.exception("LightGBM evaluation failed.")
            raise RuntimeError("LightGBM evaluation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run LightGBM evaluation and write metrics CSV.

        Returns:
            Stage response dictionary.
        """
        print("[PROGRESS] Entering LightGBMTwinEvaluator.run")

        try:
            metrics_df = self.evaluate()

            print(f"[PROGRESS] Writing LightGBM metrics CSV to: {self.metrics_csv}")
            atomic_write_csv(metrics_df, self.metrics_csv)
            print("[PROGRESS] LightGBM metrics CSV written successfully")

            response = {
                "status": "success",
                "message": (
                    "LightGBM digital twin evaluation completed for raw X_s sensors only."
                ),
                "output_file": str(self.metrics_csv),
                "records_count": int(len(metrics_df)),
                "target_type": "raw_X_s_only",
                "model": "lightgbm",
            }

            print(f"[PROGRESS] LightGBM evaluator response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] LightGBM evaluator stage failed: {exc}")
            logger.exception("LightGBM evaluator stage failed.")
            raise RuntimeError("LightGBM evaluator stage failed.") from exc


def run_lightgbm_evaluator() -> Dict[str, object]:
    """
    Execute LightGBM evaluator.

    Returns:
        Stage response dictionary.
    """
    evaluator = LightGBMTwinEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    print("[PROGRESS] lightgbm_evaluator.py execution started")
    result = run_lightgbm_evaluator()
    print("[PROGRESS] lightgbm_evaluator.py execution finished successfully")
    print(result)