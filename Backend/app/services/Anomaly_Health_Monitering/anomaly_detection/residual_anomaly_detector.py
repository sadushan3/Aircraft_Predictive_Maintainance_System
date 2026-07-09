"""
Residual threshold anomaly detector for CA-EDT-AHMA.

Role:
Context-aware residual anomaly scoring.

Training:
Input = dev residuals + gmm_context_id
Fit = context-specific residual thresholds from dev only.

Thresholds:
watch    = 90th percentile
warning  = 95th percentile
critical = 99th percentile

Reads:
outputs/Anomaly_Health_Monitering/residuals.csv

Writes:
outputs/Anomaly_Health_Monitering/residual_anomaly_scores.csv

Saves:
models/anomaly/residual_thresholds.json

Memory-safe version:
- Does not load full residuals.csv into RAM.
- Fits thresholds chunk-by-chunk using dev rows only.
- Scores all rows chunk-by-chunk.
- Writes output to temporary CSV first.
- Replaces final CSV only after successful completion.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "anomaly_detection/residual_anomaly_detector.py"
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


from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_json,
    read_json_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_abs_residual_columns


logger = get_logger(__name__)


class ResidualAnomalyDetector:
    """
    Memory-safe context-aware residual threshold detector.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize residual anomaly detector.

        Args:
            chunk_size: Number of residual rows processed per chunk.
        """
        print("[PROGRESS] Entering ResidualAnomalyDetector.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "RESIDUAL_ANOMALY_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("RESIDUAL_ANOMALY_CHUNK_SIZE must be positive.")

        self.threshold_path: Path = Config.RESIDUAL_THRESHOLDS_PATH
        self.output_csv: Path = Config.RESIDUAL_ANOMALY_CSV

        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Residual CSV: {Config.RESIDUALS_CSV}")
        print(f"[PROGRESS] Threshold JSON: {self.threshold_path}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")

    # ==================================================================================
    # Basic helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV data rows without loading the file.

        Args:
            path: CSV path.

        Returns:
            Data row count excluding header.
        """
        print(f"[PROGRESS] Counting rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)

        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read CSV header only as empty DataFrame.
        """
        print(f"[PROGRESS] Reading CSV header from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV column names only.
        """
        print(f"[PROGRESS] Reading CSV columns from: {path}")
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
        print(f"[PROGRESS] Validating columns for {label}")

        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _get_abs_residual_columns(self, residual_header_df: pd.DataFrame) -> List[str]:
        """
        Get true absolute residual columns only.

        This excludes temporal features such as:
        resfeat_abs_residual_Xs_*_rolling_mean_5
        """
        all_abs_columns = get_abs_residual_columns(residual_header_df)

        abs_residual_columns = [
            column
            for column in all_abs_columns
            if column.startswith("abs_residual_Xs_")
            and not column.startswith("resfeat_")
        ]

        if not abs_residual_columns:
            # Fallback if utility returns nothing.
            abs_residual_columns = [
                column
                for column in residual_header_df.columns
                if column.startswith("abs_residual_Xs_")
            ]

        if not abs_residual_columns:
            raise ValueError("No abs_residual_Xs_* columns found in residuals.csv.")

        print(f"[PROGRESS] Absolute residual column count: {len(abs_residual_columns)}")
        print(f"[PROGRESS] Absolute residual columns: {abs_residual_columns}")

        return abs_residual_columns

    def _validate_threshold_payload(self, payload: Dict[str, object]) -> Dict[str, Dict[str, float]]:
        """
        Validate loaded threshold payload.

        Args:
            payload: JSON payload.

        Returns:
            Threshold dictionary.
        """
        thresholds = payload.get("thresholds", {})

        if not isinstance(thresholds, dict) or not thresholds:
            raise ValueError("Residual threshold payload has no thresholds.")

        if "global" not in thresholds:
            raise KeyError("Residual threshold payload is missing global thresholds.")

        required_levels = ["watch", "warning", "critical"]

        for context_key, context_thresholds in thresholds.items():
            for level in required_levels:
                if level not in context_thresholds:
                    raise KeyError(
                        f"Threshold level '{level}' missing for context '{context_key}'."
                    )

            watch = float(context_thresholds["watch"])
            warning = float(context_thresholds["warning"])
            critical = float(context_thresholds["critical"])

            if not (watch <= warning <= critical):
                raise ValueError(
                    f"Invalid threshold order for context {context_key}: "
                    f"watch={watch}, warning={warning}, critical={critical}"
                )

        print("[PROGRESS] Residual threshold payload validated successfully")

        return {
            str(context_key): {
                "watch": float(context_thresholds["watch"]),
                "warning": float(context_thresholds["warning"]),
                "critical": float(context_thresholds["critical"]),
            }
            for context_key, context_thresholds in thresholds.items()
        }

    # ==================================================================================
    # Threshold fitting
    # ==================================================================================

    def fit_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Fit context-specific residual thresholds using dev residuals only.

        Returns:
            Threshold dictionary.
        """
        print("[PROGRESS] Entering ResidualAnomalyDetector.fit_thresholds")

        try:
            started = perf_counter()

            if not Config.RESIDUALS_CSV.exists():
                raise FileNotFoundError(f"Residual CSV not found: {Config.RESIDUALS_CSV}")

            residual_header_df = self._read_header_df(Config.RESIDUALS_CSV)
            residual_columns = list(residual_header_df.columns)

            merge_columns = ["unit_id", "cycle", "split"]
            required_columns = merge_columns + ["gmm_context_id"]

            self._validate_columns(
                available_columns=residual_columns,
                required_columns=required_columns,
                label="residuals.csv",
            )

            abs_residual_columns = self._get_abs_residual_columns(residual_header_df)

            usecols = merge_columns + ["gmm_context_id"] + abs_residual_columns

            context_arrays: Dict[str, List[np.ndarray]] = {}
            dev_rows_seen = 0
            total_rows_seen = 0
            chunk_index = 0

            print("[PROGRESS] Starting chunked dev threshold fitting")
            print(f"[PROGRESS] Threshold percentiles: {Config.RESIDUAL_PERCENTILES}")

            for chunk in pd.read_csv(
                Config.RESIDUALS_CSV,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                total_rows_seen += len(chunk)

                print("=" * 100)
                print(f"[PROGRESS] Fitting threshold chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")
                print(f"[PROGRESS] Total rows scanned: {total_rows_seen}")

                dev_mask = chunk["split"] == Config.DEV_SPLIT_NAME
                dev_rows = int(dev_mask.sum())
                dev_rows_seen += dev_rows

                print(f"[PROGRESS] Dev rows in chunk: {dev_rows}")
                print(f"[PROGRESS] Total dev rows seen: {dev_rows_seen}")

                if dev_rows == 0:
                    del chunk
                    gc.collect()
                    continue

                dev_chunk = chunk.loc[dev_mask, ["gmm_context_id"] + abs_residual_columns]

                total_abs_residual = (
                    dev_chunk[abs_residual_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum(axis=1)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                context_ids = dev_chunk["gmm_context_id"].to_numpy(copy=False)

                for context_id in np.unique(context_ids):
                    context_key = str(int(context_id))
                    context_mask = context_ids == context_id
                    context_values = total_abs_residual[context_mask].astype(
                        np.float32,
                        copy=True,
                    )

                    if context_key not in context_arrays:
                        context_arrays[context_key] = []

                    context_arrays[context_key].append(context_values)

                    print(
                        f"[PROGRESS] Context {context_key}: appended "
                        f"{len(context_values)} dev residual scores"
                    )

                del chunk
                del dev_chunk
                del total_abs_residual
                del context_ids
                gc.collect()

            if dev_rows_seen <= 0:
                raise ValueError("No dev residual rows found. Cannot fit thresholds.")

            if not context_arrays:
                raise ValueError("No context residual arrays collected.")

            thresholds: Dict[str, Dict[str, float]] = {}
            global_arrays: List[np.ndarray] = []

            print("[PROGRESS] Calculating percentile thresholds per GMM context")

            for context_key, arrays in sorted(context_arrays.items(), key=lambda item: int(item[0])):
                context_values = np.concatenate(arrays).astype(np.float32, copy=False)

                if len(context_values) == 0:
                    continue

                global_arrays.append(context_values)

                thresholds[context_key] = {
                    level: float(np.percentile(context_values, percentile))
                    for level, percentile in Config.RESIDUAL_PERCENTILES.items()
                }

                print(f"[PROGRESS] Context {context_key} thresholds: {thresholds[context_key]}")

            if not global_arrays:
                raise ValueError("No global residual values available for thresholds.")

            global_values = np.concatenate(global_arrays).astype(np.float32, copy=False)

            thresholds["global"] = {
                level: float(np.percentile(global_values, percentile))
                for level, percentile in Config.RESIDUAL_PERCENTILES.items()
            }

            print(f"[PROGRESS] Global residual thresholds: {thresholds['global']}")

            duration = perf_counter() - started

            threshold_payload = {
                "thresholds": thresholds,
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "target_type": "raw_X_s_only",
                "score_input": "sum_abs_residual_raw_X_s",
                "percentiles": Config.RESIDUAL_PERCENTILES,
                "dev_rows_used": int(dev_rows_seen),
                "total_rows_scanned": int(total_rows_seen),
                "abs_residual_columns": abs_residual_columns,
                "context_count": int(len([key for key in thresholds.keys() if key != "global"])),
                "duration_seconds": float(duration),
                "leakage_audit": {
                    "thresholds_fit_on_dev_only": True,
                    "test_rows_used_for_threshold_fitting": 0,
                    "uses_y_targets": False,
                    "uses_t_degradation_as_input": False,
                },
            }

            print(f"[PROGRESS] Writing residual thresholds JSON to: {self.threshold_path}")
            atomic_write_json(threshold_payload, self.threshold_path)

            print("[PROGRESS] Residual thresholds fitted successfully")
            print(f"[PROGRESS] Threshold fitting duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Threshold fitting duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Residual thresholds fitted on dev split only. dev_rows=%s contexts=%s",
                dev_rows_seen,
                len(thresholds),
            )

            return thresholds

        except Exception as exc:
            print(f"[ERROR] Residual threshold fitting failed: {exc}")
            logger.exception("Residual threshold fitting failed.")
            raise RuntimeError("Residual threshold fitting failed.") from exc

    def load_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Load saved residual thresholds.

        Returns:
            Threshold dictionary.
        """
        print("[PROGRESS] Entering ResidualAnomalyDetector.load_thresholds")

        try:
            if not self.threshold_path.exists():
                print("[PROGRESS] Threshold file not found. Fitting thresholds now.")
                return self.fit_thresholds()

            data = read_json_required(self.threshold_path)
            thresholds = self._validate_threshold_payload(data)

            print(f"[PROGRESS] Loaded residual thresholds from: {self.threshold_path}")
            return thresholds

        except Exception as exc:
            print(f"[ERROR] Failed to load residual thresholds: {exc}")
            logger.exception("Failed to load residual thresholds.")
            raise RuntimeError("Failed to load residual thresholds.") from exc

    # ==================================================================================
    # Scoring
    # ==================================================================================

    def _threshold_arrays_for_chunk(
        self,
        context_ids: np.ndarray,
        thresholds: Dict[str, Dict[str, float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build watch/warning/critical threshold arrays for a chunk.

        Args:
            context_ids: gmm_context_id array.
            thresholds: Threshold dictionary.

        Returns:
            Tuple of watch, warning, critical arrays.
        """
        global_thresholds = thresholds["global"]

        watch_thresholds = np.full(
            len(context_ids),
            float(global_thresholds["watch"]),
            dtype=np.float32,
        )
        warning_thresholds = np.full(
            len(context_ids),
            float(global_thresholds["warning"]),
            dtype=np.float32,
        )
        critical_thresholds = np.full(
            len(context_ids),
            float(global_thresholds["critical"]),
            dtype=np.float32,
        )

        for context_id in np.unique(context_ids):
            context_key = str(int(context_id))
            context_thresholds = thresholds.get(context_key, global_thresholds)
            mask = context_ids == context_id

            watch_thresholds[mask] = float(context_thresholds["watch"])
            warning_thresholds[mask] = float(context_thresholds["warning"])
            critical_thresholds[mask] = float(context_thresholds["critical"])

        return watch_thresholds, warning_thresholds, critical_thresholds

    def score(self) -> int:
        """
        Score residual anomalies for dev and test using chunked residual data.

        Returns:
            Number of scored rows written.
        """
        print("[PROGRESS] Entering ResidualAnomalyDetector.score")

        try:
            started = perf_counter()

            if not Config.RESIDUALS_CSV.exists():
                raise FileNotFoundError(f"Residual CSV not found: {Config.RESIDUALS_CSV}")

            expected_rows = self._count_csv_rows(Config.RESIDUALS_CSV)

            residual_header_df = self._read_header_df(Config.RESIDUALS_CSV)
            residual_columns = list(residual_header_df.columns)

            merge_columns = ["unit_id", "cycle", "split"]
            required_columns = merge_columns + ["gmm_context_id"]

            self._validate_columns(
                available_columns=residual_columns,
                required_columns=required_columns,
                label="residuals.csv",
            )

            abs_residual_columns = self._get_abs_residual_columns(residual_header_df)

            thresholds = self.load_thresholds()
            self._validate_threshold_payload({"thresholds": thresholds})

            usecols = merge_columns + ["gmm_context_id"] + abs_residual_columns

            output_path = self.output_csv
            temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary residual anomaly CSV")
                temp_output_path.unlink()

            print(f"[PROGRESS] Final residual anomaly CSV: {output_path}")
            print(f"[PROGRESS] Temporary residual anomaly CSV: {temp_output_path}")

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            alert_counts = {
                "Normal": 0,
                "Watch": 0,
                "Warning": 0,
                "Critical": 0,
            }

            print("[PROGRESS] Starting chunked residual anomaly scoring")

            for chunk in pd.read_csv(
                Config.RESIDUALS_CSV,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1

                print("=" * 100)
                print(f"[PROGRESS] Scoring residual anomaly chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                context_ids = chunk["gmm_context_id"].to_numpy(copy=False)

                total_abs_residual = (
                    chunk[abs_residual_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .sum(axis=1)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                watch_thresholds, warning_thresholds, critical_thresholds = (
                    self._threshold_arrays_for_chunk(
                        context_ids=context_ids,
                        thresholds=thresholds,
                    )
                )

                safe_critical = np.maximum(critical_thresholds, 1e-9)

                residual_scores = np.minimum(
                    total_abs_residual / safe_critical,
                    1.0,
                ).astype(np.float32, copy=False)

                alert_levels = np.full(
                    len(chunk),
                    "Normal",
                    dtype=object,
                )

                alert_levels[total_abs_residual >= watch_thresholds] = "Watch"
                alert_levels[total_abs_residual >= warning_thresholds] = "Warning"
                alert_levels[total_abs_residual >= critical_thresholds] = "Critical"

                unique_levels, unique_counts = np.unique(alert_levels, return_counts=True)

                for level, count in zip(unique_levels, unique_counts):
                    alert_counts[str(level)] = alert_counts.get(str(level), 0) + int(count)

                result_chunk = chunk[merge_columns].copy()
                result_chunk["gmm_context_id"] = chunk["gmm_context_id"].astype(int).values
                result_chunk["total_abs_residual"] = total_abs_residual
                result_chunk["residual_anomaly_score"] = residual_scores
                result_chunk["residual_alert_level"] = alert_levels
                result_chunk["watch_threshold"] = watch_thresholds
                result_chunk["warning_threshold"] = warning_thresholds
                result_chunk["critical_threshold"] = critical_thresholds

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                print(f"[PROGRESS] Total residual anomaly rows written: {total_rows_written}")
                print(f"[PROGRESS] Running alert counts: {alert_counts}")

                del chunk
                del context_ids
                del total_abs_residual
                del watch_thresholds
                del warning_thresholds
                del critical_thresholds
                del safe_critical
                del residual_scores
                del alert_levels
                del result_chunk
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All residual anomaly chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Residual anomaly row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final CSV will not be replaced."
                )

            os.replace(temp_output_path, output_path)

            duration = perf_counter() - started

            score_metadata = {
                "status": "success",
                "output_file": str(output_path),
                "records_count": int(total_rows_written),
                "alert_counts": alert_counts,
                "threshold_file": str(self.threshold_path),
                "chunk_size": int(self.chunk_size),
                "score_input": "sum_abs_residual_raw_X_s",
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
            }

            metadata_path = Config.REPORT_DIR / "residual_anomaly_score_summary.json"
            atomic_write_json(score_metadata, metadata_path)

            print("[PROGRESS] Residual anomaly CSV written successfully")
            print(f"[PROGRESS] Alert counts: {alert_counts}")
            print(f"[PROGRESS] Score metadata written to: {metadata_path}")
            print(f"[PROGRESS] Scoring duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Scoring duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Residual anomaly scoring completed. rows=%s alerts=%s",
                total_rows_written,
                alert_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Residual anomaly scoring failed: {exc}")
            logger.exception("Residual anomaly scoring failed.")
            raise RuntimeError("Residual anomaly scoring failed.") from exc

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run residual anomaly detector.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering ResidualAnomalyDetector.run")

        try:
            thresholds = self.fit_thresholds()
            records_count = self.score()

            response = {
                "status": "success",
                "message": "Residual anomaly scores generated using dev-fitted thresholds.",
                "output_file": str(self.output_csv),
                "threshold_file": str(self.threshold_path),
                "records_count": int(records_count),
                "threshold_context_count": int(
                    len([key for key in thresholds.keys() if key != "global"])
                ),
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "target_type": "raw_X_s_only",
            }

            print(f"[PROGRESS] Residual anomaly detector response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Residual anomaly detector stage failed: {exc}")
            logger.exception("Residual anomaly detector stage failed.")
            raise RuntimeError("Residual anomaly detector stage failed.") from exc


def run_residual_anomaly_detection() -> Dict[str, object]:
    """
    Execute residual anomaly detection.
    """
    print("[PROGRESS] Entering run_residual_anomaly_detection")

    detector = ResidualAnomalyDetector()
    return detector.run()


if __name__ == "__main__":
    print("[PROGRESS] residual_anomaly_detector.py execution started")
    result = run_residual_anomaly_detection()
    print("[PROGRESS] residual_anomaly_detector.py execution finished successfully")
    print(result)