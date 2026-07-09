"""
Residual calculator for CA-EDT-AHMA.

Correct research version:
1. Residuals are calculated only for raw measured X_s sensors.
2. Additional residual temporal features are created for anomaly detection.
3. Digital Twin target remains raw X_s only.

Residual formula:
residual_Xs_* = actual_raw_Xs_* - ensemble_predicted_raw_Xs_*

Additional anomaly features:
resfeat_residual_Xs_*_rolling_mean_5
resfeat_residual_Xs_*_rolling_std_5
resfeat_residual_Xs_*_trend
resfeat_abs_residual_Xs_*_rolling_mean_5

Memory-safe version:
- Does not load full scaled/context/ensemble data into one giant DataFrame.
- Reads aligned chunks.
- Writes residual output chunk by chunk.
- Uses temporary CSV and replaces final file only after successful completion.
- Validates input row counts before processing.
- Validates output row count before replacing previous residual output.

Important:
- This file assumes scaled_features.csv, context_clusters.csv, and
  ensemble_predictions.csv are row-aligned and generated from the same ordered base data.
- Temporal rolling features assume rows are processed in stable split/unit_id/cycle order.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/digital_twin/residual_calculator.py")

from pathlib import Path
from typing import Dict, List, Tuple
from time import perf_counter
import gc

import os as _os
import sys as _sys

import numpy as np
import pandas as pd


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    print("[PROGRESS] Running residual_calculator.py as standalone script")

    _backend_root = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..")
    )

    print(f"[PROGRESS] Resolved backend root path: {_backend_root}")

    if _backend_root not in _sys.path:
        print("[PROGRESS] Backend root not found in sys.path. Adding it now.")
        _sys.path.append(_backend_root)
    else:
        print("[PROGRESS] Backend root already exists in sys.path")


from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_raw_xs_columns


logger = get_logger(__name__)

print("[PROGRESS] Imports completed successfully for residual_calculator.py")


class ResidualCalculator:
    """
    Digital twin residual calculator.

    Uses final 4-model ensemble predictions:
    - ensemble_predicted_Xs_*

    Produces:
    - residual_Xs_*
    - abs_residual_Xs_*
    - residual temporal features for anomaly models
    """

    def __init__(self, rolling_window: int | None = None, chunk_size: int | None = None) -> None:
        """
        Initialize residual calculator.

        Args:
            rolling_window: Window size for residual temporal features.
            chunk_size: CSV chunk size for memory-safe processing.
        """
        print("[PROGRESS] Entering residual_calculator.py::ResidualCalculator.__init__")

        Config.create_directories()

        self.rolling_window = int(
            rolling_window
            if rolling_window is not None
            else getattr(Config, "ROLLING_WINDOW", 5)
        )

        self.chunk_size = int(
            chunk_size
            if chunk_size is not None
            else getattr(Config, "RESIDUAL_CHUNK_SIZE", 25_000)
        )

        if self.rolling_window <= 1:
            raise ValueError("rolling_window must be greater than 1.")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        print(f"[PROGRESS] Rolling window set to: {self.rolling_window}")
        print(f"[PROGRESS] Chunk size set to: {self.chunk_size}")
        print("[PROGRESS] ResidualCalculator initialized successfully")

    # ==================================================================================
    # Header / validation helpers
    # ==================================================================================

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read CSV header only as empty DataFrame.

        Args:
            path: CSV path.

        Returns:
            Empty DataFrame with columns.
        """
        print(f"[PROGRESS] Reading header only from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read only CSV column names.

        Args:
            path: CSV path.

        Returns:
            Column list.
        """
        print(f"[PROGRESS] Reading header columns only from: {path}")

        columns = list(pd.read_csv(path, nrows=0).columns)

        print(f"[PROGRESS] Column count for {path}: {len(columns)}")
        return columns

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
        print(f"[PROGRESS] Validating required columns for {label}")

        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            print(f"[ERROR] Missing columns for {label}: {missing}")
            raise KeyError(f"Missing columns for {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _verify_key_alignment(
        self,
        base_chunk: pd.DataFrame,
        other_chunk: pd.DataFrame,
        merge_columns: List[str],
        label: str,
    ) -> None:
        """
        Verify chunk row alignment by unit_id, cycle, split.

        Args:
            base_chunk: Base chunk, usually scaled_features chunk.
            other_chunk: Other chunk to compare.
            merge_columns: Row identity columns.
            label: Label for error message.
        """
        if len(base_chunk) != len(other_chunk):
            print(
                f"[ERROR] Row count mismatch for {label}: "
                f"base={len(base_chunk)}, other={len(other_chunk)}"
            )
            raise ValueError(f"Row count mismatch for {label}")

        base_keys = base_chunk[merge_columns].reset_index(drop=True)
        other_keys = other_chunk[merge_columns].reset_index(drop=True)

        if not base_keys.equals(other_keys):
            print(f"[ERROR] Row-key alignment failed for {label}")
            raise ValueError(
                f"Files are not row-aligned for {label}. "
                "Regenerate context and ensemble outputs using the same input order."
            )

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV data rows without loading the file.

        Args:
            path: CSV path.

        Returns:
            Number of data rows, excluding header.
        """
        print(f"[PROGRESS] Counting CSV rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        if row_count < 0:
            row_count = 0

        print(f"[PROGRESS] CSV row count for {path.name}: {row_count}")
        return int(row_count)

    def _validate_input_row_counts(self) -> int:
        """
        Validate that scaled, context, and ensemble CSVs have the same number of rows.

        This prevents Python zip() from silently stopping at the shortest input file.

        Returns:
            Expected total row count.
        """
        print("[PROGRESS] Validating input row counts before residual calculation")

        scaled_rows = self._count_csv_rows(Config.SCALED_CSV)
        context_rows = self._count_csv_rows(Config.CONTEXT_CSV)
        ensemble_rows = self._count_csv_rows(Config.ENSEMBLE_PREDICTIONS_CSV)

        print("[PROGRESS] Input row-count summary")
        print(f"[PROGRESS] scaled_features rows: {scaled_rows}")
        print(f"[PROGRESS] context_clusters rows: {context_rows}")
        print(f"[PROGRESS] ensemble_predictions rows: {ensemble_rows}")

        if not (scaled_rows == context_rows == ensemble_rows):
            raise ValueError(
                "Input row counts do not match. "
                f"scaled={scaled_rows}, "
                f"context={context_rows}, "
                f"ensemble={ensemble_rows}. "
                "Regenerate context and ensemble outputs before residual calculation."
            )

        if scaled_rows <= 0:
            raise ValueError("Input CSV files contain zero data rows.")

        print("[PROGRESS] Input row counts validated successfully")
        return int(scaled_rows)

    # ==================================================================================
    # Temporal residual feature generation
    # ==================================================================================

    def _update_temporal_features_for_chunk(
        self,
        chunk_result: pd.DataFrame,
        raw_xs_columns: List[str],
        state: Dict[Tuple[object, object], Dict[str, Dict[str, object]]],
    ) -> pd.DataFrame:
        """
        Add residual temporal features to one chunk while carrying rolling state
        across chunks.

        The streaming version assumes rows are processed in split/unit_id/cycle order.
        If upstream outputs were generated from the same ordered base CSV, this remains stable.

        Args:
            chunk_result: Current residual chunk.
            raw_xs_columns: Raw measured X_s sensor columns.
            state: Rolling-state dictionary keyed by (split, unit_id).

        Returns:
            Chunk with temporal residual features.
        """
        print("[PROGRESS] Adding residual temporal features for current chunk")

        group_columns = ["split", "unit_id"]
        output = chunk_result.copy()

        for sensor in raw_xs_columns:
            residual_col = f"residual_{sensor}"
            abs_residual_col = f"abs_residual_{sensor}"

            if residual_col not in output.columns:
                raise KeyError(f"Missing residual column: {residual_col}")

            if abs_residual_col not in output.columns:
                raise KeyError(f"Missing absolute residual column: {abs_residual_col}")

            rolling_mean_col = (
                f"resfeat_{residual_col}_rolling_mean_{self.rolling_window}"
            )
            rolling_std_col = (
                f"resfeat_{residual_col}_rolling_std_{self.rolling_window}"
            )
            trend_col = f"resfeat_{residual_col}_trend"
            abs_rolling_mean_col = (
                f"resfeat_{abs_residual_col}_rolling_mean_{self.rolling_window}"
            )

            output[rolling_mean_col] = np.zeros(len(output), dtype=np.float32)
            output[rolling_std_col] = np.zeros(len(output), dtype=np.float32)
            output[trend_col] = np.zeros(len(output), dtype=np.float32)
            output[abs_rolling_mean_col] = np.zeros(len(output), dtype=np.float32)

        for group_key, group_index in output.groupby(group_columns, sort=False).groups.items():
            if group_key not in state:
                state[group_key] = {}

            for sensor in raw_xs_columns:
                residual_col = f"residual_{sensor}"
                abs_residual_col = f"abs_residual_{sensor}"

                rolling_mean_col = (
                    f"resfeat_{residual_col}_rolling_mean_{self.rolling_window}"
                )
                rolling_std_col = (
                    f"resfeat_{residual_col}_rolling_std_{self.rolling_window}"
                )
                trend_col = f"resfeat_{residual_col}_trend"
                abs_rolling_mean_col = (
                    f"resfeat_{abs_residual_col}_rolling_mean_{self.rolling_window}"
                )

                sensor_state = state[group_key].get(
                    sensor,
                    {
                        "residual_window": [],
                        "abs_window": [],
                        "last_residual": None,
                    },
                )

                current_residuals = output.loc[group_index, residual_col].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                current_abs_residuals = output.loc[group_index, abs_residual_col].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )

                previous_residual_window = np.asarray(
                    sensor_state["residual_window"],
                    dtype=np.float32,
                )
                previous_abs_window = np.asarray(
                    sensor_state["abs_window"],
                    dtype=np.float32,
                )

                combined_residuals = np.concatenate(
                    [previous_residual_window, current_residuals]
                )
                combined_abs_residuals = np.concatenate(
                    [previous_abs_window, current_abs_residuals]
                )

                rolling_mean = (
                    pd.Series(combined_residuals)
                    .rolling(window=self.rolling_window, min_periods=1)
                    .mean()
                    .to_numpy(dtype=np.float32)
                )[-len(current_residuals):]

                rolling_std = (
                    pd.Series(combined_residuals)
                    .rolling(window=self.rolling_window, min_periods=1)
                    .std()
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )[-len(current_residuals):]

                abs_rolling_mean = (
                    pd.Series(combined_abs_residuals)
                    .rolling(window=self.rolling_window, min_periods=1)
                    .mean()
                    .to_numpy(dtype=np.float32)
                )[-len(current_abs_residuals):]

                trend = np.zeros(len(current_residuals), dtype=np.float32)

                if len(current_residuals) > 0:
                    last_residual = sensor_state["last_residual"]

                    if last_residual is None:
                        trend[0] = 0.0
                    else:
                        trend[0] = current_residuals[0] - float(last_residual)

                    if len(current_residuals) > 1:
                        trend[1:] = np.diff(current_residuals)

                output.loc[group_index, rolling_mean_col] = rolling_mean
                output.loc[group_index, rolling_std_col] = rolling_std
                output.loc[group_index, trend_col] = trend
                output.loc[group_index, abs_rolling_mean_col] = abs_rolling_mean

                keep_count = max(self.rolling_window - 1, 1)

                state[group_key][sensor] = {
                    "residual_window": combined_residuals[-keep_count:].tolist(),
                    "abs_window": combined_abs_residuals[-keep_count:].tolist(),
                    "last_residual": float(current_residuals[-1])
                    if len(current_residuals) > 0
                    else sensor_state["last_residual"],
                }

        resfeat_columns = [
            column for column in output.columns if column.startswith("resfeat_")
        ]

        if resfeat_columns:
            output[resfeat_columns] = output[resfeat_columns].replace(
                [np.inf, -np.inf],
                np.nan,
            )
            output[resfeat_columns] = output[resfeat_columns].fillna(0.0)

        print(f"[PROGRESS] Residual temporal feature count in chunk: {len(resfeat_columns)}")
        return output

    # ==================================================================================
    # Main residual calculation
    # ==================================================================================

    def calculate(self) -> int:
        """
        Calculate raw X_s residuals and residual temporal features.

        Memory-safe version:
        - Reads scaled/context/ensemble CSVs in chunks.
        - Verifies row alignment.
        - Writes residual output chunk by chunk.
        - Validates row counts before and after processing.

        Returns:
            Number of residual records written.
        """
        print("[PROGRESS] Entering residual_calculator.py::ResidualCalculator.calculate")

        try:
            stage_start = perf_counter()

            merge_columns = ["unit_id", "cycle", "split"]
            context_columns = [
                "kmeans_context_id",
                "gmm_context_id",
                "context_confidence",
            ]

            print("[PROGRESS] Validating input files exist")

            for path_label, path in [
                ("scaled CSV", Config.SCALED_CSV),
                ("context CSV", Config.CONTEXT_CSV),
                ("ensemble predictions CSV", Config.ENSEMBLE_PREDICTIONS_CSV),
            ]:
                if not path.exists():
                    raise FileNotFoundError(f"{path_label} not found: {path}")

            expected_total_rows = self._validate_input_row_counts()

            print("[PROGRESS] Reading CSV headers")
            scaled_header_df = self._read_header_df(Config.SCALED_CSV)
            scaled_columns = list(scaled_header_df.columns)
            context_columns_available = self._read_header_columns(Config.CONTEXT_CSV)
            ensemble_columns_available = self._read_header_columns(
                Config.ENSEMBLE_PREDICTIONS_CSV
            )

            print("[PROGRESS] Extracting raw X_s sensors from scaled CSV header")
            raw_xs_columns = get_raw_xs_columns(scaled_header_df)

            if not raw_xs_columns:
                print("[ERROR] No raw X_s columns found")
                raise ValueError("No raw X_s columns found.")

            print(f"[PROGRESS] Raw X_s residual sensor count: {len(raw_xs_columns)}")
            print(f"[PROGRESS] Raw X_s residual sensors: {raw_xs_columns}")

            ensemble_prediction_columns = [
                f"ensemble_predicted_{sensor}" for sensor in raw_xs_columns
            ]

            self._validate_columns(
                available_columns=scaled_columns,
                required_columns=merge_columns + raw_xs_columns,
                label="scaled CSV",
            )
            self._validate_columns(
                available_columns=context_columns_available,
                required_columns=merge_columns + context_columns,
                label="context CSV",
            )
            self._validate_columns(
                available_columns=ensemble_columns_available,
                required_columns=merge_columns + ensemble_prediction_columns,
                label="ensemble predictions CSV",
            )

            scaled_usecols = merge_columns + raw_xs_columns
            context_usecols = merge_columns + context_columns
            ensemble_usecols = merge_columns + ensemble_prediction_columns

            print("[PROGRESS] Creating chunk iterators")
            print(f"[PROGRESS] Chunk size: {self.chunk_size}")

            scaled_iter = pd.read_csv(
                Config.SCALED_CSV,
                usecols=scaled_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )
            context_iter = pd.read_csv(
                Config.CONTEXT_CSV,
                usecols=context_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )
            ensemble_iter = pd.read_csv(
                Config.ENSEMBLE_PREDICTIONS_CSV,
                usecols=ensemble_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            output_path = Config.RESIDUALS_CSV
            temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

            print(f"[PROGRESS] Final residual CSV path: {output_path}")
            print(f"[PROGRESS] Temporary residual CSV path: {temp_output_path}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary residual file")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            temporal_state: Dict[
                Tuple[object, object],
                Dict[str, Dict[str, object]],
            ] = {}

            print("[PROGRESS] Starting memory-safe residual calculation")

            for scaled_chunk, context_chunk, ensemble_chunk in zip(
                scaled_iter,
                context_iter,
                ensemble_iter,
            ):
                chunk_index += 1

                print("=" * 100)
                print(f"[PROGRESS] Processing residual chunk #{chunk_index}")
                print(f"[PROGRESS] scaled_chunk shape: {scaled_chunk.shape}")
                print(f"[PROGRESS] context_chunk shape: {context_chunk.shape}")
                print(f"[PROGRESS] ensemble_chunk shape: {ensemble_chunk.shape}")

                self._verify_key_alignment(
                    base_chunk=scaled_chunk,
                    other_chunk=context_chunk,
                    merge_columns=merge_columns,
                    label="context CSV",
                )
                self._verify_key_alignment(
                    base_chunk=scaled_chunk,
                    other_chunk=ensemble_chunk,
                    merge_columns=merge_columns,
                    label="ensemble predictions CSV",
                )

                result_chunk = scaled_chunk[merge_columns].copy()

                for column in context_columns:
                    result_chunk[column] = context_chunk[column].values

                for sensor in raw_xs_columns:
                    pred_col = f"ensemble_predicted_{sensor}"
                    residual_col = f"residual_{sensor}"
                    abs_residual_col = f"abs_residual_{sensor}"

                    actual_values = scaled_chunk[sensor].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
                    predicted_values = ensemble_chunk[pred_col].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )

                    residual_values = actual_values - predicted_values

                    result_chunk[sensor] = actual_values
                    result_chunk[pred_col] = predicted_values
                    result_chunk[residual_col] = residual_values.astype(
                        np.float32,
                        copy=False,
                    )
                    result_chunk[abs_residual_col] = np.abs(residual_values).astype(
                        np.float32,
                        copy=False,
                    )

                    del actual_values
                    del predicted_values
                    del residual_values

                print(f"[PROGRESS] Raw residual chunk shape before resfeat: {result_chunk.shape}")

                result_chunk = self._update_temporal_features_for_chunk(
                    chunk_result=result_chunk,
                    raw_xs_columns=raw_xs_columns,
                    state=temporal_state,
                )

                print(f"[PROGRESS] Residual chunk shape after resfeat: {result_chunk.shape}")

                print("[PROGRESS] Writing residual chunk to temporary CSV")
                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                print(f"[PROGRESS] Total residual rows written so far: {total_rows_written}")

                del scaled_chunk
                del context_chunk
                del ensemble_chunk
                del result_chunk
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All residual chunks completed successfully")
            print(f"[PROGRESS] Total residual rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected residual rows: {expected_total_rows}")

            if total_rows_written != expected_total_rows:
                raise ValueError(
                    "Residual output row count mismatch. "
                    f"written={total_rows_written}, expected={expected_total_rows}. "
                    "Final residual CSV will not be replaced."
                )

            print("[PROGRESS] Replacing final residual CSV with completed temporary CSV")
            _os.replace(temp_output_path, output_path)
            print("[PROGRESS] Residual CSV written successfully")

            duration = perf_counter() - stage_start

            print(f"[PROGRESS] Residual records written: {total_rows_written}")
            print(f"[PROGRESS] Residual calculation duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Residual calculation duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Residual calculation completed. rows=%s raw_sensors=%s",
                total_rows_written,
                len(raw_xs_columns),
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Residual calculation failed: {exc}")
            logger.exception("Residual calculation failed.")
            raise RuntimeError("Residual calculation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run residual calculation.

        Returns:
            Stage response dictionary.
        """
        print("[PROGRESS] Entering residual_calculator.py::ResidualCalculator.run")

        try:
            records_count = self.calculate()

            response = {
                "status": "success",
                "message": (
                    "Residuals generated using raw X_s sensors only. "
                    "Residual temporal features generated for anomaly models."
                ),
                "output_file": str(Config.RESIDUALS_CSV),
                "records_count": int(records_count),
                "residual_source": "four_model_ensemble",
                "prediction_prefix": "ensemble_predicted_",
                "target_type": "raw_X_s_only",
            }

            print(f"[PROGRESS] Residual calculator response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Residual calculator stage failed: {exc}")
            logger.exception("Residual calculator stage failed.")
            raise RuntimeError("Residual calculator stage failed.") from exc


def run_residual_calculation() -> Dict[str, object]:
    """
    Execute residual calculation.

    Returns:
        Stage response dictionary.
    """
    print("[PROGRESS] Entering residual_calculator.py::run_residual_calculation")

    service = ResidualCalculator()
    return service.run()


if __name__ == "__main__":
    print("[PROGRESS] residual_calculator.py execution started from __main__")
    result = run_residual_calculation()
    print("[PROGRESS] residual_calculator.py execution finished successfully")
    print(result)