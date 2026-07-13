"""
Sensor residual ranking for CA-EDT-AHMA.

Role:
Identify top contributing measured sensors based on absolute residuals.

Formula:
sensor_contribution = abs(sensor_residual) / sum(abs(all_sensor_residuals))

Reads:
outputs/Anomaly_Health_Monitering/residuals.csv

Writes:
outputs/Anomaly_Health_Monitering/sensor_residual_ranking.csv
reports/sensor_residual_ranking_summary.json

Memory-safe:
- Does not load full residuals.csv into RAM.
- Reads residuals.csv in chunks.
- Uses vectorized NumPy top-k ranking.
- Writes to temporary CSV first.
- Replaces final CSV only after successful completion.

Important:
- Does not train a model.
- Does not predict RUL.
- Does not use Y_dev/Y_test.
- Does not make maintenance decisions.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "explainability/sensor_residual_ranking.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List
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


class SensorResidualRanking:
    """
    Memory-safe top-k sensor residual contribution ranking.
    """

    def __init__(self, top_k: int = 5, chunk_size: int = 250_000) -> None:
        """
        Initialize sensor residual ranking service.

        Args:
            top_k: Number of top contributing sensors to keep.
            chunk_size: Rows processed per chunk.
        """
        print("[PROGRESS] Entering SensorResidualRanking.__init__")

        Config.create_directories()

        self.top_k = int(getattr(Config, "SENSOR_RANKING_TOP_K", top_k))
        self.chunk_size = int(
            getattr(Config, "SENSOR_RESIDUAL_RANKING_CHUNK_SIZE", chunk_size)
        )

        if self.top_k <= 0:
            raise ValueError("SENSOR_RANKING_TOP_K must be positive.")

        if self.chunk_size <= 0:
            raise ValueError("SENSOR_RESIDUAL_RANKING_CHUNK_SIZE must be positive.")

        self.input_csv: Path = Config.RESIDUALS_CSV

        self.output_csv: Path = getattr(
            Config,
            "SENSOR_RESIDUAL_RANKING_CSV",
            Config.OUTPUT_DIR / "sensor_residual_ranking.csv",
        )

        self.summary_json: Path = getattr(
            Config,
            "SENSOR_RESIDUAL_RANKING_SUMMARY_JSON",
            Config.REPORT_DIR / "sensor_residual_ranking_summary.json",
        )

        print(f"[PROGRESS] Input CSV: {self.input_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Top K: {self.top_k}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")

    # ==================================================================================
    # Helpers
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

    def _get_raw_abs_residual_columns(self, columns: List[str]) -> List[str]:
        """
        Select raw absolute residual sensor columns only.

        Excludes engineered temporal residual features such as rolling mean/std/trend.
        """
        abs_columns: List[str] = []

        excluded_tokens = [
            "rolling",
            "trend",
            "mean",
            "std",
            "delta",
            "slope",
            "lag",
            "temporal",
        ]

        for column in columns:
            if not column.startswith("abs_residual_"):
                continue

            lower = column.lower()

            if any(token in lower for token in excluded_tokens):
                continue

            abs_columns.append(column)

        if not abs_columns:
            raise ValueError(
                "No raw abs_residual_ sensor columns found in residuals.csv."
            )

        print(f"[PROGRESS] Raw absolute residual column count: {len(abs_columns)}")
        print(f"[PROGRESS] Raw absolute residual columns: {abs_columns}")

        return abs_columns

    def _build_usecols(
        self,
        columns: List[str],
        abs_residual_columns: List[str],
    ) -> List[str]:
        """
        Build residuals.csv usecols.
        """
        required_columns = ["unit_id", "cycle", "split"]

        missing = [
            column
            for column in required_columns + abs_residual_columns
            if column not in columns
        ]

        if missing:
            raise KeyError(f"Missing required residual ranking columns: {missing}")

        usecols = list(required_columns)

        optional_columns = [
            "gmm_context_id",
            "context_confidence",
        ]

        for column in optional_columns:
            if column in columns:
                usecols.append(column)

        usecols.extend(abs_residual_columns)

        return usecols

    def _sensor_name_from_abs_column(self, column: str) -> str:
        """
        Convert abs_residual_Xs_T24 -> Xs_T24.
        """
        return str(column).replace("abs_residual_", "")

    # ==================================================================================
    # Vectorized ranking
    # ==================================================================================

    def _rank_chunk(
        self,
        chunk: pd.DataFrame,
        abs_residual_columns: List[str],
    ) -> pd.DataFrame:
        """
        Rank top-k sensor residual contributions for one chunk.
        """
        abs_matrix = (
            chunk[abs_residual_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

        abs_matrix = np.abs(abs_matrix)

        total_abs_residual = np.sum(abs_matrix, axis=1, dtype=np.float32)

        denominator = np.maximum(total_abs_residual, 1e-12).reshape(-1, 1)
        contribution_matrix = abs_matrix / denominator

        sensor_names = np.asarray(
            [self._sensor_name_from_abs_column(column) for column in abs_residual_columns],
            dtype=object,
        )

        top_count = min(self.top_k, len(abs_residual_columns))

        top_indices = np.argsort(-contribution_matrix, axis=1)[:, :top_count]

        top_values = np.take_along_axis(
            contribution_matrix,
            top_indices,
            axis=1,
        )

        top_sensors = sensor_names[top_indices]

        result = chunk[["unit_id", "cycle", "split"]].copy()

        if "gmm_context_id" in chunk.columns:
            result["gmm_context_id"] = chunk["gmm_context_id"].values

        if "context_confidence" in chunk.columns:
            result["context_confidence"] = chunk["context_confidence"].values

        result["total_abs_residual"] = total_abs_residual.astype(np.float32)

        zero_mask = total_abs_residual <= 1e-12

        for index in range(self.top_k):
            sensor_col = f"rank_{index + 1}_sensor"
            contribution_col = f"rank_{index + 1}_contribution"

            if index < top_count:
                result[sensor_col] = top_sensors[:, index]
                result[contribution_col] = top_values[:, index].astype(np.float32)
            else:
                result[sensor_col] = "none"
                result[contribution_col] = 0.0

            if np.any(zero_mask):
                result.loc[zero_mask, sensor_col] = "none"
                result.loc[zero_mask, contribution_col] = 0.0

        top_sum = np.zeros(len(result), dtype=np.float32)

        for index in range(self.top_k):
            top_sum += result[f"rank_{index + 1}_contribution"].to_numpy(
                dtype=np.float32,
                copy=False,
            )

        result["topk_contribution_sum"] = top_sum

        return result

    # ==================================================================================
    # Main
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run memory-safe sensor residual ranking.
        """
        print("[PROGRESS] Entering SensorResidualRanking.run")

        try:
            started = perf_counter()

            expected_rows = self._count_csv_rows(self.input_csv)

            if expected_rows <= 0:
                raise ValueError("residuals.csv contains zero rows.")

            columns = self._read_header_columns(self.input_csv)
            abs_residual_columns = self._get_raw_abs_residual_columns(columns)
            usecols = self._build_usecols(columns, abs_residual_columns)

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary sensor ranking CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            top_sensor_counts: Dict[str, int] = {}
            top1_contribution_sum = 0.0
            topk_contribution_sum = 0.0
            total_abs_residual_sum = 0.0

            print("[PROGRESS] Starting memory-safe sensor residual ranking")

            for chunk in pd.read_csv(
                self.input_csv,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                chunk = chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Sensor ranking chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                ranking_chunk = self._rank_chunk(
                    chunk=chunk,
                    abs_residual_columns=abs_residual_columns,
                )

                ranking_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(ranking_chunk)

                top1_values = ranking_chunk["rank_1_sensor"].astype(str).to_numpy(dtype=object)
                unique_sensors, unique_counts = np.unique(top1_values, return_counts=True)

                for sensor, count in zip(unique_sensors, unique_counts):
                    top_sensor_counts[str(sensor)] = (
                        top_sensor_counts.get(str(sensor), 0) + int(count)
                    )

                top1_contribution_sum += float(
                    ranking_chunk["rank_1_contribution"].sum()
                )
                topk_contribution_sum += float(
                    ranking_chunk["topk_contribution_sum"].sum()
                )
                total_abs_residual_sum += float(
                    ranking_chunk["total_abs_residual"].sum()
                )

                print(f"[PROGRESS] Total ranking rows written: {total_rows_written}")
                print(f"[PROGRESS] Running top-1 sensor count sample: {dict(list(top_sensor_counts.items())[:5])}")

                del chunk
                del ranking_chunk
                del top1_values
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All sensor ranking chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Sensor residual ranking row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final sensor_residual_ranking.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "message": "Sensor residual ranking generated.",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "top_k": int(self.top_k),
                "abs_residual_sensor_count": int(len(abs_residual_columns)),
                "abs_residual_columns": abs_residual_columns,
                "top_1_sensor_counts": top_sensor_counts,
                "average_rank_1_contribution": float(
                    top1_contribution_sum / max(total_rows_written, 1)
                ),
                "average_topk_contribution_sum": float(
                    topk_contribution_sum / max(total_rows_written, 1)
                ),
                "average_total_abs_residual": float(
                    total_abs_residual_sum / max(total_rows_written, 1)
                ),
                "chunk_size": int(self.chunk_size),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_residuals_only": True,
                },
            }

            print(f"[PROGRESS] Writing sensor ranking summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            response = {
                "status": "success",
                "message": "Sensor residual ranking generated.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(total_rows_written),
            }

            print(f"[PROGRESS] Sensor residual ranking response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Sensor residual ranking failed: {exc}")
            logger.exception("Sensor residual ranking failed.")
            raise RuntimeError("Sensor residual ranking failed.") from exc


def run_sensor_residual_ranking() -> Dict[str, object]:
    """
    Execute sensor residual ranking.
    """
    print("[PROGRESS] Entering run_sensor_residual_ranking")

    ranking = SensorResidualRanking()
    return ranking.run()


if __name__ == "__main__":
    print("[PROGRESS] sensor_residual_ranking.py execution started")
    result = run_sensor_residual_ranking()
    print("[PROGRESS] sensor_residual_ranking.py execution finished successfully")
    print(result)