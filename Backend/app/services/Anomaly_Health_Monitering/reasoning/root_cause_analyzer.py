"""
Root-cause analyzer for CA-EDT-AHMA.

Role:
Rank top contributing measured sensors using residual contribution.

Formula:
sensor_contribution = abs(sensor_residual) / sum(abs(all_sensor_residuals))

Output:
top_sensor_1
top_sensor_2
top_sensor_3
contribution_1
contribution_2
contribution_3
root_cause_pattern
inspection_focus

Important:
- This module identifies likely contributing sensor patterns.
- It does not make final maintenance decisions.
- It does not predict RUL.
- It does not use Y_dev/Y_test.
- It does not use T_dev/T_test.

Reads:
outputs/Anomaly_Health_Monitering/residuals.csv
outputs/Anomaly_Health_Monitering/health_states.csv

Writes:
outputs/Anomaly_Health_Monitering/root_cause_analysis.csv
reports/root_cause_summary.json

Memory-safe:
- Does not load full CSV files into RAM.
- Reads residuals.csv and health_states.csv in aligned chunks.
- Validates row-key alignment using unit_id/cycle/split.
- Uses vectorized top-3 residual contribution ranking.
- Writes to temporary CSV first.
- Replaces final root_cause_analysis.csv only after successful completion.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "reasoning/root_cause_analyzer.py"
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
from app.services.Anomaly_Health_Monitering.reasoning.rule_engine import ReasoningRuleEngine
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class RootCauseAnalyzer:
    """
    Memory-safe residual-based root-cause analyzer.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize root-cause analyzer.

        Args:
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering RootCauseAnalyzer.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "ROOT_CAUSE_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("ROOT_CAUSE_CHUNK_SIZE must be positive.")

        self.residual_csv: Path = Config.RESIDUALS_CSV
        self.health_states_csv: Path = Config.HEALTH_STATES_CSV
        self.output_csv: Path = Config.ROOT_CAUSE_CSV

        self.summary_json: Path = getattr(
            Config,
            "ROOT_CAUSE_SUMMARY_JSON",
            Config.REPORT_DIR / "root_cause_summary.json",
        )

        self.rule_engine = ReasoningRuleEngine()
        self.pattern_cache: Dict[Tuple[str, str, str], Tuple[str, str]] = {}

        print(f"[PROGRESS] Residual CSV: {self.residual_csv}")
        print(f"[PROGRESS] Health states CSV: {self.health_states_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")

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

    def _get_abs_residual_columns(self, columns: List[str]) -> List[str]:
        """
        Select absolute residual columns.

        Expected pattern:
        abs_residual_Xs_*
        """
        abs_residual_columns = [
            column
            for column in columns
            if column.startswith("abs_residual_")
        ]

        if not abs_residual_columns:
            raise ValueError(
                "No absolute residual columns found. "
                "Expected columns like abs_residual_Xs_T24."
            )

        print(f"[PROGRESS] Absolute residual column count: {len(abs_residual_columns)}")
        print(f"[PROGRESS] Absolute residual columns: {abs_residual_columns}")

        return abs_residual_columns

    def _sensor_name_from_abs_column(self, column: str) -> str:
        """
        Convert abs_residual_Xs_T24 -> Xs_T24.
        """
        return column.replace("abs_residual_", "")

    def _build_residual_usecols(
        self,
        columns: List[str],
        abs_residual_columns: List[str],
    ) -> List[str]:
        """
        Build residuals.csv usecols.
        """
        required_columns = ["unit_id", "cycle", "split"]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns + abs_residual_columns,
            label="residuals.csv",
        )

        optional_columns = [
            "gmm_context_id",
            "total_abs_residual",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        usecols.extend(abs_residual_columns)

        print(f"[PROGRESS] Root-cause residual usecols count: {len(usecols)}")
        return usecols

    def _build_health_usecols(self, columns: List[str]) -> List[str]:
        """
        Build health_states.csv usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "final_anomaly_score",
            "alert_level",
            "health_index",
            "health_state",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="health_states.csv",
        )

        optional_columns = [
            "gmm_context_id",
            "remaining_health_percentage",
            "anomaly_persistence_score",
            "residual_trend_score",
            "health_state_rank",
            "health_state_explanation",
            "detector_agreement_count",
            "detector_agreement_ratio",
            "dominant_detector",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Root-cause health usecols: {usecols}")
        return usecols

    def _verify_key_alignment(
        self,
        residual_chunk: pd.DataFrame,
        health_chunk: pd.DataFrame,
    ) -> None:
        """
        Verify row alignment using unit_id, cycle, split.
        """
        merge_columns = ["unit_id", "cycle", "split"]

        if len(residual_chunk) != len(health_chunk):
            raise ValueError(
                "Root-cause chunk row count mismatch. "
                f"residual_rows={len(residual_chunk)}, health_rows={len(health_chunk)}"
            )

        residual_keys = residual_chunk[merge_columns].reset_index(drop=True)
        health_keys = health_chunk[merge_columns].reset_index(drop=True)

        if not residual_keys.equals(health_keys):
            raise ValueError(
                "Row-key alignment failed between residuals.csv and health_states.csv. "
                "Regenerate health states from the same anomaly/residual row order."
            )

    # ==================================================================================
    # Vectorized contribution logic
    # ==================================================================================

    def _calculate_top3_contributions(
        self,
        residual_chunk: pd.DataFrame,
        abs_residual_columns: List[str],
    ) -> pd.DataFrame:
        """
        Calculate top-3 sensor contributions for a chunk.
        """
        abs_matrix = (
            residual_chunk[abs_residual_columns]
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

        top_count = min(3, len(abs_residual_columns))

        top_indices = np.argsort(-contribution_matrix, axis=1)[:, :top_count]

        top_values = np.take_along_axis(
            contribution_matrix,
            top_indices,
            axis=1,
        )

        top_sensor_values = sensor_names[top_indices]

        result = pd.DataFrame(
            {
                "total_abs_residual": total_abs_residual.astype(np.float32),
                "top_sensor_1": top_sensor_values[:, 0],
                "contribution_1": top_values[:, 0].astype(np.float32),
            }
        )

        if top_count >= 2:
            result["top_sensor_2"] = top_sensor_values[:, 1]
            result["contribution_2"] = top_values[:, 1].astype(np.float32)
        else:
            result["top_sensor_2"] = "none"
            result["contribution_2"] = 0.0

        if top_count >= 3:
            result["top_sensor_3"] = top_sensor_values[:, 2]
            result["contribution_3"] = top_values[:, 2].astype(np.float32)
        else:
            result["top_sensor_3"] = "none"
            result["contribution_3"] = 0.0

        zero_mask = total_abs_residual <= 1e-12

        if np.any(zero_mask):
            result.loc[zero_mask, "top_sensor_1"] = "none"
            result.loc[zero_mask, "top_sensor_2"] = "none"
            result.loc[zero_mask, "top_sensor_3"] = "none"
            result.loc[zero_mask, "contribution_1"] = 0.0
            result.loc[zero_mask, "contribution_2"] = 0.0
            result.loc[zero_mask, "contribution_3"] = 0.0

        result["top3_contribution_sum"] = (
            result["contribution_1"].astype(np.float32)
            + result["contribution_2"].astype(np.float32)
            + result["contribution_3"].astype(np.float32)
        )

        return result

    def _infer_patterns_for_chunk(self, top_df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer root-cause pattern and inspection focus using rule engine.

        Uses cache to avoid repeated rule-engine calls for repeated top sensor triples.
        """
        root_patterns: List[str] = []
        inspection_focuses: List[str] = []

        for sensors in zip(
            top_df["top_sensor_1"].astype(str),
            top_df["top_sensor_2"].astype(str),
            top_df["top_sensor_3"].astype(str),
        ):
            if sensors not in self.pattern_cache:
                sensor_list = list(sensors)

                root_pattern = self.rule_engine.infer_root_cause_pattern(sensor_list)
                inspection_focus = self.rule_engine.recommend_inspection_focus(root_pattern)

                self.pattern_cache[sensors] = (root_pattern, inspection_focus)

            cached_pattern, cached_focus = self.pattern_cache[sensors]

            root_patterns.append(cached_pattern)
            inspection_focuses.append(cached_focus)

        top_df["root_cause_pattern"] = root_patterns
        top_df["inspection_focus"] = inspection_focuses

        return top_df

    # ==================================================================================
    # Main analysis
    # ==================================================================================

    def analyze_file(self) -> int:
        """
        Run memory-safe root-cause analysis.

        Returns:
            Number of rows written.
        """
        print("[PROGRESS] Entering RootCauseAnalyzer.analyze_file")

        try:
            started = perf_counter()

            if not self.residual_csv.exists():
                raise FileNotFoundError(f"Residual CSV not found: {self.residual_csv}")

            if not self.health_states_csv.exists():
                raise FileNotFoundError(
                    f"Health states CSV not found: {self.health_states_csv}"
                )

            residual_rows = self._count_csv_rows(self.residual_csv)
            health_rows = self._count_csv_rows(self.health_states_csv)

            if residual_rows != health_rows:
                raise ValueError(
                    "Root-cause input row counts do not match. "
                    f"residual_rows={residual_rows}, health_rows={health_rows}"
                )

            if residual_rows <= 0:
                raise ValueError("Root-cause input files contain zero rows.")

            residual_columns = self._read_header_columns(self.residual_csv)
            health_columns = self._read_header_columns(self.health_states_csv)

            abs_residual_columns = self._get_abs_residual_columns(residual_columns)

            residual_usecols = self._build_residual_usecols(
                columns=residual_columns,
                abs_residual_columns=abs_residual_columns,
            )

            health_usecols = self._build_health_usecols(health_columns)

            residual_iter = pd.read_csv(
                self.residual_csv,
                usecols=residual_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            health_iter = pd.read_csv(
                self.health_states_csv,
                usecols=health_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary root-cause CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            pattern_counts: Dict[str, int] = {}
            focus_counts: Dict[str, int] = {}
            top_sensor_counts: Dict[str, int] = {}

            split_pattern_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {},
                Config.TEST_SPLIT_NAME: {},
            }

            contribution_1_sum = 0.0
            contribution_2_sum = 0.0
            contribution_3_sum = 0.0
            top3_sum_total = 0.0
            total_abs_residual_sum = 0.0

            print("[PROGRESS] Starting memory-safe root-cause analysis")

            for residual_chunk, health_chunk in zip(residual_iter, health_iter):
                chunk_index += 1

                residual_chunk = residual_chunk.reset_index(drop=True)
                health_chunk = health_chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Root-cause chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(residual_chunk)}")

                self._verify_key_alignment(
                    residual_chunk=residual_chunk,
                    health_chunk=health_chunk,
                )

                top_df = self._calculate_top3_contributions(
                    residual_chunk=residual_chunk,
                    abs_residual_columns=abs_residual_columns,
                )

                top_df = self._infer_patterns_for_chunk(top_df)

                result_chunk = residual_chunk[
                    [
                        "unit_id",
                        "cycle",
                        "split",
                    ]
                ].copy()

                if "gmm_context_id" in health_chunk.columns:
                    result_chunk["gmm_context_id"] = health_chunk["gmm_context_id"].values
                elif "gmm_context_id" in residual_chunk.columns:
                    result_chunk["gmm_context_id"] = residual_chunk["gmm_context_id"].values

                health_copy_columns = [
                    "final_anomaly_score",
                    "alert_level",
                    "health_index",
                    "health_state",
                    "remaining_health_percentage",
                    "anomaly_persistence_score",
                    "residual_trend_score",
                    "health_state_rank",
                    "health_state_explanation",
                    "detector_agreement_count",
                    "detector_agreement_ratio",
                    "dominant_detector",
                ]

                for column in health_copy_columns:
                    if column in health_chunk.columns:
                        result_chunk[column] = health_chunk[column].values

                result_chunk["total_abs_residual"] = top_df["total_abs_residual"].values

                result_chunk["top_sensor_1"] = top_df["top_sensor_1"].values
                result_chunk["top_sensor_2"] = top_df["top_sensor_2"].values
                result_chunk["top_sensor_3"] = top_df["top_sensor_3"].values

                result_chunk["contribution_1"] = top_df["contribution_1"].round(6).values
                result_chunk["contribution_2"] = top_df["contribution_2"].round(6).values
                result_chunk["contribution_3"] = top_df["contribution_3"].round(6).values
                result_chunk["top3_contribution_sum"] = (
                    top_df["top3_contribution_sum"].round(6).values
                )

                result_chunk["root_cause_pattern"] = top_df["root_cause_pattern"].values
                result_chunk["inspection_focus"] = top_df["inspection_focus"].values

                result_chunk["maintenance_decision"] = "Not generated by this component"
                result_chunk["component_role"] = "Root-cause sensor attribution and inspection focus"

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_patterns, pattern_count_values = np.unique(
                    result_chunk["root_cause_pattern"].astype(str).to_numpy(dtype=object),
                    return_counts=True,
                )

                for pattern, count in zip(unique_patterns, pattern_count_values):
                    pattern_counts[str(pattern)] = pattern_counts.get(str(pattern), 0) + int(count)

                unique_focuses, focus_count_values = np.unique(
                    result_chunk["inspection_focus"].astype(str).to_numpy(dtype=object),
                    return_counts=True,
                )

                for focus, count in zip(unique_focuses, focus_count_values):
                    focus_counts[str(focus)] = focus_counts.get(str(focus), 0) + int(count)

                unique_sensors, sensor_count_values = np.unique(
                    result_chunk["top_sensor_1"].astype(str).to_numpy(dtype=object),
                    return_counts=True,
                )

                for sensor, count in zip(unique_sensors, sensor_count_values):
                    top_sensor_counts[str(sensor)] = top_sensor_counts.get(str(sensor), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_patterns = (
                        result_chunk.loc[split_mask, "root_cause_pattern"]
                        .astype(str)
                        .to_numpy(dtype=object)
                    )

                    split_unique, split_counts = np.unique(
                        split_patterns,
                        return_counts=True,
                    )

                    for pattern, count in zip(split_unique, split_counts):
                        split_pattern_counts[split][str(pattern)] = (
                            split_pattern_counts[split].get(str(pattern), 0) + int(count)
                        )

                contribution_1_sum += float(result_chunk["contribution_1"].sum())
                contribution_2_sum += float(result_chunk["contribution_2"].sum())
                contribution_3_sum += float(result_chunk["contribution_3"].sum())
                top3_sum_total += float(result_chunk["top3_contribution_sum"].sum())
                total_abs_residual_sum += float(result_chunk["total_abs_residual"].sum())

                print(f"[PROGRESS] Total root-cause rows written: {total_rows_written}")
                print(f"[PROGRESS] Running top sensor count sample: {dict(list(top_sensor_counts.items())[:5])}")

                del residual_chunk
                del health_chunk
                del top_df
                del result_chunk
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All root-cause chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {residual_rows}")

            if total_rows_written != residual_rows:
                raise ValueError(
                    "Root-cause row count mismatch. "
                    f"written={total_rows_written}, expected={residual_rows}. "
                    "Final root_cause_analysis.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "abs_residual_sensor_count": int(len(abs_residual_columns)),
                "abs_residual_columns": abs_residual_columns,
                "average_contribution_1": float(contribution_1_sum / max(total_rows_written, 1)),
                "average_contribution_2": float(contribution_2_sum / max(total_rows_written, 1)),
                "average_contribution_3": float(contribution_3_sum / max(total_rows_written, 1)),
                "average_top3_contribution_sum": float(top3_sum_total / max(total_rows_written, 1)),
                "average_total_abs_residual": float(total_abs_residual_sum / max(total_rows_written, 1)),
                "pattern_counts": pattern_counts,
                "focus_counts": focus_counts,
                "top_sensor_1_counts": top_sensor_counts,
                "split_pattern_counts": split_pattern_counts,
                "cached_pattern_count": int(len(self.pattern_cache)),
                "chunk_size": int(self.chunk_size),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_residual_contributions": True,
                    "uses_health_states_for_context": True,
                },
            }

            print(f"[PROGRESS] Writing root-cause summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Root-cause analysis completed successfully")
            print(f"[PROGRESS] Pattern counts: {pattern_counts}")
            print(f"[PROGRESS] Focus counts: {focus_counts}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Root-cause analysis completed. rows=%s patterns=%s",
                total_rows_written,
                pattern_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Root-cause analysis failed: {exc}")
            logger.exception("Root-cause analysis failed.")
            raise RuntimeError("Root-cause analysis failed.") from exc

    # ==================================================================================
    # Compatibility helper
    # ==================================================================================

    def analyze(self) -> int:
        """
        Production-safe analyze method.

        Returns:
            Number of rows written.
        """
        return self.analyze_file()

    def run(self) -> Dict[str, object]:
        """
        Run root-cause analyzer.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering RootCauseAnalyzer.run")

        try:
            records_count = self.analyze_file()

            response = {
                "status": "success",
                "message": "Root-cause sensor attribution completed.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Root-cause analyzer response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Root-cause analyzer stage failed: {exc}")
            logger.exception("Root-cause analyzer stage failed.")
            raise RuntimeError("Root-cause analyzer stage failed.") from exc


def run_root_cause_analysis() -> Dict[str, object]:
    """
    Execute root-cause analysis.
    """
    print("[PROGRESS] Entering run_root_cause_analysis")

    analyzer = RootCauseAnalyzer()
    return analyzer.run()


if __name__ == "__main__":
    print("[PROGRESS] root_cause_analyzer.py execution started")
    result = run_root_cause_analysis()
    print("[PROGRESS] root_cause_analyzer.py execution finished successfully")
    print(result)