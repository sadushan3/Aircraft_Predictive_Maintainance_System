"""
Subsystem explainer for CA-EDT-AHMA.

Role:
Map top contributing sensors and residual patterns to broad subsystem-level
explanations for dashboard and human-readable reports.

Important:
- Subsystem labels are explanation support, not confirmed physical causality.
- This module does not make maintenance decisions.
- This module does not predict RUL.
- This module does not use Y_dev/Y_test.
- This module does not use T_dev/T_test.

Reads:
outputs/Anomaly_Health_Monitering/root_cause_analysis.csv

Writes:
outputs/Anomaly_Health_Monitering/subsystem_explanations.csv
reports/subsystem_explanations_summary.json

Memory-safe:
- Does not load full root_cause_analysis.csv into RAM.
- Reads root_cause_analysis.csv in chunks.
- Uses chunked subsystem mapping.
- Writes to temporary CSV first.
- Replaces final CSV only after successful completion.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "explainability/subsystem_explainer.py"
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


class SubsystemExplainer:
    """
    Memory-safe subsystem-level explanation generator.
    """

    def __init__(self, chunk_size: int = 250_000) -> None:
        """
        Initialize subsystem explainer.

        Args:
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering SubsystemExplainer.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "SUBSYSTEM_EXPLAINER_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("SUBSYSTEM_EXPLAINER_CHUNK_SIZE must be positive.")

        self.input_csv: Path = Config.ROOT_CAUSE_CSV

        self.output_csv: Path = getattr(
            Config,
            "SUBSYSTEM_EXPLANATIONS_CSV",
            Config.OUTPUT_DIR / "subsystem_explanations.csv",
        )

        self.summary_json: Path = getattr(
            Config,
            "SUBSYSTEM_EXPLANATIONS_SUMMARY_JSON",
            Config.REPORT_DIR / "subsystem_explanations_summary.json",
        )

        self.write_text: bool = bool(
            getattr(Config, "SUBSYSTEM_EXPLAINER_WRITE_TEXT", True)
        )

        print(f"[PROGRESS] Input CSV: {self.input_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Write text: {self.write_text}")

    # ==================================================================================
    # File helpers
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

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        return list(pd.read_csv(path, nrows=0).columns)

    def _build_usecols(self, columns: List[str]) -> List[str]:
        """
        Build root_cause_analysis.csv usecols.
        """
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "top_sensor_1",
            "top_sensor_2",
            "top_sensor_3",
            "root_cause_pattern",
            "inspection_focus",
        ]

        missing = [column for column in required_columns if column not in columns]

        if missing:
            raise KeyError(f"Missing required subsystem explanation columns: {missing}")

        optional_columns = [
            "gmm_context_id",
            "alert_level",
            "final_anomaly_score",
            "health_index",
            "health_state",
            "contribution_1",
            "contribution_2",
            "contribution_3",
            "top3_contribution_sum",
            "total_abs_residual",
            "dominant_detector",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Subsystem explainer usecols: {usecols}")
        return usecols

    # ==================================================================================
    # Sensor-to-subsystem mapping
    # ==================================================================================

    def sensor_to_subsystem(self, sensor_name: str) -> str:
        """
        Map one sensor name to broad subsystem category.

        Covers measured N-CMAPSS sensors used in this component:
        - Temperature: T24, T30, T48, T50
        - Pressure: P2, P15, P21, P24, Ps30, P40, P50
        - Rotational: Nf, Nc
        - Fuel/flow: Wf

        Args:
            sensor_name: Sensor name.

        Returns:
            Broad subsystem label.
        """
        name = (
            str(sensor_name)
            .lower()
            .replace("abs_residual_", "")
            .replace("residual_", "")
            .replace("xs_", "")
            .replace("x_s_", "")
            .replace("xv_", "")
            .replace("x_v_", "")
            .strip()
        )

        if name in {"none", "nan", "", "unknown"}:
            return "unknown_subsystem"

        # Hot-section / turbine temperature.
        if any(token in name for token in ["t48", "t50"]):
            return "hot_section_thermal_subsystem"

        # Compressor / inlet temperature.
        if any(token in name for token in ["t24", "t30", "t2", "temp", "temperature"]):
            return "compressor_thermal_subsystem"

        # Core / turbine pressure.
        if any(token in name for token in ["ps30", "p40", "p50"]):
            return "core_pressure_subsystem"

        # Fan / LPC / compressor pressure.
        if any(token in name for token in ["p2", "p15", "p21", "p24", "p30", "press", "pressure"]):
            return "fan_lpc_pressure_subsystem"

        # Fuel / flow.
        if any(token in name for token in ["wf", "fuel", "flow"]):
            return "fuel_flow_subsystem"

        # Fan rotational speed.
        if "nf" in name:
            return "fan_rotational_subsystem"

        # Core rotational speed.
        if "nc" in name:
            return "core_rotational_subsystem"

        if any(token in name for token in ["speed", "shaft", "n1", "n2"]):
            return "rotational_subsystem"

        if any(token in name for token in ["eff", "efficiency", "epr", "bpr"]):
            return "efficiency_subsystem"

        return "general_sensor_subsystem"

    def _map_series_to_subsystem(self, series: pd.Series) -> pd.Series:
        """
        Map a sensor-name Series to subsystem labels.

        This is chunked, so using map() here is safe.
        """
        return series.astype(str).map(self.sensor_to_subsystem)

    # ==================================================================================
    # Explanation logic
    # ==================================================================================

    def _explain_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Generate subsystem explanations for one chunk.
        """
        result = chunk.copy()

        result["subsystem_1"] = self._map_series_to_subsystem(result["top_sensor_1"])
        result["subsystem_2"] = self._map_series_to_subsystem(result["top_sensor_2"])
        result["subsystem_3"] = self._map_series_to_subsystem(result["top_sensor_3"])

        result["primary_subsystem"] = result["subsystem_1"]

        if self.write_text:
            result["subsystem_explanation"] = (
                "Leading residual contribution is associated with "
                + result["primary_subsystem"].astype(str)
                + ". Supporting contributors are "
                + result["subsystem_2"].astype(str)
                + " and "
                + result["subsystem_3"].astype(str)
                + ". Pattern label: "
                + result["root_cause_pattern"].astype(str)
                + ". Inspection focus: "
                + result["inspection_focus"].astype(str)
                + " This is explanation support only, not a maintenance decision."
            )

        result["hard_causal_claim"] = False
        result["maintenance_decision"] = "Not generated by this component"
        result["component_role"] = "Subsystem-level explanation support"

        return result

    # ==================================================================================
    # Main run
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run memory-safe subsystem explanation generation.
        """
        print("[PROGRESS] Entering SubsystemExplainer.run")

        try:
            started = perf_counter()

            expected_rows = self._count_csv_rows(self.input_csv)

            if expected_rows <= 0:
                raise ValueError("root_cause_analysis.csv contains zero rows.")

            columns = self._read_header_columns(self.input_csv)
            usecols = self._build_usecols(columns)

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary subsystem explanation CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            subsystem_counts: Dict[str, int] = {}
            pattern_counts: Dict[str, int] = {}

            print("[PROGRESS] Starting memory-safe subsystem explanation generation")

            for chunk in pd.read_csv(
                self.input_csv,
                usecols=usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                chunk = chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Subsystem explanation chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                explanation_chunk = self._explain_chunk(chunk)

                output_columns = [
                    "unit_id",
                    "cycle",
                    "split",
                ]

                if "gmm_context_id" in explanation_chunk.columns:
                    output_columns.append("gmm_context_id")

                passthrough_columns = [
                    "alert_level",
                    "final_anomaly_score",
                    "health_index",
                    "health_state",
                    "root_cause_pattern",
                    "inspection_focus",
                    "top_sensor_1",
                    "top_sensor_2",
                    "top_sensor_3",
                    "contribution_1",
                    "contribution_2",
                    "contribution_3",
                    "top3_contribution_sum",
                    "total_abs_residual",
                    "dominant_detector",
                ]

                for column in passthrough_columns:
                    if column in explanation_chunk.columns and column not in output_columns:
                        output_columns.append(column)

                output_columns.extend(
                    [
                        "subsystem_1",
                        "subsystem_2",
                        "subsystem_3",
                        "primary_subsystem",
                    ]
                )

                if self.write_text and "subsystem_explanation" in explanation_chunk.columns:
                    output_columns.append("subsystem_explanation")

                output_columns.extend(
                    [
                        "hard_causal_claim",
                        "maintenance_decision",
                        "component_role",
                    ]
                )

                result_chunk = explanation_chunk[output_columns]

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_subsystems, subsystem_count_values = np.unique(
                    result_chunk["primary_subsystem"].astype(str).to_numpy(dtype=object),
                    return_counts=True,
                )

                for subsystem, count in zip(unique_subsystems, subsystem_count_values):
                    subsystem_counts[str(subsystem)] = (
                        subsystem_counts.get(str(subsystem), 0) + int(count)
                    )

                unique_patterns, pattern_count_values = np.unique(
                    result_chunk["root_cause_pattern"].astype(str).to_numpy(dtype=object),
                    return_counts=True,
                )

                for pattern, count in zip(unique_patterns, pattern_count_values):
                    pattern_counts[str(pattern)] = (
                        pattern_counts.get(str(pattern), 0) + int(count)
                    )

                print(f"[PROGRESS] Total subsystem explanation rows written: {total_rows_written}")
                print(f"[PROGRESS] Running subsystem counts: {subsystem_counts}")

                del chunk
                del explanation_chunk
                del result_chunk
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All subsystem explanation chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Subsystem explanation row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final subsystem_explanations.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "message": "Subsystem explanations generated.",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "primary_subsystem_counts": subsystem_counts,
                "root_cause_pattern_counts": pattern_counts,
                "write_text": bool(self.write_text),
                "chunk_size": int(self.chunk_size),
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_root_cause_analysis_only": True,
                },
            }

            print(f"[PROGRESS] Writing subsystem explanation summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            response = {
                "status": "success",
                "message": "Subsystem explanations generated.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(total_rows_written),
            }

            print(f"[PROGRESS] Subsystem explainer response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Subsystem explainer failed: {exc}")
            logger.exception("Subsystem explainer failed.")
            raise RuntimeError("Subsystem explainer failed.") from exc


def run_subsystem_explainer() -> Dict[str, object]:
    """
    Execute subsystem explanation generation.
    """
    print("[PROGRESS] Entering run_subsystem_explainer")

    explainer = SubsystemExplainer()
    return explainer.run()


if __name__ == "__main__":
    print("[PROGRESS] subsystem_explainer.py execution started")
    result = run_subsystem_explainer()
    print("[PROGRESS] subsystem_explainer.py execution finished successfully")
    print(result)