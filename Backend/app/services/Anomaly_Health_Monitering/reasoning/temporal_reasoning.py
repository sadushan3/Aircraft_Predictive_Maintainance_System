"""
Temporal reasoning for CA-EDT-AHMA.

Role:
Analyze how anomaly severity, health, and root-cause evidence evolve over time.

Reads:
outputs/Anomaly_Health_Monitering/health_states.csv
outputs/Anomaly_Health_Monitering/root_cause_analysis.csv

Writes:
outputs/Anomaly_Health_Monitering/temporal_reasoning.csv
reports/temporal_reasoning_summary.json

Memory-safe:
- Does not load full CSVs into RAM.
- Reads health_states.csv and root_cause_analysis.csv in aligned chunks.
- Maintains rolling temporal state per split/unit_id.
- Uses vectorized temporal pattern classification.
- Writes to temporary CSV first.
- Replaces final CSV only after successful completion.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "reasoning/temporal_reasoning.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple, Optional
import gc
import os
import sys

import numpy as np
import pandas as pd


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


class TemporalReasoning:
    """
    Memory-safe temporal anomaly reasoning engine.
    """

    def __init__(self, window: Optional[int] = None, chunk_size: int = 25_000) -> None:
        print("[PROGRESS] Entering TemporalReasoning.__init__")

        Config.create_directories()

        self.window = int(
            window
            if window is not None
            else getattr(Config, "TEMPORAL_REASONING_WINDOW", getattr(Config, "ROLLING_WINDOW", 5))
        )

        self.chunk_size = int(
            getattr(Config, "TEMPORAL_REASONING_CHUNK_SIZE", chunk_size)
        )

        if self.window <= 1:
            raise ValueError("TEMPORAL_REASONING_WINDOW must be greater than 1.")

        if self.chunk_size <= 0:
            raise ValueError("TEMPORAL_REASONING_CHUNK_SIZE must be positive.")

        self.anomaly_threshold = float(
            getattr(Config, "TEMPORAL_REASONING_ANOMALY_THRESHOLD", 0.40)
        )

        self.health_states_csv: Path = Config.HEALTH_STATES_CSV
        self.root_cause_csv: Path = Config.ROOT_CAUSE_CSV

        self.output_csv: Path = getattr(
            Config,
            "TEMPORAL_REASONING_CSV",
            Config.OUTPUT_DIR / "temporal_reasoning.csv",
        )

        self.summary_json: Path = getattr(
            Config,
            "TEMPORAL_REASONING_SUMMARY_JSON",
            Config.REPORT_DIR / "temporal_reasoning_summary.json",
        )

        print(f"[PROGRESS] Health states CSV: {self.health_states_csv}")
        print(f"[PROGRESS] Root-cause CSV: {self.root_cause_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Window: {self.window}")
        print(f"[PROGRESS] Anomaly threshold: {self.anomaly_threshold}")

    # ==================================================================================
    # Helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        print(f"[PROGRESS] Counting rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)
        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _read_header_columns(self, path: Path) -> List[str]:
        print(f"[PROGRESS] Reading header columns from: {path}")
        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        missing = [
            column
            for column in required_columns
            if column not in available_columns
        ]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _build_health_usecols(self, columns: List[str]) -> List[str]:
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
            "detector_agreement_count",
            "detector_agreement_ratio",
            "dominant_detector",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Temporal health usecols: {usecols}")
        return usecols

    def _build_root_cause_usecols(self, columns: List[str]) -> List[str]:
        required_columns = [
            "unit_id",
            "cycle",
            "split",
            "root_cause_pattern",
            "top_sensor_1",
            "top_sensor_2",
            "top_sensor_3",
            "inspection_focus",
        ]

        self._validate_columns(
            available_columns=columns,
            required_columns=required_columns,
            label="root_cause_analysis.csv",
        )

        optional_columns = [
            "contribution_1",
            "contribution_2",
            "contribution_3",
            "top3_contribution_sum",
            "total_abs_residual",
        ]

        usecols = list(required_columns)

        for column in optional_columns:
            if column in columns and column not in usecols:
                usecols.append(column)

        print(f"[PROGRESS] Temporal root-cause usecols: {usecols}")
        return usecols

    def _verify_key_alignment(
        self,
        health_chunk: pd.DataFrame,
        root_chunk: pd.DataFrame,
    ) -> None:
        merge_columns = ["unit_id", "cycle", "split"]

        if len(health_chunk) != len(root_chunk):
            raise ValueError(
                "Temporal reasoning chunk row count mismatch. "
                f"health_rows={len(health_chunk)}, root_rows={len(root_chunk)}"
            )

        health_keys = health_chunk[merge_columns].reset_index(drop=True)
        root_keys = root_chunk[merge_columns].reset_index(drop=True)

        if not health_keys.equals(root_keys):
            raise ValueError(
                "Row-key alignment failed between health_states.csv and "
                "root_cause_analysis.csv. Regenerate both from the same row order."
            )

    # ==================================================================================
    # Temporal feature calculation
    # ==================================================================================

    def _calculate_temporal_features(
        self,
        health_chunk: pd.DataFrame,
        state: Dict[Tuple[object, object], Dict[str, object]],
    ) -> pd.DataFrame:
        result = health_chunk.copy()

        result["final_anomaly_score"] = (
            result["final_anomaly_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .clip(0.0, 1.0)
        )

        result["health_index"] = (
            result["health_index"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .clip(0.0, 100.0)
        )

        result["anomaly_score_rolling_mean"] = np.zeros(len(result), dtype=np.float32)
        result["anomaly_score_delta"] = np.zeros(len(result), dtype=np.float32)
        result["health_index_delta"] = np.zeros(len(result), dtype=np.float32)
        result["anomaly_persistence_score_temporal"] = np.zeros(len(result), dtype=np.float32)

        for group_key, group_index in result.groupby(["split", "unit_id"], sort=False).groups.items():
            anomaly_values = result.loc[group_index, "final_anomaly_score"].to_numpy(
                dtype=np.float32,
                copy=False,
            )

            health_values = result.loc[group_index, "health_index"].to_numpy(
                dtype=np.float32,
                copy=False,
            )

            anomaly_flags = (anomaly_values >= self.anomaly_threshold).astype(np.float32)

            group_state = state.get(
                group_key,
                {
                    "anomaly_window": [],
                    "flag_window": [],
                    "last_anomaly": None,
                    "last_health": None,
                },
            )

            previous_anomaly_window = np.asarray(
                group_state["anomaly_window"],
                dtype=np.float32,
            )
            previous_flag_window = np.asarray(
                group_state["flag_window"],
                dtype=np.float32,
            )

            combined_anomaly = np.concatenate([previous_anomaly_window, anomaly_values])
            combined_flags = np.concatenate([previous_flag_window, anomaly_flags])

            rolling_anomaly_mean = (
                pd.Series(combined_anomaly)
                .rolling(window=self.window, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )[-len(anomaly_values):]

            rolling_persistence = (
                pd.Series(combined_flags)
                .rolling(window=self.window, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )[-len(anomaly_values):]

            anomaly_delta = np.zeros(len(anomaly_values), dtype=np.float32)
            health_delta = np.zeros(len(health_values), dtype=np.float32)

            if len(anomaly_values) > 0:
                if group_state["last_anomaly"] is None:
                    anomaly_delta[0] = 0.0
                else:
                    anomaly_delta[0] = anomaly_values[0] - float(group_state["last_anomaly"])

                if group_state["last_health"] is None:
                    health_delta[0] = 0.0
                else:
                    health_delta[0] = health_values[0] - float(group_state["last_health"])

                if len(anomaly_values) > 1:
                    anomaly_delta[1:] = np.diff(anomaly_values)

                if len(health_values) > 1:
                    health_delta[1:] = np.diff(health_values)

            result.loc[group_index, "anomaly_score_rolling_mean"] = rolling_anomaly_mean
            result.loc[group_index, "anomaly_score_delta"] = anomaly_delta
            result.loc[group_index, "health_index_delta"] = health_delta
            result.loc[group_index, "anomaly_persistence_score_temporal"] = rolling_persistence

            keep_count = max(self.window - 1, 1)

            state[group_key] = {
                "anomaly_window": combined_anomaly[-keep_count:].tolist(),
                "flag_window": combined_flags[-keep_count:].tolist(),
                "last_anomaly": float(anomaly_values[-1]) if len(anomaly_values) > 0 else group_state["last_anomaly"],
                "last_health": float(health_values[-1]) if len(health_values) > 0 else group_state["last_health"],
            }

        return result

    def _classify_temporal_pattern_array(
        self,
        anomaly_delta: np.ndarray,
        health_delta: np.ndarray,
        persistence: np.ndarray,
    ) -> np.ndarray:
        patterns = np.full(len(anomaly_delta), "Stable_Behaviour", dtype=object)

        recovering_mask = (anomaly_delta < -0.05) & (health_delta >= 0.0)
        intermittent_mask = persistence >= 0.40
        increasing_mask = (anomaly_delta > 0.05) & (health_delta < 0.0)
        persistent_mask = (persistence >= 0.80) & (health_delta < -1.0)

        patterns[recovering_mask] = "Recovering_Behaviour"
        patterns[intermittent_mask] = "Intermittent_Anomaly"
        patterns[increasing_mask] = "Increasing_Anomaly"
        patterns[persistent_mask] = "Persistent_Deterioration"

        return patterns

    def _build_reasoning_text(
        self,
        temporal_patterns: np.ndarray,
        persistence: np.ndarray,
        health_delta: np.ndarray,
        anomaly_delta: np.ndarray,
        root_patterns: np.ndarray,
    ) -> List[str]:
        texts: List[str] = []

        for pattern, pers, h_delta, a_delta, root_pattern in zip(
            temporal_patterns,
            persistence,
            health_delta,
            anomaly_delta,
            root_patterns,
        ):
            texts.append(
                (
                    f"Temporal pattern is {pattern}. Recent anomaly persistence is "
                    f"{float(pers):.2f}. Anomaly score change is {float(a_delta):.4f}. "
                    f"Health index change from previous cycle is {float(h_delta):.2f}. "
                    f"Root-cause evidence suggests {root_pattern}. "
                    f"This is reasoning support only, not a maintenance decision."
                )
            )

        return texts

    # ==================================================================================
    # Main reasoning
    # ==================================================================================

    def reason_file(self) -> int:
        print("[PROGRESS] Entering TemporalReasoning.reason_file")

        try:
            started = perf_counter()

            if not self.health_states_csv.exists():
                raise FileNotFoundError(
                    f"Health states CSV not found: {self.health_states_csv}"
                )

            if not self.root_cause_csv.exists():
                raise FileNotFoundError(
                    f"Root-cause CSV not found: {self.root_cause_csv}"
                )

            health_rows = self._count_csv_rows(self.health_states_csv)
            root_rows = self._count_csv_rows(self.root_cause_csv)

            if health_rows != root_rows:
                raise ValueError(
                    "Temporal reasoning input row counts do not match. "
                    f"health_rows={health_rows}, root_rows={root_rows}"
                )

            if health_rows <= 0:
                raise ValueError("Temporal reasoning input files contain zero rows.")

            health_columns = self._read_header_columns(self.health_states_csv)
            root_columns = self._read_header_columns(self.root_cause_csv)

            health_usecols = self._build_health_usecols(health_columns)
            root_usecols = self._build_root_cause_usecols(root_columns)

            health_iter = pd.read_csv(
                self.health_states_csv,
                usecols=health_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            root_iter = pd.read_csv(
                self.root_cause_csv,
                usecols=root_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary temporal reasoning CSV")
                temp_output_path.unlink()

            state: Dict[Tuple[object, object], Dict[str, object]] = {}

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            pattern_counts: Dict[str, int] = {}
            split_pattern_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {},
                Config.TEST_SPLIT_NAME: {},
            }

            persistence_sum = 0.0
            anomaly_delta_sum = 0.0
            health_delta_sum = 0.0

            print("[PROGRESS] Starting memory-safe temporal reasoning")

            for health_chunk, root_chunk in zip(health_iter, root_iter):
                chunk_index += 1

                health_chunk = health_chunk.reset_index(drop=True)
                root_chunk = root_chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] Temporal reasoning chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(health_chunk)}")

                self._verify_key_alignment(health_chunk, root_chunk)

                temporal_chunk = self._calculate_temporal_features(
                    health_chunk=health_chunk,
                    state=state,
                )

                anomaly_delta = temporal_chunk["anomaly_score_delta"].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                health_delta = temporal_chunk["health_index_delta"].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                persistence = temporal_chunk["anomaly_persistence_score_temporal"].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )

                temporal_patterns = self._classify_temporal_pattern_array(
                    anomaly_delta=anomaly_delta,
                    health_delta=health_delta,
                    persistence=persistence,
                )

                root_patterns = root_chunk["root_cause_pattern"].astype(str).to_numpy(dtype=object)

                temporal_text = self._build_reasoning_text(
                    temporal_patterns=temporal_patterns,
                    persistence=persistence,
                    health_delta=health_delta,
                    anomaly_delta=anomaly_delta,
                    root_patterns=root_patterns,
                )

                result_chunk = temporal_chunk[
                    [
                        "unit_id",
                        "cycle",
                        "split",
                    ]
                ].copy()

                if "gmm_context_id" in temporal_chunk.columns:
                    result_chunk["gmm_context_id"] = temporal_chunk["gmm_context_id"].values

                result_chunk["final_anomaly_score"] = temporal_chunk["final_anomaly_score"].values
                result_chunk["alert_level"] = temporal_chunk["alert_level"].values
                result_chunk["health_index"] = temporal_chunk["health_index"].values
                result_chunk["health_state"] = temporal_chunk["health_state"].values

                result_chunk["anomaly_score_rolling_mean"] = temporal_chunk[
                    "anomaly_score_rolling_mean"
                ].values
                result_chunk["anomaly_score_delta"] = anomaly_delta
                result_chunk["health_index_delta"] = health_delta
                result_chunk["anomaly_persistence_score_temporal"] = persistence

                result_chunk["temporal_pattern"] = temporal_patterns
                result_chunk["temporal_reasoning_text"] = temporal_text

                result_chunk["root_cause_pattern"] = root_chunk["root_cause_pattern"].values
                result_chunk["top_sensor_1"] = root_chunk["top_sensor_1"].values
                result_chunk["top_sensor_2"] = root_chunk["top_sensor_2"].values
                result_chunk["top_sensor_3"] = root_chunk["top_sensor_3"].values
                result_chunk["inspection_focus"] = root_chunk["inspection_focus"].values

                optional_root_columns = [
                    "contribution_1",
                    "contribution_2",
                    "contribution_3",
                    "top3_contribution_sum",
                    "total_abs_residual",
                ]

                for column in optional_root_columns:
                    if column in root_chunk.columns:
                        result_chunk[column] = root_chunk[column].values

                optional_health_columns = [
                    "remaining_health_percentage",
                    "anomaly_persistence_score",
                    "residual_trend_score",
                    "health_state_rank",
                    "detector_agreement_count",
                    "detector_agreement_ratio",
                    "dominant_detector",
                ]

                for column in optional_health_columns:
                    if column in temporal_chunk.columns:
                        result_chunk[column] = temporal_chunk[column].values

                result_chunk["maintenance_decision"] = "Not generated by this component"
                result_chunk["component_role"] = "Temporal anomaly and health reasoning support"

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_patterns, pattern_count_values = np.unique(
                    temporal_patterns,
                    return_counts=True,
                )

                for pattern, count in zip(unique_patterns, pattern_count_values):
                    pattern_counts[str(pattern)] = pattern_counts.get(str(pattern), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = result_chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_patterns = result_chunk.loc[split_mask, "temporal_pattern"].astype(str).to_numpy(dtype=object)
                    split_unique, split_counts = np.unique(split_patterns, return_counts=True)

                    for pattern, count in zip(split_unique, split_counts):
                        split_pattern_counts[split][str(pattern)] = (
                            split_pattern_counts[split].get(str(pattern), 0) + int(count)
                        )

                persistence_sum += float(np.sum(persistence, dtype=np.float64))
                anomaly_delta_sum += float(np.sum(anomaly_delta, dtype=np.float64))
                health_delta_sum += float(np.sum(health_delta, dtype=np.float64))

                print(f"[PROGRESS] Total temporal reasoning rows written: {total_rows_written}")
                print(f"[PROGRESS] Running temporal pattern counts: {pattern_counts}")

                del health_chunk
                del root_chunk
                del temporal_chunk
                del result_chunk
                del anomaly_delta
                del health_delta
                del persistence
                del temporal_patterns
                del root_patterns
                del temporal_text
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All temporal reasoning chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {health_rows}")

            if total_rows_written != health_rows:
                raise ValueError(
                    "Temporal reasoning row count mismatch. "
                    f"written={total_rows_written}, expected={health_rows}. "
                    "Final temporal_reasoning.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "window": int(self.window),
                "anomaly_threshold": float(self.anomaly_threshold),
                "temporal_pattern_counts": pattern_counts,
                "temporal_pattern_ratios": {
                    pattern: float(count / max(total_rows_written, 1))
                    for pattern, count in pattern_counts.items()
                },
                "split_pattern_counts": split_pattern_counts,
                "average_anomaly_persistence_score_temporal": float(
                    persistence_sum / max(total_rows_written, 1)
                ),
                "average_anomaly_score_delta": float(
                    anomaly_delta_sum / max(total_rows_written, 1)
                ),
                "average_health_index_delta": float(
                    health_delta_sum / max(total_rows_written, 1)
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
                    "uses_health_states": True,
                    "uses_root_cause_analysis": True,
                },
            }

            print(f"[PROGRESS] Writing temporal reasoning summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Temporal reasoning completed successfully")
            print(f"[PROGRESS] Temporal pattern counts: {pattern_counts}")
            print(f"[PROGRESS] Split pattern counts: {split_pattern_counts}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Temporal reasoning completed. rows=%s patterns=%s",
                total_rows_written,
                pattern_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Temporal reasoning failed: {exc}")
            logger.exception("Temporal reasoning failed.")
            raise RuntimeError("Temporal reasoning failed.") from exc

    def reason(self) -> int:
        """
        Production-safe reasoning method.
        """
        return self.reason_file()

    def run(self) -> Dict[str, object]:
        print("[PROGRESS] Entering TemporalReasoning.run")

        try:
            records_count = self.reason_file()

            response = {
                "status": "success",
                "message": "Temporal reasoning completed.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Temporal reasoning response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Temporal reasoning stage failed: {exc}")
            logger.exception("Temporal reasoning stage failed.")
            raise RuntimeError("Temporal reasoning stage failed.") from exc


def run_temporal_reasoning() -> Dict[str, object]:
    print("[PROGRESS] Entering run_temporal_reasoning")

    service = TemporalReasoning()
    return service.run()


if __name__ == "__main__":
    print("[PROGRESS] temporal_reasoning.py execution started")
    result = run_temporal_reasoning()
    print("[PROGRESS] temporal_reasoning.py execution finished successfully")
    print(result)