"""
Severity classifier for CA-EDT-AHMA.

Role:
Classify final fused anomaly score into:
- Normal
- Watch
- Warning
- Critical

Reads:
outputs/Anomaly_Health_Monitering/anomaly_fusion.csv

Writes:
outputs/Anomaly_Health_Monitering/anomaly_fusion.csv

Memory-safe:
- Does not load full anomaly_fusion.csv into RAM.
- Reads in chunks.
- Writes to a temporary CSV first.
- Replaces final anomaly_fusion.csv only after successful completion.

Important:
- This file does not train any model.
- This file does not change final_anomaly_score.
- It only refreshes/enriches alert severity columns.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "anomaly_detection/severity_classifier.py"
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
        sys.path.insert(0, BACKEND_ROOT)


from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class SeverityClassifier:
    """
    Memory-safe final anomaly severity classifier.
    """

    def __init__(self, chunk_size: int = 25_000) -> None:
        """
        Initialize severity classifier.

        Args:
            chunk_size: Number of anomaly fusion rows processed per chunk.
        """
        print("[PROGRESS] Entering SeverityClassifier.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "SEVERITY_CLASSIFICATION_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("SEVERITY_CLASSIFICATION_CHUNK_SIZE must be positive.")

        self.input_csv: Path = Config.ANOMALY_FUSION_CSV
        self.output_csv: Path = Config.ANOMALY_FUSION_CSV
        self.summary_json: Path = Config.REPORT_DIR / "severity_classification_summary.json"

        self.watch_threshold = float(
            getattr(Config, "SEVERITY_WATCH_THRESHOLD", 0.40)
        )
        self.warning_threshold = float(
            getattr(Config, "SEVERITY_WARNING_THRESHOLD", 0.65)
        )
        self.critical_threshold = float(
            getattr(Config, "SEVERITY_CRITICAL_THRESHOLD", 0.85)
        )

        if not (
            0.0 <= self.watch_threshold
            <= self.warning_threshold
            <= self.critical_threshold
            <= 1.0
        ):
            raise ValueError(
                "Severity thresholds must satisfy: "
                "0 <= watch <= warning <= critical <= 1."
            )

        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Input/output CSV: {self.output_csv}")
        print(f"[PROGRESS] Watch threshold: {self.watch_threshold}")
        print(f"[PROGRESS] Warning threshold: {self.warning_threshold}")
        print(f"[PROGRESS] Critical threshold: {self.critical_threshold}")

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

    def _classify_alert_array(self, scores: np.ndarray) -> np.ndarray:
        """
        Vectorized alert classification.

        Args:
            scores: final_anomaly_score values.

        Returns:
            Alert-level string array.
        """
        alert_levels = np.full(len(scores), "Normal", dtype=object)

        alert_levels[scores >= self.watch_threshold] = "Watch"
        alert_levels[scores >= self.warning_threshold] = "Warning"
        alert_levels[scores >= self.critical_threshold] = "Critical"

        return alert_levels

    def _severity_rank_array(self, alert_levels: np.ndarray) -> np.ndarray:
        """
        Convert alert levels to numeric rank.
        """
        ranks = np.zeros(len(alert_levels), dtype=np.int8)

        ranks[alert_levels == "Watch"] = 1
        ranks[alert_levels == "Warning"] = 2
        ranks[alert_levels == "Critical"] = 3

        return ranks

    def _severity_description_array(self, alert_levels: np.ndarray) -> np.ndarray:
        """
        Human-readable severity description.
        """
        descriptions = np.full(
            len(alert_levels),
            "Normal operating behavior",
            dtype=object,
        )

        descriptions[alert_levels == "Watch"] = "Early abnormal behavior detected"
        descriptions[alert_levels == "Warning"] = "Strong anomaly pattern detected"
        descriptions[alert_levels == "Critical"] = "Critical anomaly pattern detected"

        return descriptions

    # ==================================================================================
    # Main classification
    # ==================================================================================

    def classify_file(self) -> int:
        """
        Classify severity for anomaly_fusion.csv.

        Returns:
            Number of rows written.
        """
        print("[PROGRESS] Entering SeverityClassifier.classify_file")

        try:
            started = perf_counter()

            if not self.input_csv.exists():
                raise FileNotFoundError(
                    f"Anomaly fusion CSV not found: {self.input_csv}"
                )

            expected_rows = self._count_csv_rows(self.input_csv)

            if expected_rows <= 0:
                raise ValueError("anomaly_fusion.csv contains zero rows.")

            columns = self._read_header_columns(self.input_csv)

            self._validate_columns(
                available_columns=columns,
                required_columns=[
                    "unit_id",
                    "cycle",
                    "split",
                    "final_anomaly_score",
                ],
                label="anomaly_fusion.csv",
            )

            temp_output_path = self.output_csv.with_suffix(
                self.output_csv.suffix + ".severity.tmp"
            )

            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary severity CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            alert_counts = {
                "Normal": 0,
                "Watch": 0,
                "Warning": 0,
                "Critical": 0,
            }

            split_alert_counts: Dict[str, Dict[str, int]] = {
                Config.DEV_SPLIT_NAME: {
                    "Normal": 0,
                    "Watch": 0,
                    "Warning": 0,
                    "Critical": 0,
                },
                Config.TEST_SPLIT_NAME: {
                    "Normal": 0,
                    "Watch": 0,
                    "Warning": 0,
                    "Critical": 0,
                },
            }

            score_sum = 0.0

            print("[PROGRESS] Starting chunked severity classification")

            for chunk in pd.read_csv(
                self.input_csv,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1

                print("=" * 100)
                print(f"[PROGRESS] Severity chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                scores = (
                    chunk["final_anomaly_score"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )

                scores = np.clip(scores, 0.0, 1.0)

                alert_levels = self._classify_alert_array(scores)
                severity_ranks = self._severity_rank_array(alert_levels)
                severity_descriptions = self._severity_description_array(alert_levels)

                chunk["alert_level"] = alert_levels
                chunk["severity_rank"] = severity_ranks
                chunk["severity_description"] = severity_descriptions

                chunk["severity_watch_threshold"] = float(self.watch_threshold)
                chunk["severity_warning_threshold"] = float(self.warning_threshold)
                chunk["severity_critical_threshold"] = float(self.critical_threshold)

                chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(chunk)

                unique_alerts, unique_counts = np.unique(alert_levels, return_counts=True)

                for level, count in zip(unique_alerts, unique_counts):
                    alert_counts[str(level)] = alert_counts.get(str(level), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_alerts = alert_levels[split_mask.to_numpy()]
                    split_unique, split_counts = np.unique(split_alerts, return_counts=True)

                    for level, count in zip(split_unique, split_counts):
                        split_alert_counts[split][str(level)] = (
                            split_alert_counts[split].get(str(level), 0) + int(count)
                        )

                score_sum += float(np.sum(scores, dtype=np.float64))

                print(f"[PROGRESS] Total severity rows written: {total_rows_written}")
                print(f"[PROGRESS] Running alert counts: {alert_counts}")

                del chunk
                del scores
                del alert_levels
                del severity_ranks
                del severity_descriptions
                gc.collect()

            print("=" * 100)
            print("[PROGRESS] All severity chunks completed")
            print(f"[PROGRESS] Rows written: {total_rows_written}")
            print(f"[PROGRESS] Expected rows: {expected_rows}")

            if total_rows_written != expected_rows:
                raise ValueError(
                    "Severity classification row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final anomaly_fusion.csv will not be replaced."
                )

            os.replace(temp_output_path, self.output_csv)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "output_file": str(self.output_csv),
                "records_count": int(total_rows_written),
                "alert_counts": alert_counts,
                "split_alert_counts": split_alert_counts,
                "final_anomaly_score_mean": float(
                    score_sum / max(total_rows_written, 1)
                ),
                "thresholds": {
                    "watch": float(self.watch_threshold),
                    "warning": float(self.warning_threshold),
                    "critical": float(self.critical_threshold),
                },
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_refit_anomaly_detectors": True,
                    "does_not_use_y_targets": True,
                    "does_not_use_t_degradation_as_input": True,
                },
            }

            print(f"[PROGRESS] Writing severity summary JSON to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] Severity classification completed successfully")
            print(f"[PROGRESS] Alert counts: {alert_counts}")
            print(f"[PROGRESS] Split alert counts: {split_alert_counts}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info(
                "Severity classification completed. rows=%s alerts=%s",
                total_rows_written,
                alert_counts,
            )

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] Severity classification failed: {exc}")
            logger.exception("Severity classification failed.")
            raise RuntimeError("Severity classification failed.") from exc

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        In-memory helper for small DataFrames only.

        Kept for compatibility with older code, but the recommended production path
        is classify_file().
        """
        print("[PROGRESS] Entering SeverityClassifier.classify")

        try:
            result = df.copy()

            if "final_anomaly_score" not in result.columns:
                raise KeyError(
                    "final_anomaly_score is required for severity classification."
                )

            scores = (
                result["final_anomaly_score"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )

            scores = np.clip(scores, 0.0, 1.0)

            alert_levels = self._classify_alert_array(scores)

            result["alert_level"] = alert_levels
            result["severity_rank"] = self._severity_rank_array(alert_levels)
            result["severity_description"] = self._severity_description_array(alert_levels)

            return result

        except Exception as exc:
            logger.exception("Severity classification failed.")
            raise RuntimeError("Severity classification failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run severity classification stage.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering SeverityClassifier.run")

        try:
            records_count = self.classify_file()

            response = {
                "status": "success",
                "message": "Severity classification completed.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
            }

            print(f"[PROGRESS] Severity classifier response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Severity classifier stage failed: {exc}")
            logger.exception("Severity classifier stage failed.")
            raise RuntimeError("Severity classifier stage failed.") from exc


def run_severity_classification() -> Dict[str, object]:
    """
    Execute severity classification.
    """
    print("[PROGRESS] Entering run_severity_classification")

    classifier = SeverityClassifier()
    return classifier.run()


if __name__ == "__main__":
    print("[PROGRESS] severity_classifier.py execution started")
    result = run_severity_classification()
    print("[PROGRESS] severity_classifier.py execution finished successfully")
    print(result)