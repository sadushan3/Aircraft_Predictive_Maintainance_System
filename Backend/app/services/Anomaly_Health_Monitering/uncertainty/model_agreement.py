"""
Model agreement calculator for CA-EDT-AHMA.

Role:
Estimate uncertainty from disagreement among:
- Random Forest Digital Twin
- XGBoost Digital Twin
- LightGBM Digital Twin

Formula:
For each measured sensor:
    sensor_disagreement = std(RF prediction, XGBoost prediction, LightGBM prediction)

Then:
    model_disagreement = mean(sensor_disagreement across sensors)

Normalization:
    Fit normalization threshold using dev split only:
        dev_threshold = percentile(model_disagreement on dev split)

    normalized_model_disagreement =
        model_disagreement / dev_threshold

    model_agreement_score =
        1 - normalized_model_disagreement

Important:
- This module does not train a model.
- This module does not use Y_dev/Y_test.
- This module does not use T_dev/T_test.
- This module does not predict RUL.
- This module does not make maintenance decisions.
- Normalization is fitted using dev split only to avoid test leakage.

Reads:
outputs/Anomaly_Health_Monitering/rf_predictions.csv
outputs/Anomaly_Health_Monitering/xgb_predictions.csv
outputs/Anomaly_Health_Monitering/lgbm_predictions.csv

Writes:
outputs/Anomaly_Health_Monitering/model_agreement.csv
reports/model_agreement_summary.json
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "uncertainty/model_agreement.py"
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


class ModelAgreementCalculator:
    """
    Memory-safe model agreement calculator.

    This implementation never performs full DataFrame merges.
    Prediction files are processed in aligned chunks.
    """

    def __init__(self, chunk_size: int = 250_000) -> None:
        """
        Initialize model agreement calculator.

        Args:
            chunk_size: Number of rows processed per chunk.
        """
        print("[PROGRESS] Entering ModelAgreementCalculator.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "MODEL_AGREEMENT_CHUNK_SIZE", chunk_size)
        )

        if self.chunk_size <= 0:
            raise ValueError("MODEL_AGREEMENT_CHUNK_SIZE must be positive.")

        self.normalization_percentile = float(
            getattr(Config, "MODEL_AGREEMENT_NORMALIZATION_PERCENTILE", 99.0)
        )

        if not (0.0 < self.normalization_percentile <= 100.0):
            raise ValueError(
                "MODEL_AGREEMENT_NORMALIZATION_PERCENTILE must be in (0, 100]."
            )

        self.rf_csv: Path = Config.RF_PREDICTIONS_CSV
        self.xgb_csv: Path = Config.XGB_PREDICTIONS_CSV
        self.lgbm_csv: Path = Config.LGBM_PREDICTIONS_CSV

        self.output_csv: Path = Config.MODEL_AGREEMENT_CSV

        self.summary_json: Path = getattr(
            Config,
            "MODEL_AGREEMENT_SUMMARY_JSON",
            Config.REPORT_DIR / "model_agreement_summary.json",
        )

        print(f"[PROGRESS] RF predictions CSV: {self.rf_csv}")
        print(f"[PROGRESS] XGB predictions CSV: {self.xgb_csv}")
        print(f"[PROGRESS] LGBM predictions CSV: {self.lgbm_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Normalization percentile: {self.normalization_percentile}")

    # ==================================================================================
    # File helpers
    # ==================================================================================

    def _count_csv_rows(self, path: Path) -> int:
        """
        Count CSV data rows without loading the file.
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

    def _infer_sensors_from_rf_columns(self, rf_columns: List[str]) -> List[str]:
        """
        Infer measured target sensor names from RF prediction columns.

        Args:
            rf_columns: RF prediction CSV column names.

        Returns:
            List of sensor names without rf_predicted_ prefix.
        """
        sensors = [
            column.replace("rf_predicted_", "")
            for column in rf_columns
            if column.startswith("rf_predicted_")
        ]

        sensors = sorted(sensors)

        if not sensors:
            raise ValueError("No rf_predicted_ columns found in RF predictions CSV.")

        print(f"[PROGRESS] Inferred target sensor count: {len(sensors)}")
        print(f"[PROGRESS] Inferred target sensors: {sensors}")

        return sensors

    def _build_usecols(
        self,
        rf_columns: List[str],
        xgb_columns: List[str],
        lgbm_columns: List[str],
        sensors: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Build usecols for RF, XGB, and LGBM prediction files.
        """
        key_columns = ["unit_id", "cycle", "split"]

        for label, columns in [
            ("rf_predictions.csv", rf_columns),
            ("xgb_predictions.csv", xgb_columns),
            ("lgbm_predictions.csv", lgbm_columns),
        ]:
            missing_keys = [column for column in key_columns if column not in columns]

            if missing_keys:
                raise KeyError(f"Missing key columns in {label}: {missing_keys}")

        rf_prediction_columns = [f"rf_predicted_{sensor}" for sensor in sensors]
        xgb_prediction_columns = [f"xgb_predicted_{sensor}" for sensor in sensors]
        lgbm_prediction_columns = [f"lgbm_predicted_{sensor}" for sensor in sensors]

        missing_rf = [
            column for column in rf_prediction_columns if column not in rf_columns
        ]
        missing_xgb = [
            column for column in xgb_prediction_columns if column not in xgb_columns
        ]
        missing_lgbm = [
            column for column in lgbm_prediction_columns if column not in lgbm_columns
        ]

        if missing_rf:
            raise KeyError(f"Missing RF prediction columns: {missing_rf}")

        if missing_xgb:
            raise KeyError(f"Missing XGB prediction columns: {missing_xgb}")

        if missing_lgbm:
            raise KeyError(f"Missing LGBM prediction columns: {missing_lgbm}")

        rf_usecols = key_columns + rf_prediction_columns
        xgb_usecols = key_columns + xgb_prediction_columns
        lgbm_usecols = key_columns + lgbm_prediction_columns

        print(f"[PROGRESS] RF usecols count: {len(rf_usecols)}")
        print(f"[PROGRESS] XGB usecols count: {len(xgb_usecols)}")
        print(f"[PROGRESS] LGBM usecols count: {len(lgbm_usecols)}")

        return rf_usecols, xgb_usecols, lgbm_usecols

    def _verify_key_alignment(
        self,
        rf_chunk: pd.DataFrame,
        xgb_chunk: pd.DataFrame,
        lgbm_chunk: pd.DataFrame,
    ) -> None:
        """
        Verify that prediction chunks are row-aligned.
        """
        key_columns = ["unit_id", "cycle", "split"]

        if len(rf_chunk) != len(xgb_chunk) or len(rf_chunk) != len(lgbm_chunk):
            raise ValueError(
                "Prediction chunk row mismatch: "
                f"rf={len(rf_chunk)}, xgb={len(xgb_chunk)}, lgbm={len(lgbm_chunk)}"
            )

        rf_keys = rf_chunk[key_columns].reset_index(drop=True)
        xgb_keys = xgb_chunk[key_columns].reset_index(drop=True)
        lgbm_keys = lgbm_chunk[key_columns].reset_index(drop=True)

        if not rf_keys.equals(xgb_keys):
            raise ValueError(
                "Row-key alignment failed between RF and XGB predictions. "
                "Regenerate prediction CSVs using the same ordered input."
            )

        if not rf_keys.equals(lgbm_keys):
            raise ValueError(
                "Row-key alignment failed between RF and LGBM predictions. "
                "Regenerate prediction CSVs using the same ordered input."
            )

    # ==================================================================================
    # Disagreement calculation
    # ==================================================================================

    def _compute_model_disagreement(
        self,
        rf_chunk: pd.DataFrame,
        xgb_chunk: pd.DataFrame,
        lgbm_chunk: pd.DataFrame,
        sensors: List[str],
    ) -> np.ndarray:
        """
        Compute average model disagreement across all target sensors for one chunk.

        Args:
            rf_chunk: RF prediction chunk.
            xgb_chunk: XGBoost prediction chunk.
            lgbm_chunk: LightGBM prediction chunk.
            sensors: Target sensor names.

        Returns:
            Float32 array of model_disagreement values.
        """
        row_count = len(rf_chunk)
        model_disagreement = np.zeros(row_count, dtype=np.float32)

        for sensor in sensors:
            rf_col = f"rf_predicted_{sensor}"
            xgb_col = f"xgb_predicted_{sensor}"
            lgbm_col = f"lgbm_predicted_{sensor}"

            prediction_stack = np.column_stack(
                [
                    rf_chunk[rf_col].to_numpy(dtype=np.float32, copy=False),
                    xgb_chunk[xgb_col].to_numpy(dtype=np.float32, copy=False),
                    lgbm_chunk[lgbm_col].to_numpy(dtype=np.float32, copy=False),
                ]
            )

            prediction_stack = np.nan_to_num(
                prediction_stack,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            sensor_std = np.std(prediction_stack, axis=1).astype(np.float32)
            model_disagreement += sensor_std

            del prediction_stack
            del sensor_std

        model_disagreement = model_disagreement / float(len(sensors))

        return model_disagreement.astype(np.float32)

    # ==================================================================================
    # Pass 1: dev-only normalization
    # ==================================================================================

    def _fit_dev_normalization_threshold(
        self,
        rf_usecols: List[str],
        xgb_usecols: List[str],
        lgbm_usecols: List[str],
        sensors: List[str],
    ) -> Dict[str, object]:
        """
        Fit disagreement normalization threshold using dev split only.
        """
        print("[PROGRESS] Entering _fit_dev_normalization_threshold")
        print("[PROGRESS] Starting pass 1: dev-only disagreement collection")

        dev_disagreement_batches: List[np.ndarray] = []
        total_rows_seen = 0
        dev_rows_seen = 0
        chunk_index = 0

        rf_iter = pd.read_csv(
            self.rf_csv,
            usecols=rf_usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        )

        xgb_iter = pd.read_csv(
            self.xgb_csv,
            usecols=xgb_usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        )

        lgbm_iter = pd.read_csv(
            self.lgbm_csv,
            usecols=lgbm_usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        )

        for rf_chunk, xgb_chunk, lgbm_chunk in zip(rf_iter, xgb_iter, lgbm_iter):
            chunk_index += 1

            rf_chunk = rf_chunk.reset_index(drop=True)
            xgb_chunk = xgb_chunk.reset_index(drop=True)
            lgbm_chunk = lgbm_chunk.reset_index(drop=True)

            print("=" * 100)
            print(f"[PROGRESS] Agreement pass 1 chunk #{chunk_index}")
            print(f"[PROGRESS] Chunk rows: {len(rf_chunk)}")

            self._verify_key_alignment(rf_chunk, xgb_chunk, lgbm_chunk)

            disagreement = self._compute_model_disagreement(
                rf_chunk=rf_chunk,
                xgb_chunk=xgb_chunk,
                lgbm_chunk=lgbm_chunk,
                sensors=sensors,
            )

            split_values = rf_chunk["split"].astype(str).to_numpy(dtype=object)
            dev_mask = split_values == Config.DEV_SPLIT_NAME

            if np.any(dev_mask):
                dev_values = disagreement[dev_mask].astype(np.float32, copy=True)
                dev_disagreement_batches.append(dev_values)
                dev_rows_seen += int(len(dev_values))

            total_rows_seen += int(len(rf_chunk))

            print(f"[PROGRESS] Total rows seen: {total_rows_seen}")
            print(f"[PROGRESS] Dev rows collected: {dev_rows_seen}")

            del rf_chunk
            del xgb_chunk
            del lgbm_chunk
            del disagreement
            del split_values
            del dev_mask
            gc.collect()

        if not dev_disagreement_batches:
            raise ValueError(
                "No dev split rows found. Cannot fit model agreement normalization."
            )

        dev_disagreement = np.concatenate(dev_disagreement_batches).astype(np.float32)

        dev_threshold = float(
            np.percentile(dev_disagreement, self.normalization_percentile)
        )

        if not np.isfinite(dev_threshold) or dev_threshold <= 1e-12:
            dev_threshold = float(np.max(dev_disagreement))

        if not np.isfinite(dev_threshold) or dev_threshold <= 1e-12:
            dev_threshold = 1.0

        summary = {
            "fit_split": Config.DEV_SPLIT_NAME,
            "normalization_method": "dev_only_percentile",
            "normalization_percentile": float(self.normalization_percentile),
            "normalization_threshold": float(dev_threshold),
            "dev_rows_seen": int(dev_rows_seen),
            "total_rows_seen": int(total_rows_seen),
            "dev_disagreement_min": float(np.min(dev_disagreement)),
            "dev_disagreement_max": float(np.max(dev_disagreement)),
            "dev_disagreement_mean": float(np.mean(dev_disagreement)),
            "dev_disagreement_std": float(np.std(dev_disagreement)),
        }

        print(f"[PROGRESS] Dev-only normalization summary: {summary}")

        del dev_disagreement_batches
        del dev_disagreement
        gc.collect()

        return summary

    # ==================================================================================
    # Pass 2: scoring and writing
    # ==================================================================================

    def _write_agreement_scores(
        self,
        rf_usecols: List[str],
        xgb_usecols: List[str],
        lgbm_usecols: List[str],
        sensors: List[str],
        normalization_summary: Dict[str, object],
        expected_rows: int,
    ) -> Dict[str, object]:
        """
        Compute model agreement scores and write final CSV safely.
        """
        print("[PROGRESS] Entering _write_agreement_scores")
        print("[PROGRESS] Starting pass 2: model agreement scoring")

        normalization_threshold = float(
            normalization_summary["normalization_threshold"]
        )

        temp_output_path = self.output_csv.with_suffix(
            self.output_csv.suffix + ".tmp"
        )

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        if temp_output_path.exists():
            print("[PROGRESS] Removing old temporary model agreement CSV")
            temp_output_path.unlink()

        rf_iter = pd.read_csv(
            self.rf_csv,
            usecols=rf_usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        )

        xgb_iter = pd.read_csv(
            self.xgb_csv,
            usecols=xgb_usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        )

        lgbm_iter = pd.read_csv(
            self.lgbm_csv,
            usecols=lgbm_usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        )

        first_batch = True
        total_rows_written = 0
        chunk_index = 0

        split_rows: Dict[str, int] = {}
        agreement_sum_by_split: Dict[str, float] = {}
        disagreement_sum_by_split: Dict[str, float] = {}

        min_disagreement = float("inf")
        max_disagreement = float("-inf")
        agreement_sum_total = 0.0
        disagreement_sum_total = 0.0

        for rf_chunk, xgb_chunk, lgbm_chunk in zip(rf_iter, xgb_iter, lgbm_iter):
            chunk_index += 1

            rf_chunk = rf_chunk.reset_index(drop=True)
            xgb_chunk = xgb_chunk.reset_index(drop=True)
            lgbm_chunk = lgbm_chunk.reset_index(drop=True)

            print("=" * 100)
            print(f"[PROGRESS] Agreement pass 2 chunk #{chunk_index}")
            print(f"[PROGRESS] Chunk rows: {len(rf_chunk)}")

            self._verify_key_alignment(rf_chunk, xgb_chunk, lgbm_chunk)

            model_disagreement = self._compute_model_disagreement(
                rf_chunk=rf_chunk,
                xgb_chunk=xgb_chunk,
                lgbm_chunk=lgbm_chunk,
                sensors=sensors,
            )

            normalized_disagreement = np.clip(
                model_disagreement / max(normalization_threshold, 1e-12),
                0.0,
                1.0,
            ).astype(np.float32)

            model_agreement_score = np.clip(
                1.0 - normalized_disagreement,
                0.0,
                1.0,
            ).astype(np.float32)

            result_chunk = rf_chunk[["unit_id", "cycle", "split"]].copy()
            result_chunk["model_disagreement"] = model_disagreement
            result_chunk["normalized_model_disagreement"] = normalized_disagreement
            result_chunk["model_agreement_score"] = model_agreement_score
            result_chunk["uncertainty_from_model_disagreement"] = normalized_disagreement
            result_chunk["agreement_normalization_threshold"] = normalization_threshold
            result_chunk["agreement_normalization_percentile"] = self.normalization_percentile
            result_chunk["agreement_fit_split"] = Config.DEV_SPLIT_NAME
            result_chunk["agreement_sensor_count"] = int(len(sensors))

            result_chunk.to_csv(
                temp_output_path,
                mode="w" if first_batch else "a",
                header=first_batch,
                index=False,
            )

            first_batch = False
            total_rows_written += int(len(result_chunk))

            min_disagreement = min(min_disagreement, float(np.min(model_disagreement)))
            max_disagreement = max(max_disagreement, float(np.max(model_disagreement)))
            agreement_sum_total += float(np.sum(model_agreement_score, dtype=np.float64))
            disagreement_sum_total += float(np.sum(model_disagreement, dtype=np.float64))

            for split in result_chunk["split"].astype(str).unique():
                split_mask = result_chunk["split"].astype(str) == split
                split_count = int(split_mask.sum())

                split_rows[split] = split_rows.get(split, 0) + split_count
                agreement_sum_by_split[split] = agreement_sum_by_split.get(split, 0.0) + float(
                    result_chunk.loc[split_mask, "model_agreement_score"].sum()
                )
                disagreement_sum_by_split[split] = disagreement_sum_by_split.get(split, 0.0) + float(
                    result_chunk.loc[split_mask, "model_disagreement"].sum()
                )

            print(f"[PROGRESS] Total model agreement rows written: {total_rows_written}")

            del rf_chunk
            del xgb_chunk
            del lgbm_chunk
            del model_disagreement
            del normalized_disagreement
            del model_agreement_score
            del result_chunk
            gc.collect()

        print("=" * 100)
        print("[PROGRESS] All model agreement chunks completed")
        print(f"[PROGRESS] Rows written: {total_rows_written}")
        print(f"[PROGRESS] Expected rows: {expected_rows}")

        if total_rows_written != expected_rows:
            raise ValueError(
                "Model agreement row count mismatch. "
                f"written={total_rows_written}, expected={expected_rows}. "
                "Final model_agreement.csv will not be replaced."
            )

        os.replace(temp_output_path, self.output_csv)

        split_summary: Dict[str, Dict[str, float]] = {}

        for split, count in split_rows.items():
            split_summary[split] = {
                "rows": int(count),
                "average_model_agreement_score": float(
                    agreement_sum_by_split.get(split, 0.0) / max(count, 1)
                ),
                "average_model_disagreement": float(
                    disagreement_sum_by_split.get(split, 0.0) / max(count, 1)
                ),
            }

        score_summary = {
            "records_count": int(total_rows_written),
            "model_disagreement_min": float(min_disagreement),
            "model_disagreement_max": float(max_disagreement),
            "average_model_disagreement": float(
                disagreement_sum_total / max(total_rows_written, 1)
            ),
            "average_model_agreement_score": float(
                agreement_sum_total / max(total_rows_written, 1)
            ),
            "split_summary": split_summary,
        }

        return score_summary

    # ==================================================================================
    # Main
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run memory-safe model agreement calculation.
        """
        print("[PROGRESS] Entering ModelAgreementCalculator.run")

        try:
            started = perf_counter()

            rf_rows = self._count_csv_rows(self.rf_csv)
            xgb_rows = self._count_csv_rows(self.xgb_csv)
            lgbm_rows = self._count_csv_rows(self.lgbm_csv)

            if rf_rows <= 0:
                raise ValueError("rf_predictions.csv contains zero rows.")

            if rf_rows != xgb_rows or rf_rows != lgbm_rows:
                raise ValueError(
                    "Prediction CSV row-count mismatch: "
                    f"rf={rf_rows}, xgb={xgb_rows}, lgbm={lgbm_rows}"
                )

            rf_columns = self._read_header_columns(self.rf_csv)
            xgb_columns = self._read_header_columns(self.xgb_csv)
            lgbm_columns = self._read_header_columns(self.lgbm_csv)

            sensors = self._infer_sensors_from_rf_columns(rf_columns)

            rf_usecols, xgb_usecols, lgbm_usecols = self._build_usecols(
                rf_columns=rf_columns,
                xgb_columns=xgb_columns,
                lgbm_columns=lgbm_columns,
                sensors=sensors,
            )

            normalization_summary = self._fit_dev_normalization_threshold(
                rf_usecols=rf_usecols,
                xgb_usecols=xgb_usecols,
                lgbm_usecols=lgbm_usecols,
                sensors=sensors,
            )

            score_summary = self._write_agreement_scores(
                rf_usecols=rf_usecols,
                xgb_usecols=xgb_usecols,
                lgbm_usecols=lgbm_usecols,
                sensors=sensors,
                normalization_summary=normalization_summary,
                expected_rows=rf_rows,
            )

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "message": "Model agreement calculated from RF, XGBoost, and LightGBM predictions.",
                "output_file": str(self.output_csv),
                "records_count": int(score_summary["records_count"]),
                "sensor_count": int(len(sensors)),
                "sensors": sensors,
                "chunk_size": int(self.chunk_size),
                "normalization": normalization_summary,
                "score_summary": score_summary,
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "normalization_fit_split": Config.DEV_SPLIT_NAME,
                    "test_split_used_for_threshold_fit": False,
                },
            }

            print(f"[PROGRESS] Writing model agreement summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            response = {
                "status": "success",
                "message": "Model agreement calculated from RF, XGBoost, and LightGBM predictions.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(score_summary["records_count"]),
            }

            print(f"[PROGRESS] Model agreement response: {response}")

            logger.info(
                "Model agreement calculation completed. rows=%s",
                score_summary["records_count"],
            )

            return response

        except Exception as exc:
            print(f"[ERROR] Model agreement stage failed: {exc}")
            logger.exception("Model agreement stage failed.")
            raise RuntimeError("Model agreement stage failed.") from exc


def run_model_agreement() -> Dict[str, object]:
    """
    Execute model agreement calculation.
    """
    print("[PROGRESS] Entering run_model_agreement")

    calculator = ModelAgreementCalculator()
    return calculator.run()


if __name__ == "__main__":
    print("[PROGRESS] model_agreement.py execution started")
    result = run_model_agreement()
    print("[PROGRESS] model_agreement.py execution finished successfully")
    print(result)