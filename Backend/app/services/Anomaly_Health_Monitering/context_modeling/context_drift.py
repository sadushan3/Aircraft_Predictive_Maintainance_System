"""
Context drift detection for CA-EDT-AHMA.

This module compares GMM likelihood behavior between dev and test contexts.

It does not refit on test.
The GMM is trained only on dev and then used to score dev/test context likelihood.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/context_modeling/context_drift.py")
import gc
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    print("[PROGRESS] Running context_drift.py as standalone script")
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    print(f"[PROGRESS] Resolved backend root path: {_backend_root}")

    if _backend_root not in _sys.path:
        print("[PROGRESS] Backend root not found in sys.path. Adding it now.")
        _sys.path.append(_backend_root)
    else:
        print("[PROGRESS] Backend root already exists in sys.path")

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    load_joblib_required,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)

print("[PROGRESS] Imports completed successfully for context_drift.py")


class ContextDriftDetector:
    """
    Context drift detector using dev-fitted GMM likelihood.
    """

    def __init__(self) -> None:
        """
        Initialize context drift detector.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/context_drift.py::__init__")
        print("[PROGRESS] Creating required project directories if they do not exist")
        Config.create_directories()
        print("[PROGRESS] ContextDriftDetector initialized successfully")

    def score_likelihood(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score likelihood under the dev-fitted GMM.

        This version uses batch scoring to avoid RAM explosion for very large datasets.

        Args:
            df: Scaled features DataFrame.

        Returns:
            pd.DataFrame: Drift scoring DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/context_drift.py::score_likelihood")
        print(f"[PROGRESS] Input DataFrame received with shape: {df.shape}")
        print(f"[PROGRESS] Input DataFrame columns count: {len(df.columns)}")

        try:
            print(f"[PROGRESS] Loading dev-fitted GMM model from: {Config.GMM_MODEL_PATH}")
            payload = load_joblib_required(Config.GMM_MODEL_PATH)
            print("[PROGRESS] GMM model payload loaded successfully")

            model = payload["model"]
            feature_columns: List[str] = payload["feature_columns"]

            print("[PROGRESS] Extracted GMM model from payload")
            print(f"[PROGRESS] Number of GMM feature columns expected: {len(feature_columns)}")
            print(f"[PROGRESS] First 10 GMM feature columns: {feature_columns[:10]}")

            print("[PROGRESS] Checking whether all required GMM feature columns exist in input DataFrame")
            missing = [column for column in feature_columns if column not in df.columns]
            if missing:
                print(f"[ERROR] Missing GMM feature columns: {missing}")
                raise KeyError(f"Missing GMM feature columns for drift detection: {missing}")

            print("[PROGRESS] All required GMM feature columns are available")

            total_rows = len(df)
            batch_size = 100_000

            print(f"[PROGRESS] Total rows for GMM likelihood scoring: {total_rows}")
            print(f"[PROGRESS] GMM likelihood scoring batch size: {batch_size}")
            print("[PROGRESS] Starting memory-safe batch GMM log-likelihood scoring")

            likelihood = np.empty(total_rows, dtype=np.float64)

            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)

                print(
                    "[PROGRESS] Scoring GMM likelihood batch: "
                    f"start={start_idx}, end={end_idx}, rows={end_idx - start_idx}"
                )

                batch_x = df.iloc[start_idx:end_idx][feature_columns]
                likelihood[start_idx:end_idx] = model.score_samples(batch_x)
                del batch_x

                print(
                    "[PROGRESS] Completed likelihood batch: "
                    f"start={start_idx}, end={end_idx}"
                )

            gc.collect()

            print("[PROGRESS] Memory-safe batch GMM log-likelihood scoring completed")
            print(f"[PROGRESS] Likelihood array length: {len(likelihood)}")
            print(
                "[PROGRESS] Likelihood summary: "
                f"min={float(np.min(likelihood))}, "
                f"max={float(np.max(likelihood))}, "
                f"mean={float(np.mean(likelihood))}"
            )

            print("[PROGRESS] Creating context drift result DataFrame with unit_id, cycle, and split")
            result = df[["unit_id", "cycle", "split"]].copy()
            result["gmm_log_likelihood"] = likelihood
            print(f"[PROGRESS] Result DataFrame created with shape: {result.shape}")

            print(f"[PROGRESS] Identifying dev split rows using split name: {Config.DEV_SPLIT_NAME}")
            dev_mask = result["split"] == Config.DEV_SPLIT_NAME
            dev_likelihood = result.loc[dev_mask, "gmm_log_likelihood"]

            print(f"[PROGRESS] Dev likelihood rows count: {len(dev_likelihood)}")
            print(f"[PROGRESS] Non-dev likelihood rows count: {len(result) - len(dev_likelihood)}")

            if dev_likelihood.empty:
                print("[ERROR] No dev likelihood values found. Cannot calculate drift baseline.")
                raise ValueError("No dev likelihood values available for drift baseline.")

            print("[PROGRESS] Calculating dev likelihood percentile thresholds")
            lower_threshold = float(np.percentile(dev_likelihood, 1.0))
            upper_threshold = float(np.percentile(dev_likelihood, 99.0))

            print(f"[PROGRESS] Dev likelihood lower threshold 1%: {lower_threshold}")
            print(f"[PROGRESS] Dev likelihood upper threshold 99%: {upper_threshold}")

            print("[PROGRESS] Calculating context drift scores")
            result["context_drift_score"] = np.where(
                result["gmm_log_likelihood"] < lower_threshold,
                np.minimum(
                    1.0,
                    np.abs(result["gmm_log_likelihood"] - lower_threshold)
                    / max(abs(lower_threshold), 1e-9),
                ),
                0.0,
            )
            print("[PROGRESS] Context drift scores calculated successfully")

            print("[PROGRESS] Assigning context drift labels")
            result["context_drift_label"] = np.where(
                result["gmm_log_likelihood"] < lower_threshold,
                "Potential_Context_Drift",
                "No_Context_Drift",
            )
            print("[PROGRESS] Context drift labels assigned successfully")

            print("[PROGRESS] Adding dev likelihood threshold columns to result DataFrame")
            result["dev_likelihood_lower_threshold"] = lower_threshold
            result["dev_likelihood_upper_threshold"] = upper_threshold

            drift_count = int((result["context_drift_label"] == "Potential_Context_Drift").sum())
            no_drift_count = int((result["context_drift_label"] == "No_Context_Drift").sum())

            print(f"[PROGRESS] Potential_Context_Drift count: {drift_count}")
            print(f"[PROGRESS] No_Context_Drift count: {no_drift_count}")
            print(f"[PROGRESS] Final context drift result shape: {result.shape}")

            logger.info("Context drift scoring completed with rows=%s.", len(result))
            print("[PROGRESS] Context drift scoring completed successfully")

            return result

        except Exception as exc:
            print(f"[ERROR] Context drift scoring failed with error: {exc}")
            logger.exception("Context drift scoring failed.")
            raise RuntimeError("Context drift scoring failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run context drift detection.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/context_drift.py::run")
        try:
            print(f"[PROGRESS] Reading scaled CSV file from: {Config.SCALED_CSV}")
            scaled_df = read_csv_required(Config.SCALED_CSV)
            print("[PROGRESS] Scaled CSV loaded successfully")
            print(f"[PROGRESS] Scaled DataFrame shape: {scaled_df.shape}")

            print("[PROGRESS] Starting likelihood scoring and drift detection")
            drift_df = self.score_likelihood(scaled_df)
            print("[PROGRESS] Likelihood scoring and drift detection completed")

            output_path: Path = Config.OUTPUT_DIR / "context_drift.csv"
            print(f"[PROGRESS] Writing context drift output CSV to: {output_path}")
            atomic_write_csv(drift_df, output_path)
            print("[PROGRESS] Context drift output CSV written successfully")

            response = {
                "status": "success",
                "message": "Context drift detection completed using dev-fitted GMM.",
                "output_file": str(output_path),
                "records_count": len(drift_df),
            }

            print(f"[PROGRESS] Context drift stage response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Context drift stage failed with error: {exc}")
            logger.exception("Context drift stage failed.")
            raise RuntimeError("Context drift stage failed.") from exc


def run_context_drift_detection() -> Dict[str, object]:
    """
    Execute context drift detection.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/context_drift.py::run_context_drift_detection")
    print("[PROGRESS] Creating ContextDriftDetector instance")
    detector = ContextDriftDetector()

    print("[PROGRESS] Running ContextDriftDetector")
    result = detector.run()

    print("[PROGRESS] Context drift detection function completed")
    return result


if __name__ == "__main__":
    print("[PROGRESS] context_drift.py execution started from __main__")
    result = run_context_drift_detection()
    print("[PROGRESS] context_drift.py execution finished successfully")
    print(result)
