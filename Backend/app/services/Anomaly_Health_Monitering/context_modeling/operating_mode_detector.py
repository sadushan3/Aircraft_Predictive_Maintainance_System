"""
Operating mode detector for CA-EDT-AHMA.

This module combines:
1. K-Means baseline context model
2. GMM final probabilistic context model

Final output:
data/outputs/context_clusters.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/context_modeling/operating_mode_detector.py")
from typing import Dict, List

import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    print("[PROGRESS] Running operating_mode_detector.py as standalone script")
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    print(f"[PROGRESS] Resolved backend root path: {_backend_root}")

    if _backend_root not in _sys.path:
        print("[PROGRESS] Backend root not found in sys.path. Adding it now.")
        _sys.path.append(_backend_root)
    else:
        print("[PROGRESS] Backend root already exists in sys.path")

from app.config.Anomaly_Health_Monitering.config import Config
from app.services.Anomaly_Health_Monitering.context_modeling.gmm_context import GMMContextModel
from app.services.Anomaly_Health_Monitering.context_modeling.kmeans_context import KMeansContextModel
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_csv, read_csv_required
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_w_columns

logger = get_logger(__name__)

print("[PROGRESS] Imports completed successfully for operating_mode_detector.py")


class OperatingModeDetector:
    """
    Combined operating mode detector.
    """

    def __init__(self) -> None:
        """
        Initialize operating mode detector.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/operating_mode_detector.py::__init__")
        print("[PROGRESS] Creating required project directories if they do not exist")
        Config.create_directories()

        print("[PROGRESS] Creating KMeansContextModel service instance")
        self.kmeans_service = KMeansContextModel()
        print("[PROGRESS] KMeansContextModel service instance created successfully")

        print("[PROGRESS] Creating GMMContextModel service instance")
        self.gmm_service = GMMContextModel()
        print("[PROGRESS] GMMContextModel service instance created successfully")

        print("[PROGRESS] OperatingModeDetector initialized successfully")

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit context models on dev and predict for all splits.

        Args:
            df: Scaled features DataFrame.

        Returns:
            pd.DataFrame: Context cluster DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/operating_mode_detector.py::fit_predict")
        print(f"[PROGRESS] Input DataFrame shape: {df.shape}")
        print(f"[PROGRESS] Input DataFrame columns count: {len(df.columns)}")

        try:
            print("[PROGRESS] Extracting W operating-condition feature columns")
            w_columns = get_w_columns(df)
            print(f"[PROGRESS] Number of W columns found: {len(w_columns)}")
            print(f"[PROGRESS] W columns: {w_columns}")

            if not w_columns:
                print("[ERROR] No W operating-condition features found")
                raise ValueError("No W operating-condition features found.")

            print("[PROGRESS] Starting K-Means training on dev split")
            self.kmeans_service.train(df)
            print("[PROGRESS] K-Means training completed successfully")

            print("[PROGRESS] Starting GMM training on dev split")
            self.gmm_service.train(df)
            print("[PROGRESS] GMM training completed successfully")

            print("[PROGRESS] Starting K-Means prediction for all splits")
            kmeans_df = self.kmeans_service.predict(df)
            print("[PROGRESS] K-Means prediction completed successfully")
            print(f"[PROGRESS] K-Means output DataFrame shape: {kmeans_df.shape}")

            print("[PROGRESS] Starting GMM prediction for all splits")
            gmm_df = self.gmm_service.predict(df)
            print("[PROGRESS] GMM prediction completed successfully")
            print(f"[PROGRESS] GMM output DataFrame shape: {gmm_df.shape}")

            merge_columns = ["unit_id", "cycle", "split"]
            print(f"[PROGRESS] Merge columns: {merge_columns}")

            print("[PROGRESS] Checking duplicate merge keys before combining K-Means and GMM outputs")
            kmeans_duplicate_count = int(kmeans_df.duplicated(merge_columns).sum())
            gmm_duplicate_count = int(gmm_df.duplicated(merge_columns).sum())

            print(f"[PROGRESS] Duplicate unit_id/cycle/split rows in kmeans_df: {kmeans_duplicate_count}")
            print(f"[PROGRESS] Duplicate unit_id/cycle/split rows in gmm_df: {gmm_duplicate_count}")

            if kmeans_duplicate_count > 0 or gmm_duplicate_count > 0:
                print("[WARNING] unit_id + cycle + split is not unique")
                print("[WARNING] A normal merge can create a many-to-many RAM explosion")
                print("[WARNING] Using temporary occurrence index to make merge one-to-one safe")

            print("[PROGRESS] Preparing K-Means DataFrame for safe merge")
            kmeans_merge_df = kmeans_df[merge_columns + ["kmeans_context_id"]].copy()
            print(f"[PROGRESS] K-Means merge DataFrame shape before occurrence index: {kmeans_merge_df.shape}")

            print("[PROGRESS] Adding temporary merge occurrence index to K-Means output")
            kmeans_merge_df["__merge_occurrence"] = kmeans_merge_df.groupby(merge_columns).cumcount()
            print("[PROGRESS] Temporary merge occurrence index added to K-Means output")

            print("[PROGRESS] Preparing GMM DataFrame for safe merge")
            gmm_merge_df = gmm_df[
                merge_columns
                + [
                    "gmm_context_id",
                    "context_probability",
                    "context_confidence",
                ]
            ].copy()
            print(f"[PROGRESS] GMM merge DataFrame shape before occurrence index: {gmm_merge_df.shape}")

            print("[PROGRESS] Adding temporary merge occurrence index to GMM output")
            gmm_merge_df["__merge_occurrence"] = gmm_merge_df.groupby(merge_columns).cumcount()
            print("[PROGRESS] Temporary merge occurrence index added to GMM output")

            safe_merge_columns = merge_columns + ["__merge_occurrence"]
            print(f"[PROGRESS] Safe merge columns: {safe_merge_columns}")

            print("[PROGRESS] Performing safe one-to-one merge between K-Means and GMM outputs")
            context_df = kmeans_merge_df.merge(
                gmm_merge_df,
                on=safe_merge_columns,
                how="left",
                validate="one_to_one",
            )

            print("[PROGRESS] Safe merge completed successfully")
            print(f"[PROGRESS] Context DataFrame shape after safe merge: {context_df.shape}")

            print("[PROGRESS] Dropping temporary merge occurrence index from final output")
            context_df = context_df.drop(columns=["__merge_occurrence"])
            print("[PROGRESS] Temporary merge occurrence index dropped successfully")

            print("[PROGRESS] Checking for missing GMM values after merge")
            missing_gmm_count = int(context_df["gmm_context_id"].isna().sum())
            print(f"[PROGRESS] Missing GMM context rows after merge: {missing_gmm_count}")

            if missing_gmm_count > 0:
                print("[WARNING] Some K-Means rows did not receive matching GMM values")

            print("[PROGRESS] Adding W operating-condition columns to final context output")
            for column in w_columns:
                print(f"[PROGRESS] Adding W column to context output: {column}")

                if len(context_df) == len(df):
                    context_df[column] = df[column].values
                else:
                    print(
                        "[WARNING] context_df and df row counts differ. "
                        "Using index-aligned assignment fallback."
                    )
                    context_df[column] = df[column]

            print("[PROGRESS] W operating-condition columns added successfully")
            print(f"[PROGRESS] Context DataFrame shape after adding W columns: {context_df.shape}")

            print("[PROGRESS] Sorting context output by split, unit_id, and cycle")
            context_df = context_df.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)
            print("[PROGRESS] Sorting completed successfully")

            print("[PROGRESS] Final operating context output summary")
            print(f"[PROGRESS] Final context DataFrame shape: {context_df.shape}")
            print(f"[PROGRESS] Final context DataFrame columns: {list(context_df.columns)}")

            print("[PROGRESS] K-Means context ID distribution:")
            print(context_df["kmeans_context_id"].value_counts().sort_index().to_dict())

            print("[PROGRESS] GMM context ID distribution:")
            print(context_df["gmm_context_id"].value_counts().sort_index().to_dict())

            print(
                "[PROGRESS] Context probability summary: "
                f"min={float(context_df['context_probability'].min())}, "
                f"max={float(context_df['context_probability'].max())}, "
                f"mean={float(context_df['context_probability'].mean())}"
            )

            print(
                "[PROGRESS] Context confidence summary: "
                f"min={float(context_df['context_confidence'].min())}, "
                f"max={float(context_df['context_confidence'].max())}, "
                f"mean={float(context_df['context_confidence'].mean())}"
            )

            logger.info("Operating context detection completed with rows=%s.", len(context_df))
            print("[PROGRESS] Operating mode fit_predict completed successfully")

            return context_df

        except Exception as exc:
            print(f"[ERROR] Operating mode detection failed with error: {exc}")
            logger.exception("Operating mode detection failed.")
            raise RuntimeError("Operating mode detection failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run full context modeling stage.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/operating_mode_detector.py::run")
        try:
            print(f"[PROGRESS] Reading scaled CSV file from: {Config.SCALED_CSV}")
            scaled_df = read_csv_required(Config.SCALED_CSV)
            print("[PROGRESS] Scaled CSV loaded successfully")
            print(f"[PROGRESS] Scaled DataFrame shape: {scaled_df.shape}")
            print(f"[PROGRESS] Scaled DataFrame columns count: {len(scaled_df.columns)}")

            print("[PROGRESS] Starting combined operating mode detection")
            context_df = self.fit_predict(scaled_df)
            print("[PROGRESS] Combined operating mode detection completed successfully")

            print(f"[PROGRESS] Writing operating context output CSV to: {Config.CONTEXT_CSV}")
            atomic_write_csv(context_df, Config.CONTEXT_CSV)
            print("[PROGRESS] Operating context output CSV written successfully")

            response = {
                "status": "success",
                "message": "Operating context detection completed using K-Means and GMM.",
                "output_file": str(Config.CONTEXT_CSV),
                "records_count": len(context_df),
            }

            print(f"[PROGRESS] Operating mode detector stage response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] Operating mode detector stage failed with error: {exc}")
            logger.exception("Operating mode detector stage failed.")
            raise RuntimeError("Operating mode detector stage failed.") from exc


def run_operating_mode_detection() -> Dict[str, object]:
    """
    Execute operating mode detection.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/operating_mode_detector.py::run_operating_mode_detection")
    print("[PROGRESS] Creating OperatingModeDetector instance")
    detector = OperatingModeDetector()

    print("[PROGRESS] Running OperatingModeDetector")
    result = detector.run()

    print("[PROGRESS] Operating mode detection function completed")
    return result


if __name__ == "__main__":
    print("[PROGRESS] operating_mode_detector.py execution started from __main__")
    result = run_operating_mode_detection()
    print("[PROGRESS] operating_mode_detector.py execution finished successfully")
    print(result)