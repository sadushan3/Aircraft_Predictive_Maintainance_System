"""
K-Means operating-context model for CA-EDT-AHMA.

Role:
Baseline operating-context clustering model.

Training:
Input = W_dev
Target = None

Inference:
Input = W_dev and W_test
Output = kmeans_context_id

Saved model:
models/context/kmeans_context.pkl

CSV output:
data/outputs/context_clusters.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/context_modeling/kmeans_context.py")
import gc
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    print("[PROGRESS] Running kmeans_context.py as standalone script")
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    print(f"[PROGRESS] Resolved backend root path: {_backend_root}")

    if _backend_root not in _sys.path:
        print("[PROGRESS] Backend root not found in sys.path. Adding it now.")
        _sys.path.append(_backend_root)
    else:
        print("[PROGRESS] Backend root already exists in sys.path")

from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_save_joblib,
    atomic_write_csv,
    load_joblib_required,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_w_columns

logger = get_logger(__name__)

print("[PROGRESS] Imports completed successfully for kmeans_context.py")


class KMeansContextModel:
    """
    K-Means baseline context model.
    """

    def __init__(self) -> None:
        """
        Initialize K-Means context model service.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/kmeans_context.py::__init__")
        print("[PROGRESS] Creating required project directories if they do not exist")
        Config.create_directories()
        print("[PROGRESS] KMeansContextModel initialized successfully")

    def _validate_w_features(self, df: pd.DataFrame) -> List[str]:
        """
        Validate and return W operating-condition columns.

        Args:
            df: Scaled features DataFrame.

        Returns:
            List[str]: W feature columns.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/kmeans_context.py::_validate_w_features")
        print(f"[PROGRESS] Validating W features from DataFrame with shape: {df.shape}")
        print(f"[PROGRESS] Total input columns count: {len(df.columns)}")

        try:
            print("[PROGRESS] Searching for W operating-condition columns")
            w_columns = get_w_columns(df)

            print(f"[PROGRESS] Number of W operating-condition columns found: {len(w_columns)}")
            print(f"[PROGRESS] W operating-condition columns: {w_columns}")

            if not w_columns:
                print("[ERROR] No W operating-condition columns found")
                raise ValueError("No W operating-condition columns found for K-Means context modeling.")

            print("[PROGRESS] W feature validation completed successfully")
            return w_columns

        except Exception as exc:
            print(f"[ERROR] W feature validation failed for K-Means with error: {exc}")
            logger.exception("W feature validation failed for K-Means.")
            raise RuntimeError("W feature validation failed for K-Means.") from exc

    def train(self, df: pd.DataFrame) -> Dict[str, object]:
        """
        Train K-Means using dev split only.

        Args:
            df: Scaled features DataFrame.

        Returns:
            Dict[str, object]: Trained model payload.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/kmeans_context.py::train")
        print(f"[PROGRESS] Training input DataFrame shape: {df.shape}")

        try:
            print("[PROGRESS] Validating W operating-condition features for K-Means training")
            w_columns = self._validate_w_features(df)

            print(f"[PROGRESS] Creating dev split mask using split name: {Config.DEV_SPLIT_NAME}")
            dev_mask = df["split"] == Config.DEV_SPLIT_NAME
            print(f"[PROGRESS] Dev rows found for K-Means training: {int(dev_mask.sum())}")
            print(f"[PROGRESS] Non-dev rows ignored during K-Means training: {int((~dev_mask).sum())}")

            if dev_mask.sum() == 0:
                print("[ERROR] No dev rows found. K-Means training cannot continue.")
                raise ValueError("No dev rows found. Cannot train K-Means.")

            print("[PROGRESS] Extracting W_dev feature matrix for K-Means training")
            x_dev = df.loc[dev_mask, w_columns]
            print(f"[PROGRESS] K-Means training matrix shape: {x_dev.shape}")

            print("[PROGRESS] Initializing K-Means model")
            print(f"[PROGRESS] K-Means n_clusters: {Config.CONTEXT_CLUSTER_COUNT}")
            print(f"[PROGRESS] K-Means random_state: {Config.RANDOM_SEED}")
            print(f"[PROGRESS] K-Means n_init: {int(Config.KMEANS_PARAMS['n_init'])}")
            print(f"[PROGRESS] K-Means max_iter: {int(Config.KMEANS_PARAMS['max_iter'])}")

            model = KMeans(
                n_clusters=Config.CONTEXT_CLUSTER_COUNT,
                random_state=Config.RANDOM_SEED,
                n_init=int(Config.KMEANS_PARAMS["n_init"]),
                max_iter=int(Config.KMEANS_PARAMS["max_iter"]),
            )

            print("[PROGRESS] Starting K-Means model fitting on dev split only")
            model.fit(x_dev)
            print("[PROGRESS] K-Means model fitting completed successfully")

            print(f"[PROGRESS] K-Means inertia: {float(model.inertia_)}")
            print(f"[PROGRESS] K-Means cluster centers shape: {model.cluster_centers_.shape}")

            print("[PROGRESS] Creating K-Means model payload")
            payload: Dict[str, object] = {
                "model": model,
                "feature_columns": w_columns,
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "predict_only",
            }

            print(f"[PROGRESS] Saving K-Means model payload to: {Config.KMEANS_MODEL_PATH}")
            atomic_save_joblib(payload, Config.KMEANS_MODEL_PATH)
            print("[PROGRESS] K-Means model payload saved successfully")

            logger.info(
                "K-Means trained on dev split only. rows=%s features=%s",
                int(dev_mask.sum()),
                len(w_columns),
            )

            print("[PROGRESS] K-Means training stage completed successfully")
            return payload

        except Exception as exc:
            print(f"[ERROR] K-Means training failed with error: {exc}")
            logger.exception("K-Means training failed.")
            raise RuntimeError("K-Means training failed.") from exc

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict K-Means context IDs for dev and test using saved model.

        Args:
            df: Scaled features DataFrame.

        Returns:
            pd.DataFrame: Context DataFrame with kmeans_context_id.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/kmeans_context.py::predict")
        print(f"[PROGRESS] Prediction input DataFrame shape: {df.shape}")

        try:
            print(f"[PROGRESS] Loading saved K-Means model from: {Config.KMEANS_MODEL_PATH}")
            payload = load_joblib_required(Config.KMEANS_MODEL_PATH)
            print("[PROGRESS] Saved K-Means model payload loaded successfully")

            model: KMeans = payload["model"]
            feature_columns: List[str] = payload["feature_columns"]

            print("[PROGRESS] Extracted K-Means model and feature columns from payload")
            print(f"[PROGRESS] Number of K-Means feature columns expected: {len(feature_columns)}")
            print(f"[PROGRESS] K-Means feature columns: {feature_columns}")

            print("[PROGRESS] Checking whether all required K-Means feature columns exist in input DataFrame")
            missing = [column for column in feature_columns if column not in df.columns]
            if missing:
                print(f"[ERROR] Missing K-Means feature columns: {missing}")
                raise KeyError(f"Missing K-Means feature columns: {missing}")

            print("[PROGRESS] All required K-Means feature columns are available")

            print("[PROGRESS] Creating context result DataFrame with unit_id, cycle, and split")
            result = df[["unit_id", "cycle", "split"]].copy()
            print(f"[PROGRESS] Base result DataFrame shape: {result.shape}")

            print("[PROGRESS] Predicting K-Means context IDs for all rows")
            predictions = np.empty(len(df), dtype=int)
            batch_size = 100_000

            for start in range(0, len(df), batch_size):
                end = min(start + batch_size, len(df))
                batch_x = df.iloc[start:end][feature_columns]
                predictions[start:end] = model.predict(batch_x)
                del batch_x

            result["kmeans_context_id"] = predictions
            del predictions
            gc.collect()
            print("[PROGRESS] K-Means context ID prediction completed")

            print("[PROGRESS] K-Means context ID distribution:")
            print(result["kmeans_context_id"].value_counts().sort_index().to_dict())

            print("[PROGRESS] Adding W feature columns to context output DataFrame")
            for column in feature_columns:
                result[column] = df[column]

            print(f"[PROGRESS] Final K-Means context result shape: {result.shape}")

            logger.info("K-Means context prediction completed for rows=%s.", len(result))
            print("[PROGRESS] K-Means prediction stage completed successfully")
            return result

        except Exception as exc:
            print(f"[ERROR] K-Means prediction failed with error: {exc}")
            logger.exception("K-Means prediction failed.")
            raise RuntimeError("K-Means prediction failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Train and predict K-Means context model.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/kmeans_context.py::run")
        try:
            print(f"[PROGRESS] Reading scaled CSV file from: {Config.SCALED_CSV}")
            scaled_df = read_csv_required(Config.SCALED_CSV)
            print("[PROGRESS] Scaled CSV loaded successfully")
            print(f"[PROGRESS] Scaled DataFrame shape: {scaled_df.shape}")
            print(f"[PROGRESS] Scaled DataFrame columns count: {len(scaled_df.columns)}")

            print("[PROGRESS] Starting K-Means training")
            self.train(scaled_df)
            print("[PROGRESS] K-Means training completed")

            print("[PROGRESS] Starting K-Means context prediction")
            context_df = self.predict(scaled_df)
            print("[PROGRESS] K-Means context prediction completed")

            print(f"[PROGRESS] Writing K-Means context CSV output to: {Config.CONTEXT_CSV}")
            atomic_write_csv(context_df, Config.CONTEXT_CSV)
            print("[PROGRESS] K-Means context CSV written successfully")

            response = {
                "status": "success",
                "message": "K-Means context model trained on dev and inferred for all splits.",
                "output_file": str(Config.CONTEXT_CSV),
                "records_count": len(context_df),
            }

            print(f"[PROGRESS] K-Means context stage response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] K-Means context stage failed with error: {exc}")
            logger.exception("K-Means context stage failed.")
            raise RuntimeError("K-Means context stage failed.") from exc


def run_kmeans_context() -> Dict[str, object]:
    """
    Execute K-Means context modeling.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/kmeans_context.py::run_kmeans_context")
    print("[PROGRESS] Creating KMeansContextModel service instance")
    service = KMeansContextModel()

    print("[PROGRESS] Running KMeansContextModel service")
    result = service.run()

    print("[PROGRESS] K-Means context modeling function completed")
    return result


if __name__ == "__main__":
    print("[PROGRESS] kmeans_context.py execution started from __main__")
    result = run_kmeans_context()
    print("[PROGRESS] kmeans_context.py execution finished successfully")
    print(result)
