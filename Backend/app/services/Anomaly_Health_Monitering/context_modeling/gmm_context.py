"""
Gaussian Mixture Model operating-context model for CA-EDT-AHMA.

Role:
Final probabilistic operating-context model.

Training:
Input = W_dev
Target = None

Inference:
Input = W_dev and W_test
Output = gmm_context_id, context_probability, context_confidence

Saved model:
models/context/gmm_context.pkl

CSV output:
data/outputs/context_clusters.csv
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/context_modeling/gmm_context.py")
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    print("[PROGRESS] Running gmm_context.py as standalone script")
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    print(f"[PROGRESS] Resolved backend root path: {_backend_root}")

    if _backend_root not in _sys.path:
        print("[PROGRESS] Backend root not found in sys.path. Adding it now.")
        _sys.path.append(_backend_root)
    else:
        print("[PROGRESS] Backend root already exists in sys.path")

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_save_joblib,
    atomic_write_csv,
    load_joblib_required,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_w_columns

logger = get_logger(__name__)

print("[PROGRESS] Imports completed successfully for gmm_context.py")


class GMMContextModel:
    """
    Gaussian Mixture Model probabilistic context detector.
    """

    def __init__(self) -> None:
        """
        Initialize GMM context model service.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/gmm_context.py::__init__")
        print("[PROGRESS] Creating required project directories if they do not exist")
        Config.create_directories()
        print("[PROGRESS] GMMContextModel initialized successfully")

    def _validate_w_features(self, df: pd.DataFrame) -> List[str]:
        """
        Validate W operating-condition features.

        Args:
            df: Scaled features DataFrame.

        Returns:
            List[str]: W feature columns.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/gmm_context.py::_validate_w_features")
        print(f"[PROGRESS] Validating W features from DataFrame with shape: {df.shape}")
        print(f"[PROGRESS] Total input columns count: {len(df.columns)}")

        try:
            print("[PROGRESS] Searching for W operating-condition columns")
            w_columns = get_w_columns(df)

            print(f"[PROGRESS] Number of W operating-condition columns found: {len(w_columns)}")
            print(f"[PROGRESS] W operating-condition columns: {w_columns}")

            if not w_columns:
                print("[ERROR] No W operating-condition columns found")
                raise ValueError("No W operating-condition columns found for GMM context modeling.")

            print("[PROGRESS] W feature validation completed successfully")
            return w_columns

        except Exception as exc:
            print(f"[ERROR] W feature validation failed for GMM with error: {exc}")
            logger.exception("W feature validation failed for GMM.")
            raise RuntimeError("W feature validation failed for GMM.") from exc

    def train(self, df: pd.DataFrame) -> Dict[str, object]:
        """
        Train GMM using dev split only.

        Args:
            df: Scaled features DataFrame.

        Returns:
            Dict[str, object]: Trained model payload.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/gmm_context.py::train")
        print(f"[PROGRESS] Training input DataFrame shape: {df.shape}")

        try:
            print("[PROGRESS] Validating W operating-condition features for GMM training")
            w_columns = self._validate_w_features(df)

            print(f"[PROGRESS] Creating dev split mask using split name: {Config.DEV_SPLIT_NAME}")
            dev_mask = df["split"] == Config.DEV_SPLIT_NAME

            print(f"[PROGRESS] Dev rows found for GMM training: {int(dev_mask.sum())}")
            print(f"[PROGRESS] Non-dev rows ignored during GMM training: {int((~dev_mask).sum())}")

            if dev_mask.sum() == 0:
                print("[ERROR] No dev rows found. GMM training cannot continue.")
                raise ValueError("No dev rows found. Cannot train GMM.")

            print("[PROGRESS] Extracting W_dev feature matrix for GMM training")
            x_dev = df.loc[dev_mask, w_columns]
            print(f"[PROGRESS] GMM training matrix shape: {x_dev.shape}")

            print("[PROGRESS] Initializing Gaussian Mixture Model")
            print(f"[PROGRESS] GMM n_components: {Config.CONTEXT_CLUSTER_COUNT}")
            print(f"[PROGRESS] GMM covariance_type: {Config.GMM_COVARIANCE_TYPE}")
            print(f"[PROGRESS] GMM random_state: {Config.RANDOM_SEED}")
            print(f"[PROGRESS] GMM max_iter: {int(Config.GMM_PARAMS['max_iter'])}")
            print(f"[PROGRESS] GMM init_params: {str(Config.GMM_PARAMS['init_params'])}")

            model = GaussianMixture(
                n_components=Config.CONTEXT_CLUSTER_COUNT,
                covariance_type=Config.GMM_COVARIANCE_TYPE,
                random_state=Config.RANDOM_SEED,
                max_iter=int(Config.GMM_PARAMS["max_iter"]),
                init_params=str(Config.GMM_PARAMS["init_params"]),
            )

            print("[PROGRESS] Starting GMM model fitting on dev split only")
            model.fit(x_dev)
            print("[PROGRESS] GMM model fitting completed successfully")

            print(f"[PROGRESS] GMM converged: {model.converged_}")
            print(f"[PROGRESS] GMM iterations used: {model.n_iter_}")
            print(f"[PROGRESS] GMM lower bound: {float(model.lower_bound_)}")
            print(f"[PROGRESS] GMM weights shape: {model.weights_.shape}")
            print(f"[PROGRESS] GMM means shape: {model.means_.shape}")

            print("[PROGRESS] Creating GMM model payload")
            payload: Dict[str, object] = {
                "model": model,
                "feature_columns": w_columns,
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "predict_only",
            }

            print(f"[PROGRESS] Saving GMM model payload to: {Config.GMM_MODEL_PATH}")
            atomic_save_joblib(payload, Config.GMM_MODEL_PATH)
            print("[PROGRESS] GMM model payload saved successfully")

            logger.info(
                "GMM trained on dev split only. rows=%s features=%s",
                int(dev_mask.sum()),
                len(w_columns),
            )

            print("[PROGRESS] GMM training stage completed successfully")
            return payload

        except Exception as exc:
            print(f"[ERROR] GMM training failed with error: {exc}")
            logger.exception("GMM training failed.")
            raise RuntimeError("GMM training failed.") from exc

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict GMM context ID and confidence for dev and test.

        This version uses batch prediction to avoid RAM explosion for very large datasets.

        Args:
            df: Scaled features DataFrame.

        Returns:
            pd.DataFrame: GMM context output.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/gmm_context.py::predict")
        print(f"[PROGRESS] Prediction input DataFrame shape: {df.shape}")

        try:
            print(f"[PROGRESS] Loading saved GMM model from: {Config.GMM_MODEL_PATH}")
            payload = load_joblib_required(Config.GMM_MODEL_PATH)
            print("[PROGRESS] Saved GMM model payload loaded successfully")

            model: GaussianMixture = payload["model"]
            feature_columns: List[str] = payload["feature_columns"]

            print("[PROGRESS] Extracted GMM model and feature columns from payload")
            print(f"[PROGRESS] Number of GMM feature columns expected: {len(feature_columns)}")
            print(f"[PROGRESS] GMM feature columns: {feature_columns}")

            print("[PROGRESS] Checking whether all required GMM feature columns exist in input DataFrame")
            missing = [column for column in feature_columns if column not in df.columns]
            if missing:
                print(f"[ERROR] Missing GMM feature columns: {missing}")
                raise KeyError(f"Missing GMM feature columns: {missing}")

            print("[PROGRESS] All required GMM feature columns are available")

            total_rows = len(df)
            batch_size = 100_000

            print(f"[PROGRESS] Total rows for GMM prediction: {total_rows}")
            print(f"[PROGRESS] GMM prediction batch size: {batch_size}")
            print("[PROGRESS] Starting memory-safe batch GMM probability prediction")

            context_ids = np.empty(total_rows, dtype=np.int64)
            context_probability = np.empty(total_rows, dtype=np.float64)

            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)

                print(
                    "[PROGRESS] Predicting GMM batch: "
                    f"start={start_idx}, end={end_idx}, rows={end_idx - start_idx}"
                )

                batch_x = df.iloc[start_idx:end_idx][feature_columns]
                batch_probabilities = model.predict_proba(batch_x)

                context_ids[start_idx:end_idx] = np.argmax(batch_probabilities, axis=1)
                context_probability[start_idx:end_idx] = np.max(batch_probabilities, axis=1)

                print(
                    "[PROGRESS] Completed GMM batch: "
                    f"start={start_idx}, end={end_idx}"
                )

            print("[PROGRESS] Memory-safe batch GMM prediction completed successfully")

            print(
                "[PROGRESS] Context probability summary: "
                f"min={float(np.min(context_probability))}, "
                f"max={float(np.max(context_probability))}, "
                f"mean={float(np.mean(context_probability))}"
            )

            print("[PROGRESS] Creating GMM context result DataFrame with unit_id, cycle, and split")
            result = df[["unit_id", "cycle", "split"]].copy()
            print(f"[PROGRESS] Base result DataFrame shape: {result.shape}")

            print("[PROGRESS] Adding GMM context output columns")
            result["gmm_context_id"] = context_ids
            result["context_probability"] = context_probability
            result["context_confidence"] = np.clip(context_probability, 0.0, 1.0)
            print("[PROGRESS] GMM context output columns added successfully")

            print("[PROGRESS] GMM context ID distribution:")
            print(result["gmm_context_id"].value_counts().sort_index().to_dict())

            print(
                "[PROGRESS] Context confidence summary: "
                f"min={float(result['context_confidence'].min())}, "
                f"max={float(result['context_confidence'].max())}, "
                f"mean={float(result['context_confidence'].mean())}"
            )

            print("[PROGRESS] Adding W feature columns to GMM context output DataFrame")
            for column in feature_columns:
                result[column] = df[column].values

            print(f"[PROGRESS] Final GMM context result shape: {result.shape}")

            logger.info("GMM context prediction completed for rows=%s.", len(result))
            print("[PROGRESS] GMM prediction stage completed successfully")
            return result

        except Exception as exc:
            print(f"[ERROR] GMM prediction failed with error: {exc}")
            logger.exception("GMM prediction failed.")
            raise RuntimeError("GMM prediction failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Train and predict GMM context model.

        If context_clusters.csv already contains K-Means output, this stage
        enriches it with GMM columns. If not, it creates a valid GMM-only
        context output.

        Important:
        This uses a temporary occurrence index during merge to prevent
        many-to-many merge explosion when unit_id + cycle + split is duplicated.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/gmm_context.py::run")
        try:
            print(f"[PROGRESS] Reading scaled CSV file from: {Config.SCALED_CSV}")
            scaled_df = read_csv_required(Config.SCALED_CSV)
            print("[PROGRESS] Scaled CSV loaded successfully")
            print(f"[PROGRESS] Scaled DataFrame shape: {scaled_df.shape}")
            print(f"[PROGRESS] Scaled DataFrame columns count: {len(scaled_df.columns)}")

            merge_columns = ["unit_id", "cycle", "split"]
            print(f"[PROGRESS] Merge columns: {merge_columns}")

            print("[PROGRESS] Checking duplicate merge keys in scaled DataFrame")
            duplicate_key_count = int(scaled_df.duplicated(merge_columns).sum())
            print(f"[PROGRESS] Duplicate unit_id/cycle/split rows in scaled_df: {duplicate_key_count}")

            if duplicate_key_count > 0:
                print("[WARNING] unit_id + cycle + split is not unique in scaled_df")
                print("[WARNING] A normal merge can create a many-to-many RAM explosion")
                print("[WARNING] Safe merge mode will use temporary occurrence index")

            print("[PROGRESS] Starting GMM training")
            self.train(scaled_df)
            print("[PROGRESS] GMM training completed")

            print("[PROGRESS] Starting GMM context prediction")
            gmm_df = self.predict(scaled_df)
            print("[PROGRESS] GMM context prediction completed")
            print(f"[PROGRESS] GMM output DataFrame shape: {gmm_df.shape}")

            print("[PROGRESS] Adding temporary merge occurrence index to GMM output")
            gmm_df = gmm_df.copy()
            gmm_df["__merge_occurrence"] = gmm_df.groupby(merge_columns).cumcount()
            print("[PROGRESS] Temporary merge occurrence index added to GMM output")

            print(f"[PROGRESS] Checking whether existing context CSV exists at: {Config.CONTEXT_CSV}")
            if Config.CONTEXT_CSV.exists():
                print("[PROGRESS] Existing context CSV found. Reading previous context output.")
                previous_context = read_csv_required(Config.CONTEXT_CSV)
                print(f"[PROGRESS] Previous context DataFrame shape: {previous_context.shape}")
                print(f"[PROGRESS] Previous context columns: {list(previous_context.columns)}")

                print("[PROGRESS] Checking duplicate merge keys in previous context DataFrame")
                previous_duplicate_key_count = int(previous_context.duplicated(merge_columns).sum())
                print(
                    "[PROGRESS] Duplicate unit_id/cycle/split rows in previous_context: "
                    f"{previous_duplicate_key_count}"
                )

                if previous_duplicate_key_count > 0:
                    print("[WARNING] unit_id + cycle + split is not unique in previous_context")
                    print("[WARNING] Using occurrence index to prevent many-to-many merge")

                print("[PROGRESS] Adding temporary merge occurrence index to previous context")
                previous_context = previous_context.copy()
                previous_context["__merge_occurrence"] = previous_context.groupby(merge_columns).cumcount()
                print("[PROGRESS] Temporary merge occurrence index added to previous context")

                safe_merge_columns = merge_columns + ["__merge_occurrence"]
                print(f"[PROGRESS] Safe merge columns: {safe_merge_columns}")

                print("[PROGRESS] Keeping merge columns and existing kmeans_context_id if available")
                keep_columns = [
                    column
                    for column in previous_context.columns
                    if column in safe_merge_columns or column == "kmeans_context_id"
                ]
                print(f"[PROGRESS] Columns kept from previous context: {keep_columns}")

                print("[PROGRESS] Performing safe one-to-one merge between previous context and GMM output")
                context_df = previous_context[keep_columns].merge(
                    gmm_df,
                    on=safe_merge_columns,
                    how="left",
                    validate="one_to_one",
                )
                print("[PROGRESS] Safe context merge completed successfully")
                print(f"[PROGRESS] Merged context DataFrame shape: {context_df.shape}")

                print("[PROGRESS] Dropping temporary merge occurrence index from final merged output")
                context_df = context_df.drop(columns=["__merge_occurrence"])
                print("[PROGRESS] Temporary merge occurrence index dropped successfully")

            else:
                print("[PROGRESS] No existing context CSV found. Creating GMM-only context output.")
                context_df = gmm_df.copy()
                context_df["kmeans_context_id"] = -1
                print("[PROGRESS] Added placeholder kmeans_context_id = -1")

                if "__merge_occurrence" in context_df.columns:
                    print("[PROGRESS] Dropping temporary merge occurrence index from GMM-only output")
                    context_df = context_df.drop(columns=["__merge_occurrence"])
                    print("[PROGRESS] Temporary merge occurrence index dropped successfully")

                print(f"[PROGRESS] GMM-only context DataFrame shape: {context_df.shape}")

            required_columns = [
                "unit_id",
                "cycle",
                "split",
                "kmeans_context_id",
                "gmm_context_id",
                "context_probability",
                "context_confidence",
            ]

            print(f"[PROGRESS] Required context output columns: {required_columns}")
            print("[PROGRESS] Checking and filling missing required columns if needed")

            for column in required_columns:
                if column not in context_df.columns:
                    print(f"[PROGRESS] Missing required column detected: {column}")
                    context_df[column] = -1 if column.endswith("_id") else 0.0
                    print(f"[PROGRESS] Filled missing column {column}")

            print("[PROGRESS] Extracting W operating-condition columns from scaled DataFrame")
            w_columns = get_w_columns(scaled_df)
            print(f"[PROGRESS] Number of W columns found for final output: {len(w_columns)}")
            print(f"[PROGRESS] W columns for final output: {w_columns}")

            print("[PROGRESS] Ensuring W columns are available in final context output")
            for column in w_columns:
                if column not in context_df.columns:
                    print(f"[PROGRESS] Adding missing W column to context output: {column}")

                    if len(context_df) == len(scaled_df):
                        context_df[column] = scaled_df[column].values
                    else:
                        print(
                            "[WARNING] context_df and scaled_df row counts differ. "
                            "Using index-aligned assignment fallback."
                        )
                        context_df[column] = scaled_df[column]

            print("[PROGRESS] Reordering final context output columns")
            context_df = context_df[
                required_columns + [column for column in w_columns if column in context_df.columns]
            ]

            print(f"[PROGRESS] Final context DataFrame shape after column ordering: {context_df.shape}")
            print(f"[PROGRESS] Final context DataFrame columns: {list(context_df.columns)}")

            print("[PROGRESS] Final GMM context ID distribution:")
            print(context_df["gmm_context_id"].value_counts().sort_index().to_dict())

            print(
                "[PROGRESS] Final context confidence summary: "
                f"min={float(context_df['context_confidence'].min())}, "
                f"max={float(context_df['context_confidence'].max())}, "
                f"mean={float(context_df['context_confidence'].mean())}"
            )

            print(f"[PROGRESS] Writing final context CSV output to: {Config.CONTEXT_CSV}")
            atomic_write_csv(context_df, Config.CONTEXT_CSV)
            print("[PROGRESS] Final context CSV written successfully")

            response = {
                "status": "success",
                "message": "GMM context model trained on dev and inferred for all splits.",
                "output_file": str(Config.CONTEXT_CSV),
                "records_count": len(context_df),
            }

            print(f"[PROGRESS] GMM context stage response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] GMM context stage failed with error: {exc}")
            logger.exception("GMM context stage failed.")
            raise RuntimeError("GMM context stage failed.") from exc


def run_gmm_context() -> Dict[str, object]:
    """
    Execute GMM context modeling.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/context_modeling/gmm_context.py::run_gmm_context")
    print("[PROGRESS] Creating GMMContextModel service instance")
    service = GMMContextModel()

    print("[PROGRESS] Running GMMContextModel service")
    result = service.run()

    print("[PROGRESS] GMM context modeling function completed")
    return result


if __name__ == "__main__":
    print("[PROGRESS] gmm_context.py execution started from __main__")
    result = run_gmm_context()
    print("[PROGRESS] gmm_context.py execution finished successfully")
    print(result)