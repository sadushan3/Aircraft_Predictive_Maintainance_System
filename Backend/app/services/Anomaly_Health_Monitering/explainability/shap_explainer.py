"""
SHAP explainer for CA-EDT-AHMA.

Role:
Explain digital twin predictions and support root-cause reasoning.

Models:
- Random Forest Digital Twin
- XGBoost Digital Twin
- LightGBM Digital Twin

Reads:
processed/scaled_features.csv
outputs/Anomaly_Health_Monitering/context_clusters.csv
models/digital_twin/random_forest_twin.pkl
models/digital_twin/xgboost_twin.pkl
models/digital_twin/lightgbm_twin.pkl

Writes:
outputs/Anomaly_Health_Monitering/shap_explanations.csv
reports/shap_summary.json

Important:
- SHAP is calculated on a bounded sample.
- This module does not generate row-level SHAP for 7.6M records.
- This module explains global feature importance of digital twin inputs.
- It does not use Y_dev/Y_test.
- It does not use T_dev/T_test.
- It does not predict RUL.
- It does not make maintenance decisions.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "explainability/shap_explainer.py"
)

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple
import gc
import math
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


from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    load_joblib_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class SHAPExplainer:
    """
    Memory-safe bounded-sample SHAP explainer for tree-based digital twins.
    """

    def __init__(self, sample_size: int = 300, chunk_size: int = 250_000) -> None:
        """
        Initialize SHAP explainer.

        Args:
            sample_size: Maximum rows used for SHAP calculation.
            chunk_size: Rows scanned per chunk while sampling.
        """
        print("[PROGRESS] Entering SHAPExplainer.__init__")

        Config.create_directories()

        self.sample_size = int(getattr(Config, "SHAP_SAMPLE_SIZE", sample_size))
        self.chunk_size = int(getattr(Config, "SHAP_SAMPLE_CHUNK_SIZE", chunk_size))

        if self.sample_size <= 0:
            raise ValueError("SHAP_SAMPLE_SIZE must be positive.")

        if self.chunk_size <= 0:
            raise ValueError("SHAP_SAMPLE_CHUNK_SIZE must be positive.")

        self.max_targets_per_model = getattr(Config, "SHAP_MAX_TARGETS_PER_MODEL", None)

        if self.max_targets_per_model is not None:
            self.max_targets_per_model = int(self.max_targets_per_model)

            if self.max_targets_per_model <= 0:
                self.max_targets_per_model = None

        self.scaled_csv: Path = Config.SCALED_CSV
        self.context_csv: Path = Config.CONTEXT_CSV

        self.output_csv: Path = Config.SHAP_CSV

        self.summary_json: Path = getattr(
            Config,
            "SHAP_SUMMARY_JSON",
            Config.REPORT_DIR / "shap_summary.json",
        )

        print(f"[PROGRESS] Scaled CSV: {self.scaled_csv}")
        print(f"[PROGRESS] Context CSV: {self.context_csv}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")
        print(f"[PROGRESS] Summary JSON: {self.summary_json}")
        print(f"[PROGRESS] Sample size: {self.sample_size}")
        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Max targets per model: {self.max_targets_per_model}")

    # ==================================================================================
    # File helpers
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

    def _get_w_columns(self, columns: List[str]) -> List[str]:
        """
        Get W operating-condition columns.
        """
        return [column for column in columns if column.startswith("W_")]

    def _get_xv_columns(self, columns: List[str]) -> List[str]:
        """
        Get X_v / Xv virtual sensor columns.
        """
        return [
            column
            for column in columns
            if column.startswith("Xv_")
            or column.startswith("X_v_")
            or column.startswith("X_v")
        ]

    def _verify_key_alignment(
        self,
        scaled_chunk: pd.DataFrame,
        context_chunk: pd.DataFrame,
    ) -> None:
        """
        Verify aligned row keys between scaled_features.csv and context_clusters.csv.
        """
        merge_columns = ["unit_id", "cycle", "split"]

        if len(scaled_chunk) != len(context_chunk):
            raise ValueError(
                "SHAP sample chunk row mismatch between scaled and context files."
            )

        scaled_keys = scaled_chunk[merge_columns].reset_index(drop=True)
        context_keys = context_chunk[merge_columns].reset_index(drop=True)

        if not scaled_keys.equals(context_keys):
            raise ValueError(
                "SHAP sample key alignment failed between scaled_features.csv and "
                "context_clusters.csv. Regenerate outputs using the same row order."
            )

    # ==================================================================================
    # Data preparation / bounded sampling
    # ==================================================================================

    def prepare_explanation_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare bounded feature sample for SHAP without loading the full dataset.
        """
        print("[PROGRESS] Entering SHAPExplainer.prepare_explanation_data")

        try:
            expected_rows = self._count_csv_rows(self.scaled_csv)
            context_rows = self._count_csv_rows(self.context_csv)

            if expected_rows <= 0:
                raise ValueError("scaled_features.csv contains zero rows.")

            if context_rows != expected_rows:
                raise ValueError(
                    f"context_clusters.csv row mismatch: {context_rows} != {expected_rows}"
                )

            scaled_columns = self._read_header_columns(self.scaled_csv)
            context_columns = self._read_header_columns(self.context_csv)

            w_columns = self._get_w_columns(scaled_columns)
            xv_columns = self._get_xv_columns(scaled_columns)

            if not w_columns:
                raise ValueError("No W operating-condition columns found for SHAP.")

            if not xv_columns:
                raise ValueError("No X_v virtual sensor columns found for SHAP.")

            if "gmm_context_id" not in context_columns:
                raise KeyError("gmm_context_id is required in context_clusters.csv for SHAP.")

            feature_columns = w_columns + xv_columns + ["gmm_context_id"]

            scaled_usecols = ["unit_id", "cycle", "split"] + w_columns + xv_columns

            context_usecols = [
                "unit_id",
                "cycle",
                "split",
                "gmm_context_id",
            ]

            if "context_confidence" in context_columns:
                context_usecols.append("context_confidence")

            rng = np.random.default_rng(Config.RANDOM_SEED)

            sample_frames: List[pd.DataFrame] = []
            chunk_index = 0
            rows_seen = 0

            scaled_iter = pd.read_csv(
                self.scaled_csv,
                usecols=scaled_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            context_iter = pd.read_csv(
                self.context_csv,
                usecols=context_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            print("[PROGRESS] Starting bounded SHAP sampling")

            for scaled_chunk, context_chunk in zip(scaled_iter, context_iter):
                chunk_index += 1
                rows_seen += len(scaled_chunk)

                scaled_chunk = scaled_chunk.reset_index(drop=True)
                context_chunk = context_chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] SHAP sampling chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(scaled_chunk)}")
                print(f"[PROGRESS] Rows seen: {rows_seen}")

                self._verify_key_alignment(scaled_chunk, context_chunk)

                combined_chunk = scaled_chunk.copy()
                combined_chunk["gmm_context_id"] = context_chunk["gmm_context_id"].values

                if "context_confidence" in context_chunk.columns:
                    combined_chunk["context_confidence"] = context_chunk["context_confidence"].values

                # Oversample slightly per chunk, then downsample globally at the end.
                target_sample_from_chunk = int(
                    math.ceil(
                        self.sample_size
                        * len(combined_chunk)
                        / max(expected_rows, 1)
                        * 4.0
                    )
                )

                target_sample_from_chunk = max(1, target_sample_from_chunk)
                target_sample_from_chunk = min(target_sample_from_chunk, len(combined_chunk))

                sampled_indices = rng.choice(
                    len(combined_chunk),
                    size=target_sample_from_chunk,
                    replace=False,
                )

                sample_frames.append(combined_chunk.iloc[sampled_indices].copy())

                del scaled_chunk
                del context_chunk
                del combined_chunk
                del sampled_indices
                gc.collect()

            if not sample_frames:
                raise ValueError("No SHAP sample rows collected.")

            sample_df = pd.concat(sample_frames, axis=0, ignore_index=True)

            if len(sample_df) > self.sample_size:
                sample_df = sample_df.sample(
                    n=self.sample_size,
                    random_state=Config.RANDOM_SEED,
                ).reset_index(drop=True)

            sample_df = sample_df.sort_values(
                ["split", "unit_id", "cycle"]
            ).reset_index(drop=True)

            print(f"[PROGRESS] Final SHAP sample rows: {len(sample_df)}")
            print(f"[PROGRESS] SHAP feature count: {len(feature_columns)}")
            print(f"[PROGRESS] SHAP feature columns: {feature_columns}")

            logger.info(
                "Prepared SHAP sample. rows=%s features=%s",
                len(sample_df),
                len(feature_columns),
            )

            return sample_df, feature_columns

        except Exception as exc:
            print(f"[ERROR] Failed to prepare SHAP data: {exc}")
            logger.exception("Failed to prepare SHAP explanation data.")
            raise RuntimeError("Failed to prepare SHAP explanation data.") from exc

    # ==================================================================================
    # Model payload helpers
    # ==================================================================================

    def _extract_model_payload(
        self,
        payload: Any,
        fallback_feature_columns: List[str],
    ) -> Tuple[Any, List[str], List[str]]:
        """
        Extract model, feature columns, and target columns from saved payload.

        Supports:
        - Dictionary payload with keys model/feature_columns/target_columns.
        - Direct model object fallback.
        """
        if isinstance(payload, dict):
            if "model" not in payload:
                raise KeyError("Model payload dictionary does not contain key: model")

            model = payload["model"]
            feature_columns = list(payload.get("feature_columns", fallback_feature_columns))
            target_columns = list(payload.get("target_columns", []))

            return model, feature_columns, target_columns

        model = payload
        feature_columns = list(fallback_feature_columns)
        target_columns = []

        return model, feature_columns, target_columns

    def _make_shap_compatible(self) -> None:
        """
        Compatibility patch for older SHAP versions with newer NumPy versions.
        """
        if not hasattr(np, "bool"):
            np.bool = bool  # type: ignore[attr-defined]

        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]

        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]

    def _shap_arrays_from_values(self, shap_values: Any) -> List[np.ndarray]:
        """
        Normalize SHAP output into a list of 2D arrays.

        Handles:
        - Single-output array: (n_samples, n_features)
        - Multi-output array: (n_samples, n_features, n_outputs)
        - List of arrays.
        """
        if isinstance(shap_values, list):
            arrays = [np.asarray(values, dtype=np.float64) for values in shap_values]
        else:
            shap_array = np.asarray(shap_values, dtype=np.float64)

            if shap_array.ndim == 3:
                arrays = [
                    shap_array[:, :, target_index]
                    for target_index in range(shap_array.shape[2])
                ]
            else:
                arrays = [shap_array]

        if self.max_targets_per_model is not None:
            arrays = arrays[: self.max_targets_per_model]

        return arrays

    # ==================================================================================
    # SHAP calculations
    # ==================================================================================

    def _mean_abs_shap_for_model(
        self,
        model_payload: Any,
        sample_df: pd.DataFrame,
        model_name: str,
        fallback_feature_columns: List[str],
    ) -> pd.DataFrame:
        """
        Calculate mean absolute SHAP values for one model.

        Handles:
        - MultiOutputRegressor models: explain each target estimator.
        - Direct tree models such as RandomForestRegressor: explain full model once.
        """
        print(f"[PROGRESS] Entering SHAP calculation for {model_name}")

        try:
            import shap

            self._make_shap_compatible()

            model, feature_columns, target_columns = self._extract_model_payload(
                payload=model_payload,
                fallback_feature_columns=fallback_feature_columns,
            )

            missing = [
                column
                for column in feature_columns
                if column not in sample_df.columns
            ]

            if missing:
                raise KeyError(f"Missing SHAP feature columns for {model_name}: {missing}")

            x_sample = sample_df[feature_columns]

            feature_importance_accumulator = np.zeros(
                len(feature_columns),
                dtype=np.float64,
            )

            explained_count = 0

            is_multioutput_wrapper = (
                model.__class__.__name__ == "MultiOutputRegressor"
                and hasattr(model, "estimators_")
            )

            if is_multioutput_wrapper:
                estimators = list(model.estimators_)

                if self.max_targets_per_model is not None:
                    estimators = estimators[: self.max_targets_per_model]

                for estimator_index, estimator in enumerate(estimators):
                    print(
                        f"[PROGRESS] SHAP {model_name} target estimator "
                        f"{estimator_index + 1}/{len(estimators)}"
                    )

                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(x_sample)
                    shap_arrays = self._shap_arrays_from_values(shap_values)

                    for output_index, shap_array in enumerate(shap_arrays):
                        if shap_array.ndim != 2:
                            logger.warning(
                                "Skipping unexpected SHAP shape for %s estimator %s output %s: %s",
                                model_name,
                                estimator_index,
                                output_index,
                                shap_array.shape,
                            )
                            continue

                        if shap_array.shape[1] != len(feature_columns):
                            logger.warning(
                                "Skipping SHAP feature mismatch for %s estimator %s output %s: %s",
                                model_name,
                                estimator_index,
                                output_index,
                                shap_array.shape,
                            )
                            continue

                        feature_importance_accumulator += np.mean(
                            np.abs(shap_array),
                            axis=0,
                        )
                        explained_count += 1

                    del shap_values
                    del shap_arrays
                    del explainer
                    gc.collect()

            else:
                print(f"[PROGRESS] SHAP {model_name} full tree model")

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_sample)
                shap_arrays = self._shap_arrays_from_values(shap_values)

                for output_index, shap_array in enumerate(shap_arrays):
                    print(
                        f"[PROGRESS] SHAP {model_name} output "
                        f"{output_index + 1}/{len(shap_arrays)}"
                    )

                    if shap_array.ndim != 2:
                        logger.warning(
                            "Skipping unexpected SHAP shape for %s output %s: %s",
                            model_name,
                            output_index,
                            shap_array.shape,
                        )
                        continue

                    if shap_array.shape[1] != len(feature_columns):
                        logger.warning(
                            "Skipping SHAP feature mismatch for %s output %s: %s",
                            model_name,
                            output_index,
                            shap_array.shape,
                        )
                        continue

                    feature_importance_accumulator += np.mean(
                        np.abs(shap_array),
                        axis=0,
                    )
                    explained_count += 1

                del shap_values
                del shap_arrays
                del explainer
                gc.collect()

            if explained_count == 0:
                raise ValueError(f"No valid SHAP values calculated for {model_name}.")

            feature_importance = feature_importance_accumulator / float(explained_count)

            result = pd.DataFrame(
                {
                    "model": model_name,
                    "feature": feature_columns,
                    "mean_abs_shap": feature_importance,
                    "targets_explained": int(explained_count),
                    "available_target_count": int(len(target_columns)),
                    "sample_size": int(len(sample_df)),
                }
            )

            result = result.sort_values(
                "mean_abs_shap",
                ascending=False,
            ).reset_index(drop=True)

            logger.info("SHAP explanation completed for %s.", model_name)
            return result

        except Exception as exc:
            print(f"[ERROR] SHAP calculation failed for {model_name}: {exc}")
            logger.exception("SHAP calculation failed for model: %s", model_name)
            raise RuntimeError(f"SHAP calculation failed for model: {model_name}") from exc

    # ==================================================================================
    # Main
    # ==================================================================================

    def explain(self) -> pd.DataFrame:
        """
        Generate SHAP explanation DataFrame for available digital twin models.
        """
        print("[PROGRESS] Entering SHAPExplainer.explain")

        try:
            started = perf_counter()

            sample_df, feature_columns = self.prepare_explanation_data()

            model_paths = {
                "random_forest": Config.RF_MODEL_PATH,
                "xgboost": Config.XGB_MODEL_PATH,
                "lightgbm": Config.LGBM_MODEL_PATH,
            }

            frames: List[pd.DataFrame] = []
            models_explained: List[str] = []
            models_skipped: Dict[str, str] = {}

            for model_name, model_path in model_paths.items():
                print("=" * 100)
                print(f"[PROGRESS] Preparing SHAP for model: {model_name}")
                print(f"[PROGRESS] Model path: {model_path}")

                if not model_path.exists():
                    reason = f"missing model file: {model_path}"
                    print(f"[WARNING] Skipping {model_name}: {reason}")
                    models_skipped[model_name] = reason
                    continue

                try:
                    payload = load_joblib_required(model_path)

                    model_shap_df = self._mean_abs_shap_for_model(
                        model_payload=payload,
                        sample_df=sample_df,
                        model_name=model_name,
                        fallback_feature_columns=feature_columns,
                    )

                    frames.append(model_shap_df)
                    models_explained.append(model_name)

                except Exception as model_exc:
                    reason = str(model_exc)
                    print(f"[WARNING] Skipping SHAP for {model_name}: {reason}")
                    models_skipped[model_name] = reason

            if not frames:
                raise FileNotFoundError(
                    "No digital twin models were successfully explained by SHAP."
                )

            shap_df = pd.concat(frames, axis=0, ignore_index=True)

            summary_df = (
                shap_df.groupby("feature", as_index=False)["mean_abs_shap"]
                .mean()
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )

            top_features = summary_df.head(20).to_dict(orient="records")

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "sample_size": int(len(sample_df)),
                "feature_count": int(len(feature_columns)),
                "models_explained": models_explained,
                "models_skipped": models_skipped,
                "top_features": top_features,
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "explanation_scope": (
                    "Mean absolute SHAP feature importance across available "
                    "tree-based digital twin models using a bounded sample."
                ),
                "leakage_audit": {
                    "bounded_sample_only": True,
                    "does_not_generate_row_level_shap_for_full_dataset": True,
                    "does_not_train_model": True,
                    "does_not_predict_rul": True,
                    "does_not_make_maintenance_decisions": True,
                    "does_not_use_y_dev_y_test": True,
                    "does_not_use_t_dev_t_test": True,
                    "uses_scaled_features": True,
                    "uses_context_clusters": True,
                },
            }

            print(f"[PROGRESS] Writing SHAP summary to: {self.summary_json}")
            atomic_write_json(summary, self.summary_json)

            logger.info("Combined SHAP explanation completed. rows=%s", len(shap_df))
            return shap_df

        except Exception as exc:
            print(f"[ERROR] SHAP explanation stage failed: {exc}")
            logger.exception("SHAP explanation stage failed.")
            raise RuntimeError("SHAP explanation stage failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run SHAP explainer.
        """
        print("[PROGRESS] Entering SHAPExplainer.run")

        try:
            shap_df = self.explain()
            atomic_write_csv(shap_df, self.output_csv)

            response = {
                "status": "success",
                "message": "SHAP explanations generated for digital twin models.",
                "output_file": str(self.output_csv),
                "summary_file": str(self.summary_json),
                "records_count": int(len(shap_df)),
            }

            print(f"[PROGRESS] SHAP explainer response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] SHAP explainer run failed: {exc}")
            logger.exception("SHAP explainer run failed.")
            raise RuntimeError("SHAP explainer run failed.") from exc


def run_shap_explainer() -> Dict[str, object]:
    """
    Execute SHAP explanation stage.
    """
    print("[PROGRESS] Entering run_shap_explainer")

    explainer = SHAPExplainer()
    return explainer.run()


if __name__ == "__main__":
    print("[PROGRESS] shap_explainer.py execution started")
    result = run_shap_explainer()
    print("[PROGRESS] shap_explainer.py execution finished successfully")
    print(result)