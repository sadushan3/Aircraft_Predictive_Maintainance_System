"""
SHAP explainer for CA-EDT-AHMA.

Role:
Explain digital twin predictions and support root-cause reasoning.

Models:
- Random Forest Digital Twin
- XGBoost Digital Twin
- LightGBM Digital Twin

Reads:
data/processed/scaled_features.csv
data/outputs/context_clusters.csv
models/digital_twin/random_forest_twin.pkl
models/digital_twin/xgboost_twin.pkl
models/digital_twin/lightgbm_twin.pkl

Writes:
data/outputs/shap_explanations.csv
data/outputs/shap_summary.json

Important:
SHAP is calculated on a bounded sample to keep the pipeline executable on
large NASA N-CMAPSS files.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/explainability/shap_explainer.py")
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    load_joblib_required,
    read_csv_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_w_columns, get_xv_columns

logger = get_logger(__name__)


class SHAPExplainer:
    """
    SHAP explainer for tree-based digital twin models.
    """

    def __init__(self, sample_size: int = 500) -> None:
        """
        Initialize SHAP explainer.

        Args:
            sample_size: Maximum rows used for SHAP calculation.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/shap_explainer.py::__init__")
        Config.create_directories()

        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")

        self.sample_size = sample_size

    def prepare_explanation_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature data for SHAP.

        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame and feature columns.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/shap_explainer.py::prepare_explanation_data")
        try:
            scaled_df = read_csv_required(Config.SCALED_CSV)
            context_df = read_csv_required(Config.CONTEXT_CSV)

            merge_columns = ["unit_id", "cycle", "split"]

            df = scaled_df.merge(
                context_df[merge_columns + ["gmm_context_id", "context_confidence"]],
                on=merge_columns,
                how="left",
            )

            w_columns = get_w_columns(df)
            xv_columns = get_xv_columns(df)

            if not w_columns:
                raise ValueError("No W features found for SHAP explanation.")
            if not xv_columns:
                raise ValueError("No X_v features found for SHAP explanation.")
            if "gmm_context_id" not in df.columns:
                raise KeyError("gmm_context_id is required for SHAP explanation.")

            feature_columns = w_columns + xv_columns + ["gmm_context_id"]

            df = df.sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)

            if len(df) > self.sample_size:
                df_sample = df.sample(
                    n=self.sample_size,
                    random_state=Config.RANDOM_SEED,
                ).sort_values(["split", "unit_id", "cycle"]).reset_index(drop=True)
            else:
                df_sample = df.copy()

            logger.info(
                "Prepared SHAP explanation data. rows=%s features=%s",
                len(df_sample),
                len(feature_columns),
            )

            return df_sample, feature_columns

        except Exception as exc:
            logger.exception("Failed to prepare SHAP explanation data.")
            raise RuntimeError("Failed to prepare SHAP explanation data.") from exc

    def _mean_abs_shap_for_model(
        self,
        model_payload: Dict[str, object],
        x_sample: pd.DataFrame,
        model_name: str,
    ) -> pd.DataFrame:
        """
        Calculate mean absolute SHAP values for one model payload.

        Args:
            model_payload: Saved model payload.
            x_sample: Feature sample.
            model_name: Model label.

        Returns:
            pd.DataFrame: Feature importance DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/shap_explainer.py::_mean_abs_shap_for_model")
        try:
            import shap

            model = model_payload["model"]
            feature_columns: List[str] = model_payload["feature_columns"]
            target_columns: List[str] = model_payload["target_columns"]

            missing = [column for column in feature_columns if column not in x_sample.columns]
            if missing:
                raise KeyError(f"Missing SHAP feature columns for {model_name}: {missing}")

            x_values = x_sample[feature_columns]

            feature_importance_accumulator = np.zeros(len(feature_columns), dtype=float)
            target_count = 0

            if hasattr(model, "estimators_"):
                estimators = list(model.estimators_)
            else:
                estimators = [model]

            for estimator_index, estimator in enumerate(estimators):
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(x_values)

                if isinstance(shap_values, list):
                    shap_array = np.asarray(shap_values[0], dtype=float)
                else:
                    shap_array = np.asarray(shap_values, dtype=float)

                if shap_array.ndim == 3:
                    shap_array = shap_array[:, :, 0]

                if shap_array.ndim != 2:
                    logger.warning(
                        "Skipping unexpected SHAP output shape for %s estimator %s: %s",
                        model_name,
                        estimator_index,
                        shap_array.shape,
                    )
                    continue

                feature_importance_accumulator += np.mean(np.abs(shap_array), axis=0)
                target_count += 1

            if target_count == 0:
                raise ValueError(f"No valid SHAP values calculated for {model_name}.")

            feature_importance = feature_importance_accumulator / target_count

            result = pd.DataFrame(
                {
                    "model": model_name,
                    "feature": feature_columns,
                    "mean_abs_shap": feature_importance,
                    "targets_explained": min(target_count, len(target_columns)),
                    "sample_size": len(x_sample),
                }
            )

            result = result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

            logger.info("SHAP explanation completed for %s.", model_name)
            return result

        except Exception as exc:
            logger.exception("SHAP calculation failed for model: %s", model_name)
            raise RuntimeError(f"SHAP calculation failed for model: {model_name}") from exc

    def explain(self) -> pd.DataFrame:
        """
        Generate SHAP explanation DataFrame for all digital twin models.

        Returns:
            pd.DataFrame: SHAP explanation DataFrame.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/shap_explainer.py::explain")
        try:
            sample_df, feature_columns = self.prepare_explanation_data()
            x_sample = sample_df[feature_columns]

            model_paths = {
                "random_forest": Config.RF_MODEL_PATH,
                "xgboost": Config.XGB_MODEL_PATH,
                "lightgbm": Config.LGBM_MODEL_PATH,
            }

            frames: List[pd.DataFrame] = []

            for model_name, model_path in model_paths.items():
                if not model_path.exists():
                    logger.warning("Skipping SHAP for missing model: %s", model_path)
                    continue

                payload = load_joblib_required(model_path)
                model_shap_df = self._mean_abs_shap_for_model(
                    model_payload=payload,
                    x_sample=x_sample,
                    model_name=model_name,
                )
                frames.append(model_shap_df)

            if not frames:
                raise FileNotFoundError("No digital twin models found for SHAP explanation.")

            shap_df = pd.concat(frames, axis=0, ignore_index=True)

            summary_df = (
                shap_df.groupby("feature", as_index=False)["mean_abs_shap"]
                .mean()
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )

            top_features = summary_df.head(20).to_dict(orient="records")

            atomic_write_json(
                {
                    "status": "success",
                    "sample_size": int(len(sample_df)),
                    "top_features": top_features,
                    "explanation_scope": (
                        "Mean absolute SHAP feature importance across available "
                        "tree-based digital twin models."
                    ),
                },
                Config.OUTPUT_DIR / "shap_summary.json",
            )

            logger.info("Combined SHAP explanation completed. rows=%s", len(shap_df))
            return shap_df

        except Exception as exc:
            logger.exception("SHAP explanation stage failed.")
            raise RuntimeError("SHAP explanation stage failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run SHAP explainer.

        Returns:
            Dict[str, object]: Stage response.
        """
        print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/shap_explainer.py::run")
        try:
            shap_df = self.explain()
            atomic_write_csv(shap_df, Config.SHAP_CSV)

            return {
                "status": "success",
                "message": "SHAP explanations generated for digital twin models.",
                "output_file": str(Config.SHAP_CSV),
                "records_count": len(shap_df),
            }

        except Exception as exc:
            logger.exception("SHAP explainer run failed.")
            raise RuntimeError("SHAP explainer run failed.") from exc


def run_shap_explainer() -> Dict[str, object]:
    """
    Execute SHAP explanation stage.

    Returns:
        Dict[str, object]: Stage response.
    """
    print("[PROGRESS] Entering Backend/app/services/Anomaly_Health_Monitering/explainability/shap_explainer.py::run_shap_explainer")
    explainer = SHAPExplainer()
    return explainer.run()


if __name__ == "__main__":
    result = run_shap_explainer()
    print(result)