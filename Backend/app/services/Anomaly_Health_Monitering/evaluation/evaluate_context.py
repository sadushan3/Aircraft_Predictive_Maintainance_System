"""
Memory-safe context modeling evaluation for CA-EDT-AHMA.

Metrics:
- Sampled Silhouette score for K-Means context IDs
- Sampled Silhouette score for GMM context IDs
- Sampled Davies-Bouldin score for K-Means context IDs
- Sampled Davies-Bouldin score for GMM context IDs
- Chunked GMM average log likelihood
- Average context confidence

Reads:
- outputs/Anomaly_Health_Monitering/context_clusters.csv
- processed/scaled_features.csv only if W columns are not already in context CSV
- models/context/gmm_context.pkl

Writes:
- metrics/evaluate_context.csv
- reports/evaluate_context_summary.json

Why sampled?
Silhouette score is not safe on millions of rows because it requires pairwise
distance calculations. Sampling is the correct practical evaluation method.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/evaluation/evaluate_context.py")

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Tuple
import gc
import os as _os
import sys as _sys

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score


# ======================================================================================
# Standalone script support
# ======================================================================================

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..")
    )
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)


from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import (
    atomic_write_csv,
    atomic_write_json,
    load_joblib_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import get_w_columns


logger = get_logger(__name__)


class ContextEvaluator:
    """
    Memory-safe evaluator for operating context models.

    Important:
    - Does not load full scaled/context CSVs into RAM.
    - Does not perform unsafe many-to-many merges.
    - Uses sampling for silhouette and Davies-Bouldin.
    - Uses chunked accumulation for GMM average log likelihood.
    """

    def __init__(
        self,
        chunk_size: int = 50_000,
        sample_size_per_split: int = 20_000,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize context evaluator.

        Args:
            chunk_size: Number of rows read per CSV chunk.
            sample_size_per_split: Max sampled rows per split for clustering metrics.
            random_seed: Random seed for reproducible sampling.
        """
        print("[PROGRESS] Entering ContextEvaluator.__init__")

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "CONTEXT_EVAL_CHUNK_SIZE", chunk_size)
        )
        self.sample_size_per_split = int(
            getattr(Config, "CONTEXT_EVAL_SAMPLE_SIZE_PER_SPLIT", sample_size_per_split)
        )
        self.random_seed = int(
            getattr(Config, "RANDOM_SEED", 42) if random_seed is None else random_seed
        )

        if self.chunk_size <= 0:
            raise ValueError("CONTEXT_EVAL_CHUNK_SIZE must be positive.")

        if self.sample_size_per_split <= 0:
            raise ValueError("CONTEXT_EVAL_SAMPLE_SIZE_PER_SPLIT must be positive.")

        self.rng = np.random.default_rng(self.random_seed)

        print(f"[PROGRESS] Context eval chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Context eval sample size per split: {self.sample_size_per_split}")
        print(f"[PROGRESS] Context eval random seed: {self.random_seed}")

    # ==================================================================================
    # Header helpers
    # ==================================================================================

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read CSV header as an empty DataFrame.

        Args:
            path: CSV path.

        Returns:
            Empty DataFrame containing only columns.
        """
        print(f"[PROGRESS] Reading CSV header from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV column names only.

        Args:
            path: CSV path.

        Returns:
            Column list.
        """
        print(f"[PROGRESS] Reading CSV columns from: {path}")
        return list(pd.read_csv(path, nrows=0).columns)

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        """
        Validate that required columns exist.

        Args:
            available_columns: Existing columns.
            required_columns: Required columns.
            label: Human-readable label.
        """
        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def _verify_key_alignment(
        self,
        scaled_chunk: pd.DataFrame,
        context_chunk: pd.DataFrame,
        merge_columns: List[str],
    ) -> None:
        """
        Verify row alignment if scaled and context chunks are both needed.

        Args:
            scaled_chunk: Chunk from scaled_features.csv.
            context_chunk: Chunk from context_clusters.csv.
            merge_columns: Row identity columns.
        """
        if len(scaled_chunk) != len(context_chunk):
            raise ValueError(
                "Chunk row count mismatch: "
                f"scaled={len(scaled_chunk)}, context={len(context_chunk)}"
            )

        scaled_keys = scaled_chunk[merge_columns].reset_index(drop=True)
        context_keys = context_chunk[merge_columns].reset_index(drop=True)

        if not scaled_keys.equals(context_keys):
            raise ValueError(
                "scaled_features.csv and context_clusters.csv are not row-aligned. "
                "Regenerate context outputs using the same input order."
            )

    # ==================================================================================
    # Chunk iterator
    # ==================================================================================

    def _prepare_w_columns(self) -> Tuple[List[str], bool]:
        """
        Determine W columns and whether they are already available in context CSV.

        Returns:
            Tuple:
            - W column names
            - True if W columns are available in context CSV
        """
        context_header_df = self._read_header_df(Config.CONTEXT_CSV)
        context_columns = list(context_header_df.columns)

        merge_columns = ["unit_id", "cycle", "split"]
        required_context_columns = merge_columns + [
            "kmeans_context_id",
            "gmm_context_id",
            "context_confidence",
        ]

        self._validate_columns(
            available_columns=context_columns,
            required_columns=required_context_columns,
            label="context_clusters.csv",
        )

        context_w_columns = get_w_columns(context_header_df)

        if context_w_columns:
            print("[PROGRESS] W columns found directly in context_clusters.csv")
            print(f"[PROGRESS] W columns: {context_w_columns}")
            return context_w_columns, True

        print("[WARNING] W columns not found in context_clusters.csv")
        print("[PROGRESS] Falling back to scaled_features.csv for W columns")

        scaled_header_df = self._read_header_df(Config.SCALED_CSV)
        scaled_columns = list(scaled_header_df.columns)

        scaled_w_columns = get_w_columns(scaled_header_df)

        if not scaled_w_columns:
            raise ValueError("No W operating-condition columns found in scaled_features.csv.")

        self._validate_columns(
            available_columns=scaled_columns,
            required_columns=merge_columns + scaled_w_columns,
            label="scaled_features.csv",
        )

        print(f"[PROGRESS] W columns from scaled_features.csv: {scaled_w_columns}")
        return scaled_w_columns, False

    def _iter_context_chunks(
        self,
        w_columns: List[str],
        w_in_context_csv: bool,
    ) -> Iterator[pd.DataFrame]:
        """
        Yield memory-safe context evaluation chunks.

        Args:
            w_columns: Operating-condition columns.
            w_in_context_csv: Whether W columns exist in context CSV.

        Yields:
            DataFrame with merge columns, context labels, confidence, and W columns.
        """
        merge_columns = ["unit_id", "cycle", "split"]

        context_base_columns = merge_columns + [
            "kmeans_context_id",
            "gmm_context_id",
            "context_confidence",
        ]

        if w_in_context_csv:
            context_usecols = context_base_columns + w_columns

            for context_chunk in pd.read_csv(
                Config.CONTEXT_CSV,
                usecols=context_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                yield context_chunk

                del context_chunk
                gc.collect()

        else:
            context_usecols = context_base_columns
            scaled_usecols = merge_columns + w_columns

            context_iter = pd.read_csv(
                Config.CONTEXT_CSV,
                usecols=context_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            scaled_iter = pd.read_csv(
                Config.SCALED_CSV,
                usecols=scaled_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            )

            for scaled_chunk, context_chunk in zip(scaled_iter, context_iter):
                self._verify_key_alignment(
                    scaled_chunk=scaled_chunk,
                    context_chunk=context_chunk,
                    merge_columns=merge_columns,
                )

                for column in w_columns:
                    context_chunk[column] = scaled_chunk[column].values

                yield context_chunk

                del scaled_chunk
                del context_chunk
                gc.collect()

    # ==================================================================================
    # Sampling and metrics
    # ==================================================================================

    def _sample_for_split(
        self,
        split_df: pd.DataFrame,
        current_sample: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Update a bounded sample for one split.

        This keeps memory bounded. It is not a perfect reservoir sampler, but it is
        stable and safe for large CSV evaluation.

        Args:
            split_df: Current split rows from chunk.
            current_sample: Existing sample.

        Returns:
            Updated sample DataFrame.
        """
        if split_df.empty:
            return current_sample if current_sample is not None else split_df

        if len(split_df) > self.sample_size_per_split:
            split_df = split_df.sample(
                n=self.sample_size_per_split,
                random_state=self.random_seed,
            )

        if current_sample is None or current_sample.empty:
            combined = split_df.copy()
        else:
            combined = pd.concat([current_sample, split_df], ignore_index=True)

        if len(combined) > self.sample_size_per_split:
            combined = combined.sample(
                n=self.sample_size_per_split,
                random_state=self.random_seed,
            ).reset_index(drop=True)

        return combined

    def _safe_silhouette(self, x: pd.DataFrame, labels: pd.Series) -> float:
        """
        Safely calculate silhouette score on a sample.

        Args:
            x: Feature matrix.
            labels: Cluster labels.

        Returns:
            Silhouette score or 0.0 if invalid.
        """
        print("[PROGRESS] Entering ContextEvaluator._safe_silhouette")

        try:
            clean = pd.concat([x, labels.rename("__label")], axis=1).dropna()

            if clean.empty:
                return 0.0

            clean_labels = clean["__label"]
            clean_x = clean.drop(columns=["__label"])

            if clean_labels.nunique() < 2 or len(clean_labels) <= clean_labels.nunique():
                return 0.0

            return float(silhouette_score(clean_x, clean_labels))

        except Exception as exc:
            print(f"[WARNING] Silhouette calculation failed safely: {exc}")
            return 0.0

    def _safe_davies_bouldin(self, x: pd.DataFrame, labels: pd.Series) -> float:
        """
        Safely calculate Davies-Bouldin score on a sample.

        Args:
            x: Feature matrix.
            labels: Cluster labels.

        Returns:
            Davies-Bouldin score or 0.0 if invalid.
        """
        print("[PROGRESS] Entering ContextEvaluator._safe_davies_bouldin")

        try:
            clean = pd.concat([x, labels.rename("__label")], axis=1).dropna()

            if clean.empty:
                return 0.0

            clean_labels = clean["__label"]
            clean_x = clean.drop(columns=["__label"])

            if clean_labels.nunique() < 2 or len(clean_labels) <= clean_labels.nunique():
                return 0.0

            return float(davies_bouldin_score(clean_x, clean_labels))

        except Exception as exc:
            print(f"[WARNING] Davies-Bouldin calculation failed safely: {exc}")
            return 0.0

    def _load_gmm_payload(self) -> Tuple[object, List[str]]:
        """
        Load dev-fitted GMM model payload.

        Returns:
            Tuple of model and feature columns.
        """
        print("[PROGRESS] Loading dev-fitted GMM payload")

        payload = load_joblib_required(Config.GMM_MODEL_PATH)
        model = payload["model"]
        feature_columns: List[str] = payload.get("feature_columns", [])

        if not feature_columns:
            raise KeyError("GMM payload does not contain feature_columns.")

        print(f"[PROGRESS] GMM feature columns: {feature_columns}")
        return model, feature_columns

    # ==================================================================================
    # Main evaluation
    # ==================================================================================

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate context modeling.

        Returns:
            Context metrics DataFrame.
        """
        print("[PROGRESS] Entering ContextEvaluator.evaluate")

        try:
            started = perf_counter()

            if not Config.CONTEXT_CSV.exists():
                raise FileNotFoundError(f"Context CSV not found: {Config.CONTEXT_CSV}")

            if not Config.GMM_MODEL_PATH.exists():
                raise FileNotFoundError(f"GMM model not found: {Config.GMM_MODEL_PATH}")

            w_columns, w_in_context_csv = self._prepare_w_columns()
            gmm_model, gmm_feature_columns = self._load_gmm_payload()

            missing_gmm_features = [
                column for column in gmm_feature_columns if column not in w_columns
            ]

            if missing_gmm_features:
                print(
                    "[WARNING] Some GMM payload feature columns are not in W columns: "
                    f"{missing_gmm_features}"
                )
                print("[WARNING] Falling back to available W columns for likelihood scoring")
                gmm_score_columns = w_columns
            else:
                gmm_score_columns = gmm_feature_columns

            split_names = [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]

            split_samples: Dict[str, pd.DataFrame | None] = {
                split: None for split in split_names
            }

            split_confidence_sum: Dict[str, float] = {
                split: 0.0 for split in split_names
            }
            split_confidence_count: Dict[str, int] = {
                split: 0 for split in split_names
            }

            split_likelihood_sum: Dict[str, float] = {
                split: 0.0 for split in split_names
            }
            split_likelihood_count: Dict[str, int] = {
                split: 0 for split in split_names
            }

            total_rows_seen = 0
            chunk_index = 0

            for chunk in self._iter_context_chunks(
                w_columns=w_columns,
                w_in_context_csv=w_in_context_csv,
            ):
                chunk_index += 1
                total_rows_seen += len(chunk)

                print("=" * 100)
                print(f"[PROGRESS] Context eval chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")
                print(f"[PROGRESS] Total rows seen: {total_rows_seen}")

                for split in split_names:
                    split_mask = chunk["split"] == split
                    split_chunk = chunk.loc[split_mask].copy()
                    split_rows = len(split_chunk)

                    print(f"[PROGRESS] Split={split}, rows in chunk={split_rows}")

                    if split_rows == 0:
                        del split_chunk
                        continue

                    split_confidence_sum[split] += float(
                        split_chunk["context_confidence"].sum()
                    )
                    split_confidence_count[split] += int(split_rows)

                    score_input = (
                        split_chunk[gmm_score_columns]
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.0)
                        .astype(np.float32)
                    )

                    try:
                        log_likelihood_values = gmm_model.score_samples(score_input)
                        split_likelihood_sum[split] += float(
                            np.sum(log_likelihood_values, dtype=np.float64)
                        )
                        split_likelihood_count[split] += int(len(log_likelihood_values))
                    except Exception as exc:
                        print(
                            f"[WARNING] GMM score_samples failed for split={split}: {exc}"
                        )

                    sample_columns = (
                        ["unit_id", "cycle", "split"]
                        + w_columns
                        + [
                            "kmeans_context_id",
                            "gmm_context_id",
                            "context_confidence",
                        ]
                    )

                    split_samples[split] = self._sample_for_split(
                        split_df=split_chunk[sample_columns],
                        current_sample=split_samples[split],
                    )

                    del score_input
                    del split_chunk
                    gc.collect()

                del chunk
                gc.collect()

            records: List[Dict[str, object]] = []

            for split in split_names:
                sample_df = split_samples[split]

                if sample_df is None or sample_df.empty:
                    logger.warning("No sampled rows found for split=%s.", split)
                    continue

                print("=" * 100)
                print(f"[PROGRESS] Calculating sampled context metrics for split={split}")
                print(f"[PROGRESS] Sample rows: {len(sample_df)}")

                x = (
                    sample_df[w_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .astype(np.float32)
                )

                kmeans_labels = sample_df["kmeans_context_id"]
                gmm_labels = sample_df["gmm_context_id"]

                kmeans_silhouette = self._safe_silhouette(x, kmeans_labels)
                gmm_silhouette = self._safe_silhouette(x, gmm_labels)

                kmeans_db = self._safe_davies_bouldin(x, kmeans_labels)
                gmm_db = self._safe_davies_bouldin(x, gmm_labels)

                avg_confidence = (
                    split_confidence_sum[split] / max(split_confidence_count[split], 1)
                )

                avg_log_likelihood = (
                    split_likelihood_sum[split] / max(split_likelihood_count[split], 1)
                )

                records.append(
                    {
                        "split": split,
                        "sample_rows": int(len(sample_df)),
                        "total_rows_seen_for_split": int(split_confidence_count[split]),
                        "kmeans_silhouette_sampled": float(kmeans_silhouette),
                        "gmm_silhouette_sampled": float(gmm_silhouette),
                        "kmeans_davies_bouldin_sampled": float(kmeans_db),
                        "gmm_davies_bouldin_sampled": float(gmm_db),
                        "gmm_average_log_likelihood": float(avg_log_likelihood),
                        "average_context_confidence": float(avg_confidence),
                        "w_columns_used": ",".join(w_columns),
                        "gmm_score_columns_used": ",".join(gmm_score_columns),
                        "evaluation_mode": "chunked_confidence_likelihood_plus_sampled_cluster_metrics",
                    }
                )

                del sample_df
                gc.collect()

            metrics_df = pd.DataFrame(records)

            duration = perf_counter() - started

            print("[PROGRESS] Context evaluation completed")
            print(f"[PROGRESS] Metrics rows: {len(metrics_df)}")
            print(f"[PROGRESS] Total rows scanned: {total_rows_seen}")
            print(f"[PROGRESS] Duration seconds: {duration:.2f}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            logger.info("Context evaluation completed. rows=%s", len(metrics_df))
            return metrics_df

        except Exception as exc:
            logger.exception("Context evaluation failed.")
            raise RuntimeError("Context evaluation failed.") from exc

    def summarize(self, metrics_df: pd.DataFrame) -> Dict[str, object]:
        """
        Summarize context evaluation.

        Args:
            metrics_df: Metrics DataFrame.

        Returns:
            Summary dictionary.
        """
        print("[PROGRESS] Entering ContextEvaluator.summarize")

        try:
            if metrics_df.empty:
                return {
                    "status": "warning",
                    "message": "No context metrics were generated.",
                }

            return {
                "status": "success",
                "evaluation_mode": "memory_safe_sampled_context_evaluation",
                "average_gmm_context_confidence": float(
                    metrics_df["average_context_confidence"].mean()
                ),
                "average_gmm_silhouette_sampled": float(
                    metrics_df["gmm_silhouette_sampled"].mean()
                ),
                "average_kmeans_silhouette_sampled": float(
                    metrics_df["kmeans_silhouette_sampled"].mean()
                ),
                "average_gmm_davies_bouldin_sampled": float(
                    metrics_df["gmm_davies_bouldin_sampled"].mean()
                ),
                "average_kmeans_davies_bouldin_sampled": float(
                    metrics_df["kmeans_davies_bouldin_sampled"].mean()
                ),
                "average_gmm_log_likelihood": float(
                    metrics_df["gmm_average_log_likelihood"].mean()
                ),
                "records_count": int(len(metrics_df)),
            }

        except Exception as exc:
            logger.exception("Context summary generation failed.")
            raise RuntimeError("Context summary generation failed.") from exc

    def run(self) -> Dict[str, object]:
        """
        Run context evaluation.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering ContextEvaluator.run")

        try:
            metrics_df = self.evaluate()

            output_path: Path = Config.METRIC_DIR / "evaluate_context.csv"
            summary_path: Path = Config.REPORT_DIR / "evaluate_context_summary.json"

            print(f"[PROGRESS] Writing context metrics CSV to: {output_path}")
            atomic_write_csv(metrics_df, output_path)

            summary = self.summarize(metrics_df)

            print(f"[PROGRESS] Writing context summary JSON to: {summary_path}")
            atomic_write_json(summary, summary_path)

            response = {
                "status": "success",
                "message": "Memory-safe context modeling evaluation completed.",
                "output_file": str(output_path),
                "summary_file": str(summary_path),
                "records_count": int(len(metrics_df)),
                "metrics": summary,
            }

            print(f"[PROGRESS] Context evaluator response: {response}")
            return response

        except Exception as exc:
            logger.exception("Context evaluator stage failed.")
            raise RuntimeError("Context evaluator stage failed.") from exc


def run_context_evaluation() -> Dict[str, object]:
    """
    Execute context evaluation.

    Returns:
        Stage response.
    """
    print("[PROGRESS] Entering run_context_evaluation")
    evaluator = ContextEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    result = run_context_evaluation()
    print(result)