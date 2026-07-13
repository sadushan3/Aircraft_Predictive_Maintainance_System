"""
MLP Digital Twin Regressor for CA-EDT-AHMA.

CA-EDT-AHMA:
Context-Aware Ensemble Digital Twin for Explainable Health Monitoring
and Anomaly Reasoning.

Model:
- Multi-Layer Perceptron Digital Twin Regressor

Library:
- TensorFlow/Keras

Research-correct data usage:
- Input  = W + X_v + gmm_context_id
- Target = raw measured X_s sensors only
- Train  = dev split only
- Test   = prediction/evaluation only
- Ignore = Y_dev/Y_test
- Ignore = T_dev/T_test as model input
- Do not use X_s as input features

Memory-safety:
- Does not load full scaled_features.csv into RAM.
- Reads scaled_features.csv and context_clusters.csv in chunks.
- Uses tf.data streaming for full-dev mini-batch training.
- Repeats tf.data datasets safely when steps_per_epoch is used.
- Writes predictions batch-by-batch to a temporary CSV.
- Replaces final prediction CSV only after successful completion.

Output:
- mlp_twin_predictions.csv / TF_PREDICTIONS_CSV alias

Prediction columns:
- tf_predicted_Xs_T24
- tf_predicted_Xs_T30
- ...
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Generator, List, Tuple
import gc
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


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
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger
from app.utils.Anomaly_Health_Monitering.model_utils import (
    get_raw_xs_columns,
    get_w_columns,
    get_xv_columns,
)


logger = get_logger(__name__)


class MLPDigitalTwin:
    """
    Full-dev memory-safe MLP Digital Twin Regressor.

    This class uses TensorFlow/Keras as the implementation framework.
    The model itself is an MLP regressor.
    """

    def __init__(
        self,
        train_chunk_size: int = 50_000,
        prediction_chunk_size: int = 50_000,
        batch_size: int = 4096,
        epochs: int = 20,
    ) -> None:
        """
        Initialize MLP Digital Twin.

        Args:
            train_chunk_size: Number of CSV rows read per training chunk.
            prediction_chunk_size: Number of CSV rows read per prediction chunk.
            batch_size: TensorFlow mini-batch size.
            epochs: Maximum number of training epochs.
        """
        Config.create_directories()

        self.train_chunk_size = int(
            getattr(
                Config,
                "MLP_TWIN_TRAIN_CHUNK_SIZE",
                getattr(Config, "TF_TRAIN_CHUNK_SIZE", train_chunk_size),
            )
        )
        self.prediction_chunk_size = int(
            getattr(
                Config,
                "MLP_TWIN_PREDICTION_BATCH_SIZE",
                getattr(Config, "TF_PREDICTION_BATCH_SIZE", prediction_chunk_size),
            )
        )
        self.batch_size = int(
            getattr(
                Config,
                "MLP_TWIN_BATCH_SIZE",
                getattr(Config, "TF_BATCH_SIZE", batch_size),
            )
        )
        self.epochs = int(
            getattr(
                Config,
                "MLP_TWIN_EPOCHS",
                getattr(Config, "TF_EPOCHS", epochs),
            )
        )
        self.learning_rate = float(
            getattr(
                Config,
                "MLP_TWIN_LEARNING_RATE",
                getattr(Config, "TF_LEARNING_RATE", 0.001),
            )
        )
        self.validation_fraction = float(
            getattr(
                Config,
                "MLP_TWIN_VALIDATION_FRACTION",
                getattr(Config, "TF_VALIDATION_FRACTION", 0.10),
            )
        )
        self.random_seed = int(
            getattr(
                Config,
                "MLP_TWIN_RANDOM_SEED",
                getattr(Config, "TF_RANDOM_SEED", getattr(Config, "RANDOM_SEED", 42)),
            )
        )

        self.hidden_units: Tuple[int, ...] = tuple(
            getattr(Config, "MLP_TWIN_HIDDEN_UNITS", (256, 256, 128, 64))
        )
        self.dropout_rate = float(getattr(Config, "MLP_TWIN_DROPOUT_RATE", 0.10))
        self.use_batch_norm = bool(getattr(Config, "MLP_TWIN_USE_BATCH_NORM", True))
        self.early_stopping_patience = int(
            getattr(Config, "MLP_TWIN_EARLY_STOPPING_PATIENCE", 4)
        )
        self.reduce_lr_patience = int(getattr(Config, "MLP_TWIN_REDUCE_LR_PATIENCE", 2))
        self.min_learning_rate = float(getattr(Config, "MLP_TWIN_MIN_LEARNING_RATE", 1e-6))

        self.model_path: Path = getattr(
            Config,
            "MLP_TWIN_MODEL_PATH",
            getattr(
                Config,
                "TF_MODEL_PATH",
                Config.DIGITAL_TWIN_MODEL_DIR / "mlp_digital_twin.keras",
            ),
        )
        self.metadata_path: Path = getattr(
            Config,
            "MLP_TWIN_METADATA_PATH",
            getattr(
                Config,
                "TF_MODEL_METADATA_PATH",
                Config.DIGITAL_TWIN_MODEL_DIR / "mlp_digital_twin_metadata.json",
            ),
        )
        self.predictions_csv: Path = getattr(
            Config,
            "MLP_TWIN_PREDICTIONS_CSV",
            getattr(
                Config,
                "TF_PREDICTIONS_CSV",
                Config.OUTPUT_DIR / "mlp_twin_predictions.csv",
            ),
        )

        self.prediction_prefix = str(
            getattr(Config, "ACTIVE_DIGITAL_TWIN_PREDICTION_PREFIX", "tf_predicted_")
        )

        self._validate_runtime_config()

        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

        logger.info(
            "MLPDigitalTwin initialized. train_chunk=%s prediction_chunk=%s "
            "batch_size=%s epochs=%s model_path=%s predictions_csv=%s",
            self.train_chunk_size,
            self.prediction_chunk_size,
            self.batch_size,
            self.epochs,
            self.model_path,
            self.predictions_csv,
        )

        print("[PROGRESS] MLP Digital Twin initialized")
        print(f"[PROGRESS] Train chunk size: {self.train_chunk_size}")
        print(f"[PROGRESS] Prediction chunk size: {self.prediction_chunk_size}")
        print(f"[PROGRESS] Batch size: {self.batch_size}")
        print(f"[PROGRESS] Epochs: {self.epochs}")
        print(f"[PROGRESS] Learning rate: {self.learning_rate}")
        print(f"[PROGRESS] Validation fraction: {self.validation_fraction}")
        print(f"[PROGRESS] Model path: {self.model_path}")
        print(f"[PROGRESS] Predictions CSV: {self.predictions_csv}")

    def _validate_runtime_config(self) -> None:
        """
        Validate runtime configuration values.
        """
        if self.train_chunk_size <= 0:
            raise ValueError("train_chunk_size must be positive.")

        if self.prediction_chunk_size <= 0:
            raise ValueError("prediction_chunk_size must be positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in the range [0.0, 0.5).")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

    # ==================================================================================
    # Column preparation and validation
    # ==================================================================================

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        """
        Read CSV header only.

        Args:
            path: CSV file path.

        Returns:
            Empty DataFrame containing only column names.
        """
        print(f"[PROGRESS] Reading CSV header from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        """
        Read CSV column names only.

        Args:
            path: CSV file path.

        Returns:
            List of column names.
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
        Validate required columns exist.

        Args:
            available_columns: Columns available in file.
            required_columns: Columns required by this stage.
            label: Human-readable file/stage label.
        """
        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            raise KeyError(f"Missing required columns for {label}: {missing}")

    def _verify_key_alignment(
        self,
        scaled_chunk: pd.DataFrame,
        context_chunk: pd.DataFrame,
        merge_columns: List[str],
    ) -> None:
        """
        Verify that scaled and context chunks are row-aligned.

        This avoids expensive many-to-many merges and assumes context_clusters.csv
        was generated from the same row order as scaled_features.csv.

        Args:
            scaled_chunk: Chunk from scaled_features.csv.
            context_chunk: Chunk from context_clusters.csv.
            merge_columns: Row identity columns.
        """
        if len(scaled_chunk) != len(context_chunk):
            raise ValueError(
                "Scaled/context chunk row count mismatch: "
                f"scaled={len(scaled_chunk)}, context={len(context_chunk)}"
            )

        scaled_keys = scaled_chunk[merge_columns].reset_index(drop=True)
        context_keys = context_chunk[merge_columns].reset_index(drop=True)

        if not scaled_keys.equals(context_keys):
            raise ValueError(
                "scaled_features.csv and context_clusters.csv are not row-aligned. "
                "Regenerate context_clusters.csv from the same scaled_features.csv order."
            )

    def prepare_columns(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Prepare metadata, feature, and target columns.

        Returns:
            Tuple containing:
            - merge_columns
            - feature_columns
            - target_columns
        """
        print("[PROGRESS] Preparing MLP Digital Twin columns")

        merge_columns = ["unit_id", "cycle", "split"]

        scaled_header_df = self._read_header_df(Config.SCALED_CSV)
        scaled_columns = list(scaled_header_df.columns)
        context_columns = self._read_header_columns(Config.CONTEXT_CSV)

        self._validate_columns(
            available_columns=scaled_columns,
            required_columns=merge_columns,
            label="scaled_features.csv",
        )
        self._validate_columns(
            available_columns=context_columns,
            required_columns=merge_columns + ["gmm_context_id"],
            label="context_clusters.csv",
        )

        w_columns = get_w_columns(scaled_header_df)
        xv_columns = get_xv_columns(scaled_header_df)
        target_columns = get_raw_xs_columns(scaled_header_df)

        if not w_columns:
            raise ValueError("No W operating-condition columns found.")

        if not xv_columns:
            raise ValueError("No X_v virtual sensor columns found.")

        if not target_columns:
            raise ValueError("No raw X_s target columns found.")

        feature_columns = w_columns + xv_columns + ["gmm_context_id"]

        leakage_features = [
            column
            for column in feature_columns
            if column.startswith(("Xs_", "X_s_", "Y_", "T_"))
        ]

        if leakage_features:
            raise ValueError(f"Leakage-risk input feature columns found: {leakage_features}")

        engineered_target_tokens = (
            "rolling",
            "trend",
            "lag",
            "delta",
            "diff",
            "mean",
            "std",
            "var",
            "min",
            "max",
        )

        engineered_targets = [
            column
            for column in target_columns
            if any(token in column.lower() for token in engineered_target_tokens)
        ]

        if engineered_targets:
            raise ValueError(
                "Engineered target columns found. Digital twin target must be raw X_s only: "
                f"{engineered_targets}"
            )

        print(f"[PROGRESS] Feature columns count: {len(feature_columns)}")
        print(f"[PROGRESS] Target columns count: {len(target_columns)}")
        print(f"[PROGRESS] Feature columns: {feature_columns}")
        print(f"[PROGRESS] Target columns: {target_columns}")

        return merge_columns, feature_columns, target_columns

    # ==================================================================================
    # Row counting and deterministic dev validation split
    # ==================================================================================

    def _make_validation_mask(self, row_count: int) -> np.ndarray:
        """
        Build deterministic validation mask for a dev-only chunk.

        Args:
            row_count: Number of dev rows in the current chunk.

        Returns:
            Boolean mask where True means validation row.
        """
        if self.validation_fraction <= 0.0:
            return np.zeros(row_count, dtype=bool)

        validation_period = max(int(round(1.0 / self.validation_fraction)), 1)
        local_positions = np.arange(row_count)

        return (local_positions % validation_period) == 0

    def count_rows(self, merge_columns: List[str]) -> Dict[str, int]:
        """
        Count total, dev, test, train-dev, and validation-dev rows.

        Args:
            merge_columns: Row identity columns.

        Returns:
            Row count dictionary.
        """
        print("[PROGRESS] Counting rows for MLP Digital Twin")

        counts = {
            "total": 0,
            "dev": 0,
            "test": 0,
            "train_dev": 0,
            "val_dev": 0,
        }

        chunk_index = 0

        for chunk in pd.read_csv(
            Config.SCALED_CSV,
            usecols=merge_columns,
            chunksize=self.train_chunk_size,
            low_memory=True,
        ):
            chunk_index += 1
            counts["total"] += len(chunk)

            dev_mask = chunk["split"] == Config.DEV_SPLIT_NAME
            test_mask = chunk["split"] == Config.TEST_SPLIT_NAME

            dev_rows = int(dev_mask.sum())
            test_rows = int(test_mask.sum())

            counts["dev"] += dev_rows
            counts["test"] += test_rows

            if dev_rows > 0:
                val_mask = self._make_validation_mask(dev_rows)
                val_rows = int(val_mask.sum())
                train_rows = int(dev_rows - val_rows)

                counts["val_dev"] += val_rows
                counts["train_dev"] += train_rows

            print(
                f"[PROGRESS] Count chunk {chunk_index}: "
                f"total={counts['total']} dev={counts['dev']} test={counts['test']} "
                f"train_dev={counts['train_dev']} val_dev={counts['val_dev']}"
            )

            del chunk
            gc.collect()

        if counts["dev"] <= 0:
            raise ValueError("No dev rows found.")

        if counts["train_dev"] <= 0:
            raise ValueError("No training dev rows found after validation split.")

        print(f"[PROGRESS] Final row counts: {counts}")

        return counts

    # ==================================================================================
    # tf.data dataset creation
    # ==================================================================================

    def _chunk_pair_iterator(
        self,
        merge_columns: List[str],
        feature_columns: List[str],
        target_columns: List[str],
        split_mode: str,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield X/y arrays from chunked CSVs.

        Args:
            merge_columns: Row identity columns.
            feature_columns: Model input columns.
            target_columns: Raw X_s target columns.
            split_mode: "train" or "val".

        Yields:
            Tuple of X and y NumPy arrays.
        """
        if split_mode not in {"train", "val"}:
            raise ValueError("split_mode must be 'train' or 'val'.")

        scaled_usecols = list(
            dict.fromkeys(
                merge_columns
                + [column for column in feature_columns if column != "gmm_context_id"]
                + target_columns
            )
        )
        context_usecols = merge_columns + ["gmm_context_id"]

        scaled_iter = pd.read_csv(
            Config.SCALED_CSV,
            usecols=scaled_usecols,
            chunksize=self.train_chunk_size,
            low_memory=True,
        )

        context_iter = pd.read_csv(
            Config.CONTEXT_CSV,
            usecols=context_usecols,
            chunksize=self.train_chunk_size,
            low_memory=True,
        )

        for scaled_chunk, context_chunk in zip(scaled_iter, context_iter):
            self._verify_key_alignment(
                scaled_chunk=scaled_chunk,
                context_chunk=context_chunk,
                merge_columns=merge_columns,
            )

            scaled_chunk["gmm_context_id"] = context_chunk["gmm_context_id"].values

            dev_mask = scaled_chunk["split"] == Config.DEV_SPLIT_NAME

            if not dev_mask.any():
                del scaled_chunk
                del context_chunk
                gc.collect()
                continue

            dev_chunk = scaled_chunk.loc[
                dev_mask,
                feature_columns + target_columns,
            ].copy()

            validation_mask = self._make_validation_mask(len(dev_chunk))

            if split_mode == "train":
                selected = dev_chunk.loc[~validation_mask]
            else:
                selected = dev_chunk.loc[validation_mask]

            if selected.empty:
                del scaled_chunk
                del context_chunk
                del dev_chunk
                gc.collect()
                continue

            x_array = (
                selected[feature_columns]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )

            y_array = (
                selected[target_columns]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )

            yield x_array, y_array

            del scaled_chunk
            del context_chunk
            del dev_chunk
            del selected
            del x_array
            del y_array
            gc.collect()

    def build_dataset(
        self,
        merge_columns: List[str],
        feature_columns: List[str],
        target_columns: List[str],
        split_mode: str,
        repeat: bool = False,
    ) -> tf.data.Dataset:
        """
        Build tf.data Dataset from CSV streaming generator.

        Args:
            merge_columns: Row identity columns.
            feature_columns: Model input columns.
            target_columns: Raw X_s target columns.
            split_mode: "train" or "val".
            repeat: If True, repeat the dataset so Keras has enough batches
                for steps_per_epoch * epochs.

        Returns:
            Batched TensorFlow dataset.
        """
        n_features = len(feature_columns)
        n_targets = len(target_columns)

        output_signature = (
            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, n_targets), dtype=tf.float32),
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: self._chunk_pair_iterator(
                merge_columns=merge_columns,
                feature_columns=feature_columns,
                target_columns=target_columns,
                split_mode=split_mode,
            ),
            output_signature=output_signature,
        )

        dataset = dataset.unbatch()
        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    # ==================================================================================
    # Model build/train
    # ==================================================================================

    def build_model(self, n_features: int, n_targets: int) -> tf.keras.Model:
        """
        Build and compile MLP Digital Twin model.

        Args:
            n_features: Number of input features.
            n_targets: Number of raw X_s output targets.

        Returns:
            Compiled Keras model.
        """
        print("[PROGRESS] Building MLP Digital Twin Regressor")

        inputs = tf.keras.Input(shape=(n_features,), name="digital_twin_inputs")
        x = inputs

        for layer_index, units in enumerate(self.hidden_units, start=1):
            x = tf.keras.layers.Dense(
                units,
                activation="relu",
                name=f"dense_{layer_index}_{units}",
            )(x)

            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(
                    name=f"batch_norm_{layer_index}_{units}"
                )(x)

            if self.dropout_rate > 0.0 and layer_index <= 2:
                x = tf.keras.layers.Dropout(
                    self.dropout_rate,
                    name=f"dropout_{layer_index}_{units}",
                )(x)

        outputs = tf.keras.layers.Dense(
            n_targets,
            activation="linear",
            name="raw_xs_predictions",
        )(x)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="CA_EDT_AHMA_MLP_Digital_Twin",
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )

        model.summary(print_fn=lambda line: print(f"[MODEL] {line}"))

        return model

    def train(self) -> Dict[str, object]:
        """
        Train MLP Digital Twin on dev split only.

        Returns:
            Training metadata dictionary.
        """
        print("[PROGRESS] Entering MLPDigitalTwin.train")

        try:
            started = perf_counter()

            merge_columns, feature_columns, target_columns = self.prepare_columns()
            counts = self.count_rows(merge_columns=merge_columns)

            steps_per_epoch = math.ceil(counts["train_dev"] / self.batch_size)
            validation_steps = (
                math.ceil(counts["val_dev"] / self.batch_size)
                if counts["val_dev"] > 0
                else None
            )

            print(f"[TRAINING] steps_per_epoch: {steps_per_epoch}")
            print(f"[TRAINING] validation_steps: {validation_steps}")

            train_dataset = self.build_dataset(
                merge_columns=merge_columns,
                feature_columns=feature_columns,
                target_columns=target_columns,
                split_mode="train",
                repeat=True,
            )

            val_dataset = None
            if counts["val_dev"] > 0:
                val_dataset = self.build_dataset(
                    merge_columns=merge_columns,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    split_mode="val",
                    repeat=True,
                )

            model = self.build_model(
                n_features=len(feature_columns),
                n_targets=len(target_columns),
            )

            monitor_metric = "val_loss" if val_dataset is not None else "loss"

            callbacks: List[tf.keras.callbacks.Callback] = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.model_path),
                    monitor=monitor_metric,
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor_metric,
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]

            print("[TRAINING] MLP Digital Twin full-dev training started")
            print(f"[TRAINING] train_dev rows: {counts['train_dev']}")
            print(f"[TRAINING] val_dev rows: {counts['val_dev']}")
            print(f"[TRAINING] total dev rows: {counts['dev']}")
            print("[TRAINING] test rows used for training: 0")
            print(f"[TRAINING] feature count: {len(feature_columns)}")
            print(f"[TRAINING] raw target count: {len(target_columns)}")
            print(f"[TRAINING] epochs: {self.epochs}")
            print(f"[TRAINING] batch_size: {self.batch_size}")

            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
            )

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(self.model_path)

            elapsed = perf_counter() - started

            metadata = {
                "model_name": "mlp_digital_twin_regressor",
                "model_type": "Multi-Layer Perceptron Regressor",
                "library": "TensorFlow/Keras",
                "model_path": str(self.model_path),
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "prediction_prefix": self.prediction_prefix,
                "target_type": "raw_X_s_only",
                "input_type": "W_plus_Xv_plus_gmm_context_id",
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "predict_only",
                "training_mode": "full_dev_tf_data_streaming",
                "train_rows": int(counts["train_dev"]),
                "validation_rows": int(counts["val_dev"]),
                "total_dev_rows": int(counts["dev"]),
                "test_rows": int(counts["test"]),
                "test_rows_used_for_training": 0,
                "steps_per_epoch": int(steps_per_epoch),
                "validation_steps": int(validation_steps) if validation_steps is not None else None,
                "epochs_requested": int(self.epochs),
                "epochs_completed": int(len(history.history.get("loss", []))),
                "batch_size": int(self.batch_size),
                "learning_rate": float(self.learning_rate),
                "validation_fraction": float(self.validation_fraction),
                "hidden_units": list(self.hidden_units),
                "dropout_rate": float(self.dropout_rate),
                "use_batch_norm": bool(self.use_batch_norm),
                "duration_seconds": float(elapsed),
                "history": {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                },
                "leakage_audit": {
                    "uses_y_targets": False,
                    "uses_t_degradation_as_input": False,
                    "uses_xs_as_input": False,
                    "target_is_raw_xs_only": True,
                    "train_on_test": False,
                },
            }

            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)

            print("[TRAINING] MLP Digital Twin training completed")
            print(f"[TRAINING] Model saved to: {self.model_path}")
            print(f"[TRAINING] Metadata saved to: {self.metadata_path}")
            print(f"[TRAINING] Duration seconds: {elapsed:.2f}")
            print(f"[TRAINING] Duration minutes: {elapsed / 60.0:.2f}")

            logger.info(
                "MLP Digital Twin trained. train_rows=%s val_rows=%s targets=%s duration=%.2fs",
                counts["train_dev"],
                counts["val_dev"],
                len(target_columns),
                elapsed,
            )

            return metadata

        except Exception as exc:
            print(f"[ERROR] MLP Digital Twin training failed: {exc}")
            logger.exception("MLP Digital Twin training failed.")
            raise RuntimeError("MLP Digital Twin training failed.") from exc

    # ==================================================================================
    # Prediction
    # ==================================================================================

    def _prediction_chunk_iterator(
        self,
        merge_columns: List[str],
        feature_columns: List[str],
    ) -> Generator[Tuple[pd.DataFrame, np.ndarray], None, None]:
        """
        Yield metadata and feature arrays for prediction.

        Args:
            merge_columns: Row identity columns.
            feature_columns: Model input columns.

        Yields:
            Tuple of metadata DataFrame and feature array.
        """
        scaled_usecols = list(
            dict.fromkeys(
                merge_columns
                + [column for column in feature_columns if column != "gmm_context_id"]
            )
        )
        context_usecols = merge_columns + ["gmm_context_id"]

        scaled_iter = pd.read_csv(
            Config.SCALED_CSV,
            usecols=scaled_usecols,
            chunksize=self.prediction_chunk_size,
            low_memory=True,
        )

        context_iter = pd.read_csv(
            Config.CONTEXT_CSV,
            usecols=context_usecols,
            chunksize=self.prediction_chunk_size,
            low_memory=True,
        )

        for scaled_chunk, context_chunk in zip(scaled_iter, context_iter):
            self._verify_key_alignment(
                scaled_chunk=scaled_chunk,
                context_chunk=context_chunk,
                merge_columns=merge_columns,
            )

            scaled_chunk["gmm_context_id"] = context_chunk["gmm_context_id"].values

            metadata_df = scaled_chunk[merge_columns].copy()

            x_array = (
                scaled_chunk[feature_columns]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )

            yield metadata_df, x_array

            del scaled_chunk
            del context_chunk
            del metadata_df
            del x_array
            gc.collect()

    def predict(self) -> Dict[str, object]:
        """
        Predict raw X_s values for all dev and test rows.

        Returns:
            Prediction response dictionary.
        """
        print("[PROGRESS] Entering MLPDigitalTwin.predict")

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"MLP Digital Twin model not found: {self.model_path}")

            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"MLP Digital Twin metadata not found: {self.metadata_path}"
                )

            with self.metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

            merge_columns = ["unit_id", "cycle", "split"]
            feature_columns: List[str] = metadata["feature_columns"]
            target_columns: List[str] = metadata["target_columns"]
            prediction_prefix = str(metadata.get("prediction_prefix", self.prediction_prefix))

            print(f"[PROGRESS] Loading MLP Digital Twin model from: {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)

            output_path = self.predictions_csv
            temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                temp_output_path.unlink()

            prediction_columns = [
                f"{prediction_prefix}{sensor}" for sensor in target_columns
            ]

            first_batch = True
            total_rows = 0
            chunk_index = 0

            print("[PROGRESS] Starting streaming prediction for all rows")

            for metadata_df, x_batch in self._prediction_chunk_iterator(
                merge_columns=merge_columns,
                feature_columns=feature_columns,
            ):
                chunk_index += 1
                start_row = total_rows
                end_row = total_rows + len(metadata_df)

                print("=" * 100)
                print(f"[PROGRESS] Prediction chunk #{chunk_index}")
                print(f"[PROGRESS] Rows {start_row} to {end_row}")
                print(f"[PROGRESS] X batch shape: {x_batch.shape}")

                predictions = model.predict(
                    x_batch,
                    batch_size=self.batch_size,
                    verbose=0,
                ).astype(np.float32)

                batch_result = metadata_df.copy()

                for column_index, prediction_column in enumerate(prediction_columns):
                    batch_result[prediction_column] = predictions[:, column_index]

                batch_result.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows += len(batch_result)

                print(f"[PROGRESS] Total prediction rows written: {total_rows}")

                del metadata_df
                del x_batch
                del predictions
                del batch_result
                gc.collect()

            os.replace(temp_output_path, output_path)

            print(f"[PROGRESS] Predictions saved to: {output_path}")
            print(f"[PROGRESS] Total prediction rows: {total_rows}")

            logger.info("MLP Digital Twin prediction completed. rows=%s", total_rows)

            return {
                "status": "success",
                "message": "MLP Digital Twin predictions generated for all splits.",
                "output_file": str(output_path),
                "records_count": int(total_rows),
            }

        except Exception as exc:
            print(f"[ERROR] MLP Digital Twin prediction failed: {exc}")
            logger.exception("MLP Digital Twin prediction failed.")
            raise RuntimeError("MLP Digital Twin prediction failed.") from exc

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Train and predict using the MLP Digital Twin.

        Returns:
            Stage response dictionary.
        """
        print("[PROGRESS] Entering MLPDigitalTwin.run")

        try:
            train_metadata = self.train()
            prediction_response = self.predict()

            response = {
                "status": "success",
                "message": (
                    "MLP Digital Twin Regressor trained on full dev split using "
                    "TensorFlow/Keras streaming mini-batches and inferred for all splits. "
                    "Targets are raw X_s sensors only."
                ),
                "output_file": prediction_response["output_file"],
                "records_count": prediction_response["records_count"],
                "model_name": train_metadata["model_name"],
                "library": train_metadata["library"],
                "training_mode": train_metadata["training_mode"],
                "train_rows": train_metadata["train_rows"],
                "validation_rows": train_metadata["validation_rows"],
                "target_type": train_metadata["target_type"],
            }

            print(f"[PROGRESS] MLP Digital Twin response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] MLP Digital Twin stage failed: {exc}")
            logger.exception("MLP Digital Twin stage failed.")
            raise RuntimeError("MLP Digital Twin stage failed.") from exc


# ======================================================================================
# Backward-compatible aliases
# ======================================================================================

TensorFlowTwin = MLPDigitalTwin


def run_mlp_digital_twin() -> Dict[str, object]:
    """
    Execute MLP Digital Twin stage.

    Returns:
        Stage response dictionary.
    """
    service = MLPDigitalTwin()
    return service.run()


def run_tensorflow_twin() -> Dict[str, object]:
    """
    Backward-compatible function name for old pipeline calls.

    Returns:
        Stage response dictionary.
    """
    return run_mlp_digital_twin()


if __name__ == "__main__":
    print("[PROGRESS] mlp_digital_twin.py / tensorflow_twin.py execution started")
    result = run_mlp_digital_twin()
    print("[PROGRESS] mlp_digital_twin.py / tensorflow_twin.py execution finished successfully")
    print(result)