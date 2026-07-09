"""
LSTM Autoencoder anomaly detector for CA-EDT-AHMA.

Purpose:
Deep temporal anomaly detector using residual sequence reconstruction error.

Correct research version:
Input features:
1. raw residual_Xs_* columns
2. residual temporal resfeat_* columns

Training:
- Fit scaler on dev rows only.
- Train LSTM Autoencoder on dev sequences only.
- Uses all available dev sequences through a streaming tf.data generator.
- Does not train on test.

Inference:
- Scores dev and test rows.
- Produces one row-aligned score per residual row.
- Warm-up rows without enough history receive score=0 and sequence_ready=0.

Reads:
- outputs/Anomaly_Health_Monitering/residuals.csv

Writes:
- outputs/Anomaly_Health_Monitering/lstm_autoencoder_scores.csv

Saves:
- models/anomaly/lstm_autoencoder.keras
- models/anomaly/lstm_autoencoder_scaler.pkl
- models/anomaly/lstm_autoencoder_metadata.json

Important:
- TensorFlow is the library.
- The model is LSTM Autoencoder.
- Uses residual behavior, not raw X_s as direct anomaly input.
"""

from __future__ import annotations

print(
    "[PROGRESS] Loaded Backend/app/services/Anomaly_Health_Monitering/"
    "anomaly_detection/lstm_autoencoder_detector.py"
)

from pathlib import Path
from time import perf_counter
from typing import Dict, Generator, Iterable, List, Tuple
import gc
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
except Exception:
    tf = None
    Model = object
    EarlyStopping = None
    ModelCheckpoint = None
    ReduceLROnPlateau = None
    Input = None
    LSTM = None
    RepeatVector = None
    TimeDistributed = None
    Dense = None
    Dropout = None
    Adam = None
    load_model = None


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
    atomic_save_joblib,
    atomic_write_json,
    load_joblib_required,
)
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger


logger = get_logger(__name__)


class LSTMAutoencoderDetector:
    """
    Memory-safe LSTM Autoencoder residual sequence anomaly detector.
    """

    def __init__(
        self,
        chunk_size: int = 50_000,
        sequence_length: int = 20,
        batch_size: int = 1024,
        epochs: int = 15,
    ) -> None:
        """
        Initialize LSTM Autoencoder detector.

        Args:
            chunk_size: CSV rows read per chunk.
            sequence_length: Number of timesteps per residual sequence.
            batch_size: Training/prediction batch size.
            epochs: Maximum training epochs.
        """
        print("[PROGRESS] Entering LSTMAutoencoderDetector.__init__")

        if tf is None:
            raise ImportError(
                "TensorFlow is not installed. Install tensorflow to use LSTM Autoencoder."
            )

        Config.create_directories()

        self.chunk_size = int(
            getattr(Config, "LSTM_AE_CHUNK_SIZE", chunk_size)
        )
        self.sequence_length = int(
            getattr(Config, "LSTM_AE_SEQUENCE_LENGTH", sequence_length)
        )
        self.batch_size = int(
            getattr(Config, "LSTM_AE_BATCH_SIZE", batch_size)
        )
        self.epochs = int(
            getattr(Config, "LSTM_AE_EPOCHS", epochs)
        )

        self.learning_rate = float(
            getattr(Config, "LSTM_AE_LEARNING_RATE", 0.001)
        )
        self.validation_fraction = float(
            getattr(Config, "LSTM_AE_VALIDATION_FRACTION", 0.10)
        )
        self.threshold_percentile = float(
            getattr(
                Config,
                "LSTM_AE_THRESHOLD_PERCENTILE",
                getattr(Config, "RESIDUAL_AUTOENCODER_THRESHOLD_PERCENTILE", 99.0),
            )
        )

        self.encoder_units = tuple(
            getattr(Config, "LSTM_AE_ENCODER_UNITS", (128, 64))
        )
        self.latent_units = int(
            getattr(Config, "LSTM_AE_LATENT_UNITS", 32)
        )
        self.dropout_rate = float(
            getattr(Config, "LSTM_AE_DROPOUT_RATE", 0.10)
        )

        self.early_stopping_patience = int(
            getattr(Config, "LSTM_AE_EARLY_STOPPING_PATIENCE", 4)
        )
        self.reduce_lr_patience = int(
            getattr(Config, "LSTM_AE_REDUCE_LR_PATIENCE", 2)
        )
        self.min_learning_rate = float(
            getattr(Config, "LSTM_AE_MIN_LEARNING_RATE", 1e-6)
        )

        self.random_seed = int(getattr(Config, "RANDOM_SEED", 42))

        self.model_path: Path = getattr(
            Config,
            "LSTM_AUTOENCODER_MODEL_PATH",
            Config.ANOMALY_MODEL_DIR / "lstm_autoencoder.keras",
        )
        self.scaler_path: Path = getattr(
            Config,
            "LSTM_AUTOENCODER_SCALER_PATH",
            Config.ANOMALY_MODEL_DIR / "lstm_autoencoder_scaler.pkl",
        )
        self.metadata_path: Path = getattr(
            Config,
            "LSTM_AUTOENCODER_METADATA_PATH",
            Config.ANOMALY_MODEL_DIR / "lstm_autoencoder_metadata.json",
        )
        self.output_csv: Path = getattr(
            Config,
            "LSTM_AUTOENCODER_CSV",
            Config.OUTPUT_DIR / "lstm_autoencoder_scores.csv",
        )
        self.summary_json: Path = getattr(
            Config,
            "LSTM_AUTOENCODER_SUMMARY_JSON",
            Config.REPORT_DIR / "lstm_autoencoder_summary.json",
        )

        if self.chunk_size <= 0:
            raise ValueError("LSTM_AE_CHUNK_SIZE must be positive.")

        if self.sequence_length <= 1:
            raise ValueError("LSTM_AE_SEQUENCE_LENGTH must be greater than 1.")

        if self.batch_size <= 0:
            raise ValueError("LSTM_AE_BATCH_SIZE must be positive.")

        if self.epochs <= 0:
            raise ValueError("LSTM_AE_EPOCHS must be positive.")

        tf.keras.utils.set_random_seed(self.random_seed)

        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

        print(f"[PROGRESS] Chunk size: {self.chunk_size}")
        print(f"[PROGRESS] Sequence length: {self.sequence_length}")
        print(f"[PROGRESS] Batch size: {self.batch_size}")
        print(f"[PROGRESS] Epochs: {self.epochs}")
        print(f"[PROGRESS] Validation fraction: {self.validation_fraction}")
        print(f"[PROGRESS] Threshold percentile: {self.threshold_percentile}")
        print(f"[PROGRESS] Encoder units: {self.encoder_units}")
        print(f"[PROGRESS] Latent units: {self.latent_units}")
        print(f"[PROGRESS] Dropout rate: {self.dropout_rate}")
        print(f"[PROGRESS] Model path: {self.model_path}")
        print(f"[PROGRESS] Scaler path: {self.scaler_path}")
        print(f"[PROGRESS] Output CSV: {self.output_csv}")

    # ==================================================================================
    # Header and column helpers
    # ==================================================================================

    def _read_header_df(self, path: Path) -> pd.DataFrame:
        print(f"[PROGRESS] Reading header DataFrame from: {path}")
        return pd.read_csv(path, nrows=0)

    def _read_header_columns(self, path: Path) -> List[str]:
        print(f"[PROGRESS] Reading header columns from: {path}")
        return list(pd.read_csv(path, nrows=0).columns)

    def _count_csv_rows(self, path: Path) -> int:
        print(f"[PROGRESS] Counting CSV rows safely: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1

        row_count = max(int(row_count), 0)
        print(f"[PROGRESS] Row count for {path.name}: {row_count}")
        return row_count

    def _validate_columns(
        self,
        available_columns: List[str],
        required_columns: List[str],
        label: str,
    ) -> None:
        missing = [column for column in required_columns if column not in available_columns]

        if missing:
            print(f"[ERROR] Missing columns in {label}: {missing}")
            raise KeyError(f"Missing columns in {label}: {missing}")

        print(f"[PROGRESS] Required columns validated for {label}")

    def get_anomaly_feature_columns_from_header(self) -> List[str]:
        """
        Select LSTM Autoencoder anomaly features.

        Includes:
        - residual_Xs_*
        - resfeat_*

        Excludes:
        - abs_residual_Xs_* direct magnitude columns
        - actual Xs_* values
        - ensemble_predicted_* values
        - metadata/context columns
        """
        print("[PROGRESS] Selecting LSTM Autoencoder feature columns")

        header_df = self._read_header_df(Config.RESIDUALS_CSV)
        columns = list(header_df.columns)

        self._validate_columns(
            available_columns=columns,
            required_columns=["unit_id", "cycle", "split"],
            label="residuals.csv",
        )

        raw_residual_columns = [
            column for column in columns if column.startswith("residual_Xs_")
        ]

        resfeat_columns = [
            column for column in columns if column.startswith("resfeat_")
        ]

        feature_columns = raw_residual_columns + resfeat_columns

        invalid_features = [
            column
            for column in feature_columns
            if column.startswith("abs_residual_")
            or column.startswith("Xs_")
            or column.startswith("ensemble_predicted_")
            or column in {"unit_id", "cycle", "split", "gmm_context_id"}
        ]

        if invalid_features:
            raise ValueError(
                "Invalid/leakage LSTM Autoencoder feature columns found: "
                f"{invalid_features}"
            )

        if not feature_columns:
            raise ValueError(
                "No LSTM Autoencoder features found. Expected residual_Xs_* and resfeat_*."
            )

        print(f"[PROGRESS] Raw residual feature count: {len(raw_residual_columns)}")
        print(f"[PROGRESS] Residual temporal feature count: {len(resfeat_columns)}")
        print(f"[PROGRESS] Total LSTM feature count: {len(feature_columns)}")
        print(f"[PROGRESS] LSTM feature columns: {feature_columns}")

        return feature_columns

    def _prepare_feature_array(
        self,
        chunk: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        return (
            chunk[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

    # ==================================================================================
    # Scaler and sequence counting
    # ==================================================================================

    def fit_scaler_and_count_sequences(
        self,
        feature_columns: List[str],
    ) -> Tuple[StandardScaler, Dict[str, int]]:
        """
        Fit StandardScaler on dev rows only and count total dev sequences.

        Returns:
            Fitted scaler and count dictionary.
        """
        print("[TRAINING] Fitting LSTM scaler on dev rows only and counting sequences")

        scaler = StandardScaler()
        usecols = ["unit_id", "cycle", "split"] + feature_columns

        unit_row_counts: Dict[Tuple[str, object], int] = {}
        total_rows_seen = 0
        total_dev_rows_seen = 0
        chunk_index = 0

        for chunk in pd.read_csv(
            Config.RESIDUALS_CSV,
            usecols=usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        ):
            chunk_index += 1
            total_rows_seen += len(chunk)

            print("=" * 100)
            print(f"[TRAINING] Scaler/count chunk #{chunk_index}")
            print(f"[TRAINING] Chunk rows: {len(chunk)}")
            print(f"[TRAINING] Total rows scanned: {total_rows_seen}")

            dev_mask = chunk["split"] == Config.DEV_SPLIT_NAME
            dev_chunk = chunk.loc[dev_mask]

            dev_rows = len(dev_chunk)
            total_dev_rows_seen += dev_rows

            print(f"[TRAINING] Dev rows in chunk: {dev_rows}")
            print(f"[TRAINING] Total dev rows seen: {total_dev_rows_seen}")

            if dev_rows > 0:
                x_dev = self._prepare_feature_array(dev_chunk, feature_columns)
                scaler.partial_fit(x_dev)

                for group_key, group_df in dev_chunk.groupby(["split", "unit_id"], sort=False):
                    unit_row_counts[group_key] = unit_row_counts.get(group_key, 0) + len(group_df)

                del x_dev

            del chunk
            del dev_chunk
            gc.collect()

        if total_dev_rows_seen <= 0:
            raise ValueError("No dev rows found. Cannot train LSTM Autoencoder.")

        total_sequences = 0
        for row_count in unit_row_counts.values():
            total_sequences += max(row_count - self.sequence_length + 1, 0)

        if total_sequences <= 0:
            raise ValueError(
                "No dev sequences available. Increase data rows or reduce sequence length."
            )

        if self.validation_fraction > 0.0:
            validation_every = max(int(round(1.0 / self.validation_fraction)), 2)
            validation_sequences = int(math.ceil(total_sequences / validation_every))
        else:
            validation_every = 0
            validation_sequences = 0

        train_sequences = total_sequences - validation_sequences

        if train_sequences <= 0:
            raise ValueError("No training sequences left after validation split.")

        counts = {
            "total_rows_seen": int(total_rows_seen),
            "dev_rows_seen": int(total_dev_rows_seen),
            "total_dev_sequences": int(total_sequences),
            "train_sequences": int(train_sequences),
            "validation_sequences": int(validation_sequences),
            "validation_every": int(validation_every),
            "unit_group_count": int(len(unit_row_counts)),
        }

        print("[TRAINING] LSTM sequence count summary:")
        print(json.dumps(counts, indent=2))

        return scaler, counts

    # ==================================================================================
    # Sequence generator
    # ==================================================================================

    def _iter_sequence_batches(
        self,
        feature_columns: List[str],
        scaler: StandardScaler,
        mode: str,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield batches of dev sequences.

        Args:
            feature_columns: LSTM input columns.
            scaler: Dev-fitted scaler.
            mode: train, val, or dev_all.

        Yields:
            (x_batch, y_batch), where y_batch equals x_batch.
        """
        if mode not in {"train", "val", "dev_all"}:
            raise ValueError(f"Unsupported sequence generator mode: {mode}")

        usecols = ["unit_id", "cycle", "split"] + feature_columns
        state: Dict[Tuple[str, object], np.ndarray] = {}

        sequence_global_index = 0
        batch: List[np.ndarray] = []

        validation_every = max(
            int(round(1.0 / self.validation_fraction)),
            2,
        ) if self.validation_fraction > 0.0 else 0

        for chunk in pd.read_csv(
            Config.RESIDUALS_CSV,
            usecols=usecols,
            chunksize=self.chunk_size,
            low_memory=True,
        ):
            dev_chunk = chunk.loc[chunk["split"] == Config.DEV_SPLIT_NAME]

            if dev_chunk.empty:
                del chunk
                del dev_chunk
                gc.collect()
                continue

            for group_key, group_df in dev_chunk.groupby(["split", "unit_id"], sort=False):
                x_current = self._prepare_feature_array(group_df, feature_columns)
                x_current_scaled = scaler.transform(x_current).astype(np.float32, copy=False)

                previous_tail = state.get(
                    group_key,
                    np.empty((0, len(feature_columns)), dtype=np.float32),
                )

                combined = np.vstack([previous_tail, x_current_scaled])
                previous_len = len(previous_tail)

                for end_index in range(
                    max(self.sequence_length - 1, previous_len),
                    len(combined),
                ):
                    sequence = combined[
                        end_index - self.sequence_length + 1 : end_index + 1,
                        :,
                    ]

                    if sequence.shape[0] != self.sequence_length:
                        continue

                    is_validation = (
                        validation_every > 0
                        and sequence_global_index % validation_every == 0
                    )

                    should_emit = (
                        mode == "dev_all"
                        or (mode == "val" and is_validation)
                        or (mode == "train" and not is_validation)
                    )

                    if should_emit:
                        batch.append(sequence.astype(np.float32, copy=True))

                        if len(batch) >= self.batch_size:
                            x_batch = np.asarray(batch, dtype=np.float32)
                            yield x_batch, x_batch
                            batch.clear()

                    sequence_global_index += 1

                keep_count = max(self.sequence_length - 1, 1)
                state[group_key] = combined[-keep_count:, :].astype(np.float32, copy=True)

                del x_current
                del x_current_scaled
                del previous_tail
                del combined

            del chunk
            del dev_chunk
            gc.collect()

        if batch:
            x_batch = np.asarray(batch, dtype=np.float32)
            yield x_batch, x_batch
            batch.clear()

    def build_tf_dataset(
        self,
        feature_columns: List[str],
        scaler: StandardScaler,
        mode: str,
    ) -> tf.data.Dataset:
        """
        Build a repeating TensorFlow dataset from streaming CSV sequence batches.
        """
        output_signature = (
            tf.TensorSpec(
                shape=(None, self.sequence_length, len(feature_columns)),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(None, self.sequence_length, len(feature_columns)),
                dtype=tf.float32,
            ),
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: self._iter_sequence_batches(
                feature_columns=feature_columns,
                scaler=scaler,
                mode=mode,
            ),
            output_signature=output_signature,
        )

        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    # ==================================================================================
    # Model
    # ==================================================================================

    def build_model(self, n_features: int) -> Model:
        """
        Build LSTM Autoencoder model.
        """
        print("[TRAINING] Building LSTM Autoencoder model")

        input_layer = Input(shape=(self.sequence_length, n_features), name="residual_sequence_input")

        x = input_layer

        for index, units in enumerate(self.encoder_units):
            return_sequences = index < len(self.encoder_units) - 1
            x = LSTM(
                units,
                activation="tanh",
                return_sequences=return_sequences,
                name=f"encoder_lstm_{index + 1}",
            )(x)

            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate, name=f"encoder_dropout_{index + 1}")(x)

        latent = Dense(self.latent_units, activation="relu", name="latent_vector")(x)

        x = RepeatVector(self.sequence_length, name="repeat_latent")(latent)

        decoder_units = tuple(reversed(self.encoder_units))

        for index, units in enumerate(decoder_units):
            x = LSTM(
                units,
                activation="tanh",
                return_sequences=True,
                name=f"decoder_lstm_{index + 1}",
            )(x)

            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate, name=f"decoder_dropout_{index + 1}")(x)

        output_layer = TimeDistributed(
            Dense(n_features),
            name="reconstructed_sequence",
        )(x)

        model = Model(inputs=input_layer, outputs=output_layer, name="lstm_residual_autoencoder")

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        model.summary(print_fn=lambda line: print(f"[MODEL] {line}"))

        return model

    # ==================================================================================
    # Training
    # ==================================================================================

    def train(self) -> Dict[str, object]:
        """
        Train LSTM Autoencoder on dev sequences only.

        Returns:
            Metadata payload.
        """
        print("[TRAINING] Entering LSTMAutoencoderDetector.train")

        try:
            started = perf_counter()

            if not Config.RESIDUALS_CSV.exists():
                raise FileNotFoundError(f"Residual CSV not found: {Config.RESIDUALS_CSV}")

            feature_columns = self.get_anomaly_feature_columns_from_header()

            scaler, counts = self.fit_scaler_and_count_sequences(feature_columns)

            atomic_save_joblib(
                {
                    "scaler": scaler,
                    "feature_columns": feature_columns,
                    "fit_split": Config.DEV_SPLIT_NAME,
                    "feature_type": "raw_residual_plus_residual_temporal_features",
                },
                self.scaler_path,
            )

            train_steps = int(math.ceil(counts["train_sequences"] / self.batch_size))
            validation_steps = int(math.ceil(counts["validation_sequences"] / self.batch_size))

            train_steps = max(train_steps, 1)

            print(f"[TRAINING] Train steps per epoch: {train_steps}")
            print(f"[TRAINING] Validation steps: {validation_steps}")

            train_dataset = self.build_tf_dataset(
                feature_columns=feature_columns,
                scaler=scaler,
                mode="train",
            )

            validation_dataset = None

            if counts["validation_sequences"] > 0 and validation_steps > 0:
                validation_dataset = self.build_tf_dataset(
                    feature_columns=feature_columns,
                    scaler=scaler,
                    mode="val",
                )

            model = self.build_model(n_features=len(feature_columns))

            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            callbacks = [
                ModelCheckpoint(
                    filepath=str(self.model_path),
                    monitor="val_loss" if validation_dataset is not None else "loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                ),
                EarlyStopping(
                    monitor="val_loss" if validation_dataset is not None else "loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss" if validation_dataset is not None else "loss",
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
            ]

            print("[TRAINING] LSTM Autoencoder training started")
            print("[TRAINING] Fit split: dev only")
            print("[TRAINING] Test usage: score only")
            print("[TRAINING] Training uses all streaming dev sequences")

            if validation_dataset is not None:
                history = model.fit(
                    train_dataset,
                    epochs=self.epochs,
                    steps_per_epoch=train_steps,
                    validation_data=validation_dataset,
                    validation_steps=validation_steps,
                    callbacks=callbacks,
                    verbose=1,
                )
            else:
                history = model.fit(
                    train_dataset,
                    epochs=self.epochs,
                    steps_per_epoch=train_steps,
                    callbacks=callbacks,
                    verbose=1,
                )

            print("[TRAINING] LSTM Autoencoder training completed")

            model.save(self.model_path)

            threshold_payload = self.fit_threshold(
                model=model,
                scaler=scaler,
                feature_columns=feature_columns,
            )

            duration = perf_counter() - started

            metadata = {
                "model_name": "lstm_residual_autoencoder",
                "library": "TensorFlow/Keras",
                "model_path": str(self.model_path),
                "scaler_path": str(self.scaler_path),
                "feature_columns": feature_columns,
                "feature_count": int(len(feature_columns)),
                "sequence_length": int(self.sequence_length),
                "batch_size": int(self.batch_size),
                "epochs_requested": int(self.epochs),
                "epochs_completed": int(len(history.history.get("loss", []))),
                "encoder_units": list(self.encoder_units),
                "latent_units": int(self.latent_units),
                "dropout_rate": float(self.dropout_rate),
                "learning_rate": float(self.learning_rate),
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "training_mode": "all_dev_sequences_streamed_tf_data",
                "counts": counts,
                "history": {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                },
                "threshold": threshold_payload,
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
                "leakage_audit": {
                    "fit_on_dev_only": True,
                    "test_rows_used_for_fit": 0,
                    "uses_y_targets": False,
                    "uses_t_degradation_as_input": False,
                    "uses_actual_xs_as_features": False,
                    "uses_predicted_xs_as_features": False,
                },
            }

            atomic_write_json(metadata, self.metadata_path)

            print(f"[TRAINING] Metadata saved to: {self.metadata_path}")
            print(f"[TRAINING] Total duration minutes: {duration / 60.0:.2f}")

            return metadata

        except Exception as exc:
            print(f"[ERROR] LSTM Autoencoder training failed: {exc}")
            logger.exception("LSTM Autoencoder training failed.")
            raise RuntimeError("LSTM Autoencoder training failed.") from exc

    # ==================================================================================
    # Threshold fitting
    # ==================================================================================

    def fit_threshold(
        self,
        model: Model,
        scaler: StandardScaler,
        feature_columns: List[str],
    ) -> Dict[str, object]:
        """
        Fit reconstruction-error threshold from dev sequences only.
        """
        print("[TRAINING] Fitting LSTM Autoencoder threshold from dev sequences only")

        started = perf_counter()

        errors: List[np.ndarray] = []
        sequence_count = 0

        for x_batch, _ in self._iter_sequence_batches(
            feature_columns=feature_columns,
            scaler=scaler,
            mode="dev_all",
        ):
            reconstruction = model.predict(
                x_batch,
                batch_size=self.batch_size,
                verbose=0,
            )

            batch_errors = np.mean(
                np.square(x_batch - reconstruction),
                axis=(1, 2),
            ).astype(np.float32)

            errors.append(batch_errors)
            sequence_count += len(batch_errors)

            print(f"[TRAINING] Threshold fitting dev sequences processed: {sequence_count}")

            del x_batch
            del reconstruction
            del batch_errors
            gc.collect()

        if not errors:
            raise ValueError("No dev reconstruction errors available for threshold fitting.")

        all_errors = np.concatenate(errors).astype(np.float32, copy=False)

        threshold = float(np.percentile(all_errors, self.threshold_percentile))

        duration = perf_counter() - started

        payload = {
            "threshold": threshold,
            "threshold_percentile": float(self.threshold_percentile),
            "dev_sequence_count": int(sequence_count),
            "error_min": float(np.min(all_errors)),
            "error_max": float(np.max(all_errors)),
            "error_mean": float(np.mean(all_errors)),
            "error_std": float(np.std(all_errors)),
            "fit_split": Config.DEV_SPLIT_NAME,
            "test_usage": "score_only",
            "duration_seconds": float(duration),
        }

        print("[TRAINING] LSTM Autoencoder threshold payload:")
        print(json.dumps(payload, indent=2))

        return payload

    # ==================================================================================
    # Scoring
    # ==================================================================================

    def _flush_score_batch(
        self,
        model: Model,
        sequences: List[np.ndarray],
        positions: List[int],
        raw_errors: np.ndarray,
    ) -> None:
        """
        Predict reconstruction errors for buffered sequences and write into raw_errors.
        """
        if not sequences:
            return

        x_batch = np.asarray(sequences, dtype=np.float32)

        reconstruction = model.predict(
            x_batch,
            batch_size=self.batch_size,
            verbose=0,
        )

        batch_errors = np.mean(
            np.square(x_batch - reconstruction),
            axis=(1, 2),
        ).astype(np.float32)

        for position, error in zip(positions, batch_errors):
            raw_errors[position] = float(error)

        sequences.clear()
        positions.clear()

        del x_batch
        del reconstruction
        del batch_errors
        gc.collect()

    def score(self) -> int:
        """
        Score all rows with trained LSTM Autoencoder.

        Returns:
            Number of rows written.
        """
        print("[PROGRESS] Entering LSTMAutoencoderDetector.score")

        try:
            started = perf_counter()

            if not self.model_path.exists():
                raise FileNotFoundError(f"LSTM model not found: {self.model_path}")

            if not self.scaler_path.exists():
                raise FileNotFoundError(f"LSTM scaler not found: {self.scaler_path}")

            if not self.metadata_path.exists():
                raise FileNotFoundError(f"LSTM metadata not found: {self.metadata_path}")

            expected_rows = self._count_csv_rows(Config.RESIDUALS_CSV)

            model = load_model(self.model_path)

            scaler_payload = load_joblib_required(self.scaler_path)
            scaler: StandardScaler = scaler_payload["scaler"]
            feature_columns: List[str] = scaler_payload["feature_columns"]

            with self.metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

            threshold = float(metadata["threshold"]["threshold"])

            residual_columns = self._read_header_columns(Config.RESIDUALS_CSV)

            required_usecols = ["unit_id", "cycle", "split", "gmm_context_id"] + feature_columns

            self._validate_columns(
                available_columns=residual_columns,
                required_columns=required_usecols,
                label="residuals.csv",
            )

            output_path = self.output_csv
            temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if temp_output_path.exists():
                print("[PROGRESS] Removing old temporary LSTM Autoencoder CSV")
                temp_output_path.unlink()

            first_batch = True
            total_rows_written = 0
            chunk_index = 0

            state: Dict[Tuple[str, object], np.ndarray] = {}

            label_counts = {0: 0, 1: 0}
            split_label_counts: Dict[str, Dict[int, int]] = {
                Config.DEV_SPLIT_NAME: {0: 0, 1: 0},
                Config.TEST_SPLIT_NAME: {0: 0, 1: 0},
            }

            raw_error_sum = 0.0
            ready_count = 0

            print("[PROGRESS] Starting chunked LSTM Autoencoder scoring")

            for chunk in pd.read_csv(
                Config.RESIDUALS_CSV,
                usecols=required_usecols,
                chunksize=self.chunk_size,
                low_memory=True,
            ):
                chunk_index += 1
                chunk = chunk.reset_index(drop=True)

                print("=" * 100)
                print(f"[PROGRESS] LSTM scoring chunk #{chunk_index}")
                print(f"[PROGRESS] Chunk rows: {len(chunk)}")

                raw_errors = np.zeros(len(chunk), dtype=np.float32)
                sequence_ready = np.zeros(len(chunk), dtype=np.int8)

                sequences: List[np.ndarray] = []
                positions: List[int] = []

                for group_key, group_df in chunk.groupby(["split", "unit_id"], sort=False):
                    group_positions = group_df.index.to_numpy()

                    x_current = self._prepare_feature_array(group_df, feature_columns)
                    x_current_scaled = scaler.transform(x_current).astype(np.float32, copy=False)

                    previous_tail = state.get(
                        group_key,
                        np.empty((0, len(feature_columns)), dtype=np.float32),
                    )

                    combined = np.vstack([previous_tail, x_current_scaled])
                    previous_len = len(previous_tail)

                    for end_index in range(
                        max(self.sequence_length - 1, previous_len),
                        len(combined),
                    ):
                        sequence = combined[
                            end_index - self.sequence_length + 1 : end_index + 1,
                            :,
                        ]

                        if sequence.shape[0] != self.sequence_length:
                            continue

                        current_offset = end_index - previous_len

                        if current_offset < 0 or current_offset >= len(group_positions):
                            continue

                        row_position = int(group_positions[current_offset])

                        sequence_ready[row_position] = 1
                        sequences.append(sequence.astype(np.float32, copy=True))
                        positions.append(row_position)

                        if len(sequences) >= self.batch_size:
                            self._flush_score_batch(
                                model=model,
                                sequences=sequences,
                                positions=positions,
                                raw_errors=raw_errors,
                            )

                    keep_count = max(self.sequence_length - 1, 1)
                    state[group_key] = combined[-keep_count:, :].astype(np.float32, copy=True)

                    del x_current
                    del x_current_scaled
                    del previous_tail
                    del combined

                self._flush_score_batch(
                    model=model,
                    sequences=sequences,
                    positions=positions,
                    raw_errors=raw_errors,
                )

                safe_threshold = max(threshold, 1e-12)

                scores = np.minimum(
                    raw_errors / safe_threshold,
                    1.0,
                ).astype(np.float32)

                labels = np.where(
                    (raw_errors >= threshold) & (sequence_ready == 1),
                    1,
                    0,
                ).astype(np.int8)

                result_chunk = chunk[["unit_id", "cycle", "split", "gmm_context_id"]].copy()
                result_chunk["lstm_reconstruction_error"] = raw_errors
                result_chunk["lstm_autoencoder_score"] = scores
                result_chunk["lstm_autoencoder_label"] = labels
                result_chunk["lstm_sequence_ready"] = sequence_ready
                result_chunk["lstm_threshold"] = float(threshold)
                result_chunk["lstm_sequence_length"] = int(self.sequence_length)
                result_chunk["lstm_feature_count"] = int(len(feature_columns))

                result_chunk.to_csv(
                    temp_output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False,
                )

                first_batch = False
                total_rows_written += len(result_chunk)

                unique_labels, unique_counts = np.unique(labels, return_counts=True)

                for label, count in zip(unique_labels, unique_counts):
                    label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)

                for split in [Config.DEV_SPLIT_NAME, Config.TEST_SPLIT_NAME]:
                    split_mask = chunk["split"] == split

                    if not split_mask.any():
                        continue

                    split_labels = labels[split_mask.to_numpy()]
                    split_unique, split_counts = np.unique(split_labels, return_counts=True)

                    for label, count in zip(split_unique, split_counts):
                        split_label_counts[split][int(label)] = (
                            split_label_counts[split].get(int(label), 0) + int(count)
                        )

                raw_error_sum += float(np.sum(raw_errors, dtype=np.float64))
                ready_count += int(sequence_ready.sum())

                print(f"[PROGRESS] Total LSTM rows written: {total_rows_written}")
                print(f"[PROGRESS] Running label counts: {label_counts}")
                print(f"[PROGRESS] Running ready sequence count: {ready_count}")

                del chunk
                del raw_errors
                del sequence_ready
                del scores
                del labels
                del result_chunk
                gc.collect()

            if total_rows_written != expected_rows:
                raise ValueError(
                    "LSTM Autoencoder output row count mismatch. "
                    f"written={total_rows_written}, expected={expected_rows}. "
                    "Final CSV will not be replaced."
                )

            os.replace(temp_output_path, output_path)

            duration = perf_counter() - started

            summary = {
                "status": "success",
                "output_file": str(output_path),
                "records_count": int(total_rows_written),
                "sequence_ready_count": int(ready_count),
                "feature_count": int(len(feature_columns)),
                "sequence_length": int(self.sequence_length),
                "threshold": float(threshold),
                "raw_error_mean_all_rows": float(raw_error_sum / max(total_rows_written, 1)),
                "label_counts": {
                    "normal_0": int(label_counts.get(0, 0)),
                    "anomaly_1": int(label_counts.get(1, 0)),
                },
                "split_label_counts": {
                    split: {
                        "normal_0": int(counts.get(0, 0)),
                        "anomaly_1": int(counts.get(1, 0)),
                    }
                    for split, counts in split_label_counts.items()
                },
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "duration_seconds": float(duration),
                "duration_minutes": float(duration / 60.0),
            }

            atomic_write_json(summary, self.summary_json)

            print("[PROGRESS] LSTM Autoencoder scores CSV written successfully")
            print(f"[PROGRESS] Summary JSON written to: {self.summary_json}")
            print(f"[PROGRESS] Label counts: {summary['label_counts']}")
            print(f"[PROGRESS] Split label counts: {summary['split_label_counts']}")
            print(f"[PROGRESS] Duration minutes: {duration / 60.0:.2f}")

            return int(total_rows_written)

        except Exception as exc:
            print(f"[ERROR] LSTM Autoencoder scoring failed: {exc}")
            logger.exception("LSTM Autoencoder scoring failed.")
            raise RuntimeError("LSTM Autoencoder scoring failed.") from exc

    # ==================================================================================
    # Orchestration
    # ==================================================================================

    def run(self) -> Dict[str, object]:
        """
        Run LSTM Autoencoder training and scoring.

        Returns:
            Stage response.
        """
        print("[PROGRESS] Entering LSTMAutoencoderDetector.run")

        try:
            metadata = self.train()
            records_count = self.score()

            response = {
                "status": "success",
                "message": (
                    "LSTM Autoencoder anomaly scores generated using dev-trained "
                    "residual sequence reconstruction."
                ),
                "output_file": str(self.output_csv),
                "model_file": str(self.model_path),
                "scaler_file": str(self.scaler_path),
                "metadata_file": str(self.metadata_path),
                "summary_file": str(self.summary_json),
                "records_count": int(records_count),
                "fit_split": Config.DEV_SPLIT_NAME,
                "test_usage": "score_only",
                "feature_count": int(metadata["feature_count"]),
                "sequence_length": int(metadata["sequence_length"]),
                "training_mode": metadata["training_mode"],
                "threshold": float(metadata["threshold"]["threshold"]),
            }

            print(f"[PROGRESS] LSTM Autoencoder detector response: {response}")
            return response

        except Exception as exc:
            print(f"[ERROR] LSTM Autoencoder detector stage failed: {exc}")
            logger.exception("LSTM Autoencoder detector stage failed.")
            raise RuntimeError("LSTM Autoencoder detector stage failed.") from exc


def run_lstm_autoencoder_detection() -> Dict[str, object]:
    """
    Execute LSTM Autoencoder anomaly detection.
    """
    print("[PROGRESS] Entering run_lstm_autoencoder_detection")

    detector = LSTMAutoencoderDetector()
    return detector.run()


if __name__ == "__main__":
    print("[PROGRESS] lstm_autoencoder_detector.py execution started")
    result = run_lstm_autoencoder_detection()
    print("[PROGRESS] lstm_autoencoder_detector.py execution finished successfully")
    print(result)