"""
Compact Transformer-Based Temporal Classification Model
Multi-Head Self-Attention architecture capturing non-local temporal dynamics across feature sequences.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from backend.app.models.base import BaseMLModel
from backend.app.models.schemas import ModelType


class TransformerModel(BaseMLModel):
    """
    Compact Multi-Head Self-Attention Transformer for financial time-series directional classification.
    """

    def __init__(
        self,
        sequence_length: int = 15,
        num_heads: int = 2,
        key_dim: int = 16,
        ff_dim: int = 32,
        d_model: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 30,
        batch_size: int = 32,
        patience: int = 5,
        random_state: int = 42,
        model_version: str = "baseline-v1",
        feature_names: Optional[List[str]] = None,
    ):
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

        hyperparams = {
            "sequence_length": sequence_length,
            "num_heads": num_heads,
            "key_dim": key_dim,
            "ff_dim": ff_dim,
            "d_model": d_model,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "random_state": random_state,
        }

        super().__init__(
            model_name="transformer",
            model_type=ModelType.TRANSFORMER,
            model_version=model_version,
            feature_names=feature_names,
            hyperparameters=hyperparams,
        )

        self._keras_model: Optional[keras.Model] = None
        self.training_history: Dict[str, List[float]] = {}
        self.trainable_params: int = 0
        self.non_trainable_params: int = 0
        self.total_params: int = 0

    def _build_model(self, feature_dim: int) -> keras.Model:
        """Construct Keras Functional Transformer architecture."""
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        inputs = layers.Input(shape=(self.sequence_length, feature_dim), name="sequence_input")

        # 1. Linear Projection to d_model
        x = layers.Dense(self.d_model, name="feature_projection")(inputs)

        # 2. Positional Embeddings
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        pos_embedding = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.d_model,
            name="position_embedding",
        )(positions)
        x = x + pos_embedding

        # 3. Multi-Head Self-Attention Block
        attn_out = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name="multi_head_attention",
        )(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6, name="norm_1")(x)

        # 4. Feed-Forward Network
        ffn = layers.Dense(self.ff_dim, activation="relu", name="ffn_dense_1")(x)
        if self.dropout_rate > 0.0:
            ffn = layers.Dropout(self.dropout_rate, name="ffn_dropout")(ffn)
        ffn = layers.Dense(self.d_model, name="ffn_dense_2")(ffn)

        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6, name="norm_2")(x)

        # 5. Global Temporal Pooling & Classification Head
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        x = layers.Dense(16, activation="relu", name="classifier_dense")(x)
        if self.dropout_rate > 0.0:
            x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="Transformer_Classifier")

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="roc_auc")],
        )

        self.trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
        self.non_trainable_params = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
        self.total_params = self.trainable_params + self.non_trainable_params

        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TransformerModel":
        if X_train.ndim != 3:
            raise ValueError(f"TransformerModel expects 3D sequence input (samples, seq_len, features), got ndim={X_train.ndim}")

        feature_dim = X_train.shape[2]
        self._keras_model = self._build_model(feature_dim)

        callbacks = []
        val_data = None
        if X_val is not None and y_val is not None:
            if X_val.ndim != 3:
                raise ValueError(f"Validation data must be 3D tensor, got ndim={X_val.ndim}")
            val_data = (X_val, y_val)
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )

        history = self._keras_model.fit(
            X_train,
            y_train,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        self.training_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self._keras_model is None:
            raise RuntimeError("TransformerModel must be fitted before predict_proba().")
        if X.ndim != 3:
            raise ValueError(f"TransformerModel expects 3D sequence input for inference, got ndim={X.ndim}")
        p1 = self._keras_model.predict(X, verbose=0).flatten()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        return {}

    def save(self, target_dir: Union[str, Path]) -> Path:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        if self._keras_model is not None:
            model_file = target_path / "model.keras"
            self._keras_model.save(model_file)

        history_file = target_path / "training_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2)

        config_file = target_path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_version": self.model_version,
                    "feature_names": self.feature_names,
                    "hyperparameters": self.hyperparameters,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "is_fitted": self.is_fitted,
                },
                f,
                indent=2,
            )

        return target_path

    @classmethod
    def load(cls, target_dir: Union[str, Path]) -> "TransformerModel":
        target_path = Path(target_dir)
        config_file = target_path / "config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        instance = cls(
            model_version=cfg["model_version"],
            feature_names=cfg["feature_names"],
            **cfg["hyperparameters"],
        )

        model_file = target_path / "model.keras"
        if model_file.exists():
            instance._keras_model = keras.models.load_model(model_file)
            instance.is_fitted = cfg["is_fitted"]
            instance.trainable_params = cfg.get("trainable_params", 0)
            instance.total_params = cfg.get("total_params", 0)

        history_file = target_path / "training_history.json"
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                instance.training_history = json.load(f)

        return instance
