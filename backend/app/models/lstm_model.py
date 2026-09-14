"""
Long Short-Term Memory (LSTM) Recurrent Neural Network Model
Captures temporal dependencies across historical feature sequences for forward-horizon directional prediction.
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


class LSTMModel(BaseMLModel):
    """
    Long Short-Term Memory (LSTM) Recurrent Neural Network for temporal sequence classification.
    """

    def __init__(
        self,
        sequence_length: int = 15,
        lstm_units: int = 32,
        dense_units: int = 16,
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
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

        hyperparams = {
            "sequence_length": sequence_length,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "random_state": random_state,
        }

        super().__init__(
            model_name="lstm",
            model_type=ModelType.LSTM,
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
        """Construct Keras LSTM architecture."""
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        model = keras.Sequential(name="LSTM_Classifier")
        model.add(layers.Input(shape=(self.sequence_length, feature_dim)))
        model.add(layers.LSTM(self.lstm_units, return_sequences=False))
        if self.dropout_rate > 0.0:
            model.add(layers.Dropout(self.dropout_rate))
        if self.dense_units > 0:
            model.add(layers.Dense(self.dense_units, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))

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
    ) -> "LSTMModel":
        """
        Fit LSTM model on 3D sequence tensors (samples, sequence_length, features).
        """
        if X_train.ndim != 3:
            raise ValueError(f"LSTMModel expects 3D sequence input (samples, seq_len, features), got ndim={X_train.ndim}")

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
            raise RuntimeError("LSTMModel must be fitted before predict_proba().")
        if X.ndim != 3:
            raise ValueError(f"LSTMModel expects 3D sequence input for inference, got ndim={X.ndim}")
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
    def load(cls, target_dir: Union[str, Path]) -> "LSTMModel":
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
