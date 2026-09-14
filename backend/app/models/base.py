"""
Base Machine Learning Model Abstract Interface
Defines the standard lifecycle contract for training, inference, evaluation, feature importance, and persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from backend.app.models.evaluator import ModelEvaluator
from backend.app.models.schemas import EvaluationMetrics, ModelType


class BaseMLModel(ABC):
    """
    Abstract Base Class for all machine learning models in the platform.
    """

    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        model_version: str = "baseline-v1",
        feature_names: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.model_version = model_version
        self.feature_names = feature_names or []
        self.hyperparameters = hyperparameters or {}
        self.is_fitted = False
        self._model: Any = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseMLModel":
        """Fit model strictly on training observations."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate discrete binary class predictions (0 or 1)."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probability predictions [P(y=0), P(y=1)]."""
        pass

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        future_returns: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """Evaluate fitted model on a given partition."""
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.model_name} must be fitted before evaluation.")

        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        return ModelEvaluator.evaluate(
            y_true=y,
            y_pred=y_pred,
            y_prob=y_prob,
            future_returns=future_returns,
        )

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance or normalized coefficient map."""
        pass

    @abstractmethod
    def save(self, target_dir: Union[str, Path]) -> Path:
        """Serialize model artifact to directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, target_dir: Union[str, Path]) -> "BaseMLModel":
        """Deserialize model artifact from directory."""
        pass
