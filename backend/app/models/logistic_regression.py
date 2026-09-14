"""
Logistic Regression Baseline Model
Regularized linear classifier providing interpretable baseline coefficients and calibrated probability outputs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from backend.app.models.base import BaseMLModel
from backend.app.models.schemas import ModelType


class LogisticRegressionModel(BaseMLModel):
    """
    Logistic Regression baseline model for directional price return classification.
    """

    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        model_version: str = "baseline-v1",
        feature_names: Optional[List[str]] = None,
        **extra_kwargs,
    ):
        hyperparams = {
            "C": C,
            "solver": solver,
            "max_iter": max_iter,
            "random_state": random_state,
            "class_weight": class_weight,
            **extra_kwargs,
        }
        super().__init__(
            model_name="logistic_regression",
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_version=model_version,
            feature_names=feature_names,
            hyperparameters=hyperparams,
        )
        self._model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LogisticRegressionModel":
        self._model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionModel must be fitted before predict().")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionModel must be fitted before predict_proba().")
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        coefs = self._model.coef_[0]
        if self.feature_names and len(self.feature_names) == len(coefs):
            return {name: float(round(coef, 6)) for name, coef in zip(self.feature_names, coefs)}
        return {f"feat_{i}": float(round(c, 6)) for i, c in enumerate(coefs)}

    def save(self, target_dir: Union[str, Path]) -> Path:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        model_file = target_path / "model.joblib"
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self.feature_names,
                "hyperparameters": self.hyperparameters,
                "model_version": self.model_version,
                "is_fitted": self.is_fitted,
            },
            model_file,
        )
        return model_file

    @classmethod
    def load(cls, target_dir: Union[str, Path]) -> "LogisticRegressionModel":
        target_path = Path(target_dir)
        model_file = target_path / "model.joblib"
        data = joblib.load(model_file)
        instance = cls(
            model_version=data["model_version"],
            feature_names=data["feature_names"],
            **data["hyperparameters"],
        )
        instance._model = data["model"]
        instance.is_fitted = data["is_fitted"]
        return instance
