"""
Random Forest Baseline Model
Ensemble of decision trees providing non-linear feature interaction capture and Gini feature importances.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from backend.app.models.base import BaseMLModel
from backend.app.models.schemas import ModelType


class RandomForestModel(BaseMLModel):
    """
    Random Forest baseline classifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        n_jobs: int = -1,
        model_version: str = "baseline-v1",
        feature_names: Optional[List[str]] = None,
    ):
        hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "class_weight": class_weight,
            "n_jobs": n_jobs,
        }
        super().__init__(
            model_name="random_forest",
            model_type=ModelType.RANDOM_FOREST,
            model_version=model_version,
            feature_names=feature_names,
            hyperparameters=hyperparams,
        )
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestModel":
        self._model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RandomForestModel must be fitted before predict().")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RandomForestModel must be fitted before predict_proba().")
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importances = self._model.feature_importances_
        if self.feature_names and len(self.feature_names) == len(importances):
            return {name: float(round(imp, 6)) for name, imp in zip(self.feature_names, importances)}
        return {f"feat_{i}": float(round(imp, 6)) for i, imp in enumerate(importances)}

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
    def load(cls, target_dir: Union[str, Path]) -> "RandomForestModel":
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
