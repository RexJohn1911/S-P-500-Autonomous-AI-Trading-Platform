"""
Feature Preprocessing Pipeline
Provides train-only feature standardization and imputation to prevent data leakage across evaluation splits.
"""

from pathlib import Path
from typing import Optional, Union
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """
    Standardization and imputation pipeline.
    Guarantees that statistical parameters (mean, std, median) are fitted ONLY on training data.
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        impute_strategy: str = "median",
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.impute_strategy = impute_strategy

        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "FeaturePreprocessor":
        """Fit preprocessing parameters strictly on training observations."""
        if len(X_train) == 0:
            raise ValueError("Cannot fit preprocessor on empty training set.")

        X_imputed = self.imputer.fit_transform(X_train)
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted transformation to any split without refitting parameters."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        if len(X) == 0:
            return np.empty((0, self.scaler.mean_.shape[0]))

        X_imputed = self.imputer.transform(X)
        return self.scaler.transform(X_imputed)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training set and transform in a single pass."""
        return self.fit(X_train).transform(X_train)

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize fitted preprocessor to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "imputer": self.imputer,
                "with_mean": self.with_mean,
                "with_std": self.with_std,
                "impute_strategy": self.impute_strategy,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeaturePreprocessor":
        """Load serialized preprocessor from disk."""
        data = joblib.load(filepath)
        instance = cls(
            with_mean=data["with_mean"],
            with_std=data["with_std"],
            impute_strategy=data["impute_strategy"],
        )
        instance.scaler = data["scaler"]
        instance.imputer = data["imputer"]
        instance.is_fitted = data["is_fitted"]
        return instance
