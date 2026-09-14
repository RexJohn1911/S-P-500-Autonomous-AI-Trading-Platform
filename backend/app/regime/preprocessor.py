"""
Train-Only Preprocessing for Market Regime Features.
Enforces strict zero-leakage standard scaling and imputation fitted exclusively on training periods.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from backend.app.features.models import FeatureDataset


class RegimeFeaturePreprocessor:
    """
    Feature selector, imputer, and scaler for market regime detection.
    Guarantees that statistical parameters (means, variances, medians) are computed strictly on the training partition.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True,
    ):
        self.feature_names = feature_names or [
            "return_20d",
            "rolling_volatility_20d",
            "price_vs_sma_50",
            "price_vs_sma_200",
            "vix_level",
            "vix_change_5d",
        ]
        self.scale_features = scale_features
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

    def extract_features_df(self, feature_dataset: FeatureDataset) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
        """
        Extract selected regime features and timestamps from a FeatureDataset into a clean DataFrame.
        """
        records = feature_dataset.records
        if not records:
            return pd.DataFrame(columns=self.feature_names), []

        rows = []
        timestamps = []
        for rec in records:
            timestamps.append(rec.timestamp)
            row_dict = {}
            for name in self.feature_names:
                row_dict[name] = rec.features.get(name)
            rows.append(row_dict)

        df = pd.DataFrame(rows, columns=self.feature_names)
        return df, timestamps

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "RegimeFeaturePreprocessor":
        """
        Fit imputer and scaler strictly on the training partition.
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if len(X_mat) == 0:
            raise ValueError("Cannot fit preprocessor on an empty training array.")

        # Ensure numeric float dtype and fill entirely all-NaN columns with 0.0
        X_clean = np.array(X_mat, dtype=float)
        all_nan_cols = np.isnan(X_clean).all(axis=0)
        X_clean[:, all_nan_cols] = 0.0

        # 1. Fit median imputer
        self.imputer = SimpleImputer(strategy="median", fill_value=0.0)
        X_imp = self.imputer.fit_transform(X_clean)

        # 2. Fit standard scaler if enabled
        if self.scale_features:
            self.scaler = StandardScaler()
            self.scaler.fit(X_imp)

        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform out-of-sample data using training-fitted statistics.
        """
        if not self.is_fitted or self.imputer is None:
            raise ValueError("RegimeFeaturePreprocessor must be fitted before calling transform().")

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if len(X_mat) == 0:
            return np.empty((0, len(self.feature_names)))

        X_clean = np.array(X_mat, dtype=float)
        all_nan_cols = np.isnan(X_clean).all(axis=0)
        X_clean[:, all_nan_cols] = 0.0

        # 1. Apply imputation
        X_imp = self.imputer.transform(X_clean)
        # Fallback for any remaining unobserved NaNs
        np.nan_to_num(X_imp, copy=False, nan=0.0)

        # 2. Apply scaling if enabled
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X_imp)
            return X_scaled

        return X_imp

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit on training data and return scaled/imputed matrix."""
        return self.fit(X).transform(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist fitted preprocessor state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "scale_features": self.scale_features,
                "imputer": self.imputer,
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RegimeFeaturePreprocessor":
        """Load fitted preprocessor state from disk."""
        data = joblib.load(filepath)
        instance = cls(
            feature_names=data["feature_names"],
            scale_features=data["scale_features"],
        )
        instance.imputer = data["imputer"]
        instance.scaler = data["scaler"]
        instance.is_fitted = data["is_fitted"]
        return instance
