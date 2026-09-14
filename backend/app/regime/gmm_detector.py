"""
Gaussian Mixture Model (GMM) Market Regime Detector.
Probabilistic generative density estimation with soft mixture responsibilities and BIC/AIC diagnostics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from backend.app.regime.base import BaseRegimeDetector
from backend.app.regime.schemas import DetectorType


class GMMRegimeDetector(BaseRegimeDetector):
    """
    Gaussian Mixture Model (GMM) probabilistic market regime detector.
    Provides rigorous posterior probabilities P(Regime | Features), covariance matrices, and information criteria.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
        detector_version: str = "regime-v1",
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__(
            detector_type=DetectorType.GMM,
            detector_version=detector_version,
            feature_names=feature_names or [],
            n_regimes=n_components,
            random_state=random_state,
        )
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.model: Optional[GaussianMixture] = None

    def fit(
        self,
        X: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[List[Any]] = None,
        raw_features_df: Optional[pd.DataFrame] = None,
    ) -> "GMMRegimeDetector":
        """
        Fit Gaussian Mixture density components and learn semantic mapping strictly on training partition.
        """
        if len(X) < self.n_components:
            raise ValueError(f"Insufficient training samples ({len(X)}) for {self.n_components} GMM components.")

        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
            n_init=5,
        )
        train_clusters = self.model.fit_predict(X)

        # Learn semantic mapping strictly from training metrics
        self.semantic_mapping = self._learn_semantic_mapping(
            cluster_ids=train_clusters,
            returns=returns,
            raw_features_df=raw_features_df,
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign maximum a posteriori (MAP) regime index."""
        if not self.is_fitted or self.model is None:
            raise ValueError("GMMRegimeDetector must be fitted before calling predict().")

        if len(X) == 0:
            return np.empty(0, dtype=int)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Extract soft posterior probability distribution over Gaussian components."""
        if not self.is_fitted or self.model is None:
            raise ValueError("GMMRegimeDetector must be fitted before calling predict_proba().")

        if len(X) == 0:
            return np.empty((0, self.n_components), dtype=float)

        return self.model.predict_proba(X)

    def get_bic(self, X: np.ndarray) -> float:
        """Compute Bayesian Information Criterion (lower is better)."""
        if not self.is_fitted or self.model is None:
            return float("nan")
        return float(self.model.bic(X))

    def get_aic(self, X: np.ndarray) -> float:
        """Compute Akaike Information Criterion (lower is better)."""
        if not self.is_fitted or self.model is None:
            return float("nan")
        return float(self.model.aic(X))

    def get_log_likelihood(self, X: np.ndarray) -> float:
        """Compute average per-sample log-likelihood."""
        if not self.is_fitted or self.model is None:
            return float("nan")
        return float(self.model.score(X))

    def save(self, dir_path: Path) -> Path:
        """Persist GMM parameters, semantic mapping, and configuration."""
        dir_path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            joblib.dump(self.model, dir_path / "detector.joblib")

        config = {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "random_state": self.random_state,
            "detector_version": self.detector_version,
            "feature_names": self.feature_names,
            "n_regimes": self.n_regimes,
            "semantic_mapping": self.semantic_mapping,
            "is_fitted": self.is_fitted,
        }
        with open(dir_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return dir_path

    @classmethod
    def load(cls, dir_path: Path) -> "GMMRegimeDetector":
        """Load fitted GMM model from disk."""
        with open(dir_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        instance = cls(
            n_components=cfg["n_components"],
            covariance_type=cfg.get("covariance_type", "full"),
            random_state=cfg["random_state"],
            detector_version=cfg["detector_version"],
            feature_names=cfg["feature_names"],
        )
        model_file = dir_path / "detector.joblib"
        if model_file.exists():
            instance.model = joblib.load(model_file)
        instance.semantic_mapping = {int(k): v for k, v in cfg["semantic_mapping"].items()}
        instance.is_fitted = cfg["is_fitted"]
        return instance
