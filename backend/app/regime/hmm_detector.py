"""
Hidden Markov Model (HMM) Market Regime Detector.
Sequential probabilistic regime discovery utilizing temporal Markov state transitions and Gaussian emissions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from backend.app.regime.base import BaseRegimeDetector
from backend.app.regime.schemas import DetectorType

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class HMMRegimeDetector(BaseRegimeDetector):
    """
    Hidden Markov Model (HMM) regime detector.
    Models latent temporal market regimes via transition probability matrix and Gaussian emission distributions.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        random_state: int = 42,
        detector_version: str = "regime-v1",
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__(
            detector_type=DetectorType.HMM,
            detector_version=detector_version,
            feature_names=feature_names or [],
            n_regimes=n_components,
            random_state=random_state,
        )
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.model: Optional[Any] = None

    def fit(
        self,
        X: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[List[Any]] = None,
        raw_features_df: Optional[pd.DataFrame] = None,
    ) -> "HMMRegimeDetector":
        """
        Fit Gaussian HMM parameters (transition matrix, emission means & covariances) on training partition.
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required to use HMMRegimeDetector. Install with `pip install hmmlearn`.")

        if len(X) < self.n_components * 2:
            raise ValueError(f"Insufficient training samples ({len(X)}) for {self.n_components} HMM components.")

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=200,
            random_state=self.random_state,
        )
        self.model.fit(X)
        train_clusters = self.model.predict(X)

        # Learn semantic mapping strictly from training metrics
        self.semantic_mapping = self._learn_semantic_mapping(
            cluster_ids=train_clusters,
            returns=returns,
            raw_features_df=raw_features_df,
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Decode most likely sequence of hidden states via Viterbi algorithm."""
        if not self.is_fitted or self.model is None:
            raise ValueError("HMMRegimeDetector must be fitted before calling predict().")

        if len(X) == 0:
            return np.empty(0, dtype=int)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior state probabilities via forward-backward algorithm."""
        if not self.is_fitted or self.model is None:
            raise ValueError("HMMRegimeDetector must be fitted before calling predict_proba().")

        if len(X) == 0:
            return np.empty((0, self.n_components), dtype=float)

        return self.model.predict_proba(X)

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Return learned Markov transition matrix P(S_{t+1} | S_t)."""
        if not self.is_fitted or self.model is None:
            return None
        return self.model.transmat_

    def save(self, dir_path: Path) -> Path:
        """Persist fitted HMM model, transition parameters, and configuration."""
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
    def load(cls, dir_path: Path) -> "HMMRegimeDetector":
        """Load fitted HMM model from disk."""
        with open(dir_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        instance = cls(
            n_components=cfg["n_components"],
            covariance_type=cfg.get("covariance_type", "diag"),
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
