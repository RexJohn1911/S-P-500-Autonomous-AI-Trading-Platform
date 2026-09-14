"""
K-Means Clustering Market Regime Detector.
Unsupervised clustering of quantitative market feature representations with training-calibrated semantic mapping.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from backend.app.regime.base import BaseRegimeDetector
from backend.app.regime.schemas import DetectorType


class KMeansRegimeDetector(BaseRegimeDetector):
    """
    K-Means unsupervised market regime classifier.
    Learns spatial centroids in feature space and maps cluster indices to stable semantic descriptions.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        detector_version: str = "regime-v1",
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__(
            detector_type=DetectorType.KMEANS,
            detector_version=detector_version,
            feature_names=feature_names or [],
            n_regimes=n_clusters,
            random_state=random_state,
        )
        self.n_clusters = n_clusters
        self.model: Optional[KMeans] = None

    def fit(
        self,
        X: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[List[Any]] = None,
        raw_features_df: Optional[pd.DataFrame] = None,
    ) -> "KMeansRegimeDetector":
        """
        Fit K-Means cluster centroids and learn semantic mapping strictly on training partition.
        """
        if len(X) < self.n_clusters:
            raise ValueError(f"Insufficient training samples ({len(X)}) for {self.n_clusters} clusters.")

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
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
        """Assign cluster index based on minimum distance to centroid."""
        if not self.is_fitted or self.model is None:
            raise ValueError("KMeansRegimeDetector must be fitted before calling predict().")

        if len(X) == 0:
            return np.empty(0, dtype=int)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Derive soft posterior cluster probabilities using softmax-transformed inverse centroid distances.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("KMeansRegimeDetector must be fitted before calling predict_proba().")

        if len(X) == 0:
            return np.empty((0, self.n_clusters), dtype=float)

        # Distance matrix (N x K)
        distances = cdist(X, self.model.cluster_centers_, metric="euclidean")

        # Softmax of negative normalized distances
        scaled_neg_dist = -distances / (np.std(distances) + 1e-6)
        exp_dist = np.exp(scaled_neg_dist - np.max(scaled_neg_dist, axis=1, keepdims=True))
        probs = exp_dist / np.sum(exp_dist, axis=1, keepdims=True)
        return probs

    def save(self, dir_path: Path) -> Path:
        """Persist fitted KMeans model, semantic mapping, and configuration."""
        dir_path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            joblib.dump(self.model, dir_path / "detector.joblib")

        config = {
            "n_clusters": self.n_clusters,
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
    def load(cls, dir_path: Path) -> "KMeansRegimeDetector":
        """Load fitted KMeans model from disk."""
        with open(dir_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        instance = cls(
            n_clusters=cfg["n_clusters"],
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
