"""
Rule-Based Market Regime Detector Baseline.
Interpretable, deterministic baseline utilizing moving-average trend relationships, rolling volatility, and macro thresholds.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from backend.app.regime.base import BaseRegimeDetector
from backend.app.regime.schemas import DetectorType, MarketRegimeType


class RuleBasedRegimeDetector(BaseRegimeDetector):
    """
    Transparent, deterministic rule-based regime detector.
    Uses calibrated training-period volatility and trend boundaries to classify market states.
    """

    def __init__(
        self,
        volatility_quantile_threshold: float = 0.60,
        trend_threshold: float = 0.0,
        detector_version: str = "regime-v1",
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__(
            detector_type=DetectorType.RULE_BASED,
            detector_version=detector_version,
            feature_names=feature_names or ["return_20d", "rolling_volatility_20d", "price_vs_sma_50"],
            n_regimes=4,
            random_state=42,
        )
        self.volatility_quantile_threshold = volatility_quantile_threshold
        self.trend_threshold = trend_threshold
        self.calibrated_vol_threshold: float = 0.015
        self.semantic_mapping = {
            0: MarketRegimeType.LOW_VOLATILITY_BULL.value,
            1: MarketRegimeType.HIGH_VOLATILITY_BEAR.value,
            2: MarketRegimeType.BULL_TRENDING.value,
            3: MarketRegimeType.SIDEWAYS_NEUTRAL.value,
        }

    def fit(
        self,
        X: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[List[Any]] = None,
        raw_features_df: Optional[pd.DataFrame] = None,
    ) -> "RuleBasedRegimeDetector":
        """
        Calibrate volatility quantile threshold strictly from training period observations.
        """
        if len(X) == 0:
            raise ValueError("Cannot fit RuleBasedRegimeDetector on empty training data.")

        # Identify volatility feature index or compute from returns
        vol_col_idx = -1
        for idx, name in enumerate(self.feature_names):
            if "volatility" in name or "atr" in name:
                vol_col_idx = idx
                break

        if vol_col_idx >= 0 and vol_col_idx < X.shape[1]:
            vol_values = X[:, vol_col_idx]
            self.calibrated_vol_threshold = float(np.nanpercentile(vol_values, self.volatility_quantile_threshold * 100))
        elif returns is not None and len(returns) == len(X):
            rolling_v = np.abs(np.asarray(returns, dtype=float))
            self.calibrated_vol_threshold = float(np.nanpercentile(rolling_v, self.volatility_quantile_threshold * 100))
        else:
            self.calibrated_vol_threshold = 0.0

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Deterministic rule classification:
        0: Bull Low-Vol (Trend >= threshold, Vol <= threshold)
        1: Bear High-Vol (Trend < threshold, Vol > threshold)
        2: Bull High-Vol (Trend >= threshold, Vol > threshold)
        3: Sideways / Neutral Bear (Trend < threshold, Vol <= threshold)
        """
        if not self.is_fitted:
            raise ValueError("RuleBasedRegimeDetector must be fitted before predict().")

        if len(X) == 0:
            return np.empty(0, dtype=int)

        # Extract trend and vol features (or primary dimensions)
        trend_idx = 0
        vol_idx = 1 if X.shape[1] > 1 else 0

        for idx, name in enumerate(self.feature_names):
            if "trend" in name or "sma" in name or "return" in name:
                trend_idx = idx
                break
        for idx, name in enumerate(self.feature_names):
            if "volatility" in name or "atr" in name or "vix" in name:
                vol_idx = idx
                break

        trends = X[:, trend_idx]
        vols = X[:, vol_idx]

        labels = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            t = trends[i]
            v = vols[i]
            is_bull = t >= self.trend_threshold
            is_high_vol = v > self.calibrated_vol_threshold

            if is_bull and not is_high_vol:
                labels[i] = 0  # LOW_VOL_BULL
            elif not is_bull and is_high_vol:
                labels[i] = 1  # HIGH_VOL_BEAR
            elif is_bull and is_high_vol:
                labels[i] = 2  # BULL_TRENDING
            else:
                labels[i] = 3  # SIDEWAYS_NEUTRAL

        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate calibrated confidence probabilities from rule assignments.
        """
        labels = self.predict(X)
        probs = np.full((len(X), self.n_regimes), 0.10 / (self.n_regimes - 1))
        for i in range(len(X)):
            probs[i, labels[i]] = 0.90
        return probs

    def save(self, dir_path: Path) -> Path:
        """Persist rule configuration and calibrated thresholds."""
        dir_path.mkdir(parents=True, exist_ok=True)
        config = {
            "volatility_quantile_threshold": self.volatility_quantile_threshold,
            "trend_threshold": self.trend_threshold,
            "calibrated_vol_threshold": self.calibrated_vol_threshold,
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
    def load(cls, dir_path: Path) -> "RuleBasedRegimeDetector":
        """Load rule configuration from disk."""
        with open(dir_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        instance = cls(
            volatility_quantile_threshold=cfg["volatility_quantile_threshold"],
            trend_threshold=cfg["trend_threshold"],
            detector_version=cfg["detector_version"],
            feature_names=cfg["feature_names"],
        )
        instance.calibrated_vol_threshold = cfg["calibrated_vol_threshold"]
        instance.semantic_mapping = {int(k): v for k, v in cfg["semantic_mapping"].items()}
        instance.is_fitted = cfg["is_fitted"]
        return instance
