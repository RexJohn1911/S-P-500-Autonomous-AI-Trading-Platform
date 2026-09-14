"""
Base Market Regime Detector Abstract Interface.
Defines standard lifecycle contract: fit, predict, predict_proba, detect, semantic mapping, and serialization.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.regime.schemas import (
    DetectorType,
    MarketRegimeState,
    MarketRegimeType,
    TrendState,
    VolatilityState,
)


class BaseRegimeDetector(ABC):
    """
    Abstract Base Class for all Quantitative Market Regime Detectors.
    Enforces unified interfaces for fitting on historical training partitions, probabilistic inference,
    data-driven semantic mapping, and persistence.
    """

    def __init__(
        self,
        detector_type: DetectorType,
        detector_version: str = "regime-v1",
        feature_names: Optional[List[str]] = None,
        n_regimes: int = 3,
        random_state: int = 42,
    ):
        self.detector_type = detector_type
        self.detector_version = detector_version
        self.feature_names = feature_names or []
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.semantic_mapping: Dict[int, str] = {}
        self.is_fitted: bool = False

    @property
    def name(self) -> str:
        return self.detector_type.value

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[List[datetime]] = None,
        raw_features_df: Optional[pd.DataFrame] = None,
    ) -> "BaseRegimeDetector":
        """
        Fit detector parameters and learn data-driven semantic regime mapping strictly on training partition.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict discrete integer regime IDs for given feature matrix.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probability distribution over all regime states (shape: N x n_regimes).
        """
        pass

    def detect(
        self,
        X: np.ndarray,
        timestamps: List[datetime],
        symbol: str = "SPY",
    ) -> List[MarketRegimeState]:
        """
        Execute inference and produce structured MarketRegimeState time-series.
        """
        if not self.is_fitted:
            raise ValueError(f"Detector {self.name} must be fitted before calling detect().")

        if len(X) == 0:
            return []

        cluster_ids = self.predict(X)
        probs = self.predict_proba(X)
        states: List[MarketRegimeState] = []

        for i in range(len(X)):
            cid = int(cluster_ids[i])
            reg_name = self.semantic_mapping.get(cid, f"REGIME_{cid}")
            prob_row = probs[i] if probs is not None and len(probs) > i else np.zeros(self.n_regimes)
            conf = float(prob_row[cid]) if cid < len(prob_row) else float(np.max(prob_row)) if len(prob_row) > 0 else 1.0

            prob_dict = {
                self.semantic_mapping.get(k, f"REGIME_{k}"): float(prob_row[k])
                for k in range(len(prob_row))
            }

            # Infer broad volatility and trend sub-states from semantic tag
            vol_state = VolatilityState.NORMAL
            if "HIGH_VOL" in reg_name or "CRISIS" in reg_name:
                vol_state = VolatilityState.HIGH
            elif "LOW_VOL" in reg_name:
                vol_state = VolatilityState.LOW

            trend_state = TrendState.NEUTRAL
            if "BULL" in reg_name:
                trend_state = TrendState.BULLISH
            elif "BEAR" in reg_name:
                trend_state = TrendState.BEARISH

            ts = timestamps[i] if i < len(timestamps) else datetime.utcnow()
            states.append(
                MarketRegimeState(
                    timestamp=ts,
                    symbol=symbol,
                    detector=self.name,
                    regime_id=cid,
                    regime_name=reg_name,
                    confidence=conf,
                    regime_probabilities=prob_dict,
                    volatility_state=vol_state,
                    trend_state=trend_state,
                    metadata={"detector_version": self.detector_version},
                )
            )

        return states

    def _learn_semantic_mapping(
        self,
        cluster_ids: np.ndarray,
        returns: Optional[np.ndarray] = None,
        raw_features_df: Optional[pd.DataFrame] = None,
    ) -> Dict[int, str]:
        """
        Data-driven semantic regime mapping learned strictly from training-period observations.
        Examines mean return, volatility, and trend characteristics of each cluster.
        """
        # Guarantee all clusters from 0 to n_regimes - 1 have mapped labels
        all_cluster_ids = list(range(self.n_regimes)) if self.n_regimes > 0 else sorted([int(c) for c in np.unique(cluster_ids)])
        if not all_cluster_ids:
            return {0: MarketRegimeType.SIDEWAYS_NEUTRAL.value}

        mapping: Dict[int, str] = {}
        if returns is None or len(returns) != len(cluster_ids):
            # Fallback: simple descriptive naming if returns not provided
            return {int(cid): f"REGIME_STATE_{cid}" for cid in all_cluster_ids}

        rets = np.asarray(returns, dtype=float)
        overall_mean_ret = float(np.nanmean(rets)) if len(rets) > 0 else 0.0
        overall_vol = float(np.nanstd(rets)) if len(rets) > 0 else 0.01

        cluster_stats = []
        for cid in all_cluster_ids:
            mask = (cluster_ids == cid)
            if np.sum(mask) == 0:
                cluster_stats.append({
                    "cid": int(cid),
                    "mean_ret": 0.0,
                    "vol": 0.0,
                    "trend": 0.0,
                })
                continue

            c_rets = rets[mask]
            valid_rets = c_rets[~np.isnan(c_rets)]
            c_mean = float(np.mean(valid_rets)) if len(valid_rets) > 0 else 0.0
            c_vol = float(np.std(valid_rets)) if len(valid_rets) > 0 else 0.0

            # Trend metric from raw_features_df if available
            c_trend = 0.0
            if raw_features_df is not None and "price_vs_sma_50" in raw_features_df.columns:
                series_vals = pd.to_numeric(raw_features_df.loc[mask, "price_vs_sma_50"], errors="coerce").dropna().to_numpy()
                c_trend = float(np.mean(series_vals)) if len(series_vals) > 0 else 0.0
            elif raw_features_df is not None and "return_20d" in raw_features_df.columns:
                series_vals = pd.to_numeric(raw_features_df.loc[mask, "return_20d"], errors="coerce").dropna().to_numpy()
                c_trend = float(np.mean(series_vals)) if len(series_vals) > 0 else 0.0
            else:
                c_trend = c_mean

            cluster_stats.append({
                "cid": int(cid),
                "mean_ret": c_mean,
                "vol": c_vol,
                "trend": c_trend,
            })

        # Sort clusters by return and volatility to assign distinct institutional names
        sorted_by_ret = sorted(cluster_stats, key=lambda x: x["mean_ret"])
        highest_ret_cid = sorted_by_ret[-1]["cid"]
        lowest_ret_cid = sorted_by_ret[0]["cid"]

        sorted_by_vol = sorted(cluster_stats, key=lambda x: x["vol"])
        highest_vol_cid = sorted_by_vol[-1]["cid"]
        lowest_vol_cid = sorted_by_vol[0]["cid"]

        for stat in cluster_stats:
            cid = int(stat["cid"])
            mean_r = stat["mean_ret"]
            vol_r = stat["vol"]
            trend_r = stat["trend"]

            # Classify based on relative metrics
            is_bull = (cid == highest_ret_cid and mean_r >= overall_mean_ret) or (trend_r > 0.005 and mean_r > 0)
            is_bear = (cid == lowest_ret_cid and mean_r < overall_mean_ret) or (trend_r < -0.005 and mean_r < 0)
            is_high_vol = (vol_r > overall_vol * 1.05) or (cid == highest_vol_cid and vol_r > overall_vol)
            is_low_vol = (vol_r <= overall_vol * 0.95) or (cid == lowest_vol_cid)

            if is_bull and is_low_vol:
                name = MarketRegimeType.LOW_VOLATILITY_BULL.value
            elif is_bull and is_high_vol:
                name = MarketRegimeType.BULL_TRENDING.value
            elif is_bear and is_high_vol:
                name = MarketRegimeType.HIGH_VOLATILITY_BEAR.value
            elif is_bear:
                name = MarketRegimeType.BEAR_TRENDING.value
            elif is_high_vol:
                name = MarketRegimeType.HIGH_VOLATILITY.value
            else:
                name = MarketRegimeType.SIDEWAYS_NEUTRAL.value

            mapping[cid] = name

        # Ensure no identical duplicate names if n_clusters > 1
        seen_names = {}
        for cid, name in mapping.items():
            if name in seen_names:
                mapping[cid] = f"{name}_{cid}"
            seen_names[mapping[cid]] = True

        return mapping

    @abstractmethod
    def save(self, dir_path: Path) -> Path:
        """Persist model weights and metadata to directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, dir_path: Path) -> "BaseRegimeDetector":
        """Load model weights and metadata from directory."""
        pass
