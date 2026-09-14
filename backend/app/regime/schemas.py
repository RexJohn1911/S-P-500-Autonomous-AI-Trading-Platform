"""
Market Regime Detection Domain Schemas and Data Models.
Defines regime classifications, state representations, diagnostics, statistics, and persistence metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class MarketRegimeType(str, Enum):
    """Semantic market regime classifications."""
    BULL_TRENDING = "BULL_TRENDING"
    BEAR_TRENDING = "BEAR_TRENDING"
    SIDEWAYS_NEUTRAL = "SIDEWAYS_NEUTRAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY_BULL = "LOW_VOLATILITY_BULL"
    HIGH_VOLATILITY_BEAR = "HIGH_VOLATILITY_BEAR"
    TRANSITIONAL = "TRANSITIONAL"
    UNKNOWN = "UNKNOWN"


class VolatilityState(str, Enum):
    """Categorical volatility environment."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TrendState(str, Enum):
    """Categorical market directional trend state."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class DetectorType(str, Enum):
    """Supported regime detection algorithms."""
    RULE_BASED = "rule_based"
    KMEANS = "kmeans"
    GMM = "gmm"
    HMM = "hmm"


@dataclass
class MarketRegimeState:
    """
    Time-indexed classified market regime state for an asset or market benchmark.
    Contains regime identity, confidence score, full probability distribution, and sub-states.
    """
    timestamp: datetime
    symbol: str
    detector: str
    regime_id: int
    regime_name: str
    confidence: float
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    volatility_state: Optional[VolatilityState] = None
    trend_state: Optional[TrendState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "detector": self.detector,
            "regime_id": self.regime_id,
            "regime_name": self.regime_name,
            "confidence": round(float(self.confidence), 4),
            "regime_probabilities": {k: round(float(v), 4) for k, v in self.regime_probabilities.items()},
            "volatility_state": self.volatility_state.value if self.volatility_state else None,
            "trend_state": self.trend_state.value if self.trend_state else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketRegimeState":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            detector=data["detector"],
            regime_id=data["regime_id"],
            regime_name=data["regime_name"],
            confidence=data["confidence"],
            regime_probabilities=data.get("regime_probabilities", {}),
            volatility_state=VolatilityState(data["volatility_state"]) if data.get("volatility_state") else None,
            trend_state=TrendState(data["trend_state"]) if data.get("trend_state") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegimeStatistics:
    """Descriptive statistics observed within an individual regime state."""
    regime_id: int
    regime_name: str
    observation_count: int
    percentage_of_samples: float
    mean_return: float
    median_return: float
    return_std: float
    mean_volatility: float
    median_volatility: float
    mean_vix: Optional[float] = None
    mean_duration_bars: float = 0.0
    median_duration_bars: float = 0.0
    min_duration_bars: int = 0
    max_duration_bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_id": int(self.regime_id),
            "regime_name": str(self.regime_name),
            "observation_count": int(self.observation_count),
            "percentage_of_samples": round(float(self.percentage_of_samples), 4),
            "mean_return": round(float(self.mean_return), 6),
            "median_return": round(float(self.median_return), 6),
            "return_std": round(float(self.return_std), 6),
            "mean_volatility": round(float(self.mean_volatility), 6),
            "median_volatility": round(float(self.median_volatility), 6),
            "mean_vix": round(float(self.mean_vix), 4) if self.mean_vix is not None else None,
            "mean_duration_bars": round(float(self.mean_duration_bars), 2),
            "median_duration_bars": round(float(self.median_duration_bars), 2),
            "min_duration_bars": int(self.min_duration_bars),
            "max_duration_bars": int(self.max_duration_bars),
        }


@dataclass
class TransitionMatrix:
    """
    Empirical transition matrix P(S_{t+1} | S_t) between classified market regimes.
    """
    regime_names: List[str]
    matrix: List[List[float]]  # Row i: current regime, Col j: next regime
    transition_counts: List[List[int]]
    total_transitions: int
    persistence_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_names": self.regime_names,
            "matrix": [[round(float(val), 4) for val in row] for row in self.matrix],
            "transition_counts": [[int(val) for val in row] for row in self.transition_counts],
            "total_transitions": int(self.total_transitions),
            "persistence_probabilities": {k: round(float(v), 4) for k, v in self.persistence_probabilities.items()},
        }


@dataclass
class ClusteringDiagnostics:
    """Quantitative cluster separation, compactness, and likelihood diagnostics."""
    silhouette_score: Optional[float] = None
    davies_bouldin_index: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    bic: Optional[float] = None
    aic: Optional[float] = None
    log_likelihood: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "silhouette_score": round(float(self.silhouette_score), 4) if self.silhouette_score is not None else None,
            "davies_bouldin_index": round(float(self.davies_bouldin_index), 4) if self.davies_bouldin_index is not None else None,
            "calinski_harabasz_score": round(float(self.calinski_harabasz_score), 4) if self.calinski_harabasz_score is not None else None,
            "bic": round(float(self.bic), 2) if self.bic is not None else None,
            "aic": round(float(self.aic), 2) if self.aic is not None else None,
            "log_likelihood": round(float(self.log_likelihood), 4) if self.log_likelihood is not None else None,
        }


@dataclass
class RegimeDetectorMetadata:
    """Full reproducibility and auditing metadata for a fitted regime detector artifact."""
    detector_type: str
    detector_version: str
    trained_at: datetime
    feature_names: List[str]
    n_regimes: int
    semantic_mapping: Dict[int, str]
    hyperparameters: Dict[str, Any]
    train_period_start: Optional[datetime] = None
    train_period_end: Optional[datetime] = None
    val_period_start: Optional[datetime] = None
    val_period_end: Optional[datetime] = None
    test_period_start: Optional[datetime] = None
    test_period_end: Optional[datetime] = None
    train_diagnostics: Optional[ClusteringDiagnostics] = None
    val_diagnostics: Optional[ClusteringDiagnostics] = None
    test_diagnostics: Optional[ClusteringDiagnostics] = None
    statistics_per_regime: Optional[Dict[str, Any]] = None
    transition_matrix: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_type": self.detector_type,
            "detector_version": self.detector_version,
            "trained_at": self.trained_at.isoformat(),
            "feature_names": self.feature_names,
            "n_regimes": int(self.n_regimes),
            "semantic_mapping": {str(k): str(v) for k, v in self.semantic_mapping.items()},
            "hyperparameters": {str(k): (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v) for k, v in self.hyperparameters.items()},
            "train_period_start": self.train_period_start.isoformat() if self.train_period_start else None,
            "train_period_end": self.train_period_end.isoformat() if self.train_period_end else None,
            "val_period_start": self.val_period_start.isoformat() if self.val_period_start else None,
            "val_period_end": self.val_period_end.isoformat() if self.val_period_end else None,
            "test_period_start": self.test_period_start.isoformat() if self.test_period_start else None,
            "test_period_end": self.test_period_end.isoformat() if self.test_period_end else None,
            "train_diagnostics": self.train_diagnostics.to_dict() if self.train_diagnostics else None,
            "val_diagnostics": self.val_diagnostics.to_dict() if self.val_diagnostics else None,
            "test_diagnostics": self.test_diagnostics.to_dict() if self.test_diagnostics else None,
            "statistics_per_regime": self.statistics_per_regime,
            "transition_matrix": self.transition_matrix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeDetectorMetadata":
        return cls(
            detector_type=data["detector_type"],
            detector_version=data["detector_version"],
            trained_at=datetime.fromisoformat(data["trained_at"]),
            feature_names=data["feature_names"],
            n_regimes=data["n_regimes"],
            semantic_mapping={int(k): v for k, v in data.get("semantic_mapping", {}).items()},
            hyperparameters=data.get("hyperparameters", {}),
            train_period_start=datetime.fromisoformat(data["train_period_start"]) if data.get("train_period_start") else None,
            train_period_end=datetime.fromisoformat(data["train_period_end"]) if data.get("train_period_end") else None,
            val_period_start=datetime.fromisoformat(data["val_period_start"]) if data.get("val_period_start") else None,
            val_period_end=datetime.fromisoformat(data["val_period_end"]) if data.get("val_period_end") else None,
            test_period_start=datetime.fromisoformat(data["test_period_start"]) if data.get("test_period_start") else None,
            test_period_end=datetime.fromisoformat(data["test_period_end"]) if data.get("test_period_end") else None,
            statistics_per_regime=data.get("statistics_per_regime"),
            transition_matrix=data.get("transition_matrix"),
        )
