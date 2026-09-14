"""
Signal Engine Domain Schemas and Data Models.
Defines directional signals, standardized predictions, signal candidates, reason codes, configuration, and evaluation schemas.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class SignalDirection(str, Enum):
    """
    Standardized directional trading signal states.
    LONG: Sufficient model evidence supports positive directional expectation.
    SHORT: Sufficient model evidence supports negative directional expectation.
    FLAT: Evidence is insufficient, conflicting, below confidence threshold, or filtered by regime gating.
    """
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class ReasonCode(str, Enum):
    """Auditable, machine-readable explanation codes for signal generation decisions."""
    # Consensus & Directional Triggers
    MODEL_CONSENSUS = "MODEL_CONSENSUS"
    BULLISH_ENSEMBLE_SCORE = "BULLISH_ENSEMBLE_SCORE"
    BEARISH_ENSEMBLE_SCORE = "BEARISH_ENSEMBLE_SCORE"
    
    # Abstention & Filtering Reasons
    NEUTRAL_SCORE = "NEUTRAL_SCORE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    INSUFFICIENT_ACTIVE_MODELS = "INSUFFICIENT_ACTIVE_MODELS"
    NO_PREDICTIONS_AVAILABLE = "NO_PREDICTIONS_AVAILABLE"
    
    # Regime Gating Reasons
    REGIME_PERMITTED_BULL = "REGIME_PERMITTED_BULL"
    REGIME_PERMITTED_BEAR = "REGIME_PERMITTED_BEAR"
    REGIME_BLOCKED_COUNTER_TREND = "REGIME_BLOCKED_COUNTER_TREND"
    REGIME_ELEVATED_THRESHOLD_REJECTED = "REGIME_ELEVATED_THRESHOLD_REJECTED"
    REGIME_SIDEWAYS_FILTER = "REGIME_SIDEWAYS_FILTER"
    REGIME_UNKNOWN_DEFAULT = "REGIME_UNKNOWN_DEFAULT"
    REGIME_GATING_DISABLED = "REGIME_GATING_DISABLED"


@dataclass
class StandardizedPrediction:
    """
    Standardized single-model prediction output at a specific timestamp.
    Normalizes diverse ML models into a unified schema for ensemble combination.
    
    NOTE:
    - probability_up is the raw model directional probability.
    - directional_score is centered in [-1.0, 1.0].
    - confidence is a model-derived relative conviction score in [0.0, 1.0], NOT a calibrated probability of profit.
    - expected_return is populated ONLY if the upstream model genuinely produces an expected return estimate;
      it is never fabricated from directional_score or probability_up.
    """
    symbol: str
    timestamp: datetime
    model_name: str
    probability_up: float
    directional_score: float  # Centered in [-1.0, 1.0]
    confidence: float  # Model conviction in [0.0, 1.0], not calibrated profit probability
    expected_return: Optional[float] = None
    model_type: Optional[str] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.probability_up <= 1.0):
            raise ValueError(f"probability_up must be in [0.0, 1.0], got {self.probability_up}")
        if not (-1.0 <= self.directional_score <= 1.0):
            raise ValueError(f"directional_score must be in [-1.0, 1.0], got {self.directional_score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "model_type": self.model_type,
            "probability_up": round(float(self.probability_up), 4),
            "directional_score": round(float(self.directional_score), 4),
            "confidence": round(float(self.confidence), 4),
            "expected_return": round(float(self.expected_return), 6) if self.expected_return is not None else None,
            "weight": round(float(self.weight), 4),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardizedPrediction":
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model_name=data["model_name"],
            model_type=data.get("model_type"),
            probability_up=float(data["probability_up"]),
            directional_score=float(data["directional_score"]),
            confidence=float(data["confidence"]),
            expected_return=float(data["expected_return"]) if data.get("expected_return") is not None else None,
            weight=float(data.get("weight", 1.0)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SignalCandidate:
    """
    Standardized trade signal candidate output produced by the Signal Engine.
    Represents directional expectation, conviction, and reasoning at time t.
    
    IMPORTANT ARCHITECTURAL BOUNDARIES:
    - Does NOT contain position sizing, execution orders, capital allocation, or portfolio weights.
    - confidence represents ensemble model conviction in [0.0, 1.0], NOT a calibrated probability of profit.
    - expected_return is populated ONLY when upstream models provide genuine return forecasts; never fabricated.
    - signal_horizon describes the forecast horizon of upstream predictions (in bars) and does NOT define a holding period.
    """
    timestamp: datetime
    symbol: str
    signal: SignalDirection
    signal_strength: float  # Directional strength in [-1.0, 1.0] (negative = bearish, positive = bullish)
    confidence: float  # Model conviction in [0.0, 1.0], not calibrated profit probability
    direction_probability: float = 0.5  # Aggregate probability of upward movement in [0.0, 1.0]
    expected_return: Optional[float] = None

    regime: Optional[str] = None
    model_contributions: Dict[str, float] = field(default_factory=dict)
    model_agreement: float = 0.0  # Agreement ratio in [0.0, 1.0]
    bullish_votes: int = 0
    bearish_votes: int = 0
    neutral_votes: int = 0
    active_model_count: int = 0
    reason_codes: List[str] = field(default_factory=list)
    signal_horizon: int = 5  # Forecast horizon in bars; does not mandate holding period
    signal_version: str = "signal-v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (-1.0 <= self.signal_strength <= 1.0):
            raise ValueError(f"signal_strength must be in [-1.0, 1.0], got {self.signal_strength}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not (0.0 <= self.direction_probability <= 1.0):
            raise ValueError(f"direction_probability must be in [0.0, 1.0], got {self.direction_probability}")
        if not (0.0 <= self.model_agreement <= 1.0):
            raise ValueError(f"model_agreement must be in [0.0, 1.0], got {self.model_agreement}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "signal": self.signal.value,
            "signal_strength": round(float(self.signal_strength), 4),
            "confidence": round(float(self.confidence), 4),
            "direction_probability": round(float(self.direction_probability), 4),
            "expected_return": round(float(self.expected_return), 6) if self.expected_return is not None else None,
            "regime": self.regime,
            "model_contributions": {k: round(float(v), 4) for k, v in self.model_contributions.items()},
            "model_agreement": round(float(self.model_agreement), 4),
            "bullish_votes": int(self.bullish_votes),
            "bearish_votes": int(self.bearish_votes),
            "neutral_votes": int(self.neutral_votes),
            "active_model_count": int(self.active_model_count),
            "reason_codes": self.reason_codes,
            "signal_horizon": int(self.signal_horizon),
            "signal_version": self.signal_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalCandidate":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            signal=SignalDirection(data["signal"]),
            signal_strength=float(data["signal_strength"]),
            confidence=float(data["confidence"]),
            direction_probability=float(data["direction_probability"]),
            expected_return=float(data["expected_return"]) if data.get("expected_return") is not None else None,
            regime=data.get("regime"),
            model_contributions={k: float(v) for k, v in data.get("model_contributions", {}).items()},
            model_agreement=float(data.get("model_agreement", 0.0)),
            bullish_votes=int(data.get("bullish_votes", 0)),
            bearish_votes=int(data.get("bearish_votes", 0)),
            neutral_votes=int(data.get("neutral_votes", 0)),
            active_model_count=int(data.get("active_model_count", 0)),
            reason_codes=list(data.get("reason_codes", [])),
            signal_horizon=int(data.get("signal_horizon", 5)),
            signal_version=data.get("signal_version", "signal-v1"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SignalEngineConfig:
    """
    Configuration parameters governing signal synthesis, thresholds, and regime gating.
    
    NOTE ON MODEL WEIGHTS:
    Default ensemble weights are configurable heuristic defaults and have not been established
    as optimal through out-of-sample validation. They can be configured or updated per deployment.
    """
    long_threshold: float = 0.20  # Minimum ensemble score required for LONG
    short_threshold: float = -0.20  # Maximum ensemble score required for SHORT
    min_confidence: float = 0.25  # Minimum confidence required to emit non-FLAT signal
    min_active_models: int = 1  # Minimum active model predictions required
    min_agreement: float = 0.50  # Minimum model agreement ratio required
    default_horizon: int = 5  # Forward prediction horizon in bars; does not mandate holding period
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "logistic_regression": 0.10,
        "random_forest": 0.15,
        "xgboost": 0.20,
        "lightgbm": 0.20,
        "mlp": 0.10,
        "lstm": 0.10,
        "transformer": 0.15,
    })
    regime_gating_enabled: bool = True
    counter_trend_policy: str = "filter"  # "filter" (block counter-trend), "penalize" (elevate threshold), "allow"
    sideways_confidence_multiplier: float = 1.30  # Multiplier on min_confidence in sideways / high vol regimes
    version: str = "signal-v1"

    def __post_init__(self):
        if self.short_threshold >= self.long_threshold:
            raise ValueError(f"short_threshold ({self.short_threshold}) must be strictly less than long_threshold ({self.long_threshold})")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}")
        if not (0.0 <= self.min_agreement <= 1.0):
            raise ValueError(f"min_agreement must be in [0.0, 1.0], got {self.min_agreement}")
        if self.min_active_models < 1:
            raise ValueError(f"min_active_models must be >= 1, got {self.min_active_models}")
        for k, v in self.model_weights.items():
            if v < 0:
                raise ValueError(f"Model weight for {k} must be non-negative, got {v}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "long_threshold": float(self.long_threshold),
            "short_threshold": float(self.short_threshold),
            "min_confidence": float(self.min_confidence),
            "min_active_models": int(self.min_active_models),
            "min_agreement": float(self.min_agreement),
            "default_horizon": int(self.default_horizon),
            "model_weights": {k: float(v) for k, v in self.model_weights.items()},
            "regime_gating_enabled": bool(self.regime_gating_enabled),
            "counter_trend_policy": self.counter_trend_policy,
            "sideways_confidence_multiplier": float(self.sideways_confidence_multiplier),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalEngineConfig":
        return cls(
            long_threshold=float(data.get("long_threshold", 0.20)),
            short_threshold=float(data.get("short_threshold", -0.20)),
            min_confidence=float(data.get("min_confidence", 0.25)),
            min_active_models=int(data.get("min_active_models", 1)),
            min_agreement=float(data.get("min_agreement", 0.50)),
            default_horizon=int(data.get("default_horizon", 5)),
            model_weights={k: float(v) for k, v in data.get("model_weights", {}).items()} if "model_weights" in data else None,
            regime_gating_enabled=bool(data.get("regime_gating_enabled", True)),
            counter_trend_policy=data.get("counter_trend_policy", "filter"),
            sideways_confidence_multiplier=float(data.get("sideways_confidence_multiplier", 1.30)),
            version=data.get("version", "signal-v1"),
        )


@dataclass
class SignalEvaluationReport:
    """
    Diagnostic evaluation metrics computed across a sequence of generated signals.
    
    CRITICAL DISTINCTION:
    This is statistical diagnostic signal evaluation, NOT a trading backtest or P&L execution simulation.
    Does not simulate transaction costs, bid-ask spread, slippage, order fills, or portfolio equity curves.
    """
    total_signals: int
    long_count: int
    short_count: int
    flat_count: int
    long_percentage: float
    short_percentage: float
    flat_percentage: float
    directional_accuracy: Optional[float] = None
    long_precision: Optional[float] = None
    long_recall: Optional[float] = None
    long_f1: Optional[float] = None
    short_precision: Optional[float] = None
    short_recall: Optional[float] = None
    short_f1: Optional[float] = None
    mean_forward_return_long: Optional[float] = None
    mean_forward_return_short: Optional[float] = None
    mean_forward_return_flat: Optional[float] = None
    return_spread: Optional[float] = None  # mean_return_long - mean_return_short
    mean_confidence: float = 0.0
    mean_agreement: float = 0.0
    evaluation_period_start: Optional[datetime] = None
    evaluation_period_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_signals": int(self.total_signals),
            "long_count": int(self.long_count),
            "short_count": int(self.short_count),
            "flat_count": int(self.flat_count),
            "long_percentage": round(float(self.long_percentage), 4),
            "short_percentage": round(float(self.short_percentage), 4),
            "flat_percentage": round(float(self.flat_percentage), 4),
            "directional_accuracy": round(float(self.directional_accuracy), 4) if self.directional_accuracy is not None else None,
            "long_precision": round(float(self.long_precision), 4) if self.long_precision is not None else None,
            "long_recall": round(float(self.long_recall), 4) if self.long_recall is not None else None,
            "long_f1": round(float(self.long_f1), 4) if self.long_f1 is not None else None,
            "short_precision": round(float(self.short_precision), 4) if self.short_precision is not None else None,
            "short_recall": round(float(self.short_recall), 4) if self.short_recall is not None else None,
            "short_f1": round(float(self.short_f1), 4) if self.short_f1 is not None else None,
            "mean_forward_return_long": round(float(self.mean_forward_return_long), 6) if self.mean_forward_return_long is not None else None,
            "mean_forward_return_short": round(float(self.mean_forward_return_short), 6) if self.mean_forward_return_short is not None else None,
            "mean_forward_return_flat": round(float(self.mean_forward_return_flat), 6) if self.mean_forward_return_flat is not None else None,
            "return_spread": round(float(self.return_spread), 6) if self.return_spread is not None else None,
            "mean_confidence": round(float(self.mean_confidence), 4),
            "mean_agreement": round(float(self.mean_agreement), 4),
            "evaluation_period_start": self.evaluation_period_start.isoformat() if self.evaluation_period_start else None,
            "evaluation_period_end": self.evaluation_period_end.isoformat() if self.evaluation_period_end else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalEvaluationReport":
        return cls(
            total_signals=int(data["total_signals"]),
            long_count=int(data["long_count"]),
            short_count=int(data["short_count"]),
            flat_count=int(data["flat_count"]),
            long_percentage=float(data["long_percentage"]),
            short_percentage=float(data["short_percentage"]),
            flat_percentage=float(data["flat_percentage"]),
            directional_accuracy=float(data["directional_accuracy"]) if data.get("directional_accuracy") is not None else None,
            long_precision=float(data["long_precision"]) if data.get("long_precision") is not None else None,
            long_recall=float(data["long_recall"]) if data.get("long_recall") is not None else None,
            long_f1=float(data["long_f1"]) if data.get("long_f1") is not None else None,
            short_precision=float(data["short_precision"]) if data.get("short_precision") is not None else None,
            short_recall=float(data["short_recall"]) if data.get("short_recall") is not None else None,
            short_f1=float(data["short_f1"]) if data.get("short_f1") is not None else None,
            mean_forward_return_long=float(data["mean_forward_return_long"]) if data.get("mean_forward_return_long") is not None else None,
            mean_forward_return_short=float(data["mean_forward_return_short"]) if data.get("mean_forward_return_short") is not None else None,
            mean_forward_return_flat=float(data["mean_forward_return_flat"]) if data.get("mean_forward_return_flat") is not None else None,
            return_spread=float(data["return_spread"]) if data.get("return_spread") is not None else None,
            mean_confidence=float(data.get("mean_confidence", 0.0)),
            mean_agreement=float(data.get("mean_agreement", 0.0)),
            evaluation_period_start=datetime.fromisoformat(data["evaluation_period_start"]) if data.get("evaluation_period_start") else None,
            evaluation_period_end=datetime.fromisoformat(data["evaluation_period_end"]) if data.get("evaluation_period_end") else None,
            metadata=data.get("metadata", {}),
        )
