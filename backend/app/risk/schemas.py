"""
Risk Engine Domain Schemas and Data Models (Phase 12 Hardening v1.1).
Defines risk checks, violations, assessments, metrics, adjusted targets, capital contexts,
audit records, requests, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.strategy.schemas import SignalDirection


class RiskLimitSeverity(str, Enum):
    """Severity classification for risk limits."""
    HARD = "hard"  # Must not be breached under any circumstances in final adjusted targets
    SOFT = "soft"  # Triggers warnings/diagnostics without necessarily disqualifying portfolio


class RiskViolationCode(str, Enum):
    """Standardized deterministic violation reason codes for risk checks."""
    MAX_POSITION_WEIGHT = "max_position_weight"
    MAX_GROSS_EXPOSURE = "max_gross_exposure"
    MAX_NET_EXPOSURE = "max_net_exposure"
    MAX_LONG_EXPOSURE = "max_long_exposure"
    MAX_SHORT_EXPOSURE = "max_short_exposure"
    MAX_POSITIONS = "max_positions"
    MAX_LEVERAGE = "max_leverage"
    VOLATILITY_LIMIT = "volatility_limit"
    CORRELATION_LIMIT = "correlation_limit"
    TURNOVER_LIMIT = "turnover_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_WEIGHT = "invalid_weight"
    NON_FINITE_WEIGHT = "non_finite_weight"
    DUPLICATE_SYMBOL = "duplicate_symbol"
    INVALID_CAPITAL = "invalid_capital"
    INVALID_CONTEXT = "invalid_context"
    UNSUPPORTED_MODE = "unsupported_mode"


class RiskAdjustmentAction(str, Enum):
    """Action taken by the Risk Adjustment Engine on an individual target."""
    UNCHANGED = "UNCHANGED"
    REDUCED = "REDUCED"
    REMOVED = "REMOVED"
    SCALED = "SCALED"
    REJECTED = "REJECTED"


class RiskStatus(str, Enum):
    """Overall execution status of risk evaluation."""
    VALID = "VALID"
    INVALID = "INVALID"
    RISK_REJECTED = "RISK_REJECTED"
    RISK_ADJUSTED = "RISK_ADJUSTED"


@dataclass
class PortfolioCapitalContext:
    """
    Explicit portfolio capital/equity context.
    Distinguishes portfolio equity, target weights, target notionals, and quantities.
    """
    equity: float
    base_currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.equity, (int, float)):
            raise ValueError(f"equity must be numeric, got {type(self.equity)}")
        if np.isnan(self.equity) or np.isinf(self.equity):
            raise ValueError(f"equity must be finite, got {self.equity}")
        if self.equity <= 0.0:
            raise ValueError(f"equity must be strictly positive, got {self.equity}")
        if not self.base_currency or not isinstance(self.base_currency, str):
            raise ValueError(f"base_currency must be a non-empty string, got {self.base_currency}")

    def calculate_notional(self, weight: float) -> float:
        """Calculate target notional allocation: target_weight * portfolio_equity."""
        if np.isnan(weight) or np.isinf(weight):
            raise ValueError(f"weight must be finite, got {weight}")
        return float(weight * self.equity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": round(float(self.equity), 2),
            "base_currency": self.base_currency,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioCapitalContext":
        return cls(
            equity=float(data["equity"]),
            base_currency=data.get("base_currency", "USD"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RiskCheck:
    """Represents an individual evaluated risk check."""
    name: str
    passed: bool
    severity: RiskLimitSeverity
    threshold: Optional[float] = None
    observed_value: Optional[float] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity.value,
            "threshold": round(float(self.threshold), 6) if self.threshold is not None else None,
            "observed_value": round(float(self.observed_value), 6) if self.observed_value is not None else None,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskCheck":
        return cls(
            name=data["name"],
            passed=bool(data["passed"]),
            severity=RiskLimitSeverity(data["severity"]),
            threshold=float(data["threshold"]) if data.get("threshold") is not None else None,
            observed_value=float(data["observed_value"]) if data.get("observed_value") is not None else None,
            message=data.get("message", ""),
        )


@dataclass
class RiskViolation:
    """Represents an identified risk limit breach with full audit trail."""
    code: RiskViolationCode
    severity: RiskLimitSeverity
    threshold: float
    observed_value: float
    message: str
    affected_symbols: List[str] = field(default_factory=list)
    was_adjusted: bool = False
    adjustment_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "threshold": round(float(self.threshold), 6),
            "observed_value": round(float(self.observed_value), 6) if not np.isnan(self.observed_value) else None,
            "message": self.message,
            "affected_symbols": sorted(self.affected_symbols),
            "was_adjusted": self.was_adjusted,
            "adjustment_details": self.adjustment_details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskViolation":
        obs = data.get("observed_value")
        return cls(
            code=RiskViolationCode(data["code"]),
            severity=RiskLimitSeverity(data["severity"]),
            threshold=float(data["threshold"]),
            observed_value=float(obs) if obs is not None else float("nan"),
            message=data["message"],
            affected_symbols=data.get("affected_symbols", []),
            was_adjusted=bool(data.get("was_adjusted", False)),
            adjustment_details=data.get("adjustment_details"),
        )


@dataclass
class RiskMetrics:
    """
    Calculated portfolio-level risk metrics.
    
    SCIENTIFIC INTEGRITY RULES:
    1. If point-in-time data is unavailable for higher-order metrics (volatility, correlation, drawdown),
       it is represented as None and cataloged in unavailable_metrics. Data is never fabricated.
    2. Leverage is defined explicitly as: gross_notional / portfolio_equity = sum(|weight|).
    """
    gross_exposure: float
    net_exposure: float
    total_long_weight: float
    total_short_weight: float
    active_position_count: int
    max_observed_position_weight: float
    leverage: float
    portfolio_equity: Optional[float] = None
    gross_notional: Optional[float] = None
    net_notional: Optional[float] = None
    portfolio_volatility: Optional[float] = None
    average_correlation: Optional[float] = None
    turnover: Optional[float] = None
    highly_correlated_pairs: List[Dict[str, Any]] = field(default_factory=list)
    drawdown_limit_status: str = "UNAVAILABLE"
    daily_loss_limit_status: str = "UNAVAILABLE"
    unavailable_metrics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gross_exposure": round(float(self.gross_exposure), 6),
            "net_exposure": round(float(self.net_exposure), 6),
            "total_long_weight": round(float(self.total_long_weight), 6),
            "total_short_weight": round(float(self.total_short_weight), 6),
            "active_position_count": int(self.active_position_count),
            "max_observed_position_weight": round(float(self.max_observed_position_weight), 6),
            "leverage": round(float(self.leverage), 6),
            "portfolio_equity": round(float(self.portfolio_equity), 2) if self.portfolio_equity is not None else None,
            "gross_notional": round(float(self.gross_notional), 2) if self.gross_notional is not None else None,
            "net_notional": round(float(self.net_notional), 2) if self.net_notional is not None else None,
            "portfolio_volatility": round(float(self.portfolio_volatility), 6) if self.portfolio_volatility is not None else None,
            "average_correlation": round(float(self.average_correlation), 6) if self.average_correlation is not None else None,
            "turnover": round(float(self.turnover), 6) if self.turnover is not None else None,
            "highly_correlated_pairs": self.highly_correlated_pairs,
            "drawdown_limit_status": self.drawdown_limit_status,
            "daily_loss_limit_status": self.daily_loss_limit_status,
            "unavailable_metrics": sorted(self.unavailable_metrics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskMetrics":
        return cls(
            gross_exposure=float(data["gross_exposure"]),
            net_exposure=float(data["net_exposure"]),
            total_long_weight=float(data["total_long_weight"]),
            total_short_weight=float(data["total_short_weight"]),
            active_position_count=int(data["active_position_count"]),
            max_observed_position_weight=float(data["max_observed_position_weight"]),
            leverage=float(data["leverage"]),
            portfolio_equity=float(data["portfolio_equity"]) if data.get("portfolio_equity") is not None else None,
            gross_notional=float(data["gross_notional"]) if data.get("gross_notional") is not None else None,
            net_notional=float(data["net_notional"]) if data.get("net_notional") is not None else None,
            portfolio_volatility=float(data["portfolio_volatility"]) if data.get("portfolio_volatility") is not None else None,
            average_correlation=float(data["average_correlation"]) if data.get("average_correlation") is not None else None,
            turnover=float(data["turnover"]) if data.get("turnover") is not None else None,
            highly_correlated_pairs=data.get("highly_correlated_pairs", []),
            drawdown_limit_status=data.get("drawdown_limit_status", "UNAVAILABLE"),
            daily_loss_limit_status=data.get("daily_loss_limit_status", "UNAVAILABLE"),
            unavailable_metrics=data.get("unavailable_metrics", []),
        )


@dataclass
class RiskAssessment:
    """Comprehensive compliance evaluation of a portfolio against configured limits."""
    is_compliant: bool
    checks: List[RiskCheck]
    violations: List[RiskViolation]
    metrics: RiskMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "checks": [c.to_dict() for c in self.checks],
            "violations": [v.to_dict() for v in self.violations],
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskAssessment":
        return cls(
            is_compliant=bool(data["is_compliant"]),
            checks=[RiskCheck.from_dict(c) for c in data.get("checks", [])],
            violations=[RiskViolation.from_dict(v) for v in data.get("violations", [])],
            metrics=RiskMetrics.from_dict(data["metrics"]),
        )


@dataclass
class RiskAdjustmentRecord:
    """
    Structured, deterministic audit record for an individual symbol adjustment.
    """
    symbol: str
    original_weight: float
    final_weight: float
    adjustment_amount: float
    action: RiskAdjustmentAction
    adjustment_reason: Optional[str] = None
    violation_code: Optional[RiskViolationCode] = None
    original_notional: Optional[float] = None
    final_notional: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "original_weight": round(float(self.original_weight), 6),
            "final_weight": round(float(self.final_weight), 6),
            "adjustment_amount": round(float(self.adjustment_amount), 6),
            "action": self.action.value,
            "adjustment_reason": self.adjustment_reason,
            "violation_code": self.violation_code.value if self.violation_code else None,
            "original_notional": round(float(self.original_notional), 2) if self.original_notional is not None else None,
            "final_notional": round(float(self.final_notional), 2) if self.final_notional is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskAdjustmentRecord":
        return cls(
            symbol=data["symbol"],
            original_weight=float(data["original_weight"]),
            final_weight=float(data["final_weight"]),
            adjustment_amount=float(data["adjustment_amount"]),
            action=RiskAdjustmentAction(data["action"]),
            adjustment_reason=data.get("adjustment_reason"),
            violation_code=RiskViolationCode(data["violation_code"]) if data.get("violation_code") else None,
            original_notional=float(data["original_notional"]) if data.get("original_notional") is not None else None,
            final_notional=float(data["final_notional"]) if data.get("final_notional") is not None else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class RiskAdjustedTarget:
    """
    Represents an individual target allocation after risk review/adjustment.
    Preserves audit link to original allocation, capital notional, and quantity.
    """
    symbol: str
    original_weight: float
    adjusted_weight: float
    signal_direction: SignalDirection
    signal_strength: float
    confidence: float
    action: RiskAdjustmentAction = RiskAdjustmentAction.UNCHANGED
    was_adjusted: bool = False
    adjustment_reason: Optional[str] = None
    target_notional: Optional[float] = None
    shares: Optional[float] = None  # Calculated only if explicit valid point-in-time price supplied
    expected_return: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (-1.0 <= self.adjusted_weight <= 1.0):
            raise ValueError(f"adjusted_weight must be in [-1.0, 1.0], got {self.adjusted_weight}")
        if np.isnan(self.adjusted_weight) or np.isinf(self.adjusted_weight):
            raise ValueError(f"adjusted_weight must be finite, got {self.adjusted_weight}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "original_weight": round(float(self.original_weight), 6),
            "adjusted_weight": round(float(self.adjusted_weight), 6),
            "signal_direction": self.signal_direction.value,
            "signal_strength": round(float(self.signal_strength), 4),
            "confidence": round(float(self.confidence), 4),
            "action": self.action.value,
            "was_adjusted": self.was_adjusted,
            "adjustment_reason": self.adjustment_reason,
            "target_notional": round(float(self.target_notional), 2) if self.target_notional is not None else None,
            "shares": round(float(self.shares), 4) if self.shares is not None else None,
            "expected_return": round(float(self.expected_return), 6) if self.expected_return is not None else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskAdjustedTarget":
        act_str = data.get("action")
        action = RiskAdjustmentAction(act_str) if act_str else RiskAdjustmentAction.UNCHANGED
        return cls(
            symbol=data["symbol"],
            original_weight=float(data["original_weight"]),
            adjusted_weight=float(data["adjusted_weight"]),
            signal_direction=SignalDirection(data["signal_direction"]),
            signal_strength=float(data["signal_strength"]),
            confidence=float(data["confidence"]),
            action=action,
            was_adjusted=bool(data.get("was_adjusted", False)),
            adjustment_reason=data.get("adjustment_reason"),
            target_notional=float(data["target_notional"]) if data.get("target_notional") is not None else None,
            shares=float(data["shares"]) if data.get("shares") is not None else None,
            expected_return=float(data["expected_return"]) if data.get("expected_return") is not None else None,
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class RiskContext:
    """
    Explicit point-in-time market context required for higher-order risk metrics.
    
    POINT-IN-TIME CONTRACT:
    All contextual observations in historical_returns, asset_prices, and capital must be
    valid as-of timestamp t. Strictly zero data after t is permitted.
    """
    as_of_timestamp: Optional[datetime] = None
    timestamp: Optional[datetime] = None  # Backward compatibility alias
    capital: Optional[PortfolioCapitalContext] = None
    historical_returns: Optional[Dict[str, List[float]]] = None  # symbol -> list of past returns [r_{t-k}, ..., r_t]
    historical_volatilities: Optional[Dict[str, float]] = None  # symbol -> realized rolling volatility at t
    covariance_matrix: Optional[Dict[str, Dict[str, float]]] = None  # symbol -> {symbol -> cov}
    previous_weights: Optional[Dict[str, float]] = None  # symbol -> previous target weight for turnover tracking
    asset_prices: Optional[Dict[str, float]] = None  # symbol -> point-in-time price at t (for quantity calculation)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.as_of_timestamp is None and self.timestamp is not None:
            self.as_of_timestamp = self.timestamp
        elif self.timestamp is None and self.as_of_timestamp is not None:
            self.timestamp = self.as_of_timestamp


@dataclass
class RiskEngineConfig:
    """
    Configuration governing risk assessment limits and deterministic adjustment behavior.
    
    NOTE:
    Values are conservative engineering defaults, not empirically optimized parameters.
    """
    enabled: bool = True
    max_position_weight: float = 0.10  # 10% per asset limit
    max_gross_exposure: float = 1.00  # 100% gross limit
    max_net_exposure: float = 1.00  # 100% net limit
    max_long_exposure: float = 1.00  # 100% long limit
    max_short_exposure: float = 1.00  # 100% short limit
    max_positions: int = 20  # Max active positions limit
    max_leverage: float = 1.00  # 1.0x unleveraged capital base limit
    volatility_limit: Optional[float] = None  # Optional portfolio volatility cap
    correlation_limit: Optional[float] = None  # Optional avg correlation cap
    turnover_limit: Optional[float] = None  # Optional turnover limit per rebalance
    max_daily_loss: Optional[float] = None  # Contract placeholder for Phase 13
    max_drawdown: Optional[float] = None  # Contract placeholder for Phase 13
    adjustment_enabled: bool = True  # Enable deterministic adjustment of violating targets
    version: str = "risk-v1.1"

    def __post_init__(self):
        if not (0.0 < self.max_position_weight <= 1.0):
            raise ValueError(f"max_position_weight must be in (0.0, 1.0], got {self.max_position_weight}")
        if self.max_gross_exposure <= 0.0:
            raise ValueError(f"max_gross_exposure must be > 0.0, got {self.max_gross_exposure}")
        if self.max_net_exposure <= 0.0:
            raise ValueError(f"max_net_exposure must be > 0.0, got {self.max_net_exposure}")
        if self.max_long_exposure <= 0.0:
            raise ValueError(f"max_long_exposure must be > 0.0, got {self.max_long_exposure}")
        if self.max_short_exposure <= 0.0:
            raise ValueError(f"max_short_exposure must be > 0.0, got {self.max_short_exposure}")
        if self.max_positions < 1:
            raise ValueError(f"max_positions must be >= 1, got {self.max_positions}")
        if self.max_leverage <= 0.0:
            raise ValueError(f"max_leverage must be > 0.0, got {self.max_leverage}")
        if self.volatility_limit is not None and self.volatility_limit <= 0.0:
            raise ValueError(f"volatility_limit must be > 0.0, got {self.volatility_limit}")
        if self.correlation_limit is not None and not (-1.0 <= self.correlation_limit <= 1.0):
            raise ValueError(f"correlation_limit must be in [-1.0, 1.0], got {self.correlation_limit}")
        if self.turnover_limit is not None and self.turnover_limit <= 0.0:
            raise ValueError(f"turnover_limit must be > 0.0, got {self.turnover_limit}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_position_weight": float(self.max_position_weight),
            "max_gross_exposure": float(self.max_gross_exposure),
            "max_net_exposure": float(self.max_net_exposure),
            "max_long_exposure": float(self.max_long_exposure),
            "max_short_exposure": float(self.max_short_exposure),
            "max_positions": int(self.max_positions),
            "max_leverage": float(self.max_leverage),
            "volatility_limit": float(self.volatility_limit) if self.volatility_limit is not None else None,
            "correlation_limit": float(self.correlation_limit) if self.correlation_limit is not None else None,
            "turnover_limit": float(self.turnover_limit) if self.turnover_limit is not None else None,
            "max_daily_loss": float(self.max_daily_loss) if self.max_daily_loss is not None else None,
            "max_drawdown": float(self.max_drawdown) if self.max_drawdown is not None else None,
            "adjustment_enabled": self.adjustment_enabled,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskEngineConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_position_weight=float(data.get("max_position_weight", 0.10)),
            max_gross_exposure=float(data.get("max_gross_exposure", 1.00)),
            max_net_exposure=float(data.get("max_net_exposure", 1.00)),
            max_long_exposure=float(data.get("max_long_exposure", 1.00)),
            max_short_exposure=float(data.get("max_short_exposure", 1.00)),
            max_positions=int(data.get("max_positions", 20)),
            max_leverage=float(data.get("max_leverage", 1.00)),
            volatility_limit=float(data["volatility_limit"]) if data.get("volatility_limit") is not None else None,
            correlation_limit=float(data["correlation_limit"]) if data.get("correlation_limit") is not None else None,
            turnover_limit=float(data["turnover_limit"]) if data.get("turnover_limit") is not None else None,
            max_daily_loss=float(data["max_daily_loss"]) if data.get("max_daily_loss") is not None else None,
            max_drawdown=float(data["max_drawdown"]) if data.get("max_drawdown") is not None else None,
            adjustment_enabled=bool(data.get("adjustment_enabled", True)),
            version=data.get("version", "risk-v1.1"),
        )


@dataclass
class RiskEngineRequest:
    """Request payload passed to the Risk Engine at timestamp t."""
    targets: List[PortfolioTarget]
    context: Optional[RiskContext] = None
    config: Optional[RiskEngineConfig] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskEngineResult:
    """
    Auditable result of Risk Engine assessment and constraint enforcement at timestamp t.
    
    CRITICAL DISTINCTION:
    Represents risk-evaluated / adjusted target allocations.
    Does NOT execute trades, manage broker connectivity, or simulate P&L.
    """
    timestamp: datetime
    is_compliant: bool
    was_adjusted: bool
    original_targets: List[PortfolioTarget]
    adjusted_targets: List[RiskAdjustedTarget]
    assessment: RiskAssessment
    status: RiskStatus = RiskStatus.VALID
    audit_trail: List[RiskAdjustmentRecord] = field(default_factory=list)
    rejected_positions: Dict[str, str] = field(default_factory=dict)  # symbol -> reason
    capital: Optional[PortfolioCapitalContext] = None
    construction_version: str = "portfolio-v1"
    risk_version: str = "risk-v1.1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "is_compliant": self.is_compliant,
            "was_adjusted": self.was_adjusted,
            "status": self.status.value,
            "original_targets": [t.to_dict() for t in self.original_targets],
            "adjusted_targets": [t.to_dict() for t in self.adjusted_targets],
            "assessment": self.assessment.to_dict(),
            "audit_trail": [a.to_dict() for a in self.audit_trail],
            "rejected_positions": self.rejected_positions,
            "capital": self.capital.to_dict() if self.capital else None,
            "construction_version": self.construction_version,
            "risk_version": self.risk_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskEngineResult":
        st_val = data.get("status")
        status = RiskStatus(st_val) if st_val else RiskStatus.VALID
        cap_val = data.get("capital")
        capital = PortfolioCapitalContext.from_dict(cap_val) if cap_val else None

        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            is_compliant=bool(data["is_compliant"]),
            was_adjusted=bool(data["was_adjusted"]),
            status=status,
            original_targets=[PortfolioTarget.from_dict(t) for t in data.get("original_targets", [])],
            adjusted_targets=[RiskAdjustedTarget.from_dict(t) for t in data.get("adjusted_targets", [])],
            assessment=RiskAssessment.from_dict(data["assessment"]),
            audit_trail=[RiskAdjustmentRecord.from_dict(a) for a in data.get("audit_trail", [])],
            rejected_positions=data.get("rejected_positions", {}),
            capital=capital,
            construction_version=data.get("construction_version", "portfolio-v1"),
            risk_version=data.get("risk_version", "risk-v1.1"),
            metadata=data.get("metadata", {}),
        )
