"""
Portfolio Construction Domain Schemas and Data Models (Phase 11).
Defines portfolio targets, allocation requests/results, constraints, and configuration schemas.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.strategy.schemas import SignalCandidate, SignalDirection


class PortfolioMode(str, Enum):
    """Supported portfolio directional modes."""
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"


@dataclass
class PortfolioTarget:
    """
    Standardized target allocation for an individual asset in the portfolio.
    
    IMPORTANT ARCHITECTURAL PROPERTIES:
    - target_weight: Target portfolio fraction [-1.0, 1.0].
      Positive for LONG, negative for SHORT, 0.0 for FLAT / unallocated.
    - Does NOT contain share quantities, order instructions, or broker execution state.
    - expected_return: Propagated from upstream signal only if legitimately supplied; never fabricated.
    """
    symbol: str
    target_weight: float  # Signed fraction of portfolio equity (e.g. 0.05 = 5% long, -0.05 = 5% short)
    signal_direction: SignalDirection
    signal_strength: float
    confidence: float
    expected_return: Optional[float] = None
    signal_version: str = "signal-v1"
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (-1.0 <= self.target_weight <= 1.0):
            raise ValueError(f"target_weight must be in [-1.0, 1.0], got {self.target_weight}")
        if np.isnan(self.target_weight) or np.isinf(self.target_weight):
            raise ValueError(f"target_weight must be finite, got {self.target_weight}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_weight": round(float(self.target_weight), 6),
            "signal_direction": self.signal_direction.value,
            "signal_strength": round(float(self.signal_strength), 4),
            "confidence": round(float(self.confidence), 4),
            "expected_return": round(float(self.expected_return), 6) if self.expected_return is not None else None,
            "signal_version": self.signal_version,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioTarget":
        return cls(
            symbol=data["symbol"],
            target_weight=float(data["target_weight"]),
            signal_direction=SignalDirection(data["signal_direction"]),
            signal_strength=float(data["signal_strength"]),
            confidence=float(data["confidence"]),
            expected_return=float(data["expected_return"]) if data.get("expected_return") is not None else None,
            signal_version=data.get("signal_version", "signal-v1"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class PortfolioConstructionConfig:
    """
    Configuration parameters governing portfolio construction and allocation constraints.
    
    NOTE ON CONFIGURATION VALUES:
    Defaults are conservative engineering baselines, NOT parameters tuned or optimized on historical test data.
    """
    mode: PortfolioMode = PortfolioMode.LONG_ONLY
    max_position_weight: float = 0.10  # Maximum weight per asset (10%)
    min_position_weight: float = 0.01  # Minimum weight threshold for inclusion (1%)
    max_positions: int = 20  # Maximum number of concurrent active holdings
    max_gross_exposure: float = 1.00  # Maximum sum of absolute weights (100%)
    max_net_exposure: float = 1.00  # Maximum signed sum of weights (100%)
    version: str = "portfolio-v1"

    def __post_init__(self):
        if not (0.0 < self.max_position_weight <= 1.0):
            raise ValueError(f"max_position_weight must be in (0.0, 1.0], got {self.max_position_weight}")
        if not (0.0 <= self.min_position_weight <= self.max_position_weight):
            raise ValueError(
                f"min_position_weight must be in [0.0, max_position_weight], got {self.min_position_weight}"
            )
        if self.max_positions < 1:
            raise ValueError(f"max_positions must be >= 1, got {self.max_positions}")
        if self.max_gross_exposure <= 0.0:
            raise ValueError(f"max_gross_exposure must be > 0.0, got {self.max_gross_exposure}")
        if self.max_net_exposure <= 0.0:
            raise ValueError(f"max_net_exposure must be > 0.0, got {self.max_net_exposure}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "max_position_weight": float(self.max_position_weight),
            "min_position_weight": float(self.min_position_weight),
            "max_positions": int(self.max_positions),
            "max_gross_exposure": float(self.max_gross_exposure),
            "max_net_exposure": float(self.max_net_exposure),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioConstructionConfig":
        return cls(
            mode=PortfolioMode(data.get("mode", "long_only")),
            max_position_weight=float(data.get("max_position_weight", 0.10)),
            min_position_weight=float(data.get("min_position_weight", 0.01)),
            max_positions=int(data.get("max_positions", 20)),
            max_gross_exposure=float(data.get("max_gross_exposure", 1.00)),
            max_net_exposure=float(data.get("max_net_exposure", 1.00)),
            version=data.get("version", "portfolio-v1"),
        )


@dataclass
class PortfolioConstructionRequest:
    """
    Request container passed to the Portfolio Construction Engine at timestamp t.
    """
    signals: List[SignalCandidate]
    timestamp: Optional[datetime] = None
    config: Optional[PortfolioConstructionConfig] = None
    current_weights: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioConstructionResult:
    """
    Auditable result of the Portfolio Construction process at timestamp t.
    
    CRITICAL DISTINCTION:
    Represents portfolio-level target allocations. Does NOT represent order executions,
    fills, broker orders, or simulated P&L.
    """
    timestamp: datetime
    targets: List[PortfolioTarget]
    total_long_weight: float
    total_short_weight: float
    gross_exposure: float
    net_exposure: float
    active_position_count: int
    allocated_cash_weight: float  # Unallocated / reserve cash (1.0 - gross_exposure)
    rejected_signals: Dict[str, str] = field(default_factory=dict)  # symbol -> rejection reason
    construction_version: str = "portfolio-v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "targets": [t.to_dict() for t in self.targets],
            "total_long_weight": round(float(self.total_long_weight), 6),
            "total_short_weight": round(float(self.total_short_weight), 6),
            "gross_exposure": round(float(self.gross_exposure), 6),
            "net_exposure": round(float(self.net_exposure), 6),
            "active_position_count": int(self.active_position_count),
            "allocated_cash_weight": round(float(self.allocated_cash_weight), 6),
            "rejected_signals": self.rejected_signals,
            "construction_version": self.construction_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioConstructionResult":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            targets=[PortfolioTarget.from_dict(t) for t in data.get("targets", [])],
            total_long_weight=float(data["total_long_weight"]),
            total_short_weight=float(data["total_short_weight"]),
            gross_exposure=float(data["gross_exposure"]),
            net_exposure=float(data["net_exposure"]),
            active_position_count=int(data["active_position_count"]),
            allocated_cash_weight=float(data.get("allocated_cash_weight", 0.0)),
            rejected_signals=data.get("rejected_signals", {}),
            construction_version=data.get("construction_version", "portfolio-v1"),
            metadata=data.get("metadata", {}),
        )
