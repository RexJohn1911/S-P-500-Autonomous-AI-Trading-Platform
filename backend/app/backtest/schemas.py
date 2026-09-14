"""
Backtesting Engine Domain Schemas and Data Models (Phase 13).
Defines execution records, trade records, position/portfolio states, equity curves,
drawdown points, performance metrics, benchmark metrics, configuration, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.risk.schemas import RiskAdjustedTarget


class ExecutionPriceType(str, Enum):
    """Convention used for historical trade execution price."""
    NEXT_OPEN = "next_open"  # Conservative: signal at close[t], executed at open[t+1]
    NEXT_CLOSE = "next_close"  # Signal at close[t], executed at close[t+1]
    CURRENT_CLOSE = "current_close"  # Intraday/pre-close signal executed at close[t]


class OrderSide(str, Enum):
    """Order trade side."""
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(str, Enum):
    """Holding direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class AdjustmentStatus(str, Enum):
    """Corporate action adjustment status of historical market data."""
    ADJUSTED = "adjusted"
    UNADJUSTED = "unadjusted"
    UNKNOWN = "unknown"


class DataSourceType(str, Enum):
    """Classification of data source."""
    SYNTHETIC = "synthetic"
    FIXTURE = "fixture"
    LOCAL_HISTORICAL = "local_historical"
    REAL_MARKET_HISTORICAL = "real_market_historical"


@dataclass
class DatasetProvenance:
    """
    Provenance and audit metadata for historical market datasets.
    """
    dataset_id: str
    provider: str
    symbols: List[str]
    timeframe: str = "1Day"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    adjustment_status: AdjustmentStatus = AdjustmentStatus.UNKNOWN
    data_source_type: DataSourceType = DataSourceType.SYNTHETIC
    row_count: int = 0
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "provider": self.provider,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "adjustment_status": self.adjustment_status.value,
            "data_source_type": self.data_source_type.value,
            "row_count": int(self.row_count),
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetProvenance":
        return cls(
            dataset_id=data["dataset_id"],
            provider=data.get("provider", "UNKNOWN"),
            symbols=data.get("symbols", []),
            timeframe=data.get("timeframe", "1Day"),
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            adjustment_status=AdjustmentStatus(data.get("adjustment_status", "unknown")),
            data_source_type=DataSourceType(data.get("data_source_type", "synthetic")),
            row_count=int(data.get("row_count", 0)),
            checksum=data.get("checksum"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DividendEvent:
    """
    Corporate action cash dividend event.
    """
    timestamp: datetime
    symbol: str
    amount_per_share: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "amount_per_share": round(float(self.amount_per_share), 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DividendEvent":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            amount_per_share=float(data["amount_per_share"]),
        )


@dataclass
class SplitEvent:
    """
    Corporate action stock split event (e.g. 2.0 = 2-for-1 forward split).
    """
    timestamp: datetime
    symbol: str
    split_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "split_ratio": round(float(self.split_ratio), 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitEvent":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            split_ratio=float(data["split_ratio"]),
        )


@dataclass
class CostBreakdown:
    """
    Granular cost attribution across all simulated friction types.
    """
    commission_cost: float = 0.0
    slippage_cost: float = 0.0
    spread_cost: float = 0.0
    borrow_cost: float = 0.0
    market_impact_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return float(
            self.commission_cost
            + self.slippage_cost
            + self.spread_cost
            + self.borrow_cost
            + self.market_impact_cost
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commission_cost": round(float(self.commission_cost), 2),
            "slippage_cost": round(float(self.slippage_cost), 2),
            "spread_cost": round(float(self.spread_cost), 2),
            "borrow_cost": round(float(self.borrow_cost), 2),
            "market_impact_cost": round(float(self.market_impact_cost), 2),
            "total_cost": round(float(self.total_cost), 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostBreakdown":
        return cls(
            commission_cost=float(data.get("commission_cost", 0.0)),
            slippage_cost=float(data.get("slippage_cost", 0.0)),
            spread_cost=float(data.get("spread_cost", 0.0)),
            borrow_cost=float(data.get("borrow_cost", 0.0)),
            market_impact_cost=float(data.get("market_impact_cost", 0.0)),
        )


@dataclass
class PeriodPerformance:
    """
    Periodic performance breakdown (e.g. yearly or monthly).
    """
    period_label: str
    total_return: float
    annualized_return: float
    annualized_volatility: Optional[float] = None
    max_drawdown: float = 0.0
    trade_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_label": self.period_label,
            "total_return": round(float(self.total_return), 6),
            "annualized_return": round(float(self.annualized_return), 6),
            "annualized_volatility": round(float(self.annualized_volatility), 6) if self.annualized_volatility is not None else None,
            "max_drawdown": round(float(self.max_drawdown), 6),
            "trade_count": int(self.trade_count),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeriodPerformance":
        return cls(
            period_label=data["period_label"],
            total_return=float(data["total_return"]),
            annualized_return=float(data["annualized_return"]),
            annualized_volatility=float(data["annualized_volatility"]) if data.get("annualized_volatility") is not None else None,
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            trade_count=int(data.get("trade_count", 0)),
        )


@dataclass
class DrawdownEpisode:
    """
    Structured breakdown of an individual historical drawdown episode.
    """
    start_time: datetime
    peak_time: datetime
    trough_time: datetime
    recovery_time: Optional[datetime]
    depth_pct: float
    duration_bars: int
    is_recovered: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "peak_time": self.peak_time.isoformat(),
            "trough_time": self.trough_time.isoformat(),
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
            "depth_pct": round(float(self.depth_pct), 6),
            "duration_bars": int(self.duration_bars),
            "is_recovered": self.is_recovered,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrawdownEpisode":
        return cls(
            start_time=datetime.fromisoformat(data["start_time"]),
            peak_time=datetime.fromisoformat(data["peak_time"]),
            trough_time=datetime.fromisoformat(data["trough_time"]),
            recovery_time=datetime.fromisoformat(data["recovery_time"]) if data.get("recovery_time") else None,
            depth_pct=float(data["depth_pct"]),
            duration_bars=int(data["duration_bars"]),
            is_recovered=bool(data["is_recovered"]),
        )


@dataclass
class StatisticalDiagnostics:
    """
    Statistical robustness and distribution metrics for return series.
    """
    observations_count: int
    trading_days_count: int
    positive_return_ratio: Optional[float] = None
    negative_return_ratio: Optional[float] = None
    best_period_return: Optional[float] = None
    worst_period_return: Optional[float] = None
    mean_period_return: Optional[float] = None
    median_period_return: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations_count": int(self.observations_count),
            "trading_days_count": int(self.trading_days_count),
            "positive_return_ratio": round(float(self.positive_return_ratio), 4) if self.positive_return_ratio is not None else None,
            "negative_return_ratio": round(float(self.negative_return_ratio), 4) if self.negative_return_ratio is not None else None,
            "best_period_return": round(float(self.best_period_return), 6) if self.best_period_return is not None else None,
            "worst_period_return": round(float(self.worst_period_return), 6) if self.worst_period_return is not None else None,
            "mean_period_return": round(float(self.mean_period_return), 6) if self.mean_period_return is not None else None,
            "median_period_return": round(float(self.median_period_return), 6) if self.median_period_return is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticalDiagnostics":
        return cls(
            observations_count=int(data["observations_count"]),
            trading_days_count=int(data["trading_days_count"]),
            positive_return_ratio=float(data["positive_return_ratio"]) if data.get("positive_return_ratio") is not None else None,
            negative_return_ratio=float(data["negative_return_ratio"]) if data.get("negative_return_ratio") is not None else None,
            best_period_return=float(data["best_period_return"]) if data.get("best_period_return") is not None else None,
            worst_period_return=float(data["worst_period_return"]) if data.get("worst_period_return") is not None else None,
            mean_period_return=float(data["mean_period_return"]) if data.get("mean_period_return") is not None else None,
            median_period_return=float(data["median_period_return"]) if data.get("median_period_return") is not None else None,
        )


@dataclass
class ExposurePoint:
    """
    Point-in-time portfolio exposure snapshot.
    """
    timestamp: datetime
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    cash: float
    leverage: float
    max_position_weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "gross_exposure": round(float(self.gross_exposure), 6),
            "net_exposure": round(float(self.net_exposure), 6),
            "long_exposure": round(float(self.long_exposure), 6),
            "short_exposure": round(float(self.short_exposure), 6),
            "cash": round(float(self.cash), 2),
            "leverage": round(float(self.leverage), 6),
            "max_position_weight": round(float(self.max_position_weight), 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExposurePoint":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            gross_exposure=float(data["gross_exposure"]),
            net_exposure=float(data["net_exposure"]),
            long_exposure=float(data.get("long_exposure", 0.0)),
            short_exposure=float(data.get("short_exposure", 0.0)),
            cash=float(data["cash"]),
            leverage=float(data["leverage"]),
            max_position_weight=float(data.get("max_position_weight", 0.0)),
        )


@dataclass
class OrderIntent:
    """
    Simulated order intent generated by target rebalancing before execution.
    """
    symbol: str
    side: OrderSide
    target_quantity: float
    target_notional: float
    timestamp: datetime
    order_type: str = "MARKET"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "target_quantity": round(float(self.target_quantity), 4),
            "target_notional": round(float(self.target_notional), 2),
            "timestamp": self.timestamp.isoformat(),
            "order_type": self.order_type,
        }


@dataclass
class TradeRecord:
    """
    Completed round-trip or closed trade position.
    """
    trade_id: str
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    return_pct: float
    holding_bars: int
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    side: PositionSide = PositionSide.LONG
    entry_timestamp: Optional[datetime] = None
    exit_timestamp: Optional[datetime] = None
    direction: Optional[PositionSide] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.entry_time is None and self.entry_timestamp is not None:
            self.entry_time = self.entry_timestamp
        elif self.entry_timestamp is None and self.entry_time is not None:
            self.entry_timestamp = self.entry_time

        if self.exit_time is None and self.exit_timestamp is not None:
            self.exit_time = self.exit_timestamp
        elif self.exit_timestamp is None and self.exit_time is not None:
            self.exit_timestamp = self.exit_time

        if self.direction is not None:
            self.side = self.direction
        else:
            self.direction = self.side

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "direction": self.side.value,
            "entry_time": self.entry_time.isoformat() if self.entry_time else "",
            "entry_timestamp": self.entry_time.isoformat() if self.entry_time else "",
            "exit_time": self.exit_time.isoformat() if self.exit_time else "",
            "exit_timestamp": self.exit_time.isoformat() if self.exit_time else "",
            "entry_price": round(float(self.entry_price), 4),
            "exit_price": round(float(self.exit_price), 4),
            "quantity": round(float(self.quantity), 4),
            "gross_pnl": round(float(self.gross_pnl), 2),
            "commission": round(float(self.commission), 2),
            "slippage": round(float(self.slippage), 2),
            "net_pnl": round(float(self.net_pnl), 2),
            "return_pct": round(float(self.return_pct), 6),
            "holding_bars": int(self.holding_bars),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        entry_t_str = data.get("entry_time") or data.get("entry_timestamp")
        exit_t_str = data.get("exit_time") or data.get("exit_timestamp")
        side_val = data.get("side") or data.get("direction", "LONG")
        return cls(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            side=PositionSide(side_val),
            direction=PositionSide(side_val),
            entry_time=datetime.fromisoformat(entry_t_str) if entry_t_str else None,
            exit_time=datetime.fromisoformat(exit_t_str) if exit_t_str else None,
            entry_timestamp=datetime.fromisoformat(entry_t_str) if entry_t_str else None,
            exit_timestamp=datetime.fromisoformat(exit_t_str) if exit_t_str else None,
            entry_price=float(data["entry_price"]),
            exit_price=float(data["exit_price"]),
            quantity=float(data["quantity"]),
            gross_pnl=float(data["gross_pnl"]),
            commission=float(data["commission"]),
            slippage=float(data["slippage"]),
            net_pnl=float(data["net_pnl"]),
            return_pct=float(data["return_pct"]),
            holding_bars=int(data["holding_bars"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionRecord:
    """
    Record of a simulated individual order execution at a specific timestamp.
    """
    execution_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    requested_price: float
    executed_price: float
    notional: float
    commission: float
    slippage_cost: float
    spread_cost: float = 0.0
    borrow_cost: float = 0.0
    market_impact_cost: float = 0.0
    reason: str = "REBALANCE"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def traded_notional(self) -> float:
        return self.notional

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": round(float(self.quantity), 4),
            "requested_price": round(float(self.requested_price), 4),
            "executed_price": round(float(self.executed_price), 4),
            "notional": round(float(self.notional), 2),
            "commission": round(float(self.commission), 2),
            "slippage_cost": round(float(self.slippage_cost), 2),
            "spread_cost": round(float(self.spread_cost), 2),
            "borrow_cost": round(float(self.borrow_cost), 2),
            "market_impact_cost": round(float(self.market_impact_cost), 2),
            "reason": self.reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRecord":
        return cls(
            execution_id=data["execution_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            quantity=float(data["quantity"]),
            requested_price=float(data["requested_price"]),
            executed_price=float(data["executed_price"]),
            notional=float(data["notional"]),
            commission=float(data["commission"]),
            slippage_cost=float(data["slippage_cost"]),
            spread_cost=float(data.get("spread_cost", 0.0)),
            borrow_cost=float(data.get("borrow_cost", 0.0)),
            market_impact_cost=float(data.get("market_impact_cost", 0.0)),
            reason=data.get("reason", "REBALANCE"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PositionState:
    """
    Point-in-time state of an individual asset position.
    """
    symbol: str
    side: PositionSide
    quantity: float
    avg_entry_price: float
    market_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime

    @property
    def average_entry_price(self) -> float:
        return self.avg_entry_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": round(float(self.quantity), 4),
            "avg_entry_price": round(float(self.avg_entry_price), 4),
            "market_price": round(float(self.market_price), 4),
            "market_value": round(float(self.market_value), 2),
            "weight": round(float(self.weight), 6),
            "unrealized_pnl": round(float(self.unrealized_pnl), 2),
            "realized_pnl": round(float(self.realized_pnl), 2),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionState":
        return cls(
            symbol=data["symbol"],
            side=PositionSide(data["side"]),
            quantity=float(data["quantity"]),
            avg_entry_price=float(data["avg_entry_price"]),
            market_price=float(data["market_price"]),
            market_value=float(data["market_value"]),
            weight=float(data["weight"]),
            unrealized_pnl=float(data["unrealized_pnl"]),
            realized_pnl=float(data["realized_pnl"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )


@dataclass
class PortfolioState:
    """
    Point-in-time full portfolio accounting state.
    """
    timestamp: datetime
    cash: float
    equity: float
    gross_notional: float
    net_notional: float
    gross_exposure: float
    net_exposure: float
    positions: Dict[str, PositionState] = field(default_factory=dict)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_spread_cost: float = 0.0
    total_borrow_cost: float = 0.0
    total_market_impact: float = 0.0
    turnover: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash": round(float(self.cash), 2),
            "equity": round(float(self.equity), 2),
            "gross_notional": round(float(self.gross_notional), 2),
            "net_notional": round(float(self.net_notional), 2),
            "gross_exposure": round(float(self.gross_exposure), 6),
            "net_exposure": round(float(self.net_exposure), 6),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "unrealized_pnl": round(float(self.unrealized_pnl), 2),
            "realized_pnl": round(float(self.realized_pnl), 2),
            "cumulative_pnl": round(float(self.cumulative_pnl), 2),
            "total_commission": round(float(self.total_commission), 2),
            "total_slippage": round(float(self.total_slippage), 2),
            "total_spread_cost": round(float(self.total_spread_cost), 2),
            "total_borrow_cost": round(float(self.total_borrow_cost), 2),
            "total_market_impact": round(float(self.total_market_impact), 2),
            "turnover": round(float(self.turnover), 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioState":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            cash=float(data["cash"]),
            equity=float(data["equity"]),
            gross_notional=float(data["gross_notional"]),
            net_notional=float(data["net_notional"]),
            gross_exposure=float(data["gross_exposure"]),
            net_exposure=float(data["net_exposure"]),
            positions={k: PositionState.from_dict(v) for k, v in data.get("positions", {}).items()},
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            cumulative_pnl=float(data.get("cumulative_pnl", 0.0)),
            total_commission=float(data.get("total_commission", 0.0)),
            total_slippage=float(data.get("total_slippage", 0.0)),
            total_spread_cost=float(data.get("total_spread_cost", 0.0)),
            total_borrow_cost=float(data.get("total_borrow_cost", 0.0)),
            total_market_impact=float(data.get("total_market_impact", 0.0)),
            turnover=float(data.get("turnover", 0.0)),
        )


@dataclass
class EquityPoint:
    """
    Point-in-time portfolio mark-to-market equity valuation.
    """
    timestamp: datetime
    equity: float
    cash: float
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    period_return: float = 0.0
    cumulative_return: float = 0.0
    running_peak: float = 0.0
    drawdown_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": round(float(self.equity), 2),
            "cash": round(float(self.cash), 2),
            "gross_exposure": round(float(self.gross_exposure), 6),
            "net_exposure": round(float(self.net_exposure), 6),
            "long_exposure": round(float(self.long_exposure), 6),
            "short_exposure": round(float(self.short_exposure), 6),
            "period_return": round(float(self.period_return), 6),
            "cumulative_return": round(float(self.cumulative_return), 6),
            "running_peak": round(float(self.running_peak), 2),
            "drawdown_pct": round(float(self.drawdown_pct), 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquityPoint":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            equity=float(data["equity"]),
            cash=float(data["cash"]),
            gross_exposure=float(data.get("gross_exposure", 0.0)),
            net_exposure=float(data.get("net_exposure", 0.0)),
            long_exposure=float(data.get("long_exposure", 0.0)),
            short_exposure=float(data.get("short_exposure", 0.0)),
            period_return=float(data.get("period_return", 0.0)),
            cumulative_return=float(data.get("cumulative_return", 0.0)),
            running_peak=float(data.get("running_peak", 0.0)),
            drawdown_pct=float(data.get("drawdown_pct", 0.0)),
        )


@dataclass
class DrawdownPoint:
    """
    Point-in-time drawdown evaluation.
    """
    timestamp: datetime
    equity: float
    running_peak: float
    drawdown: float
    drawdown_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": round(float(self.equity), 2),
            "running_peak": round(float(self.running_peak), 2),
            "drawdown": round(float(self.drawdown), 2),
            "drawdown_percentage": round(float(self.drawdown_percentage), 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrawdownPoint":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            equity=float(data["equity"]),
            running_peak=float(data["running_peak"]),
            drawdown=float(data["drawdown"]),
            drawdown_percentage=float(data["drawdown_percentage"]),
        )


@dataclass
class PerformanceMetrics:
    """
    Portfolio backtest performance statistics.
    
    SCIENTIFIC INTEGRITY:
    If metrics are undefined (e.g. zero volatility, zero drawdown, zero trades),
    represented as None rather than inf or fabricated numbers.
    """
    start_equity: float
    end_equity: float
    total_return: float
    annualized_return: float
    annualized_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_trade_return: Optional[float] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_turnover: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_spread_cost: float = 0.0
    total_borrow_cost: float = 0.0
    total_market_impact_cost: float = 0.0
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    period_breakdowns: List[PeriodPerformance] = field(default_factory=list)
    drawdown_episodes: List[DrawdownEpisode] = field(default_factory=list)
    statistical_diagnostics: Optional[StatisticalDiagnostics] = None

    @property
    def total_costs(self) -> float:
        return float(
            self.total_commission
            + self.total_slippage
            + self.total_spread_cost
            + self.total_borrow_cost
            + self.total_market_impact_cost
        )

    @property
    def total_commission_paid(self) -> float:
        return float(self.total_commission)

    @property
    def total_slippage_paid(self) -> float:
        return float(self.total_slippage)

    @property
    def average_trade_pnl(self) -> Optional[float]:
        return self.avg_trade_return

    @property
    def total_completed_trades(self) -> int:
        return self.total_trades

    @property
    def average_turnover(self) -> float:
        return float(self.total_turnover)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_equity": round(float(self.start_equity), 2),
            "end_equity": round(float(self.end_equity), 2),
            "total_return": round(float(self.total_return), 6),
            "annualized_return": round(float(self.annualized_return), 6),
            "annualized_volatility": round(float(self.annualized_volatility), 6) if self.annualized_volatility is not None else None,
            "sharpe_ratio": round(float(self.sharpe_ratio), 4) if self.sharpe_ratio is not None else None,
            "sortino_ratio": round(float(self.sortino_ratio), 4) if self.sortino_ratio is not None else None,
            "calmar_ratio": round(float(self.calmar_ratio), 4) if self.calmar_ratio is not None else None,
            "max_drawdown": round(float(self.max_drawdown), 6),
            "max_drawdown_duration_bars": int(self.max_drawdown_duration_bars),
            "win_rate": round(float(self.win_rate), 4) if self.win_rate is not None else None,
            "profit_factor": round(float(self.profit_factor), 4) if self.profit_factor is not None else None,
            "avg_trade_return": round(float(self.avg_trade_return), 6) if self.avg_trade_return is not None else None,
            "average_trade_pnl": round(float(self.avg_trade_return), 6) if self.avg_trade_return is not None else None,
            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "losing_trades": int(self.losing_trades),
            "total_turnover": round(float(self.total_turnover), 6),
            "total_commission": round(float(self.total_commission), 2),
            "total_slippage": round(float(self.total_slippage), 2),
            "total_spread_cost": round(float(self.total_spread_cost), 2),
            "total_borrow_cost": round(float(self.total_borrow_cost), 2),
            "total_market_impact_cost": round(float(self.total_market_impact_cost), 2),
            "total_costs": round(float(self.total_costs), 2),
            "gross_pnl": round(float(self.gross_pnl), 2) if self.gross_pnl is not None else None,
            "net_pnl": round(float(self.net_pnl), 2) if self.net_pnl is not None else None,
            "period_breakdowns": [p.to_dict() for p in self.period_breakdowns],
            "drawdown_episodes": [d.to_dict() for d in self.drawdown_episodes],
            "statistical_diagnostics": self.statistical_diagnostics.to_dict() if self.statistical_diagnostics else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        stat_data = data.get("statistical_diagnostics")
        return cls(
            start_equity=float(data["start_equity"]),
            end_equity=float(data["end_equity"]),
            total_return=float(data["total_return"]),
            annualized_return=float(data["annualized_return"]),
            annualized_volatility=float(data["annualized_volatility"]) if data.get("annualized_volatility") is not None else None,
            sharpe_ratio=float(data["sharpe_ratio"]) if data.get("sharpe_ratio") is not None else None,
            sortino_ratio=float(data["sortino_ratio"]) if data.get("sortino_ratio") is not None else None,
            calmar_ratio=float(data["calmar_ratio"]) if data.get("calmar_ratio") is not None else None,
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            max_drawdown_duration_bars=int(data.get("max_drawdown_duration_bars", 0)),
            win_rate=float(data["win_rate"]) if data.get("win_rate") is not None else None,
            profit_factor=float(data["profit_factor"]) if data.get("profit_factor") is not None else None,
            avg_trade_return=float(data["avg_trade_return"]) if data.get("avg_trade_return") is not None else None,
            total_trades=int(data.get("total_trades", 0)),
            winning_trades=int(data.get("winning_trades", 0)),
            losing_trades=int(data.get("losing_trades", 0)),
            total_turnover=float(data.get("total_turnover", 0.0)),
            total_commission=float(data.get("total_commission", 0.0)),
            total_slippage=float(data.get("total_slippage", 0.0)),
            total_spread_cost=float(data.get("total_spread_cost", 0.0)),
            total_borrow_cost=float(data.get("total_borrow_cost", 0.0)),
            total_market_impact_cost=float(data.get("total_market_impact_cost", 0.0)),
            gross_pnl=float(data["gross_pnl"]) if data.get("gross_pnl") is not None else None,
            net_pnl=float(data["net_pnl"]) if data.get("net_pnl") is not None else None,
            period_breakdowns=[PeriodPerformance.from_dict(p) for p in data.get("period_breakdowns", [])],
            drawdown_episodes=[DrawdownEpisode.from_dict(d) for d in data.get("drawdown_episodes", [])],
            statistical_diagnostics=StatisticalDiagnostics.from_dict(stat_data) if stat_data else None,
        )


@dataclass
class BenchmarkMetrics:
    """
    Performance metrics for the benchmark asset (e.g. SPY) over the exact same period.
    """
    benchmark_symbol: str
    total_return: float
    annualized_return: float
    annualized_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    alpha: Optional[float] = None
    beta: Optional[float] = None
    correlation: Optional[float] = None

    @property
    def benchmark_total_return(self) -> float:
        return self.total_return

    @property
    def benchmark_annualized_return(self) -> float:
        return self.annualized_return

    @property
    def benchmark_annualized_volatility(self) -> Optional[float]:
        return self.annualized_volatility

    @property
    def benchmark_sharpe(self) -> Optional[float]:
        return self.sharpe_ratio

    @property
    def benchmark_max_drawdown(self) -> float:
        return self.max_drawdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_symbol": self.benchmark_symbol,
            "total_return": round(float(self.total_return), 6),
            "annualized_return": round(float(self.annualized_return), 6),
            "annualized_volatility": round(float(self.annualized_volatility), 6) if self.annualized_volatility is not None else None,
            "sharpe_ratio": round(float(self.sharpe_ratio), 4) if self.sharpe_ratio is not None else None,
            "max_drawdown": round(float(self.max_drawdown), 6),
            "alpha": round(float(self.alpha), 6) if self.alpha is not None else None,
            "beta": round(float(self.beta), 4) if self.beta is not None else None,
            "correlation": round(float(self.correlation), 4) if self.correlation is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkMetrics":
        return cls(
            benchmark_symbol=data["benchmark_symbol"],
            total_return=float(data["total_return"]),
            annualized_return=float(data["annualized_return"]),
            annualized_volatility=float(data["annualized_volatility"]) if data.get("annualized_volatility") is not None else None,
            sharpe_ratio=float(data["sharpe_ratio"]) if data.get("sharpe_ratio") is not None else None,
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            alpha=float(data["alpha"]) if data.get("alpha") is not None else None,
            beta=float(data["beta"]) if data.get("beta") is not None else None,
            correlation=float(data["correlation"]) if data.get("correlation") is not None else None,
        )


@dataclass
class BacktestConfig:
    """
    Configuration parameters for historical simulation.
    """
    initial_capital: float = 100_000.0
    base_currency: str = "USD"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1Day"
    execution_price_type: ExecutionPriceType = ExecutionPriceType.NEXT_OPEN
    commission_rate: float = 0.0005  # 5 bps
    slippage_rate: float = 0.0005  # 5 bps
    bid_ask_spread_rate: float = 0.0  # Optional half-spread
    daily_borrow_rate: float = 0.0  # Annual / daily borrow rate on short market value
    market_impact_coefficient: float = 0.0  # Optional market impact multiplier
    minimum_trade_notional: float = 10.0  # $10 min order size
    allow_fractional_shares: bool = True
    benchmark_symbol: Optional[str] = "SPY"
    risk_free_rate: float = 0.0  # 0.0 = 0% risk free rate
    version: str = "backtest-v1.2"
    dataset_provenance: Optional[DatasetProvenance] = None

    def __post_init__(self):
        if not isinstance(self.initial_capital, (int, float)) or self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {self.initial_capital}")
        if np.isnan(self.initial_capital) or np.isinf(self.initial_capital):
            raise ValueError(f"initial_capital must be finite, got {self.initial_capital}")
        if self.commission_rate < 0.0:
            raise ValueError(f"commission_rate must be >= 0, got {self.commission_rate}")
        if self.slippage_rate < 0.0:
            raise ValueError(f"slippage_rate must be >= 0, got {self.slippage_rate}")
        if self.bid_ask_spread_rate < 0.0:
            raise ValueError(f"bid_ask_spread_rate must be >= 0, got {self.bid_ask_spread_rate}")
        if self.daily_borrow_rate < 0.0:
            raise ValueError(f"daily_borrow_rate must be >= 0, got {self.daily_borrow_rate}")
        if self.market_impact_coefficient < 0.0:
            raise ValueError(f"market_impact_coefficient must be >= 0, got {self.market_impact_coefficient}")
        if self.minimum_trade_notional < 0.0:
            raise ValueError(f"minimum_trade_notional must be >= 0, got {self.minimum_trade_notional}")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError(f"start_date ({self.start_date}) must be earlier than end_date ({self.end_date})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_capital": round(float(self.initial_capital), 2),
            "base_currency": self.base_currency,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "execution_price_type": self.execution_price_type.value,
            "commission_rate": float(self.commission_rate),
            "slippage_rate": float(self.slippage_rate),
            "bid_ask_spread_rate": float(self.bid_ask_spread_rate),
            "daily_borrow_rate": float(self.daily_borrow_rate),
            "market_impact_coefficient": float(self.market_impact_coefficient),
            "minimum_trade_notional": float(self.minimum_trade_notional),
            "allow_fractional_shares": self.allow_fractional_shares,
            "benchmark_symbol": self.benchmark_symbol,
            "risk_free_rate": float(self.risk_free_rate),
            "version": self.version,
            "dataset_provenance": self.dataset_provenance.to_dict() if self.dataset_provenance else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        prov_data = data.get("dataset_provenance")
        return cls(
            initial_capital=float(data["initial_capital"]),
            base_currency=data.get("base_currency", "USD"),
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            symbols=data.get("symbols", []),
            timeframe=data.get("timeframe", "1Day"),
            execution_price_type=ExecutionPriceType(data.get("execution_price_type", "next_open")),
            commission_rate=float(data.get("commission_rate", 0.0005)),
            slippage_rate=float(data.get("slippage_rate", 0.0005)),
            bid_ask_spread_rate=float(data.get("bid_ask_spread_rate", 0.0)),
            daily_borrow_rate=float(data.get("daily_borrow_rate", 0.0)),
            market_impact_coefficient=float(data.get("market_impact_coefficient", 0.0)),
            minimum_trade_notional=float(data.get("minimum_trade_notional", 10.0)),
            allow_fractional_shares=bool(data.get("allow_fractional_shares", True)),
            benchmark_symbol=data.get("benchmark_symbol", "SPY"),
            risk_free_rate=float(data.get("risk_free_rate", 0.0)),
            version=data.get("version", "backtest-v1.2"),
            dataset_provenance=DatasetProvenance.from_dict(prov_data) if prov_data else None,
        )


@dataclass
class BacktestSummary:
    """
    High-level backtest summary statistics.
    """
    backtest_id: str
    timestamp: datetime
    start_date: datetime
    end_date: datetime
    total_bars: int
    symbols: List[str]
    initial_capital: float
    final_equity: float
    total_return: float
    annualized_return: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    total_trades: int
    win_rate: Optional[float]
    benchmark_symbol: Optional[str] = None
    benchmark_return: Optional[float] = None

    @property
    def total_trading_days(self) -> int:
        return self.total_bars

    @property
    def total_pnl(self) -> float:
        return float(self.final_equity - self.initial_capital)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backtest_id": self.backtest_id,
            "timestamp": self.timestamp.isoformat(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_bars": self.total_bars,
            "total_trading_days": self.total_bars,
            "symbols": self.symbols,
            "initial_capital": round(float(self.initial_capital), 2),
            "final_equity": round(float(self.final_equity), 2),
            "total_pnl": round(float(self.total_pnl), 2),
            "total_return": round(float(self.total_return), 6),
            "annualized_return": round(float(self.annualized_return), 6),
            "sharpe_ratio": round(float(self.sharpe_ratio), 4) if self.sharpe_ratio is not None else None,
            "max_drawdown": round(float(self.max_drawdown), 6),
            "total_trades": self.total_trades,
            "win_rate": round(float(self.win_rate), 4) if self.win_rate is not None else None,
            "benchmark_symbol": self.benchmark_symbol,
            "benchmark_return": round(float(self.benchmark_return), 6) if self.benchmark_return is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestSummary":
        return cls(
            backtest_id=data["backtest_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            total_bars=int(data["total_bars"]),
            symbols=data.get("symbols", []),
            initial_capital=float(data["initial_capital"]),
            final_equity=float(data["final_equity"]),
            total_return=float(data["total_return"]),
            annualized_return=float(data["annualized_return"]),
            sharpe_ratio=float(data["sharpe_ratio"]) if data.get("sharpe_ratio") is not None else None,
            max_drawdown=float(data["max_drawdown"]),
            total_trades=int(data["total_trades"]),
            win_rate=float(data["win_rate"]) if data.get("win_rate") is not None else None,
            benchmark_symbol=data.get("benchmark_symbol"),
            benchmark_return=float(data["benchmark_return"]) if data.get("benchmark_return") is not None else None,
        )


@dataclass
class BacktestResult:
    """
    Comprehensive historical backtest output object.
    
    CRITICAL DISTINCTION:
    Represents historical backtesting simulation output.
    Does NOT execute live trades, manage broker connectivity, or place exchange orders.
    """
    backtest_id: str
    timestamp: datetime
    config: BacktestConfig
    summary: BacktestSummary
    metrics: PerformanceMetrics
    equity_curve: List[EquityPoint]
    trades: List[TradeRecord]
    executions: List[ExecutionRecord]
    portfolio_history: List[PortfolioState]
    benchmark_metrics: Optional[BenchmarkMetrics] = None
    target_history: List[Dict[str, Any]] = field(default_factory=list)
    cost_breakdown: Optional[CostBreakdown] = None
    exposure_history: List[ExposurePoint] = field(default_factory=list)
    dataset_provenance: Optional[DatasetProvenance] = None
    dividends_collected: float = 0.0
    splits_applied_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def evaluated_at(self) -> datetime:
        return self.timestamp

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "baseline-v1"))

    @property
    def signal_version(self) -> str:
        return str(self.metadata.get("signal_version", "signal-v1"))

    @property
    def portfolio_version(self) -> str:
        return str(self.metadata.get("portfolio_version", "portfolio-v1"))

    @property
    def risk_version(self) -> str:
        return str(self.metadata.get("risk_version", "risk-v1.1"))

    @property
    def disclaimer(self) -> str:
        return "Simulated historical performance is presented solely for quantitative research purposes. Past performance does not guarantee future results. PHASE 13 implements historical simulation only."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backtest_id": self.backtest_id,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config.to_dict(),
            "summary": self.summary.to_dict(),
            "metrics": self.metrics.to_dict(),
            "equity_curve": [e.to_dict() for e in self.equity_curve],
            "trades": [t.to_dict() for t in self.trades],
            "executions": [x.to_dict() for x in self.executions],
            "portfolio_history": [p.to_dict() for p in self.portfolio_history],
            "benchmark_metrics": self.benchmark_metrics.to_dict() if self.benchmark_metrics else None,
            "target_history": self.target_history,
            "cost_breakdown": self.cost_breakdown.to_dict() if self.cost_breakdown else None,
            "exposure_history": [ex.to_dict() for ex in self.exposure_history],
            "dataset_provenance": self.dataset_provenance.to_dict() if self.dataset_provenance else None,
            "dividends_collected": round(float(self.dividends_collected), 2),
            "splits_applied_count": int(self.splits_applied_count),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestResult":
        bm_data = data.get("benchmark_metrics")
        cost_data = data.get("cost_breakdown")
        prov_data = data.get("dataset_provenance")
        return cls(
            backtest_id=data["backtest_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            config=BacktestConfig.from_dict(data["config"]),
            summary=BacktestSummary.from_dict(data["summary"]),
            metrics=PerformanceMetrics.from_dict(data["metrics"]),
            equity_curve=[EquityPoint.from_dict(e) for e in data.get("equity_curve", [])],
            trades=[TradeRecord.from_dict(t) for t in data.get("trades", [])],
            executions=[ExecutionRecord.from_dict(x) for x in data.get("executions", [])],
            portfolio_history=[PortfolioState.from_dict(p) for p in data.get("portfolio_history", [])],
            benchmark_metrics=BenchmarkMetrics.from_dict(bm_data) if bm_data else None,
            target_history=data.get("target_history", []),
            cost_breakdown=CostBreakdown.from_dict(cost_data) if cost_data else None,
            exposure_history=[ExposurePoint.from_dict(ex) for ex in data.get("exposure_history", [])],
            dataset_provenance=DatasetProvenance.from_dict(prov_data) if prov_data else None,
            dividends_collected=float(data.get("dividends_collected", 0.0)),
            splits_applied_count=int(data.get("splits_applied_count", 0)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WalkForwardFoldConfig:
    """
    Configuration and date bounds for an individual walk-forward fold.
    """
    fold_idx: int
    train_start: datetime
    train_end: datetime
    val_start: Optional[datetime] = None
    val_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": int(self.fold_idx),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "val_start": self.val_start.isoformat() if self.val_start else None,
            "val_end": self.val_end.isoformat() if self.val_end else None,
            "test_start": self.test_start.isoformat() if self.test_start else None,
            "test_end": self.test_end.isoformat() if self.test_end else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WalkForwardFoldConfig":
        return cls(
            fold_idx=int(data["fold_idx"]),
            train_start=datetime.fromisoformat(data["train_start"]),
            train_end=datetime.fromisoformat(data["train_end"]),
            val_start=datetime.fromisoformat(data["val_start"]) if data.get("val_start") else None,
            val_end=datetime.fromisoformat(data["val_end"]) if data.get("val_end") else None,
            test_start=datetime.fromisoformat(data["test_start"]) if data.get("test_start") else None,
            test_end=datetime.fromisoformat(data["test_end"]) if data.get("test_end") else None,
        )


@dataclass
class WalkForwardFoldResult:
    """
    Execution output for a single walk-forward evaluation fold.
    """
    fold_idx: int
    fold_config: WalkForwardFoldConfig
    in_sample_metrics: Optional[PerformanceMetrics] = None
    validation_metrics: Optional[PerformanceMetrics] = None
    out_of_sample_metrics: Optional[PerformanceMetrics] = None
    out_of_sample_result: Optional[BacktestResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": int(self.fold_idx),
            "fold_config": self.fold_config.to_dict(),
            "in_sample_metrics": self.in_sample_metrics.to_dict() if self.in_sample_metrics else None,
            "validation_metrics": self.validation_metrics.to_dict() if self.validation_metrics else None,
            "out_of_sample_metrics": self.out_of_sample_metrics.to_dict() if self.out_of_sample_metrics else None,
            "out_of_sample_result": self.out_of_sample_result.to_dict() if self.out_of_sample_result else None,
        }


@dataclass
class WalkForwardResult:
    """
    Complete concatenated walk-forward evaluation outcome across all sequential folds.
    """
    walk_forward_id: str
    timestamp: datetime
    config: BacktestConfig
    folds: List[WalkForwardFoldResult]
    concatenated_result: BacktestResult
    summary: BacktestSummary
    overall_metrics: PerformanceMetrics
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "walk_forward_id": self.walk_forward_id,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config.to_dict(),
            "folds": [f.to_dict() for f in self.folds],
            "concatenated_result": self.concatenated_result.to_dict(),
            "summary": self.summary.to_dict(),
            "overall_metrics": self.overall_metrics.to_dict(),
            "provenance": self.provenance,
        }
