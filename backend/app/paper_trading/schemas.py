"""
Paper Trading Engine Domain Schemas (Phase 14).
Strongly typed data contracts for simulated account, positions, orders,
executions, lifecycle states, audit events, and reconciliation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Union
import numpy as np


class PaperOrderStatus(str, Enum):
    """Lifecycle states of a simulated paper order."""
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class PaperOrderType(str, Enum):
    """Permitted paper order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PaperOrderSide(str, Enum):
    """Order trade direction."""
    BUY = "BUY"
    SELL = "SELL"


class PaperTimeInForce(str, Enum):
    """Time in force policy."""
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class PaperPositionSide(str, Enum):
    """Holding direction for a position."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class PaperSessionStatus(str, Enum):
    """State of an active or completed paper trading session."""
    CREATED = "CREATED"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PaperAuditEventType(str, Enum):
    """Audit log event classification."""
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_STARTED = "SESSION_STARTED"
    MARKET_DATA_RECEIVED = "MARKET_DATA_RECEIVED"
    TARGET_GENERATED = "TARGET_GENERATED"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_FILLED = "ORDER_FILLED"
    POSITION_UPDATED = "POSITION_UPDATED"
    CASH_UPDATED = "CASH_UPDATED"
    COST_APPLIED = "COST_APPLIED"
    DIVIDEND_APPLIED = "DIVIDEND_APPLIED"
    SPLIT_APPLIED = "SPLIT_APPLIED"
    ACCOUNT_VALUED = "ACCOUNT_VALUED"
    RECONCILIATION_PASSED = "RECONCILIATION_PASSED"
    RECONCILIATION_FAILED = "RECONCILIATION_FAILED"
    SESSION_PAUSED = "SESSION_PAUSED"
    SESSION_RESUMED = "SESSION_RESUMED"
    SESSION_STOPPED = "SESSION_STOPPED"
    SESSION_FAILED = "SESSION_FAILED"


class PaperReconciliationStatus(str, Enum):
    """Reconciliation state."""
    MATCHED = "MATCHED"
    MISMATCH = "MISMATCH"


@dataclass
class PaperFifoLot:
    """Represents a single point-in-time cost lot for FIFO realization."""
    lot_id: str
    symbol: str
    quantity: float
    entry_price: float
    timestamp: datetime
    side: PaperPositionSide = PaperPositionSide.LONG

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lot_id": self.lot_id,
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "timestamp": self.timestamp.isoformat(),
            "side": self.side.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperFifoLot":
        return cls(
            lot_id=data["lot_id"],
            symbol=data["symbol"],
            quantity=float(data["quantity"]),
            entry_price=float(data["entry_price"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            side=PaperPositionSide(data.get("side", "LONG")),
        )


@dataclass
class PaperPosition:
    """
    State of a single symbol holding within the paper trading account.
    """
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    market_price: float = 0.0
    market_value: float = 0.0
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    side: PaperPositionSide = PaperPositionSide.FLAT
    opened_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    fifo_lots: List[PaperFifoLot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "avg_entry_price": float(self.avg_entry_price),
            "market_price": float(self.market_price),
            "market_value": float(self.market_value),
            "cost_basis": float(self.cost_basis),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "side": self.side.value,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "fifo_lots": [lot.to_dict() for lot in self.fifo_lots],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperPosition":
        return cls(
            symbol=data["symbol"],
            quantity=float(data.get("quantity", 0.0)),
            avg_entry_price=float(data.get("avg_entry_price", 0.0)),
            market_price=float(data.get("market_price", 0.0)),
            market_value=float(data.get("market_value", 0.0)),
            cost_basis=float(data.get("cost_basis", 0.0)),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            side=PaperPositionSide(data.get("side", "FLAT")),
            opened_at=datetime.fromisoformat(data["opened_at"]) if data.get("opened_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            fifo_lots=[PaperFifoLot.from_dict(lot) for lot in data.get("fifo_lots", [])],
        )


@dataclass
class PaperExecution:
    """
    Record of an executed order fill in the paper trading broker.
    """
    execution_id: str
    order_id: str
    client_order_id: str
    symbol: str
    side: PaperOrderSide
    quantity: float
    executed_price: float
    base_price: float
    timestamp: datetime
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    market_impact_cost: float = 0.0
    realized_pnl: float = 0.0
    execution_reference: Optional[str] = None

    @property
    def total_cost(self) -> float:
        return float(self.commission + self.spread_cost + self.slippage_cost + self.market_impact_cost)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": float(self.quantity),
            "executed_price": float(self.executed_price),
            "base_price": float(self.base_price),
            "timestamp": self.timestamp.isoformat(),
            "commission": float(self.commission),
            "spread_cost": float(self.spread_cost),
            "slippage_cost": float(self.slippage_cost),
            "market_impact_cost": float(self.market_impact_cost),
            "realized_pnl": float(self.realized_pnl),
            "execution_reference": self.execution_reference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperExecution":
        return cls(
            execution_id=data["execution_id"],
            order_id=data["order_id"],
            client_order_id=data["client_order_id"],
            symbol=data["symbol"],
            side=PaperOrderSide(data["side"]),
            quantity=float(data["quantity"]),
            executed_price=float(data["executed_price"]),
            base_price=float(data["base_price"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            commission=float(data.get("commission", 0.0)),
            spread_cost=float(data.get("spread_cost", 0.0)),
            slippage_cost=float(data.get("slippage_cost", 0.0)),
            market_impact_cost=float(data.get("market_impact_cost", 0.0)),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            execution_reference=data.get("execution_reference"),
        )


@dataclass
class PaperOrder:
    """
    Represents an order created and submitted to the simulated paper broker.
    """
    order_id: str
    client_order_id: str
    session_id: str
    symbol: str
    side: PaperOrderSide
    quantity: float
    order_type: PaperOrderType = PaperOrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: PaperTimeInForce = PaperTimeInForce.DAY
    status: PaperOrderStatus = PaperOrderStatus.CREATED
    submitted_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    requested_price: Optional[float] = None
    executed_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    market_impact_cost: float = 0.0
    borrow_cost: float = 0.0
    rejection_reason: Optional[str] = None
    execution_reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in {
            PaperOrderStatus.FILLED,
            PaperOrderStatus.REJECTED,
            PaperOrderStatus.CANCELLED,
            PaperOrderStatus.EXPIRED,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": float(self.quantity),
            "order_type": self.order_type.value,
            "limit_price": float(self.limit_price) if self.limit_price is not None else None,
            "stop_price": float(self.stop_price) if self.stop_price is not None else None,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "requested_price": float(self.requested_price) if self.requested_price is not None else None,
            "executed_price": float(self.executed_price) if self.executed_price is not None else None,
            "filled_quantity": float(self.filled_quantity),
            "commission": float(self.commission),
            "spread_cost": float(self.spread_cost),
            "slippage_cost": float(self.slippage_cost),
            "market_impact_cost": float(self.market_impact_cost),
            "borrow_cost": float(self.borrow_cost),
            "rejection_reason": self.rejection_reason,
            "execution_reference": self.execution_reference,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperOrder":
        return cls(
            order_id=data["order_id"],
            client_order_id=data["client_order_id"],
            session_id=data["session_id"],
            symbol=data["symbol"],
            side=PaperOrderSide(data["side"]),
            quantity=float(data["quantity"]),
            order_type=PaperOrderType(data.get("order_type", "MARKET")),
            limit_price=float(data["limit_price"]) if data.get("limit_price") is not None else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") is not None else None,
            time_in_force=PaperTimeInForce(data.get("time_in_force", "DAY")),
            status=PaperOrderStatus(data.get("status", "CREATED")),
            submitted_at=datetime.fromisoformat(data["submitted_at"]) if data.get("submitted_at") else None,
            accepted_at=datetime.fromisoformat(data["accepted_at"]) if data.get("accepted_at") else None,
            filled_at=datetime.fromisoformat(data["filled_at"]) if data.get("filled_at") else None,
            cancelled_at=datetime.fromisoformat(data["cancelled_at"]) if data.get("cancelled_at") else None,
            requested_price=float(data["requested_price"]) if data.get("requested_price") is not None else None,
            executed_price=float(data["executed_price"]) if data.get("executed_price") is not None else None,
            filled_quantity=float(data.get("filled_quantity", 0.0)),
            commission=float(data.get("commission", 0.0)),
            spread_cost=float(data.get("spread_cost", 0.0)),
            slippage_cost=float(data.get("slippage_cost", 0.0)),
            market_impact_cost=float(data.get("market_impact_cost", 0.0)),
            borrow_cost=float(data.get("borrow_cost", 0.0)),
            rejection_reason=data.get("rejection_reason"),
            execution_reference=data.get("execution_reference"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PaperAccountSnapshot:
    """
    Point-in-time snapshot of the paper trading account state.
    """
    snapshot_id: str
    session_id: str
    timestamp: datetime
    cash: float
    equity: float
    market_value: float
    buying_power: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    cumulative_commission: float = 0.0
    cumulative_slippage: float = 0.0
    cumulative_spread: float = 0.0
    cumulative_market_impact: float = 0.0
    cumulative_borrow_cost: float = 0.0
    total_transaction_cost: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    leverage: float = 0.0
    positions_count: int = 0
    open_orders_count: int = 0
    positions: Dict[str, PaperPosition] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "cash": float(self.cash),
            "equity": float(self.equity),
            "market_value": float(self.market_value),
            "buying_power": float(self.buying_power),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "total_pnl": float(self.total_pnl),
            "cumulative_commission": float(self.cumulative_commission),
            "cumulative_slippage": float(self.cumulative_slippage),
            "cumulative_spread": float(self.cumulative_spread),
            "cumulative_market_impact": float(self.cumulative_market_impact),
            "cumulative_borrow_cost": float(self.cumulative_borrow_cost),
            "total_transaction_cost": float(self.total_transaction_cost),
            "gross_exposure": float(self.gross_exposure),
            "net_exposure": float(self.net_exposure),
            "long_exposure": float(self.long_exposure),
            "short_exposure": float(self.short_exposure),
            "leverage": float(self.leverage),
            "positions_count": int(self.positions_count),
            "open_orders_count": int(self.open_orders_count),
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperAccountSnapshot":
        return cls(
            snapshot_id=data["snapshot_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            cash=float(data["cash"]),
            equity=float(data["equity"]),
            market_value=float(data["market_value"]),
            buying_power=float(data["buying_power"]),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            total_pnl=float(data.get("total_pnl", 0.0)),
            cumulative_commission=float(data.get("cumulative_commission", 0.0)),
            cumulative_slippage=float(data.get("cumulative_slippage", 0.0)),
            cumulative_spread=float(data.get("cumulative_spread", 0.0)),
            cumulative_market_impact=float(data.get("cumulative_market_impact", 0.0)),
            cumulative_borrow_cost=float(data.get("cumulative_borrow_cost", 0.0)),
            total_transaction_cost=float(data.get("total_transaction_cost", 0.0)),
            gross_exposure=float(data.get("gross_exposure", 0.0)),
            net_exposure=float(data.get("net_exposure", 0.0)),
            long_exposure=float(data.get("long_exposure", 0.0)),
            short_exposure=float(data.get("short_exposure", 0.0)),
            leverage=float(data.get("leverage", 0.0)),
            positions_count=int(data.get("positions_count", 0)),
            open_orders_count=int(data.get("open_orders_count", 0)),
            positions={s: PaperPosition.from_dict(p) for s, p in data.get("positions", {}).items()},
        )


@dataclass
class PaperAuditEvent:
    """
    Immutable audit event record.
    """
    event_id: str
    session_id: str
    timestamp: datetime
    event_type: PaperAuditEventType
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "description": self.description,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperAuditEvent":
        return cls(
            event_id=data["event_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=PaperAuditEventType(data["event_type"]),
            description=data["description"],
            details=data.get("details", {}),
        )


@dataclass
class PaperReconciliationReport:
    """
    Result of comparing expected internal account state vs broker ledger state.
    """
    reconciliation_id: str
    session_id: str
    timestamp: datetime
    status: PaperReconciliationStatus
    cash_expected: float
    cash_actual: float
    equity_expected: float
    equity_actual: float
    positions_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    orders_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconciliation_id": self.reconciliation_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "cash_expected": float(self.cash_expected),
            "cash_actual": float(self.cash_actual),
            "equity_expected": float(self.equity_expected),
            "equity_actual": float(self.equity_actual),
            "positions_mismatches": self.positions_mismatches,
            "orders_mismatches": self.orders_mismatches,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperReconciliationReport":
        return cls(
            reconciliation_id=data["reconciliation_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=PaperReconciliationStatus(data["status"]),
            cash_expected=float(data["cash_expected"]),
            cash_actual=float(data["cash_actual"]),
            equity_expected=float(data["equity_expected"]),
            equity_actual=float(data["equity_actual"]),
            positions_mismatches=data.get("positions_mismatches", []),
            orders_mismatches=data.get("orders_mismatches", []),
            details=data.get("details", {}),
        )


@dataclass
class PaperTradingConfig:
    """
    Configuration for a paper trading session.
    """
    initial_capital: float = 100_000.0
    base_currency: str = "USD"
    commission_rate: float = 0.0005  # 5 bps
    slippage_rate: float = 0.0005    # 5 bps
    bid_ask_spread_rate: float = 0.0002  # 2 bps half-spread
    daily_borrow_rate: float = 0.0   # Short financing rate
    market_impact_coefficient: float = 0.0
    min_order_notional: float = 10.0
    min_position_delta: float = 0.001
    max_open_orders: int = 100
    allow_short: bool = False
    allow_fractional_shares: bool = True
    idempotency_enabled: bool = True
    version: str = "paper-v1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        if self.min_order_notional < 0.0:
            raise ValueError(f"min_order_notional must be >= 0, got {self.min_order_notional}")
        if self.max_open_orders <= 0:
            raise ValueError(f"max_open_orders must be > 0, got {self.max_open_orders}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_capital": float(self.initial_capital),
            "base_currency": self.base_currency,
            "commission_rate": float(self.commission_rate),
            "slippage_rate": float(self.slippage_rate),
            "bid_ask_spread_rate": float(self.bid_ask_spread_rate),
            "daily_borrow_rate": float(self.daily_borrow_rate),
            "market_impact_coefficient": float(self.market_impact_coefficient),
            "min_order_notional": float(self.min_order_notional),
            "min_position_delta": float(self.min_position_delta),
            "max_open_orders": int(self.max_open_orders),
            "allow_short": bool(self.allow_short),
            "allow_fractional_shares": bool(self.allow_fractional_shares),
            "idempotency_enabled": bool(self.idempotency_enabled),
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperTradingConfig":
        return cls(
            initial_capital=float(data.get("initial_capital", 100000.0)),
            base_currency=data.get("base_currency", "USD"),
            commission_rate=float(data.get("commission_rate", 0.0005)),
            slippage_rate=float(data.get("slippage_rate", 0.0005)),
            bid_ask_spread_rate=float(data.get("bid_ask_spread_rate", 0.0002)),
            daily_borrow_rate=float(data.get("daily_borrow_rate", 0.0)),
            market_impact_coefficient=float(data.get("market_impact_coefficient", 0.0)),
            min_order_notional=float(data.get("min_order_notional", 10.0)),
            min_position_delta=float(data.get("min_position_delta", 0.001)),
            max_open_orders=int(data.get("max_open_orders", 100)),
            allow_short=bool(data.get("allow_short", False)),
            allow_fractional_shares=bool(data.get("allow_fractional_shares", True)),
            idempotency_enabled=bool(data.get("idempotency_enabled", True)),
            version=data.get("version", "paper-v1.0"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PaperTradingSession:
    """
    Session metadata and state tracking.
    """
    session_id: str
    config: PaperTradingConfig
    status: PaperSessionStatus = PaperSessionStatus.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    final_equity: Optional[float] = None
    total_pnl: Optional[float] = None
    total_trades: int = 0
    total_costs: float = 0.0
    reconciliation_status: PaperReconciliationStatus = PaperReconciliationStatus.MATCHED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "symbols": self.symbols,
            "final_equity": float(self.final_equity) if self.final_equity is not None else None,
            "total_pnl": float(self.total_pnl) if self.total_pnl is not None else None,
            "total_trades": int(self.total_trades),
            "total_costs": float(self.total_costs),
            "reconciliation_status": self.reconciliation_status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperTradingSession":
        return cls(
            session_id=data["session_id"],
            config=PaperTradingConfig.from_dict(data["config"]),
            status=PaperSessionStatus(data.get("status", "CREATED")),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            stopped_at=datetime.fromisoformat(data["stopped_at"]) if data.get("stopped_at") else None,
            symbols=data.get("symbols", []),
            final_equity=float(data["final_equity"]) if data.get("final_equity") is not None else None,
            total_pnl=float(data["total_pnl"]) if data.get("total_pnl") is not None else None,
            total_trades=int(data.get("total_trades", 0)),
            total_costs=float(data.get("total_costs", 0.0)),
            reconciliation_status=PaperReconciliationStatus(data.get("reconciliation_status", "MATCHED")),
        )


@dataclass
class PaperTradingResult:
    """
    Complete summary of a paper trading session execution.
    """
    session: PaperTradingSession
    account_snapshots: List[PaperAccountSnapshot]
    orders: List[PaperOrder]
    executions: List[PaperExecution]
    reconciliation_reports: List[PaperReconciliationReport]
    audit_events: List[PaperAuditEvent]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session.to_dict(),
            "account_snapshots": [s.to_dict() for s in self.account_snapshots],
            "orders": [o.to_dict() for o in self.orders],
            "executions": [e.to_dict() for e in self.executions],
            "reconciliation_reports": [r.to_dict() for r in self.reconciliation_reports],
            "audit_events": [a.to_dict() for a in self.audit_events],
            "provenance_hash": self.provenance_hash,
        }
