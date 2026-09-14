"""
Normalized Broker Domain Contracts (Phase 16).
Defines provider-independent schemas for accounts, positions, orders,
executions, quotes, and broker metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional


class BrokerExecutionMode(str, Enum):
    """Execution mode: simulated paper vs live broker."""
    PAPER = "PAPER"
    LIVE = "LIVE"


class BrokerProviderType(str, Enum):
    """Broker provider classification."""
    PAPER = "PAPER"
    ALPACA = "ALPACA"
    INTERACTIVE_BROKERS = "INTERACTIVE_BROKERS"
    CUSTOM = "CUSTOM"


class BrokerSide(str, Enum):
    """Standardized trade order side."""
    BUY = "BUY"
    SELL = "SELL"


class BrokerOrderType(str, Enum):
    """Standardized order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class BrokerTimeInForce(str, Enum):
    """Standardized order time-in-force."""
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class BrokerOrderStatus(str, Enum):
    """Normalized order lifecycle status."""
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"

    def is_terminal(self) -> bool:
        """Return True if order is in terminal state."""
        return self in {
            BrokerOrderStatus.FILLED,
            BrokerOrderStatus.CANCELLED,
            BrokerOrderStatus.REJECTED,
            BrokerOrderStatus.EXPIRED,
        }

    def is_active(self) -> bool:
        """Return True if order is open/active."""
        return self in {
            BrokerOrderStatus.SUBMITTED,
            BrokerOrderStatus.ACCEPTED,
            BrokerOrderStatus.PARTIALLY_FILLED,
            BrokerOrderStatus.CANCEL_PENDING,
        }


LEGAL_ORDER_STATUS_TRANSITIONS: Dict[BrokerOrderStatus, set[BrokerOrderStatus]] = {
    BrokerOrderStatus.CREATED: {
        BrokerOrderStatus.SUBMITTED,
        BrokerOrderStatus.ACCEPTED,
        BrokerOrderStatus.REJECTED,
        BrokerOrderStatus.CANCELLED,
    },
    BrokerOrderStatus.SUBMITTED: {
        BrokerOrderStatus.ACCEPTED,
        BrokerOrderStatus.PARTIALLY_FILLED,
        BrokerOrderStatus.FILLED,
        BrokerOrderStatus.CANCEL_PENDING,
        BrokerOrderStatus.CANCELLED,
        BrokerOrderStatus.REJECTED,
        BrokerOrderStatus.EXPIRED,
    },
    BrokerOrderStatus.ACCEPTED: {
        BrokerOrderStatus.PARTIALLY_FILLED,
        BrokerOrderStatus.FILLED,
        BrokerOrderStatus.CANCEL_PENDING,
        BrokerOrderStatus.CANCELLED,
        BrokerOrderStatus.REJECTED,
        BrokerOrderStatus.EXPIRED,
    },
    BrokerOrderStatus.PARTIALLY_FILLED: {
        BrokerOrderStatus.PARTIALLY_FILLED,
        BrokerOrderStatus.FILLED,
        BrokerOrderStatus.CANCEL_PENDING,
        BrokerOrderStatus.CANCELLED,
        BrokerOrderStatus.EXPIRED,
    },
    BrokerOrderStatus.CANCEL_PENDING: {
        BrokerOrderStatus.CANCELLED,
        BrokerOrderStatus.FILLED,
        BrokerOrderStatus.PARTIALLY_FILLED,
    },
    BrokerOrderStatus.FILLED: set(),
    BrokerOrderStatus.CANCELLED: set(),
    BrokerOrderStatus.REJECTED: set(),
    BrokerOrderStatus.EXPIRED: set(),
    BrokerOrderStatus.UNKNOWN: {
        BrokerOrderStatus.CREATED,
        BrokerOrderStatus.SUBMITTED,
        BrokerOrderStatus.ACCEPTED,
        BrokerOrderStatus.PARTIALLY_FILLED,
        BrokerOrderStatus.FILLED,
        BrokerOrderStatus.CANCEL_PENDING,
        BrokerOrderStatus.CANCELLED,
        BrokerOrderStatus.REJECTED,
        BrokerOrderStatus.EXPIRED,
    },
}


def validate_order_status_transition(
    current_status: BrokerOrderStatus,
    new_status: BrokerOrderStatus,
) -> bool:
    """
    Validate whether transitioning from current_status to new_status is legal.
    Allows idempotent same-state transitions.
    Raises ValueError if transition is illegal.
    """
    if current_status == new_status:
        return True

    allowed = LEGAL_ORDER_STATUS_TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        raise ValueError(
            f"Illegal order status transition from {current_status.value} to {new_status.value}. "
            f"Terminal or invalid state transition rejected."
        )
    return True


class BrokerPositionSide(str, Enum):
    """Position direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class BrokerStatus(str, Enum):
    """Broker operational status."""
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    DEGRADED = "DEGRADED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class OrderRequest:
    """
    Normalized, validated client order submission request.
    """
    client_order_id: str
    symbol: str
    side: BrokerSide
    quantity: float
    order_type: BrokerOrderType = BrokerOrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: BrokerTimeInForce = BrokerTimeInForce.DAY
    submitted_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Strict validation of order request fields."""
        if not self.client_order_id or not isinstance(self.client_order_id, str):
            raise ValueError("client_order_id must be a non-empty string")
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("symbol must be a non-empty string")
        self.symbol = self.symbol.upper().strip()

        if not isinstance(self.side, BrokerSide):
            try:
                self.side = BrokerSide(str(self.side).upper())
            except ValueError:
                raise ValueError(f"Invalid order side: {self.side}")

        if not isinstance(self.order_type, BrokerOrderType):
            try:
                self.order_type = BrokerOrderType(str(self.order_type).upper())
            except ValueError:
                raise ValueError(f"Invalid order type: {self.order_type}")

        if not isinstance(self.time_in_force, BrokerTimeInForce):
            try:
                self.time_in_force = BrokerTimeInForce(str(self.time_in_force).upper())
            except ValueError:
                raise ValueError(f"Invalid time in force: {self.time_in_force}")

        try:
            self.quantity = float(self.quantity)
        except (ValueError, TypeError):
            raise ValueError(f"quantity must be a valid float: {self.quantity}")

        if math.isnan(self.quantity) or math.isinf(self.quantity) or self.quantity <= 0:
            raise ValueError(f"quantity must be positive and finite: {self.quantity}")

        if self.limit_price is not None:
            try:
                self.limit_price = float(self.limit_price)
            except (ValueError, TypeError):
                raise ValueError(f"limit_price must be a valid float: {self.limit_price}")
            if math.isnan(self.limit_price) or math.isinf(self.limit_price) or self.limit_price <= 0:
                raise ValueError(f"limit_price must be positive and finite: {self.limit_price}")

        if self.stop_price is not None:
            try:
                self.stop_price = float(self.stop_price)
            except (ValueError, TypeError):
                raise ValueError(f"stop_price must be a valid float: {self.stop_price}")
            if math.isnan(self.stop_price) or math.isinf(self.stop_price) or self.stop_price <= 0:
                raise ValueError(f"stop_price must be positive and finite: {self.stop_price}")

        # Order type requirement checks
        if self.order_type in (BrokerOrderType.LIMIT, BrokerOrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"limit_price is required for order type: {self.order_type.value}")
        if self.order_type in (BrokerOrderType.STOP, BrokerOrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"stop_price is required for order type: {self.order_type.value}")

        if self.submitted_at is None:
            self.submitted_at = datetime.now(timezone.utc)
        elif self.submitted_at.tzinfo is None:
            self.submitted_at = self.submitted_at.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "metadata": self.metadata,
        }


@dataclass
class OrderUpdate:
    """Normalized order status update payload."""
    broker_order_id: str
    client_order_id: Optional[str]
    status: BrokerOrderStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        elif self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class BrokerOrder:
    """Normalized broker order representation."""
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: BrokerSide
    quantity: float
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    order_type: BrokerOrderType = BrokerOrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: BrokerTimeInForce = BrokerTimeInForce.DAY
    status: BrokerOrderStatus = BrokerOrderStatus.CREATED
    submitted_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status.is_terminal()

    def is_active(self) -> bool:
        return self.status.is_active()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "rejected_at": self.rejected_at.isoformat() if self.rejected_at else None,
            "rejection_reason": self.rejection_reason,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "currency": self.currency,
            "metadata": self.metadata,
        }


@dataclass
class BrokerExecution:
    """Normalized fill / execution record."""
    execution_id: str
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: BrokerSide
    quantity: float
    execution_price: float
    executed_at: datetime
    commission: float = 0.0
    currency: str = "USD"
    venue: str = "PAPER"
    realized_pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.executed_at.tzinfo is None:
            self.executed_at = self.executed_at.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "execution_price": self.execution_price,
            "executed_at": self.executed_at.isoformat(),
            "commission": self.commission,
            "currency": self.currency,
            "venue": self.venue,
            "realized_pnl": self.realized_pnl,
            "metadata": self.metadata,
        }


@dataclass
class BrokerPosition:
    """Normalized position representation."""
    symbol: str
    quantity: float
    average_price: float
    market_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    side: BrokerPositionSide = BrokerPositionSide.FLAT
    currency: str = "USD"
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.quantity > 0:
            self.side = BrokerPositionSide.LONG
        elif self.quantity < 0:
            self.side = BrokerPositionSide.SHORT
        else:
            self.side = BrokerPositionSide.FLAT

        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        elif self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "market_price": self.market_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "side": self.side.value,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }


@dataclass
class BrokerAccount:
    """Normalized broker account representation."""
    account_id: str
    currency: str = "USD"
    cash: float = 0.0
    buying_power: float = 0.0
    equity: float = 0.0
    available_cash: float = 0.0
    margin_used: Optional[float] = None
    margin_available: Optional[float] = None
    status: str = "ACTIVE"
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        elif self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "currency": self.currency,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "equity": self.equity,
            "available_cash": self.available_cash,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "status": self.status,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }


@dataclass
class BrokerQuote:
    """Normalized quote / top-of-book snapshot."""
    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_price: Optional[float] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        elif self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "last_price": self.last_price,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class PositionSnapshot:
    """Normalized multi-position portfolio snapshot."""
    timestamp: datetime
    positions: Dict[str, BrokerPosition] = field(default_factory=dict)
    total_market_value: float = 0.0
    total_unrealized_pnl: float = 0.0


@dataclass
class AccountSnapshot:
    """Unified point-in-time account & position snapshot."""
    account: BrokerAccount
    positions: Dict[str, BrokerPosition] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = self.account.timestamp or datetime.now(timezone.utc)


@dataclass
class BrokerConfig:
    """Configuration for broker abstraction client."""
    provider: BrokerProviderType = BrokerProviderType.PAPER
    execution_mode: BrokerExecutionMode = BrokerExecutionMode.PAPER
    timeout_seconds: float = 30.0
    max_retries: int = 3
    request_id_prefix: str = "req_"
    healthcheck_enabled: bool = True
    currency: str = "USD"
    session_id: Optional[str] = None
    initial_capital: float = 100000.0
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0005
    bid_ask_spread_rate: float = 0.0002
    daily_borrow_rate: float = 0.0
    allow_short: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "execution_mode": self.execution_mode.value,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "request_id_prefix": self.request_id_prefix,
            "healthcheck_enabled": self.healthcheck_enabled,
            "currency": self.currency,
            "session_id": self.session_id,
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "bid_ask_spread_rate": self.bid_ask_spread_rate,
            "daily_borrow_rate": self.daily_borrow_rate,
            "allow_short": self.allow_short,
        }


@dataclass
class BrokerOrderReconciliation:
    """Record of reconciliation comparison between local order state and broker-reported order."""
    broker_order_id: str
    client_order_id: str
    status_match: bool
    quantity_match: bool
    price_match: bool
    is_matched: bool
    discrepancies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "status_match": self.status_match,
            "quantity_match": self.quantity_match,
            "price_match": self.price_match,
            "is_matched": self.is_matched,
            "discrepancies": self.discrepancies,
        }


@dataclass
class BrokerAccountReconciliation:
    """Record of reconciliation comparison between local account ledger and broker-reported account."""
    cash_diff: float
    equity_diff: float
    position_count_diff: int
    is_matched: bool
    discrepancies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash_diff": self.cash_diff,
            "equity_diff": self.equity_diff,
            "position_count_diff": self.position_count_diff,
            "is_matched": self.is_matched,
            "discrepancies": self.discrepancies,
        }


def reconcile_broker_order(
    local_order: BrokerOrder,
    broker_order: BrokerOrder,
) -> BrokerOrderReconciliation:
    """Compare a local order record against a broker-reported order record."""
    discrepancies = []
    status_match = local_order.status == broker_order.status
    if not status_match:
        discrepancies.append(f"Status mismatch: local={local_order.status.value}, broker={broker_order.status.value}")

    qty_match = abs(local_order.quantity - broker_order.quantity) < 1e-6
    if not qty_match:
        discrepancies.append(f"Quantity mismatch: local={local_order.quantity}, broker={broker_order.quantity}")

    filled_match = abs(local_order.filled_quantity - broker_order.filled_quantity) < 1e-6
    if not filled_match:
        discrepancies.append(f"Filled quantity mismatch: local={local_order.filled_quantity}, broker={broker_order.filled_quantity}")

    price_match = True
    if local_order.average_fill_price is not None and broker_order.average_fill_price is not None:
        price_match = abs(local_order.average_fill_price - broker_order.average_fill_price) < 1e-4
        if not price_match:
            discrepancies.append(f"Avg fill price mismatch: local={local_order.average_fill_price}, broker={broker_order.average_fill_price}")

    is_matched = status_match and qty_match and filled_match and price_match
    return BrokerOrderReconciliation(
        broker_order_id=local_order.broker_order_id,
        client_order_id=local_order.client_order_id,
        status_match=status_match,
        quantity_match=qty_match and filled_match,
        price_match=price_match,
        is_matched=is_matched,
        discrepancies=discrepancies,
    )


def reconcile_broker_account(
    local_account: BrokerAccount,
    broker_account: BrokerAccount,
    local_positions: Optional[List[BrokerPosition]] = None,
    broker_positions: Optional[List[BrokerPosition]] = None,
    tolerance: float = 0.01,
) -> BrokerAccountReconciliation:
    """Compare local account ledger against broker-reported account state."""
    discrepancies = []
    cash_diff = float(local_account.cash - broker_account.cash)
    if abs(cash_diff) > tolerance:
        discrepancies.append(f"Cash mismatch: local={local_account.cash:.2f}, broker={broker_account.cash:.2f}, diff={cash_diff:.2f}")

    equity_diff = float(local_account.equity - broker_account.equity)
    if abs(equity_diff) > tolerance:
        discrepancies.append(f"Equity mismatch: local={local_account.equity:.2f}, broker={broker_account.equity:.2f}, diff={equity_diff:.2f}")

    loc_pos_count = len(local_positions) if local_positions is not None else 0
    brk_pos_count = len(broker_positions) if broker_positions is not None else 0
    pos_count_diff = loc_pos_count - brk_pos_count
    if pos_count_diff != 0:
        discrepancies.append(f"Position count mismatch: local={loc_pos_count}, broker={brk_pos_count}")

    is_matched = (abs(cash_diff) <= tolerance) and (abs(equity_diff) <= tolerance) and (pos_count_diff == 0)
    return BrokerAccountReconciliation(
        cash_diff=cash_diff,
        equity_diff=equity_diff,
        position_count_diff=pos_count_diff,
        is_matched=is_matched,
        discrepancies=discrepancies,
    )

