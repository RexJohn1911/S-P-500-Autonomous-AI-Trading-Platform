"""
Paper Broker Adapter (Phase 16).
Adapts the Phase 14 SimulatedPaperBroker into the standardized BaseBroker contract.
Ensures 100% backward compatibility and exact execution fidelity.
"""

from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, List, Optional

from backend.app.broker.capabilities import BrokerCapabilities, BrokerCapability
from backend.app.broker.errors import (
    BrokerAccountError,
    BrokerInvalidRequestError,
    BrokerOrderRejectedError,
    BrokerUnsupportedOperationError,
)
from backend.app.broker.health import BrokerHealthSnapshot
from backend.app.broker.interface import BaseBroker
from backend.app.broker.schemas import (
    BrokerAccount,
    BrokerConfig,
    BrokerExecution,
    BrokerExecutionMode,
    BrokerOrder,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerProviderType,
    BrokerSide,
    BrokerStatus,
    BrokerTimeInForce,
    OrderRequest,
)
from backend.app.paper_trading.account import PaperAccount
from backend.app.paper_trading.broker import SimulatedPaperBroker
from backend.app.paper_trading.schemas import (
    PaperExecution,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
    PaperPosition,
    PaperTradingConfig,
)

logger = logging.getLogger(__name__)


class PaperBrokerAdapter(BaseBroker):
    """
    Adapter implementing BaseBroker by wrapping Phase 14 SimulatedPaperBroker.
    """

    def __init__(
        self,
        config: Optional[BrokerConfig] = None,
        simulated_broker: Optional[SimulatedPaperBroker] = None,
        session_id: Optional[str] = None,
    ):
        self.config = config or BrokerConfig()
        self.session_id = session_id or self.config.session_id or f"paper_bkr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        if simulated_broker is not None:
            self.broker = simulated_broker
        else:
            paper_cfg = PaperTradingConfig(
                initial_capital=self.config.initial_capital,
                commission_rate=self.config.commission_rate,
                slippage_rate=self.config.slippage_rate,
                bid_ask_spread_rate=self.config.bid_ask_spread_rate,
                daily_borrow_rate=self.config.daily_borrow_rate,
                allow_short=self.config.allow_short,
            )
            self.broker = SimulatedPaperBroker(
                session_id=self.session_id,
                config=paper_cfg,
            )

        self._capabilities = BrokerCapabilities(
            supported={
                BrokerCapability.MARKET_ORDERS,
                BrokerCapability.LIMIT_ORDERS,
                BrokerCapability.STOP_ORDERS,
                BrokerCapability.STOP_LIMIT_ORDERS,
                BrokerCapability.SHORT_SELLING if self.broker.config.allow_short else None,
                BrokerCapability.FRACTIONAL_SHARES,
                BrokerCapability.CANCEL_ORDER,
                BrokerCapability.POSITIONS,
                BrokerCapability.ACCOUNT_DATA,
                BrokerCapability.TRADE_HISTORY,
                BrokerCapability.DIVIDENDS,
                BrokerCapability.CORPORATE_ACTIONS,
                BrokerCapability.IDEMPOTENCY,
            }
        )
        # Remove any None entries
        self._capabilities.supported = {c for c in self._capabilities.supported if c is not None}

    # =========================================================================
    # Translation Helpers
    # =========================================================================

    @staticmethod
    def _to_paper_side(side: BrokerSide) -> PaperOrderSide:
        return PaperOrderSide.BUY if side == BrokerSide.BUY else PaperOrderSide.SELL

    @staticmethod
    def _to_broker_side(side: PaperOrderSide) -> BrokerSide:
        return BrokerSide.BUY if side == PaperOrderSide.BUY else BrokerSide.SELL

    @staticmethod
    def _to_paper_type(order_type: BrokerOrderType) -> PaperOrderType:
        if order_type == BrokerOrderType.LIMIT:
            return PaperOrderType.LIMIT
        return PaperOrderType.MARKET

    @staticmethod
    def _to_broker_type(order_type: PaperOrderType) -> BrokerOrderType:
        if order_type == PaperOrderType.LIMIT:
            return BrokerOrderType.LIMIT
        return BrokerOrderType.MARKET

    @staticmethod
    def _to_broker_status(status: PaperOrderStatus) -> BrokerOrderStatus:
        mapping = {
            PaperOrderStatus.CREATED: BrokerOrderStatus.CREATED,
            PaperOrderStatus.SUBMITTED: BrokerOrderStatus.SUBMITTED,
            PaperOrderStatus.ACCEPTED: BrokerOrderStatus.ACCEPTED,
            PaperOrderStatus.PARTIALLY_FILLED: BrokerOrderStatus.PARTIALLY_FILLED,
            PaperOrderStatus.FILLED: BrokerOrderStatus.FILLED,
            PaperOrderStatus.CANCELLED: BrokerOrderStatus.CANCELLED,
            PaperOrderStatus.REJECTED: BrokerOrderStatus.REJECTED,
            PaperOrderStatus.EXPIRED: BrokerOrderStatus.EXPIRED,
        }
        return mapping.get(status, BrokerOrderStatus.UNKNOWN)

    def _to_broker_order(self, order: PaperOrder) -> BrokerOrder:
        remaining_qty = max(0.0, float(order.quantity - order.filled_quantity))
        return BrokerOrder(
            broker_order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=self._to_broker_side(order.side),
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            remaining_quantity=remaining_qty,
            order_type=self._to_broker_type(order.order_type),
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            time_in_force=BrokerTimeInForce.DAY,
            status=self._to_broker_status(order.status),
            submitted_at=order.submitted_at,
            accepted_at=order.accepted_at,
            filled_at=order.filled_at,
            cancelled_at=order.cancelled_at,
            rejected_at=None,
            rejection_reason=order.rejection_reason,
            average_fill_price=order.executed_price,
            commission=order.commission,
            currency="USD",
            metadata=order.metadata,
        )

    def _to_broker_execution(self, exec_rec: PaperExecution) -> BrokerExecution:
        return BrokerExecution(
            execution_id=exec_rec.execution_id,
            broker_order_id=exec_rec.order_id,
            client_order_id=exec_rec.client_order_id,
            symbol=exec_rec.symbol,
            side=self._to_broker_side(exec_rec.side),
            quantity=exec_rec.quantity,
            execution_price=exec_rec.executed_price,
            executed_at=exec_rec.timestamp,
            commission=exec_rec.commission,
            currency="USD",
            venue="PAPER",
            realized_pnl=exec_rec.realized_pnl,
            metadata={"spread_cost": exec_rec.spread_cost, "slippage_cost": exec_rec.slippage_cost},
        )

    def _to_broker_position(self, pos: PaperPosition) -> BrokerPosition:
        return BrokerPosition(
            symbol=pos.symbol,
            quantity=pos.quantity,
            average_price=pos.avg_entry_price,
            market_price=pos.market_price,
            market_value=pos.market_value,
            unrealized_pnl=pos.unrealized_pnl,
            realized_pnl=pos.realized_pnl,
            currency="USD",
            timestamp=getattr(pos, "updated_at", None) or getattr(pos, "last_updated", None),
            metadata=getattr(pos, "metadata", {}),
        )

    # =========================================================================
    # BaseBroker Interface Implementation
    # =========================================================================

    def get_account(self) -> BrokerAccount:
        """Retrieve current normalized broker account information."""
        acc = self.broker.account
        snap = acc.mark_to_market({}, datetime.now(timezone.utc))
        return BrokerAccount(
            account_id=acc.account_id,
            currency="USD",
            cash=snap.cash,
            buying_power=snap.buying_power,
            equity=snap.equity,
            available_cash=snap.cash,
            margin_used=0.0,
            margin_available=snap.buying_power,
            status="ACTIVE",
            timestamp=snap.timestamp,
            metadata={"session_id": self.session_id},
        )

    def get_positions(self) -> List[BrokerPosition]:
        """Retrieve all current normalized open positions."""
        paper_positions = self.broker.get_positions()
        return [self._to_broker_position(pos) for pos in paper_positions.values() if pos.quantity != 0]

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Retrieve normalized position for a specific symbol."""
        pos = self.broker.account.positions.get(symbol.upper())
        if pos and pos.quantity != 0:
            return self._to_broker_position(pos)
        return None

    def get_orders(self, status: Optional[BrokerOrderStatus] = None) -> List[BrokerOrder]:
        """Retrieve all recorded orders, optionally filtered by status."""
        all_orders = [self._to_broker_order(o) for o in self.broker.order_manager.orders.values()]
        if status is not None:
            return [o for o in all_orders if o.status == status]
        return all_orders

    def get_open_orders(self) -> List[BrokerOrder]:
        """Retrieve all currently active/open orders."""
        return [self._to_broker_order(o) for o in self.broker.get_open_orders()]

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Retrieve an order by broker_order_id."""
        order = self.broker.get_order(order_id)
        if order:
            return self._to_broker_order(order)
        return None

    def get_order_by_client_id(self, client_order_id: str) -> Optional[BrokerOrder]:
        """Retrieve an order by client_order_id."""
        order = self.broker.order_manager.get_order_by_client_id(client_order_id)
        if order:
            return self._to_broker_order(order)
        return None

    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        """
        Validate and submit an order request.
        Enforces:
        - Strict request validation
        - Capability validation (LIMIT, STOP, SHORT_SELLING, FRACTIONAL_SHARES)
        - Idempotency and conflicting parameter detection
        """
        try:
            request.validate()
        except ValueError as e:
            raise BrokerInvalidRequestError(str(e), provider="PAPER")

        # 1. Capability-aware validation
        if request.order_type == BrokerOrderType.LIMIT:
            self._capabilities.require(BrokerCapability.LIMIT_ORDERS, provider_name="PAPER")
        elif request.order_type in (BrokerOrderType.STOP, BrokerOrderType.STOP_LIMIT):
            self._capabilities.require(BrokerCapability.STOP_ORDERS, provider_name="PAPER")

        if not float(request.quantity).is_integer():
            self._capabilities.require(BrokerCapability.FRACTIONAL_SHARES, provider_name="PAPER")

        # 2. Check existing order by client_order_id (Idempotency + Conflict detection)
        existing = self.broker.order_manager.get_order_by_client_id(request.client_order_id)
        if existing:
            # Check for material conflict
            existing_side = self._to_broker_side(existing.side)
            existing_type = self._to_broker_type(existing.order_type)
            if (
                existing.symbol != request.symbol
                or existing_side != request.side
                or abs(float(existing.quantity) - float(request.quantity)) > 1e-6
                or existing_type != request.order_type
            ):
                raise BrokerInvalidRequestError(
                    f"Conflicting duplicate order request for client_order_id '{request.client_order_id}': "
                    f"existing=({existing.symbol}, {existing_side.value}, {existing.quantity}, {existing_type.value}), "
                    f"requested=({request.symbol}, {request.side.value}, {request.quantity}, {request.order_type.value})",
                    provider="PAPER",
                )
            logger.info("Idempotent order lookup: returning existing order %s", existing.order_id)
            return self._to_broker_order(existing)

        # 3. Check short capability if short sell
        if request.side == BrokerSide.SELL:
            current_pos = self.get_position(request.symbol)
            current_qty = current_pos.quantity if current_pos else 0.0
            if current_qty - request.quantity < 0:
                if not self.broker.config.allow_short or not self._capabilities.supports(BrokerCapability.SHORT_SELLING):
                    raise BrokerOrderRejectedError(
                        f"Short selling disallowed for symbol {request.symbol}",
                        provider="PAPER",
                        rejection_reason="SHORT_SELLING_DISALLOWED",
                    )

        # 4. Create PaperOrder
        order_idx = len(self.broker.order_manager.orders) + 1
        paper_order = PaperOrder(
            order_id=f"ord_{self.session_id}_{order_idx}",
            session_id=self.session_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=self._to_paper_side(request.side),
            quantity=request.quantity,
            order_type=self._to_paper_type(request.order_type),
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            submitted_at=request.submitted_at or datetime.now(timezone.utc),
            metadata=request.metadata,
        )

        submitted = self.broker.submit_order(paper_order)
        return self._to_broker_order(submitted)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel an open order by broker_order_id."""
        self._capabilities.require(BrokerCapability.CANCEL_ORDER, provider_name="PAPER")
        try:
            cancelled = self.broker.cancel_order(order_id)
            return self._to_broker_order(cancelled)
        except ValueError as e:
            raise BrokerOrderRejectedError(str(e), provider="PAPER", rejection_reason=str(e))

    def get_executions(self, order_id: Optional[str] = None) -> List[BrokerExecution]:
        """Retrieve execution records, optionally filtered by broker_order_id."""
        if order_id is not None:
            execs = [e for e in self.broker.executions if e.order_id == order_id]
        else:
            execs = self.broker.executions
        return [self._to_broker_execution(e) for e in execs]

    def execute_market_order(
        self,
        order_id: str,
        market_price: float,
        timestamp: datetime,
        reference_liquidity: Optional[float] = None,
    ) -> BrokerExecution:
        """
        Execute an accepted market order against market price in simulated paper mode.
        """
        paper_order = self.broker.get_order(order_id)
        if not paper_order:
            raise BrokerInvalidRequestError(f"Order {order_id} not found", provider="PAPER")
        
        exec_record = self.broker.execute_market_order(
            order=paper_order,
            market_price=market_price,
            timestamp=timestamp,
            reference_liquidity=reference_liquidity,
        )
        return self._to_broker_execution(exec_record)

    def health_check(self) -> BrokerHealthSnapshot:
        """Perform a liveness and status check."""
        return BrokerHealthSnapshot(
            status=BrokerStatus.CONNECTED,
            provider=BrokerProviderType.PAPER.value,
            execution_mode=BrokerExecutionMode.PAPER.value,
            connected=True,
            latency_ms=0.1,
            message="Simulated paper broker healthy and operational",
            details={
                "session_id": self.session_id,
                "account_id": self.broker.account.account_id,
                "cash": self.broker.account.cash,
                "positions_count": len(self.broker.account.positions),
            },
        )

    def get_capabilities(self) -> BrokerCapabilities:
        """Return capabilities supported by this broker."""
        return self._capabilities

    def get_status(self) -> BrokerStatus:
        """Return current operational status."""
        return BrokerStatus.CONNECTED
