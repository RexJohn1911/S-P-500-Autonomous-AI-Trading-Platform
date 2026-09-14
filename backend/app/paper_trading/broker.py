"""
Paper Trading Broker Abstraction (Phase 14).
Provides BasePaperBroker interface and SimulatedPaperBroker implementation.
STRICT SAFETY BOUNDARY:
This module contains ZERO live broker API connections, credentials, or live order routing.
It operates purely as a self-contained simulated matching engine.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from backend.app.paper_trading.account import PaperAccount
from backend.app.paper_trading.execution import PaperExecutionEngine
from backend.app.paper_trading.orders import PaperOrderManager
from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperExecution,
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    PaperTradingConfig,
)

logger = logging.getLogger(__name__)


class BasePaperBroker(ABC):
    """
    Abstract interface for paper trading brokerage operations.
    """

    @abstractmethod
    def submit_order(self, order: PaperOrder) -> PaperOrder:
        """Submit a paper order for execution."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        """Retrieve an order by ID."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> PaperOrder:
        """Cancel an open order."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[PaperOrder]:
        """Return all open orders."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, PaperPosition]:
        """Return current positions."""
        pass

    @abstractmethod
    def get_account_snapshot(self, timestamp: datetime, current_prices: Dict[str, float]) -> PaperAccountSnapshot:
        """Generate point-in-time account valuation snapshot."""
        pass


class SimulatedPaperBroker(BasePaperBroker):
    """
    In-memory simulated broker managing the order book, account ledger, and execution engine.
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[PaperTradingConfig] = None,
        account: Optional[PaperAccount] = None,
        order_manager: Optional[PaperOrderManager] = None,
        execution_engine: Optional[PaperExecutionEngine] = None,
    ):
        self.session_id = session_id
        self.config = config or PaperTradingConfig()
        self.account = account or PaperAccount(
            account_id=f"acc_{session_id}",
            session_id=session_id,
            config=self.config,
        )
        self.order_manager = order_manager or PaperOrderManager(
            session_id=session_id,
            config=self.config,
        )
        self.execution_engine = execution_engine or PaperExecutionEngine(config=self.config)
        self.executions: List[PaperExecution] = []

    def submit_order(self, order: PaperOrder) -> PaperOrder:
        """
        Submit a paper order. Idempotent by client_order_id.
        """
        existing = self.order_manager.get_order_by_client_id(order.client_order_id)
        if existing and existing.order_id != order.order_id:
            logger.info("Idempotent order submission detected: client_order_id %s matches order %s", order.client_order_id, existing.order_id)
            return existing

        if order.status == PaperOrderStatus.CREATED:
            order.status = PaperOrderStatus.SUBMITTED
            self.order_manager.register_order(order)

        if order.status == PaperOrderStatus.SUBMITTED:
            order.status = PaperOrderStatus.ACCEPTED
            order.accepted_at = order.submitted_at or datetime.now(timezone.utc)

        return order

    def execute_market_order(
        self,
        order: PaperOrder,
        market_price: float,
        timestamp: datetime,
        reference_liquidity: Optional[float] = None,
    ) -> PaperExecution:
        """
        Execute an accepted market order against market price.
        """
        # Ensure order is submitted & accepted
        if order.status in (PaperOrderStatus.CREATED, PaperOrderStatus.SUBMITTED):
            self.submit_order(order)

        # Execute fill
        execution = self.execution_engine.execute_order(
            order=order,
            base_market_price=market_price,
            timestamp=timestamp,
            reference_liquidity=reference_liquidity,
        )

        # Apply to account ledger
        realized_pnl = self.account.apply_execution(execution)
        execution.realized_pnl = realized_pnl
        self.executions.append(execution)

        return execution

    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        """Retrieve order by order_id."""
        return self.order_manager.orders.get(order_id)

    def cancel_order(self, order_id: str) -> PaperOrder:
        """Cancel an open paper order."""
        order = self.order_manager.orders.get(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")
        if order.is_terminal():
            raise ValueError(f"Cannot cancel order in terminal state: {order.status.value}")

        return self.order_manager.transition_order(
            order_id=order_id,
            new_status=PaperOrderStatus.CANCELLED,
            timestamp=datetime.now(timezone.utc),
            reason="User/System cancellation",
        )

    def get_open_orders(self) -> List[PaperOrder]:
        """Return all open orders."""
        return self.order_manager.get_open_orders()

    def get_positions(self) -> Dict[str, PaperPosition]:
        """Return active positions."""
        return {s: p for s, p in self.account.positions.items() if abs(p.quantity) >= 1e-8}

    def get_account_snapshot(
        self,
        timestamp: datetime,
        current_prices: Dict[str, float],
    ) -> PaperAccountSnapshot:
        """Generate account mark-to-market valuation snapshot."""
        snapshot = self.account.mark_to_market(current_prices=current_prices, timestamp=timestamp)
        snapshot.open_orders_count = len(self.get_open_orders())
        return snapshot
