"""
Paper Trading Execution Engine (Phase 14).
Simulates realistic trade fills applying transaction costs:
- Half-spread crossing
- Execution slippage
- Deterministic market impact
- Proportional commission
"""

from datetime import datetime, timezone
import logging
from typing import Optional, Tuple
import uuid

from backend.app.paper_trading.costs import PaperCostModel
from backend.app.paper_trading.schemas import (
    PaperExecution,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperTradingConfig,
)

logger = logging.getLogger(__name__)


class PaperExecutionEngine:
    """
    Simulates fill execution against current market prices.
    """

    def __init__(self, config: Optional[PaperTradingConfig] = None):
        self.config = config or PaperTradingConfig()
        self._exec_counter: int = 0

    def execute_order(
        self,
        order: PaperOrder,
        base_market_price: float,
        timestamp: datetime,
        reference_liquidity: Optional[float] = None,
    ) -> PaperExecution:
        """
        Execute an accepted market order, applying slippage, spread, and commission.
        """
        if order.status in {PaperOrderStatus.FILLED, PaperOrderStatus.REJECTED, PaperOrderStatus.CANCELLED}:
            raise ValueError(f"Cannot execute order in terminal state: {order.status.value}")
        if base_market_price <= 0.0:
            raise ValueError(f"Invalid market execution price: {base_market_price}")

        order_notional = order.quantity * base_market_price

        # Calculate market impact rate
        impact_rate = PaperCostModel.calculate_market_impact(
            order_notional=order_notional,
            market_impact_coefficient=self.config.market_impact_coefficient,
            reference_liquidity=reference_liquidity,
        )

        # Calculate adverse executed price
        executed_price = PaperCostModel.calculate_executed_price(
            side=order.side,
            base_price=base_market_price,
            slippage_rate=self.config.slippage_rate,
            spread_rate=self.config.bid_ask_spread_rate,
            market_impact_rate=impact_rate,
        )

        executed_notional = order.quantity * executed_price

        # Calculate cost attribution
        commission = PaperCostModel.calculate_commission(
            notional=executed_notional,
            commission_rate=self.config.commission_rate,
        )
        spread_cost = PaperCostModel.calculate_spread_cost(
            quantity=order.quantity,
            base_price=base_market_price,
            spread_rate=self.config.bid_ask_spread_rate,
        )
        slippage_cost = PaperCostModel.calculate_slippage_cost(
            quantity=order.quantity,
            base_price=base_market_price,
            executed_price=executed_price,
        )
        market_impact_cost = order.quantity * base_market_price * impact_rate

        self._exec_counter += 1
        exec_id = f"exec_{order.session_id}_{self._exec_counter}"

        execution = PaperExecution(
            execution_id=exec_id,
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            executed_price=round(executed_price, 4),
            base_price=round(base_market_price, 4),
            timestamp=timestamp,
            commission=round(commission, 4),
            spread_cost=round(spread_cost, 4),
            slippage_cost=round(slippage_cost, 4),
            market_impact_cost=round(market_impact_cost, 4),
            realized_pnl=0.0,  # Will be populated by PaperAccount
            execution_reference=f"sim_ref_{order.order_id}",
        )

        # Update order fill attributes
        order.status = PaperOrderStatus.FILLED
        order.filled_at = timestamp
        order.filled_quantity = order.quantity
        order.executed_price = execution.executed_price
        order.commission = execution.commission
        order.spread_cost = execution.spread_cost
        order.slippage_cost = execution.slippage_cost
        order.market_impact_cost = execution.market_impact_cost
        order.execution_reference = execution.execution_reference

        return execution
