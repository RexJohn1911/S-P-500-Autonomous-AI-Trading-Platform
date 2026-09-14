"""
Paper Trading Order Generation and Lifecycle State Machine (Phase 14).
Converts current positions + risk-approved targets into executable paper orders,
enforcing explicit state transitions and client order ID idempotency.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Union
import uuid

from backend.app.paper_trading.risk_guard import PaperExecutionRiskGuard
from backend.app.paper_trading.schemas import (
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
    PaperPosition,
    PaperTimeInForce,
    PaperTradingConfig,
)
from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.risk.schemas import RiskAdjustedTarget

logger = logging.getLogger(__name__)


class PaperOrderManager:
    """
    Manages order creation, generation from portfolio targets, and lifecycle transitions.
    """

    # Allowed state machine transitions
    VALID_TRANSITIONS = {
        PaperOrderStatus.CREATED: {PaperOrderStatus.SUBMITTED, PaperOrderStatus.REJECTED, PaperOrderStatus.CANCELLED},
        PaperOrderStatus.SUBMITTED: {PaperOrderStatus.ACCEPTED, PaperOrderStatus.REJECTED, PaperOrderStatus.CANCELLED},
        PaperOrderStatus.ACCEPTED: {PaperOrderStatus.PARTIALLY_FILLED, PaperOrderStatus.FILLED, PaperOrderStatus.REJECTED, PaperOrderStatus.CANCELLED, PaperOrderStatus.EXPIRED},
        PaperOrderStatus.PARTIALLY_FILLED: {PaperOrderStatus.FILLED, PaperOrderStatus.CANCELLED, PaperOrderStatus.EXPIRED},
        PaperOrderStatus.FILLED: set(),      # Terminal
        PaperOrderStatus.REJECTED: set(),    # Terminal
        PaperOrderStatus.CANCELLED: set(),   # Terminal
        PaperOrderStatus.EXPIRED: set(),     # Terminal
    }

    def __init__(self, session_id: str, config: Optional[PaperTradingConfig] = None):
        self.session_id = session_id
        self.config = config or PaperTradingConfig()
        self.orders: Dict[str, PaperOrder] = {}  # order_id -> PaperOrder
        self.client_id_map: Dict[str, str] = {}  # client_order_id -> order_id
        self._order_counter: int = 0

    def generate_orders_from_targets(
        self,
        targets: Union[List[RiskAdjustedTarget], List[PortfolioTarget], Dict[str, float]],
        current_positions: Dict[str, PaperPosition],
        current_prices: Dict[str, float],
        portfolio_equity: float,
        timestamp: datetime,
    ) -> List[PaperOrder]:
        """
        Convert risk-approved target portfolio weights into rebalancing paper orders.
        """
        if portfolio_equity <= 0:
            logger.warning("Cannot generate orders with zero or negative portfolio equity ($%.2f)", portfolio_equity)
            return []

        # Normalize targets to Dict[str, float]
        target_weights: Dict[str, float] = {}
        if isinstance(targets, dict):
            target_weights = {s: float(w) for s, w in targets.items()}
        elif isinstance(targets, list):
            for t in targets:
                if isinstance(t, RiskAdjustedTarget):
                    target_weights[t.symbol] = float(t.adjusted_weight)
                elif isinstance(t, PortfolioTarget):
                    target_weights[t.symbol] = float(t.target_weight)

        all_symbols = set(current_positions.keys()).union(set(target_weights.keys()))
        generated_orders: List[PaperOrder] = []

        for symbol in sorted(all_symbols):
            price = current_prices.get(symbol, 0.0)
            if price <= 0.0:
                logger.warning("Skipping order generation for %s: no valid price available", symbol)
                continue

            current_pos = current_positions.get(symbol)
            curr_qty = current_pos.quantity if current_pos else 0.0
            curr_weight = (curr_qty * price) / portfolio_equity if portfolio_equity > 0 else 0.0

            target_weight = target_weights.get(symbol, 0.0)
            weight_delta = target_weight - curr_weight

            # Skip negligible changes
            if abs(weight_delta) < self.config.min_position_delta and abs(target_weight) < 1e-6 and abs(curr_qty) < 1e-6:
                continue

            target_notional = target_weight * portfolio_equity
            target_qty = target_notional / price
            delta_qty = target_qty - curr_qty

            order_notional = abs(delta_qty) * price
            if order_notional < self.config.min_order_notional:
                continue

            side = PaperOrderSide.BUY if delta_qty > 0 else PaperOrderSide.SELL
            qty = abs(delta_qty)

            if not self.config.allow_fractional_shares:
                qty = float(int(qty))
                if qty <= 0:
                    continue

            self._order_counter += 1
            order_id = f"ord_{self.session_id}_{self._order_counter}"
            client_order_id = f"cli_{self.session_id}_{symbol}_{int(timestamp.timestamp())}_{self._order_counter}"

            order = PaperOrder(
                order_id=order_id,
                client_order_id=client_order_id,
                session_id=self.session_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=PaperOrderType.MARKET,
                status=PaperOrderStatus.CREATED,
                submitted_at=timestamp,
                requested_price=price,
                metadata={"target_weight": target_weight, "curr_weight": curr_weight, "weight_delta": weight_delta},
            )

            # Execution guard validation
            is_valid, rejection_reason = PaperExecutionRiskGuard.validate_order(
                order=order,
                current_market_price=price,
                config=self.config,
            )

            if not is_valid:
                order.status = PaperOrderStatus.REJECTED
                order.rejection_reason = rejection_reason
                logger.warning("Order rejected by execution risk guard: %s - %s", order.order_id, rejection_reason)
            else:
                order.status = PaperOrderStatus.SUBMITTED

            self.register_order(order)
            generated_orders.append(order)

        return generated_orders

    def register_order(self, order: PaperOrder) -> None:
        """Register order and record client_order_id for idempotency."""
        self.orders[order.order_id] = order
        self.client_id_map[order.client_order_id] = order.order_id

    def get_order_by_client_id(self, client_order_id: str) -> Optional[PaperOrder]:
        """Lookup order by client_order_id."""
        order_id = self.client_id_map.get(client_order_id)
        if order_id:
            return self.orders.get(order_id)
        return None

    def transition_order(
        self,
        order_id: str,
        new_status: PaperOrderStatus,
        timestamp: datetime,
        reason: Optional[str] = None,
    ) -> PaperOrder:
        """
        Transition order to a new status enforcing state machine invariants.
        """
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")

        order = self.orders[order_id]
        current_status = order.status

        if current_status == new_status:
            return order

        valid_next_states = self.VALID_TRANSITIONS.get(current_status, set())
        if new_status not in valid_next_states:
            raise ValueError(
                f"Invalid order state transition: {current_status.value} -> {new_status.value} for order {order_id}"
            )

        order.status = new_status
        if new_status == PaperOrderStatus.ACCEPTED:
            order.accepted_at = timestamp
        elif new_status == PaperOrderStatus.FILLED:
            order.filled_at = timestamp
        elif new_status == PaperOrderStatus.CANCELLED:
            order.cancelled_at = timestamp
        elif new_status == PaperOrderStatus.REJECTED:
            order.rejection_reason = reason

        return order

    def get_open_orders(self) -> List[PaperOrder]:
        """Return all active, non-terminal orders."""
        return [o for o in self.orders.values() if not o.is_terminal()]
