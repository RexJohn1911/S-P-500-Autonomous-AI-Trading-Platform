"""
Paper Trading Execution Safety Guard (Phase 14).
Validates execution-level integrity before order submission:
- Checks for NaN/infinite/negative order quantity
- Validates symbol and pricing validity
- Validates order side and account state
This is a final execution guard and does NOT replace the upstream Phase 12 Risk Engine.
"""

import logging
import math
from typing import Optional, Tuple
import numpy as np

from backend.app.paper_trading.schemas import (
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperTradingConfig,
)

logger = logging.getLogger(__name__)


class PaperExecutionRiskGuard:
    """
    Execution-level safety guard ensuring physical and logical validity of orders.
    """

    @classmethod
    def validate_order(
        cls,
        order: PaperOrder,
        current_market_price: Optional[float] = None,
        available_cash: Optional[float] = None,
        config: Optional[PaperTradingConfig] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate order parameters. Returns (is_valid, rejection_reason).
        """
        # 1. Symbol validation
        if not order.symbol or not isinstance(order.symbol, str) or not order.symbol.strip():
            return False, "Missing or invalid symbol"

        # 2. Quantity validation
        if order.quantity is None:
            return False, "Order quantity is None"
        if math.isnan(order.quantity) or np.isnan(order.quantity):
            return False, "Order quantity is NaN"
        if math.isinf(order.quantity) or np.isinf(order.quantity):
            return False, "Order quantity is infinite"
        if order.quantity <= 0.0:
            return False, f"Order quantity must be > 0, got {order.quantity}"

        # 3. Side validation
        if order.side not in (PaperOrderSide.BUY, PaperOrderSide.SELL):
            return False, f"Invalid order side: {order.side}"

        # 4. Price validation
        price_to_check = current_market_price or order.requested_price
        if price_to_check is not None:
            if math.isnan(price_to_check) or np.isnan(price_to_check):
                return False, "Market/requested price is NaN"
            if math.isinf(price_to_check) or np.isinf(price_to_check):
                return False, "Market/requested price is infinite"
            if price_to_check <= 0.0:
                return False, f"Price must be > 0, got {price_to_check}"

        # 5. Order Notional Threshold
        if config and price_to_check:
            notional = order.quantity * price_to_check
            if notional < config.min_order_notional:
                return False, f"Order notional ${notional:.2f} is below minimum threshold ${config.min_order_notional:.2f}"

        # 6. Cash availability for Long Buy (if cash check requested)
        if available_cash is not None and order.side == PaperOrderSide.BUY and price_to_check:
            estimated_cost = order.quantity * price_to_check
            # Allow modest 1% buffer for slippage/commissions if not short enabled
            if config and not config.allow_short and estimated_cost > available_cash * 1.05:
                return False, f"Insufficient cash: required approx ${estimated_cost:.2f}, available ${available_cash:.2f}"

        return True, None
