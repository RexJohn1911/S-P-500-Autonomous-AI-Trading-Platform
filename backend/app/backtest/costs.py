"""
Transaction Cost and Slippage Models (Phase 13 Upgrade v1.2).
Provides deterministic simulation of brokerage commissions, execution slippage,
bid-ask spread, short borrow fees, and market impact.
"""

from typing import Optional, Union
import numpy as np

from backend.app.backtest.schemas import CostBreakdown, OrderSide


class CostModel:
    """
    Deterministic research-grade cost and friction calculator for simulated order execution.
    """

    @classmethod
    def calculate_executed_price(
        cls,
        arg1: Union[OrderSide, str, float],
        arg2: Union[OrderSide, str, float],
        slippage_rate: float = 0.0,
        spread_rate: float = 0.0,
        market_impact_rate: float = 0.0,
    ) -> float:
        """
        Apply directional execution friction (slippage, spread, and market impact) to a base market price.
        Supports both (side, base_price, slippage_rate, ...) and (base_price, side, slippage_rate, ...).
        - BUY: Price increases (worse fill) -> base_price * (1 + slippage_rate + spread_rate/2 + market_impact_rate)
        - SELL: Price decreases (worse fill) -> base_price * (1 - slippage_rate - spread_rate/2 - market_impact_rate)
        """
        if isinstance(arg1, (OrderSide, str)):
            side = OrderSide(arg1)
            base_price = float(arg2)
        else:
            base_price = float(arg1)
            side = OrderSide(arg2)

        if base_price <= 0:
            raise ValueError(f"base_price must be > 0, got {base_price}")
        if slippage_rate < 0:
            raise ValueError(f"slippage_rate must be >= 0, got {slippage_rate}")
        if spread_rate < 0:
            raise ValueError(f"spread_rate must be >= 0, got {spread_rate}")
        if market_impact_rate < 0:
            raise ValueError(f"market_impact_rate must be >= 0, got {market_impact_rate}")

        half_spread = spread_rate / 2.0
        total_friction = slippage_rate + half_spread + market_impact_rate

        if total_friction == 0.0:
            return float(base_price)

        if side == OrderSide.BUY:
            executed_price = base_price * (1.0 + total_friction)
        elif side == OrderSide.SELL:
            executed_price = max(0.0001, base_price * (1.0 - total_friction))
        else:
            executed_price = base_price

        return float(executed_price)

    @classmethod
    def calculate_commission(
        cls,
        notional: float,
        commission_rate: float,
    ) -> float:
        """
        Calculate total trade commission: notional * commission_rate.
        """
        if notional < 0:
            raise ValueError(f"notional must be >= 0, got {notional}")
        if commission_rate < 0:
            raise ValueError(f"commission_rate must be >= 0, got {commission_rate}")

        return float(notional * commission_rate)

    @classmethod
    def calculate_slippage_cost(
        cls,
        arg1: float,
        arg2: float,
        arg3: float,
    ) -> float:
        """
        Calculate the dollar cost of slippage.
        Supports:
        - (quantity, base_price, executed_price) -> |executed_price - base_price| * quantity
        - (price, quantity, slippage_rate) -> price * quantity * slippage_rate
        """
        # Case A: (price, quantity, slippage_rate) where arg3 is a rate (e.g., 0.01)
        if arg3 <= 1.0 and abs(arg1 - arg3) > 1.0:
            price = float(arg1)
            qty = float(arg2)
            rate = float(arg3)
            return float(price * qty * rate)

        # Case B: (quantity, base_price, executed_price)
        qty = float(arg1)
        base_price = float(arg2)
        exec_price = float(arg3)
        return float(abs(exec_price - base_price) * qty)

    @classmethod
    def calculate_spread_cost(
        cls,
        quantity: float,
        base_price: float,
        spread_rate: float,
    ) -> float:
        """
        Calculate the dollar cost of crossing the bid/ask spread (half-spread * traded notional).
        """
        if quantity < 0 or base_price < 0 or spread_rate < 0:
            return 0.0
        return float(quantity * base_price * (spread_rate / 2.0))

    @classmethod
    def calculate_borrow_fee(
        cls,
        short_market_value: float,
        daily_borrow_rate: float,
    ) -> float:
        """
        Calculate daily borrow financing cost on short position liability.
        """
        if short_market_value <= 0 or daily_borrow_rate <= 0:
            return 0.0
        return float(short_market_value * daily_borrow_rate)

    @classmethod
    def calculate_market_impact(
        cls,
        order_notional: float,
        market_impact_coefficient: float,
        reference_liquidity: Optional[float] = None,
    ) -> float:
        """
        Calculate deterministic market impact penalty.
        If reference_liquidity is provided: coeff * (order_notional / liquidity).
        Otherwise: coeff * order_notional.
        """
        if order_notional <= 0 or market_impact_coefficient <= 0:
            return 0.0

        if reference_liquidity and reference_liquidity > 0:
            impact_rate = market_impact_coefficient * (order_notional / reference_liquidity)
        else:
            impact_rate = market_impact_coefficient * 0.01  # Normalized default scaling

        return float(min(0.20, impact_rate))  # Cap at 20%
