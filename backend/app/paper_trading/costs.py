"""
Paper Trading Transaction Cost Model (Phase 14).
Provides deterministic friction calculations for simulated execution:
- Brokerage commission
- Directional half-spread
- Execution slippage
- Market impact
- Short borrow financing fee
"""

from typing import Optional, Union
import numpy as np

from backend.app.paper_trading.schemas import PaperOrderSide


class PaperCostModel:
    """
    Deterministic cost calculator for paper trading execution.
    Preserves exact mathematical consistency with Phase 13 CostModel.
    """

    @classmethod
    def calculate_executed_price(
        cls,
        side: PaperOrderSide,
        base_price: float,
        slippage_rate: float = 0.0,
        spread_rate: float = 0.0,
        market_impact_rate: float = 0.0,
    ) -> float:
        """
        Apply adverse execution friction (slippage, spread, market impact) to base market price.
        - BUY: Price increases (worse fill) -> base_price * (1 + slippage + spread/2 + impact)
        - SELL: Price decreases (worse fill) -> base_price * (1 - slippage - spread/2 - impact)
        """
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

        if side == PaperOrderSide.BUY:
            executed_price = base_price * (1.0 + total_friction)
        elif side == PaperOrderSide.SELL:
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
        quantity: float,
        base_price: float,
        executed_price: float,
    ) -> float:
        """
        Calculate the dollar cost of slippage: |executed_price - base_price| * quantity.
        """
        if quantity < 0 or base_price <= 0 or executed_price <= 0:
            return 0.0
        return float(abs(executed_price - base_price) * quantity)

    @classmethod
    def calculate_spread_cost(
        cls,
        quantity: float,
        base_price: float,
        spread_rate: float,
    ) -> float:
        """
        Calculate the dollar cost of crossing the bid/ask spread: quantity * base_price * (spread_rate / 2).
        """
        if quantity < 0 or base_price <= 0 or spread_rate <= 0:
            return 0.0
        return float(quantity * base_price * (spread_rate / 2.0))

    @classmethod
    def calculate_market_impact(
        cls,
        order_notional: float,
        market_impact_coefficient: float,
        reference_liquidity: Optional[float] = None,
    ) -> float:
        """
        Calculate deterministic market impact penalty rate.
        """
        if order_notional <= 0 or market_impact_coefficient <= 0:
            return 0.0

        if reference_liquidity and reference_liquidity > 0:
            impact_rate = market_impact_coefficient * (order_notional / reference_liquidity)
        else:
            impact_rate = market_impact_coefficient * 0.01

        return float(min(0.20, impact_rate))

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
