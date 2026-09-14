"""
Simulated Order Execution Engine (Phase 13 Upgrade v1.2).
Translates target portfolio allocations into executable simulated orders, applies transaction costs,
slippage, bid-ask spread, and market impact, and executes fills against the portfolio ledger.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import uuid

from backend.app.backtest.costs import CostModel
from backend.app.backtest.portfolio import SimulatedPortfolio
from backend.app.backtest.schemas import (
    BacktestConfig,
    ExecutionRecord,
    OrderSide,
    PositionSide,
)
from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.risk.schemas import RiskAdjustedTarget


class SimulatedExecutionEngine:
    """
    Executes historical portfolio rebalancing orders with deterministic cost and friction models.
    """

    @classmethod
    def execute_rebalance(
        cls,
        portfolio: SimulatedPortfolio,
        targets: Union[Dict[str, float], List[PortfolioTarget], List[RiskAdjustedTarget]],
        prices: Dict[str, float],
        config: BacktestConfig,
        timestamp: datetime,
    ) -> List[ExecutionRecord]:
        """
        Execute target weights at point-in-time execution prices.
        """
        # Convert targets to normalized dict: symbol -> target_weight
        target_dict: Dict[str, float] = {}
        if isinstance(targets, dict):
            target_dict = {k: float(v) for k, v in targets.items()}
        elif isinstance(targets, list):
            for t in targets:
                if hasattr(t, "adjusted_weight"):
                    target_dict[t.symbol] = float(t.adjusted_weight)
                elif hasattr(t, "target_weight"):
                    target_dict[t.symbol] = float(t.target_weight)

        current_equity = portfolio.equity
        if current_equity <= 0:
            return []

        all_symbols = sorted(set(portfolio.positions.keys()) | set(target_dict.keys()))
        planned_orders: List[Dict] = []

        for sym in all_symbols:
            if sym not in prices or prices[sym] <= 0:
                continue

            base_px = float(prices[sym])
            target_w = target_dict.get(sym, 0.0)
            pos = portfolio.positions.get(sym)

            curr_qty = pos.quantity if pos else 0.0
            if pos and pos.side == PositionSide.SHORT:
                curr_qty = -curr_qty

            target_notional = target_w * current_equity
            target_qty = target_notional / base_px if base_px > 0 else 0.0
            delta_qty = target_qty - curr_qty

            if abs(delta_qty * base_px) < config.minimum_trade_notional:
                continue

            if delta_qty > 0:
                # Need to BUY
                side = OrderSide.BUY
                order_qty = delta_qty
            else:
                # Need to SELL
                side = OrderSide.SELL
                order_qty = abs(delta_qty)

            if not config.allow_fractional_shares:
                order_qty = float(int(order_qty))
                if order_qty <= 0:
                    continue

            planned_orders.append({
                "symbol": sym,
                "side": side,
                "quantity": order_qty,
                "base_price": base_px,
            })

        # Sort planned orders: SELLS first (to free cash), then BUYS
        planned_orders.sort(key=lambda o: 0 if o["side"] == OrderSide.SELL else 1)

        executions: List[ExecutionRecord] = []

        for order in planned_orders:
            sym = order["symbol"]
            side = order["side"]
            qty = order["quantity"]
            base_px = order["base_price"]

            order_base_notional = qty * base_px
            market_impact_rate = CostModel.calculate_market_impact(
                order_base_notional, config.market_impact_coefficient
            )

            exec_px = CostModel.calculate_executed_price(
                arg1=side,
                arg2=base_px,
                slippage_rate=config.slippage_rate,
                spread_rate=config.bid_ask_spread_rate,
                market_impact_rate=market_impact_rate,
            )
            notional = qty * exec_px
            comm = CostModel.calculate_commission(notional, config.commission_rate)

            # Isolated friction components
            slippage_px = base_px * (1.0 + (config.slippage_rate if side == OrderSide.BUY else -config.slippage_rate))
            slippage_cost = CostModel.calculate_slippage_cost(qty, base_px, slippage_px)
            spread_cost = CostModel.calculate_spread_cost(qty, base_px, config.bid_ask_spread_rate)
            impact_cost = float(order_base_notional * market_impact_rate)

            # Cash boundary protection for BUY orders in long-only portfolios
            if side == OrderSide.BUY:
                required_cash = notional + comm
                if portfolio.cash < required_cash:
                    # Scale down quantity to available cash
                    max_affordable_qty = max(0.0, (portfolio.cash - comm) / exec_px)
                    if not config.allow_fractional_shares:
                        max_affordable_qty = float(int(max_affordable_qty))
                    if max_affordable_qty <= 0 or (max_affordable_qty * exec_px) < config.minimum_trade_notional:
                        continue
                    qty = max_affordable_qty
                    notional = qty * exec_px
                    comm = CostModel.calculate_commission(notional, config.commission_rate)
                    slippage_cost = CostModel.calculate_slippage_cost(qty, base_px, slippage_px)
                    spread_cost = CostModel.calculate_spread_cost(qty, base_px, config.bid_ask_spread_rate)
                    impact_cost = float(qty * base_px * market_impact_rate)

            # Apply fill to portfolio
            portfolio.apply_fill(
                symbol=sym,
                side=side,
                quantity=qty,
                executed_price=exec_px,
                commission=comm,
                slippage_cost=slippage_cost,
                spread_cost=spread_cost,
                borrow_cost=0.0,
                market_impact_cost=impact_cost,
                timestamp=timestamp,
            )

            rec = ExecutionRecord(
                execution_id=f"exec_{timestamp.strftime('%Y%m%d%H%M%S')}_{sym}_{side.value}",
                timestamp=timestamp,
                symbol=sym,
                side=side,
                quantity=qty,
                requested_price=base_px,
                executed_price=exec_px,
                notional=notional,
                commission=comm,
                slippage_cost=slippage_cost,
                spread_cost=spread_cost,
                borrow_cost=0.0,
                market_impact_cost=impact_cost,
                reason="REBALANCE",
            )
            executions.append(rec)

        return executions
