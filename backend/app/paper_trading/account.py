"""
Paper Trading Account Ledger and Accounting Manager (Phase 14).
Provides double-entry bookkeeping, multi-lot FIFO realization, position tracking,
corporate action processing (dividends, splits), short borrow financing, and
mark-to-market valuation enforcing the fundamental identity:
    equity = cash + sum(market_value)
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.app.backtest.schemas import DividendEvent, SplitEvent
from backend.app.paper_trading.costs import PaperCostModel
from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperExecution,
    PaperFifoLot,
    PaperOrderSide,
    PaperPosition,
    PaperPositionSide,
    PaperTradingConfig,
)

logger = logging.getLogger(__name__)


class PaperAccount:
    """
    Simulated trading account managing cash, positions, FIFO lots, and P&L.
    """

    def __init__(
        self,
        account_id: str,
        session_id: str,
        config: Optional[PaperTradingConfig] = None,
    ):
        self.account_id = account_id
        self.session_id = session_id
        self.config = config or PaperTradingConfig()
        self.initial_cash: float = float(self.config.initial_capital)
        self.cash: float = float(self.config.initial_capital)
        self.positions: Dict[str, PaperPosition] = {}
        
        # Cumulative metrics
        self.realized_pnl: float = 0.0
        self.cumulative_commission: float = 0.0
        self.cumulative_slippage: float = 0.0
        self.cumulative_spread: float = 0.0
        self.cumulative_market_impact: float = 0.0
        self.cumulative_borrow_cost: float = 0.0
        self.total_trades_count: int = 0
        self._lot_counter: int = 0

    @property
    def total_transaction_cost(self) -> float:
        return float(
            self.cumulative_commission
            + self.cumulative_slippage
            + self.cumulative_spread
            + self.cumulative_market_impact
            + self.cumulative_borrow_cost
        )

    def get_position(self, symbol: str) -> PaperPosition:
        """Return position for a given symbol or an empty FLAT position."""
        if symbol not in self.positions:
            self.positions[symbol] = PaperPosition(symbol=symbol)
        return self.positions[symbol]

    def mark_to_market(
        self,
        current_prices: Dict[str, float],
        timestamp: datetime,
    ) -> PaperAccountSnapshot:
        """
        Mark all open positions to market with current point-in-time prices,
        calculating unrealized P&L, gross/net exposure, leverage, and total equity.
        """
        total_market_value = 0.0
        total_unrealized_pnl = 0.0
        long_market_value = 0.0
        short_market_value = 0.0

        for symbol, pos in self.positions.items():
            if abs(pos.quantity) < 1e-8:
                pos.quantity = 0.0
                pos.market_value = 0.0
                pos.unrealized_pnl = 0.0
                pos.cost_basis = 0.0
                pos.avg_entry_price = 0.0
                pos.side = PaperPositionSide.FLAT
                continue

            px = current_prices.get(symbol, pos.market_price if pos.market_price > 0 else pos.avg_entry_price)
            if px is None or px <= 0:
                px = pos.avg_entry_price

            pos.market_price = float(px)
            pos.market_value = float(pos.quantity * px)
            pos.updated_at = timestamp

            if pos.quantity > 0:
                pos.side = PaperPositionSide.LONG
                pos.cost_basis = float(pos.quantity * pos.avg_entry_price)
                pos.unrealized_pnl = float(pos.market_value - pos.cost_basis)
                long_market_value += pos.market_value
            else:
                pos.side = PaperPositionSide.SHORT
                pos.cost_basis = float(abs(pos.quantity) * pos.avg_entry_price)
                # Short unrealized P&L = cost_basis - abs(market_value)
                pos.unrealized_pnl = float(pos.cost_basis - abs(pos.market_value))
                short_market_value += abs(pos.market_value)

            total_market_value += pos.market_value
            total_unrealized_pnl += pos.unrealized_pnl

        equity = float(self.cash + total_market_value)
        gross_notional = long_market_value + short_market_value
        net_notional = long_market_value - short_market_value

        gross_exposure = float(gross_notional / equity) if equity > 0 else 0.0
        net_exposure = float(net_notional / equity) if equity > 0 else 0.0
        long_exposure = float(long_market_value / equity) if equity > 0 else 0.0
        short_exposure = float(short_market_value / equity) if equity > 0 else 0.0
        leverage = gross_exposure
        buying_power = max(0.0, float(self.cash)) if not self.config.allow_short else max(0.0, float(equity * 2.0 - gross_notional))

        active_positions = {s: p for s, p in self.positions.items() if abs(p.quantity) >= 1e-8}

        snapshot_id = f"snap_{self.session_id}_{int(timestamp.timestamp())}"
        return PaperAccountSnapshot(
            snapshot_id=snapshot_id,
            session_id=self.session_id,
            timestamp=timestamp,
            cash=round(float(self.cash), 4),
            equity=round(float(equity), 4),
            market_value=round(float(total_market_value), 4),
            buying_power=round(float(buying_power), 4),
            unrealized_pnl=round(float(total_unrealized_pnl), 4),
            realized_pnl=round(float(self.realized_pnl), 4),
            total_pnl=round(float(self.realized_pnl + total_unrealized_pnl), 4),
            cumulative_commission=round(float(self.cumulative_commission), 4),
            cumulative_slippage=round(float(self.cumulative_slippage), 4),
            cumulative_spread=round(float(self.cumulative_spread), 4),
            cumulative_market_impact=round(float(self.cumulative_market_impact), 4),
            cumulative_borrow_cost=round(float(self.cumulative_borrow_cost), 4),
            total_transaction_cost=round(float(self.total_transaction_cost), 4),
            gross_exposure=round(float(gross_exposure), 6),
            net_exposure=round(float(net_exposure), 6),
            long_exposure=round(float(long_exposure), 6),
            short_exposure=round(float(short_exposure), 6),
            leverage=round(float(leverage), 6),
            positions_count=len(active_positions),
            open_orders_count=0,
            positions=active_positions,
        )

    def apply_execution(self, execution: PaperExecution) -> float:
        """
        Apply executed order fill to account ledger with double-entry cash/position updating
        and FIFO lot realization.
        Returns the realized P&L from this execution.
        """
        symbol = execution.symbol
        side = execution.side
        qty = float(execution.quantity)
        price = float(execution.executed_price)
        ts = execution.timestamp

        if qty <= 0 or price <= 0:
            raise ValueError(f"Invalid execution fill: qty={qty}, price={price}")

        # Update cumulative friction costs
        self.cumulative_commission += execution.commission
        self.cumulative_slippage += execution.slippage_cost
        self.cumulative_spread += execution.spread_cost
        self.cumulative_market_impact += execution.market_impact_cost
        self.total_trades_count += 1

        position = self.get_position(symbol)
        curr_qty = position.quantity
        realized_pnl = 0.0

        if side == PaperOrderSide.BUY:
            if curr_qty >= 0:
                # Adding to existing LONG or opening new LONG
                gross_cost = qty * price
                self.cash -= (gross_cost + execution.commission + execution.spread_cost + execution.slippage_cost + execution.market_impact_cost)
                
                # Add FIFO lot
                self._lot_counter += 1
                lot = PaperFifoLot(
                    lot_id=f"lot_{symbol}_{self._lot_counter}",
                    symbol=symbol,
                    quantity=qty,
                    entry_price=price,
                    timestamp=ts,
                    side=PaperPositionSide.LONG,
                )
                position.fifo_lots.append(lot)
                
                # Update position stats
                new_qty = curr_qty + qty
                total_cost = (curr_qty * position.avg_entry_price) + (qty * price)
                position.avg_entry_price = total_cost / new_qty if new_qty > 0 else 0.0
                position.quantity = new_qty
                position.side = PaperPositionSide.LONG if new_qty > 0 else PaperPositionSide.FLAT
                if position.opened_at is None:
                    position.opened_at = ts
                position.updated_at = ts
            else:
                # Buying to cover existing SHORT position (partial or full exit or reversal)
                short_qty_to_cover = min(abs(curr_qty), qty)
                covered_realized_pnl, remaining_lots = self._match_fifo_short(position.fifo_lots, short_qty_to_cover, price)
                realized_pnl += covered_realized_pnl
                position.fifo_lots = remaining_lots

                # Cash adjustment for covering short
                # For short exit: cash was credited on entry; on cover, we pay: short_qty_to_cover * price
                self.cash -= (short_qty_to_cover * price + execution.commission + execution.spread_cost + execution.slippage_cost + execution.market_impact_cost)
                
                remaining_short_qty = abs(curr_qty) - short_qty_to_cover
                excess_buy_qty = qty - short_qty_to_cover

                if remaining_short_qty > 1e-8:
                    position.quantity = -remaining_short_qty
                    position.side = PaperPositionSide.SHORT
                    position.updated_at = ts
                elif excess_buy_qty > 1e-8:
                    # Position reversed to LONG
                    gross_cost = excess_buy_qty * price
                    self.cash -= gross_cost
                    self._lot_counter += 1
                    lot = PaperFifoLot(
                        lot_id=f"lot_{symbol}_{self._lot_counter}",
                        symbol=symbol,
                        quantity=excess_buy_qty,
                        entry_price=price,
                        timestamp=ts,
                        side=PaperPositionSide.LONG,
                    )
                    position.fifo_lots = [lot]
                    position.quantity = excess_buy_qty
                    position.avg_entry_price = price
                    position.side = PaperPositionSide.LONG
                    position.opened_at = ts
                    position.updated_at = ts
                else:
                    # Fully closed
                    position.quantity = 0.0
                    position.avg_entry_price = 0.0
                    position.side = PaperPositionSide.FLAT
                    position.updated_at = ts

        elif side == PaperOrderSide.SELL:
            if curr_qty > 0:
                # Selling from existing LONG position (partial or full exit or reversal)
                long_qty_to_sell = min(curr_qty, qty)
                sold_realized_pnl, remaining_lots = self._match_fifo_long(position.fifo_lots, long_qty_to_sell, price)
                realized_pnl += sold_realized_pnl
                position.fifo_lots = remaining_lots

                # Cash proceeds from selling long
                gross_proceeds = long_qty_to_sell * price
                self.cash += (gross_proceeds - (execution.commission + execution.spread_cost + execution.slippage_cost + execution.market_impact_cost))

                remaining_long_qty = curr_qty - long_qty_to_sell
                excess_sell_qty = qty - long_qty_to_sell

                if remaining_long_qty > 1e-8:
                    position.quantity = remaining_long_qty
                    position.side = PaperPositionSide.LONG
                    position.updated_at = ts
                elif excess_sell_qty > 1e-8 and self.config.allow_short:
                    # Position reversed to SHORT
                    gross_short_proceeds = excess_sell_qty * price
                    self.cash += gross_short_proceeds
                    self._lot_counter += 1
                    lot = PaperFifoLot(
                        lot_id=f"lot_{symbol}_{self._lot_counter}",
                        symbol=symbol,
                        quantity=excess_sell_qty,
                        entry_price=price,
                        timestamp=ts,
                        side=PaperPositionSide.SHORT,
                    )
                    position.fifo_lots = [lot]
                    position.quantity = -excess_sell_qty
                    position.avg_entry_price = price
                    position.side = PaperPositionSide.SHORT
                    position.opened_at = ts
                    position.updated_at = ts
                else:
                    # Fully closed
                    position.quantity = 0.0
                    position.avg_entry_price = 0.0
                    position.side = PaperPositionSide.FLAT
                    position.updated_at = ts
            else:
                # Opening new SHORT or adding to existing SHORT
                if not self.config.allow_short:
                    raise ValueError(f"Short selling disabled in PaperAccount for symbol {symbol}")

                gross_proceeds = qty * price
                self.cash += (gross_proceeds - (execution.commission + execution.spread_cost + execution.slippage_cost + execution.market_impact_cost))

                self._lot_counter += 1
                lot = PaperFifoLot(
                    lot_id=f"lot_{symbol}_{self._lot_counter}",
                    symbol=symbol,
                    quantity=qty,
                    entry_price=price,
                    timestamp=ts,
                    side=PaperPositionSide.SHORT,
                )
                position.fifo_lots.append(lot)

                new_short_qty = abs(curr_qty) + qty
                total_short_cost = (abs(curr_qty) * position.avg_entry_price) + (qty * price)
                position.avg_entry_price = total_short_cost / new_short_qty if new_short_qty > 0 else 0.0
                position.quantity = -new_short_qty
                position.side = PaperPositionSide.SHORT
                if position.opened_at is None:
                    position.opened_at = ts
                position.updated_at = ts

        self.realized_pnl += realized_pnl
        position.realized_pnl += realized_pnl
        execution.realized_pnl = round(realized_pnl, 4)
        return realized_pnl

    def _match_fifo_long(
        self,
        lots: List[PaperFifoLot],
        sell_qty: float,
        sell_price: float,
    ) -> Tuple[float, List[PaperFifoLot]]:
        """Match and consume FIFO lots for LONG sell orders."""
        realized_pnl = 0.0
        remaining_to_sell = sell_qty
        new_lots: List[PaperFifoLot] = []

        for lot in lots:
            if remaining_to_sell <= 1e-8:
                new_lots.append(lot)
                continue

            lot_qty = lot.quantity
            matched_qty = min(lot_qty, remaining_to_sell)
            lot_pnl = matched_qty * (sell_price - lot.entry_price)
            realized_pnl += lot_pnl
            remaining_to_sell -= matched_qty

            if lot_qty > matched_qty + 1e-8:
                # Partial lot remains
                updated_lot = PaperFifoLot(
                    lot_id=lot.lot_id,
                    symbol=lot.symbol,
                    quantity=lot_qty - matched_qty,
                    entry_price=lot.entry_price,
                    timestamp=lot.timestamp,
                    side=lot.side,
                )
                new_lots.append(updated_lot)

        return realized_pnl, new_lots

    def _match_fifo_short(
        self,
        lots: List[PaperFifoLot],
        cover_qty: float,
        cover_price: float,
    ) -> Tuple[float, List[PaperFifoLot]]:
        """Match and consume FIFO lots for SHORT buy/cover orders."""
        realized_pnl = 0.0
        remaining_to_cover = cover_qty
        new_lots: List[PaperFifoLot] = []

        for lot in lots:
            if remaining_to_cover <= 1e-8:
                new_lots.append(lot)
                continue

            lot_qty = lot.quantity
            matched_qty = min(lot_qty, remaining_to_cover)
            # Short P&L: (entry_price - cover_price) * matched_qty
            lot_pnl = matched_qty * (lot.entry_price - cover_price)
            realized_pnl += lot_pnl
            remaining_to_cover -= matched_qty

            if lot_qty > matched_qty + 1e-8:
                updated_lot = PaperFifoLot(
                    lot_id=lot.lot_id,
                    symbol=lot.symbol,
                    quantity=lot_qty - matched_qty,
                    entry_price=lot.entry_price,
                    timestamp=lot.timestamp,
                    side=lot.side,
                )
                new_lots.append(updated_lot)

        return realized_pnl, new_lots

    def apply_dividend(self, dividend: DividendEvent) -> float:
        """
        Process point-in-time cash dividend.
        For LONG positions: credit dividend cash (shares * amount).
        For SHORT positions: debit dividend equivalent (-shares * amount).
        """
        symbol = dividend.symbol
        amount = float(dividend.amount_per_share)
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]
        if abs(pos.quantity) < 1e-8 or amount <= 0:
            return 0.0

        cash_delta = float(pos.quantity * amount)
        self.cash += cash_delta
        return cash_delta

    def apply_split(self, split: SplitEvent) -> None:
        """
        Process point-in-time stock split event.
        Multiplies quantity by split ratio, divides cost basis / entry prices by split ratio.
        Preserves total position market value with zero artificial P&L.
        """
        symbol = split.symbol
        ratio = float(getattr(split, "split_ratio", getattr(split, "ratio", 1.0)))
        if symbol not in self.positions or ratio <= 0:
            return

        pos = self.positions[symbol]
        if abs(pos.quantity) < 1e-8:
            return

        pos.quantity = float(pos.quantity * ratio)
        pos.avg_entry_price = float(pos.avg_entry_price / ratio)
        if pos.market_price > 0:
            pos.market_price = float(pos.market_price / ratio)
            pos.market_value = float(pos.quantity * pos.market_price)

        for lot in pos.fifo_lots:
            lot.quantity = float(lot.quantity * ratio)
            lot.entry_price = float(lot.entry_price / ratio)

    def apply_borrow_cost(self, timestamp: datetime, borrow_rate: float) -> float:
        """
        Apply daily short borrow financing cost on short exposure liability.
        """
        if borrow_rate <= 0:
            return 0.0

        total_fee = 0.0
        for symbol, pos in self.positions.items():
            if pos.quantity < -1e-8 and pos.market_price > 0:
                short_val = abs(pos.quantity * pos.market_price)
                fee = PaperCostModel.calculate_borrow_fee(short_val, borrow_rate)
                total_fee += fee

        if total_fee > 0:
            self.cash -= total_fee
            self.cumulative_borrow_cost += total_fee

        return total_fee
