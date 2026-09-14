"""
Simulated Portfolio Accounting Subsystem (Phase 13 Upgrade v1.2).
Maintains cash balance, open positions, mark-to-market valuations, realized and unrealized P&L,
FIFO lot accounting, corporate action adjustments (dividends, splits), short borrow fees,
and granular friction attribution with mathematical precision.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import uuid

from backend.app.backtest.schemas import (
    CostBreakdown,
    DividendEvent,
    ExecutionRecord,
    ExposurePoint,
    OrderSide,
    PortfolioState,
    PositionSide,
    PositionState,
    SplitEvent,
    TradeRecord,
)


class SimulatedPortfolio:
    """
    Simulated portfolio ledger tracking cash, holdings, and P&L through time.
    """

    def __init__(self, initial_capital: float = 100_000.0, base_currency: str = "USD"):
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {initial_capital}")

        self.initial_capital = float(initial_capital)
        self.base_currency = base_currency
        self.cash = float(initial_capital)
        self.positions: Dict[str, PositionState] = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_spread_cost = 0.0
        self.total_borrow_cost = 0.0
        self.total_market_impact = 0.0
        self.dividends_collected = 0.0
        self.splits_applied_count = 0
        self.completed_trades: List[TradeRecord] = []

        # Internal tracking for completed trade generation: symbol -> list of open lots
        # Each lot: {"entry_time": dt, "entry_price": float, "quantity": float, "side": PositionSide, "entry_bar_idx": int}
        self._open_lots: Dict[str, List[Dict]] = {}
        self._current_bar_idx = 0

    @property
    def equity(self) -> float:
        """Total portfolio equity = cash + sum(market_value of open positions)."""
        mv_sum = sum(pos.market_value for pos in self.positions.values())
        return float(self.cash + mv_sum)

    @property
    def current_equity(self) -> float:
        """Alias for equity."""
        return self.equity

    @property
    def gross_notional(self) -> float:
        """Total absolute dollar market value of open positions."""
        return float(sum(abs(pos.market_value) for pos in self.positions.values()))

    @property
    def net_notional(self) -> float:
        """Total signed dollar market value of open positions."""
        return float(sum(pos.market_value for pos in self.positions.values()))

    @property
    def long_notional(self) -> float:
        """Total dollar value of long positions."""
        return float(sum(pos.market_value for pos in self.positions.values() if pos.side == PositionSide.LONG))

    @property
    def short_notional(self) -> float:
        """Total absolute liability dollar value of short positions."""
        return float(sum(abs(pos.market_value) for pos in self.positions.values() if pos.side == PositionSide.SHORT))

    @property
    def gross_exposure(self) -> float:
        eq = self.equity
        return float(self.gross_notional / eq) if eq > 0 else 0.0

    @property
    def net_exposure(self) -> float:
        eq = self.equity
        return float(self.net_notional / eq) if eq > 0 else 0.0

    @property
    def long_exposure(self) -> float:
        eq = self.equity
        return float(self.long_notional / eq) if eq > 0 else 0.0

    @property
    def short_exposure(self) -> float:
        eq = self.equity
        return float(self.short_notional / eq) if eq > 0 else 0.0

    @property
    def max_position_weight(self) -> float:
        eq = self.equity
        if eq <= 0 or not self.positions:
            return 0.0
        return float(max(abs(pos.market_value) / eq for pos in self.positions.values()))

    @property
    def trades(self) -> List[TradeRecord]:
        """Alias for completed_trades."""
        return self.completed_trades

    def set_bar_index(self, bar_idx: int) -> None:
        """Update current bar index for trade holding period tracking."""
        self._current_bar_idx = bar_idx

    def update_market_prices(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """
        Mark all open positions to market with current point-in-time prices.
        """
        total_mv = 0.0
        tot_unrealized = 0.0

        for sym, pos in list(self.positions.items()):
            if sym in prices and prices[sym] > 0:
                curr_px = float(prices[sym])
                pos.market_price = curr_px
                pos.last_updated = timestamp

                if pos.side == PositionSide.LONG:
                    pos.market_value = pos.quantity * curr_px
                    pos.unrealized_pnl = (curr_px - pos.avg_entry_price) * pos.quantity
                elif pos.side == PositionSide.SHORT:
                    # Short position market value = liability (- quantity * price)
                    pos.market_value = - (pos.quantity * curr_px)
                    pos.unrealized_pnl = (pos.avg_entry_price - curr_px) * pos.quantity

                total_mv += pos.market_value
                tot_unrealized += pos.unrealized_pnl

        # Update position weights relative to current total equity
        current_eq = self.cash + total_mv
        self.unrealized_pnl = tot_unrealized

        if current_eq > 0:
            for pos in self.positions.values():
                pos.weight = pos.market_value / current_eq
        else:
            for pos in self.positions.values():
                pos.weight = 0.0

    def apply_dividend(self, dividend: DividendEvent) -> float:
        """
        Process a cash dividend event.
        For LONG positions: credits (shares * amount_per_share) to cash.
        For SHORT positions: debits (shares * amount_per_share) from cash (borrower pays dividend).
        Returns the net cash flow.
        """
        sym = dividend.symbol
        if sym not in self.positions or dividend.amount_per_share <= 0:
            return 0.0

        pos = self.positions[sym]
        if pos.side == PositionSide.LONG:
            cash_credit = float(pos.quantity * dividend.amount_per_share)
            self.cash += cash_credit
            self.dividends_collected += cash_credit
            return cash_credit
        elif pos.side == PositionSide.SHORT:
            cash_debit = float(pos.quantity * dividend.amount_per_share)
            self.cash -= cash_debit
            return -cash_debit
        return 0.0

    def apply_stock_split(self, split: SplitEvent) -> None:
        """
        Process a stock split event without creating artificial P&L.
        Adjusts quantity (qty * ratio) and entry price (price / ratio).
        """
        sym = split.symbol
        if sym not in self.positions or split.split_ratio <= 0 or split.split_ratio == 1.0:
            return

        ratio = float(split.split_ratio)
        pos = self.positions[sym]
        pos.quantity *= ratio
        pos.avg_entry_price /= ratio
        pos.market_price /= ratio
        # Market value remains unchanged!

        # Update open lots
        lots = self._open_lots.get(sym, [])
        for lot in lots:
            lot["quantity"] *= ratio
            lot["entry_price"] /= ratio

        self.splits_applied_count += 1

    def total_equity(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Helper to get current total equity, optionally updating market prices first."""
        if current_prices:
            self.update_market_prices(current_prices, datetime.now(timezone.utc))
        return float(self.equity)

    def apply_borrow_fees(
        self,
        daily_borrow_rate: float = 0.0,
        prices: Optional[Dict[str, float]] = None,
        cost_model: Optional[Any] = None,
        days: float = 1.0,
    ) -> float:
        """
        Deduct daily borrow financing cost for active short positions.
        """
        if prices:
            self.update_market_prices(prices, datetime.now(timezone.utc))
        if cost_model and hasattr(cost_model, "annual_borrow_rate"):
            daily_borrow_rate = cost_model.annual_borrow_rate / 365.0
        elif cost_model and hasattr(cost_model, "daily_borrow_rate"):
            daily_borrow_rate = cost_model.daily_borrow_rate

        if daily_borrow_rate <= 0:
            return 0.0

        total_fee = 0.0
        for pos in self.positions.values():
            if pos.side == PositionSide.SHORT:
                short_val = abs(pos.market_value)
                fee = float(short_val * daily_borrow_rate * days)
                total_fee += fee

        if total_fee > 0:
            self.cash -= total_fee
            self.total_borrow_cost += total_fee

        return total_fee

    def apply_execution(
        self,
        symbol_or_record: Optional[Union[ExecutionRecord, str]] = None,
        side: Optional[OrderSide] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0,
        slippage_cost: float = 0.0,
        spread_cost: float = 0.0,
        borrow_cost: float = 0.0,
        market_impact_cost: float = 0.0,
        symbol: Optional[str] = None,
    ) -> None:
        """Alias for apply_fill, supporting both ExecutionRecord and explicit parameters."""
        if isinstance(symbol_or_record, ExecutionRecord):
            rec = symbol_or_record
            self.apply_fill(
                symbol=rec.symbol,
                side=rec.side,
                quantity=rec.quantity,
                executed_price=rec.executed_price,
                commission=rec.commission,
                slippage_cost=rec.slippage_cost,
                spread_cost=rec.spread_cost,
                borrow_cost=rec.borrow_cost,
                market_impact_cost=rec.market_impact_cost,
                timestamp=rec.timestamp,
            )
        else:
            sym = symbol or (str(symbol_or_record) if symbol_or_record is not None else "")
            self.apply_fill(
                symbol=sym,
                side=side or OrderSide.BUY,
                quantity=float(quantity or 0.0),
                executed_price=float(price or 0.0),
                commission=commission,
                slippage_cost=slippage_cost,
                spread_cost=spread_cost,
                borrow_cost=borrow_cost,
                market_impact_cost=market_impact_cost,
                timestamp=timestamp or datetime.now(timezone.utc),
            )

    def apply_fill(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        executed_price: float,
        commission: float,
        slippage_cost: float,
        timestamp: datetime,
        spread_cost: float = 0.0,
        borrow_cost: float = 0.0,
        market_impact_cost: float = 0.0,
    ) -> None:
        """
        Apply a simulated trade execution fill to cash, positions, and trade records.
        """
        if quantity <= 0:
            return

        traded_notional = quantity * executed_price
        self.total_commission += commission
        self.total_slippage += slippage_cost
        self.total_spread_cost += spread_cost
        self.total_borrow_cost += borrow_cost
        self.total_market_impact += market_impact_cost

        if symbol not in self._open_lots:
            self._open_lots[symbol] = []

        if side == OrderSide.BUY:
            # Cash outflow
            self.cash -= (traded_notional + commission)

            # Check if closing an existing short position
            if symbol in self.positions and self.positions[symbol].side == PositionSide.SHORT:
                pos = self.positions[symbol]
                close_qty = min(quantity, pos.quantity)
                remain_buy_qty = quantity - close_qty

                # Record completed trade and retrieve exact FIFO gross P&L
                fifo_gross_pnl = self._record_closed_trade(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    close_qty=close_qty,
                    exit_price=executed_price,
                    exit_time=timestamp,
                    commission=commission,
                    slippage=slippage_cost,
                )

                realized_on_close = fifo_gross_pnl - commission
                self.realized_pnl += realized_on_close
                pos.realized_pnl += realized_on_close

                if close_qty == pos.quantity:
                    del self.positions[symbol]
                else:
                    pos.quantity -= close_qty
                    remaining_lots = self._open_lots.get(symbol, [])
                    if remaining_lots and pos.quantity > 0:
                        pos.avg_entry_price = sum(l["entry_price"] * l["quantity"] for l in remaining_lots) / pos.quantity
                    pos.market_value = - (pos.quantity * executed_price)

                # If there are remaining shares from the buy, open a LONG position
                if remain_buy_qty > 0:
                    self._open_long_position(symbol, remain_buy_qty, executed_price, timestamp)

            # Existing LONG or new LONG
            elif symbol in self.positions and self.positions[symbol].side == PositionSide.LONG:
                pos = self.positions[symbol]
                new_qty = pos.quantity + quantity
                new_avg_price = (pos.avg_entry_price * pos.quantity + executed_price * quantity) / new_qty
                pos.quantity = new_qty
                pos.avg_entry_price = new_avg_price
                pos.market_price = executed_price
                pos.market_value = new_qty * executed_price
                pos.last_updated = timestamp
                self._open_lots[symbol].append({
                    "entry_time": timestamp,
                    "entry_price": executed_price,
                    "quantity": quantity,
                    "side": PositionSide.LONG,
                    "entry_bar_idx": self._current_bar_idx,
                })
            else:
                self._open_long_position(symbol, quantity, executed_price, timestamp)

        elif side == OrderSide.SELL:
            # Cash inflow
            self.cash += (traded_notional - commission)

            # Check if closing an existing LONG position
            if symbol in self.positions and self.positions[symbol].side == PositionSide.LONG:
                pos = self.positions[symbol]
                close_qty = min(quantity, pos.quantity)
                remain_sell_qty = quantity - close_qty

                # Record completed trade and retrieve exact FIFO gross P&L
                fifo_gross_pnl = self._record_closed_trade(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    close_qty=close_qty,
                    exit_price=executed_price,
                    exit_time=timestamp,
                    commission=commission,
                    slippage=slippage_cost,
                )

                realized_on_close = fifo_gross_pnl - commission
                self.realized_pnl += realized_on_close
                pos.realized_pnl += realized_on_close

                if close_qty == pos.quantity:
                    del self.positions[symbol]
                else:
                    pos.quantity -= close_qty
                    remaining_lots = self._open_lots.get(symbol, [])
                    if remaining_lots and pos.quantity > 0:
                        pos.avg_entry_price = sum(l["entry_price"] * l["quantity"] for l in remaining_lots) / pos.quantity
                    pos.market_value = pos.quantity * executed_price

                if remain_sell_qty > 0:
                    self._open_short_position(symbol, remain_sell_qty, executed_price, timestamp)

            # Existing SHORT or new SHORT
            elif symbol in self.positions and self.positions[symbol].side == PositionSide.SHORT:
                pos = self.positions[symbol]
                new_qty = pos.quantity + quantity
                new_avg_price = (pos.avg_entry_price * pos.quantity + executed_price * quantity) / new_qty
                pos.quantity = new_qty
                pos.avg_entry_price = new_avg_price
                pos.market_price = executed_price
                pos.market_value = - (new_qty * executed_price)
                pos.last_updated = timestamp
                self._open_lots[symbol].append({
                    "entry_time": timestamp,
                    "entry_price": executed_price,
                    "quantity": quantity,
                    "side": PositionSide.SHORT,
                    "entry_bar_idx": self._current_bar_idx,
                })
            else:
                self._open_short_position(symbol, quantity, executed_price, timestamp)

    def _open_long_position(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> None:
        self.positions[symbol] = PositionState(
            symbol=symbol,
            side=PositionSide.LONG,
            quantity=quantity,
            avg_entry_price=price,
            market_price=price,
            market_value=quantity * price,
            weight=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            last_updated=timestamp,
        )
        self._open_lots[symbol].append({
            "entry_time": timestamp,
            "entry_price": price,
            "quantity": quantity,
            "side": PositionSide.LONG,
            "entry_bar_idx": self._current_bar_idx,
        })

    def _open_short_position(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> None:
        self.positions[symbol] = PositionState(
            symbol=symbol,
            side=PositionSide.SHORT,
            quantity=quantity,
            avg_entry_price=price,
            market_price=price,
            market_value=- (quantity * price),
            weight=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            last_updated=timestamp,
        )
        self._open_lots[symbol].append({
            "entry_time": timestamp,
            "entry_price": price,
            "quantity": quantity,
            "side": PositionSide.SHORT,
            "entry_bar_idx": self._current_bar_idx,
        })

    def _record_closed_trade(
        self,
        symbol: str,
        side: PositionSide,
        close_qty: float,
        exit_price: float,
        exit_time: datetime,
        commission: float,
        slippage: float,
    ) -> float:
        """FIFO lot matching to create an accurate TradeRecord and return gross realized P&L."""
        lots = self._open_lots.get(symbol, [])
        matched_qty = 0.0
        weighted_entry_px = 0.0
        earliest_entry_time = exit_time
        holding_bars = 0

        while lots and matched_qty < close_qty - 1e-7:
            lot = lots[0]
            needed = close_qty - matched_qty
            take_qty = min(needed, lot["quantity"])

            weighted_entry_px += lot["entry_price"] * take_qty
            matched_qty += take_qty
            earliest_entry_time = min(earliest_entry_time, lot["entry_time"])
            holding_bars = max(holding_bars, self._current_bar_idx - lot.get("entry_bar_idx", self._current_bar_idx))

            lot["quantity"] -= take_qty
            if lot["quantity"] <= 1e-7:
                lots.pop(0)

        if matched_qty > 0:
            avg_entry_px = weighted_entry_px / matched_qty
            if side == PositionSide.LONG:
                gross_pnl = (exit_price - avg_entry_px) * matched_qty
                ret_pct = (exit_price / avg_entry_px) - 1.0 if avg_entry_px > 0 else 0.0
            else:
                gross_pnl = (avg_entry_px - exit_price) * matched_qty
                ret_pct = 1.0 - (exit_price / avg_entry_px) if avg_entry_px > 0 else 0.0

            net_pnl = gross_pnl - commission - slippage

            self.completed_trades.append(
                TradeRecord(
                    trade_id=f"trade_{symbol}_{exit_time.strftime('%Y%m%d%H%M%S')}_{len(self.completed_trades) + 1}",
                    symbol=symbol,
                    side=side,
                    entry_time=earliest_entry_time,
                    exit_time=exit_time,
                    entry_price=avg_entry_px,
                    exit_price=exit_price,
                    quantity=matched_qty,
                    gross_pnl=gross_pnl,
                    commission=commission,
                    slippage=slippage,
                    net_pnl=net_pnl,
                    return_pct=ret_pct,
                    holding_bars=holding_bars,
                )
            )
            return float(gross_pnl)
        return 0.0

    def get_state(self, timestamp: datetime, turnover: float = 0.0) -> PortfolioState:
        """Generate a complete Point-in-Time PortfolioState snapshot."""
        gross_notional = sum(abs(pos.market_value) for pos in self.positions.values())
        net_notional = sum(pos.market_value for pos in self.positions.values())
        eq = self.equity
        gross_exp = (gross_notional / eq) if eq > 0 else 0.0
        net_exp = (net_notional / eq) if eq > 0 else 0.0

        return PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            equity=eq,
            gross_notional=gross_notional,
            net_notional=net_notional,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            positions={k: PositionState.from_dict(v.to_dict()) for k, v in self.positions.items()},
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            cumulative_pnl=self.realized_pnl + self.unrealized_pnl,
            total_commission=self.total_commission,
            total_slippage=self.total_slippage,
            total_spread_cost=self.total_spread_cost,
            total_borrow_cost=self.total_borrow_cost,
            total_market_impact=self.total_market_impact,
            turnover=turnover,
        )

    def get_exposure_point(self, timestamp: datetime) -> ExposurePoint:
        """Generate point-in-time ExposurePoint diagnostics."""
        eq = self.equity
        gross_exp = (self.gross_notional / eq) if eq > 0 else 0.0
        net_exp = (self.net_notional / eq) if eq > 0 else 0.0
        long_exp = (self.long_notional / eq) if eq > 0 else 0.0
        short_exp = (self.short_notional / eq) if eq > 0 else 0.0
        max_pos_w = self.max_position_weight

        return ExposurePoint(
            timestamp=timestamp,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            long_exposure=long_exp,
            short_exposure=short_exp,
            cash=self.cash,
            leverage=gross_exp,
            max_position_weight=max_pos_w,
        )

    def get_cost_breakdown(self) -> CostBreakdown:
        """Generate comprehensive cost attribution breakdown."""
        return CostBreakdown(
            commission_cost=self.total_commission,
            slippage_cost=self.total_slippage,
            spread_cost=self.total_spread_cost,
            borrow_cost=self.total_borrow_cost,
            market_impact_cost=self.total_market_impact,
        )
