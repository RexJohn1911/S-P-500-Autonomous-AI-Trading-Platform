"""
Comprehensive Unit, Integration, and Empirical Tests for Phase 14 Paper Trading Engine.
Validates:
- Account initialization and capital accounting
- Double-entry cash and position tracking (LONG, SHORT, FLAT)
- FIFO multi-lot realization on partial/full exits and reversals
- Transaction friction modeling (commission, half-spread, slippage, market impact, borrow fees)
- Corporate action handling (cash dividends, stock splits)
- Order lifecycle state machine and invalid state transition guards
- Execution safety guard (NaN/infinite/negative quantity, invalid prices, cash limits)
- Idempotency of order submissions and execution events
- Ledger reconciliation engine and mismatch detection
- Session lifecycle (create, start, pause, resume, stop, complete)
- Filesystem persistence and recovery roundtrip
- Zero future leakage and causal invariance under future mutations
- Determinism across repeated runs
- Empirical local historical paper trading simulation
"""

import copy
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import pytest
import numpy as np

from backend.app.backtest.schemas import DividendEvent, SplitEvent
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.storage import ProcessedDataStorage
from backend.app.paper_trading.account import PaperAccount
from backend.app.paper_trading.broker import BasePaperBroker, SimulatedPaperBroker
from backend.app.paper_trading.costs import PaperCostModel
from backend.app.paper_trading.execution import PaperExecutionEngine
from backend.app.paper_trading.orders import PaperOrderManager
from backend.app.paper_trading.reconciliation import PaperReconciliationEngine
from backend.app.paper_trading.risk_guard import PaperExecutionRiskGuard
from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperAuditEventType,
    PaperExecution,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
    PaperPositionSide,
    PaperReconciliationStatus,
    PaperSessionStatus,
    PaperTradingConfig,
    PaperTradingResult,
    PaperTradingSession,
)
from backend.app.paper_trading.service import PaperTradingService
from backend.app.paper_trading.storage import PaperTradingStorage
from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.portfolio.service import PortfolioConstructionService
from backend.app.risk.schemas import RiskAdjustedTarget, RiskEngineConfig
from backend.app.strategy.schemas import ReasonCode, SignalCandidate, SignalDirection


def create_synthetic_bars(
    symbol: str,
    prices: list[float],
    start_date: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc),
) -> list[BarData]:
    """Helper to generate clean synthetic daily bars."""
    bars = []
    for i, p in enumerate(prices):
        ts = start_date + timedelta(days=i)
        bars.append(
            BarData(
                symbol=symbol,
                timestamp=ts,
                open=float(p),
                high=float(p * 1.01),
                low=float(p * 0.99),
                close=float(p),
                volume=100000.0,
            )
        )
    return bars


# =========================================================================
# 1. Account Initialization & Accounting Identity Tests
# =========================================================================

def test_account_initialization():
    """A. Test paper account initialization with configurable capital."""
    cfg = PaperTradingConfig(initial_capital=50000.0)
    acc = PaperAccount(account_id="acc_1", session_id="ses_1", config=cfg)
    assert acc.cash == 50000.0
    assert acc.realized_pnl == 0.0
    assert acc.total_transaction_cost == 0.0
    assert len(acc.positions) == 0

    snap = acc.mark_to_market(current_prices={}, timestamp=datetime.now(timezone.utc))
    assert snap.cash == 50000.0
    assert snap.equity == 50000.0
    assert snap.market_value == 0.0
    assert snap.unrealized_pnl == 0.0


def test_accounting_identity():
    """AK. Verify fundamental accounting identity equity = cash + market_value."""
    cfg = PaperTradingConfig(initial_capital=100000.0, commission_rate=0.0, slippage_rate=0.0, bid_ask_spread_rate=0.0)
    broker = SimulatedPaperBroker(session_id="ses_test", config=cfg)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    order = PaperOrder(
        order_id="ord_1",
        client_order_id="cli_1",
        session_id="ses_test",
        symbol="AAPL",
        side=PaperOrderSide.BUY,
        quantity=100.0,
    )
    broker.execute_market_order(order=order, market_price=150.0, timestamp=t0)

    # Price moves to 160
    snap = broker.get_account_snapshot(timestamp=t0, current_prices={"AAPL": 160.0})
    # cash = 100000 - 15000 = 85000
    # market_value = 100 * 160 = 16000
    # equity = 85000 + 16000 = 101000
    assert snap.cash == 85000.0
    assert snap.market_value == 16000.0
    assert snap.equity == 101000.0
    assert snap.equity == snap.cash + snap.market_value
    assert snap.unrealized_pnl == 1000.0


# =========================================================================
# 2. Long Buy, Long Sell, and Partial Exits
# =========================================================================

def test_long_buy_and_sell_with_fifo_realization():
    """C, D, E, I. Test long buy, partial sell, and FIFO realized P&L."""
    cfg = PaperTradingConfig(initial_capital=100000.0, commission_rate=0.001, slippage_rate=0.0, bid_ask_spread_rate=0.0)
    acc = PaperAccount(account_id="acc_1", session_id="ses_1", config=cfg)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 3, tzinfo=timezone.utc)

    # Buy 1: 100 @ 100 ($10,000 + $10 comm)
    exec1 = PaperExecution(
        execution_id="e1", order_id="o1", client_order_id="c1", symbol="AAPL",
        side=PaperOrderSide.BUY, quantity=100.0, executed_price=100.0, base_price=100.0,
        timestamp=t0, commission=10.0,
    )
    acc.apply_execution(exec1)
    assert acc.cash == 89990.0
    pos = acc.get_position("AAPL")
    assert pos.quantity == 100.0
    assert pos.avg_entry_price == 100.0
    assert len(pos.fifo_lots) == 1

    # Buy 2: 100 @ 120 ($12,000 + $12 comm)
    exec2 = PaperExecution(
        execution_id="e2", order_id="o2", client_order_id="c2", symbol="AAPL",
        side=PaperOrderSide.BUY, quantity=100.0, executed_price=120.0, base_price=120.0,
        timestamp=t1, commission=12.0,
    )
    acc.apply_execution(exec2)
    assert acc.cash == 89990.0 - 12012.0
    assert pos.quantity == 200.0
    assert pos.avg_entry_price == 110.0
    assert len(pos.fifo_lots) == 2

    # Partial Sell: 150 @ 130
    # Consumes Lot 1 (100 @ 100 -> P&L = +$3,000) and 50 from Lot 2 (50 @ 120 -> P&L = +$500)
    # Total Realized P&L = $3,500
    # Gross Proceeds = 150 * 130 = $19,500 - $19.50 comm = $19,480.50
    exec3 = PaperExecution(
        execution_id="e3", order_id="o3", client_order_id="c3", symbol="AAPL",
        side=PaperOrderSide.SELL, quantity=150.0, executed_price=130.0, base_price=130.0,
        timestamp=t2, commission=19.50,
    )
    pnl = acc.apply_execution(exec3)
    assert pnl == pytest.approx(3500.0, rel=1e-5)
    assert acc.realized_pnl == pytest.approx(3500.0, rel=1e-5)
    assert pos.quantity == 50.0
    assert len(pos.fifo_lots) == 1
    assert pos.fifo_lots[0].quantity == 50.0
    assert pos.fifo_lots[0].entry_price == 120.0


# =========================================================================
# 3. Short Positions, Short Cover, and Position Reversal
# =========================================================================

def test_short_entry_cover_and_reversal():
    """F, G, H. Test short entry, cover, and full reversal from SHORT to LONG."""
    cfg = PaperTradingConfig(initial_capital=100000.0, allow_short=True, commission_rate=0.0, slippage_rate=0.0, bid_ask_spread_rate=0.0)
    acc = PaperAccount(account_id="acc_1", session_id="ses_1", config=cfg)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 2, tzinfo=timezone.utc)

    # Short 100 @ 200 (Proceeds +$20,000 credited to cash)
    exec1 = PaperExecution(
        execution_id="e1", order_id="o1", client_order_id="c1", symbol="TSLA",
        side=PaperOrderSide.SELL, quantity=100.0, executed_price=200.0, base_price=200.0,
        timestamp=t0,
    )
    acc.apply_execution(exec1)
    assert acc.cash == 120000.0
    pos = acc.get_position("TSLA")
    assert pos.quantity == -100.0
    assert pos.side == PaperPositionSide.SHORT

    # Reversal: BUY 150 @ 180
    # Covers 100 short (P&L = (200 - 180) * 100 = +$2,000, cost to cover = $18,000)
    # Opens 50 LONG @ 180 (cost = $9,000)
    # Final cash = 120000 - 18000 - 9000 = 93000
    exec2 = PaperExecution(
        execution_id="e2", order_id="o2", client_order_id="c2", symbol="TSLA",
        side=PaperOrderSide.BUY, quantity=150.0, executed_price=180.0, base_price=180.0,
        timestamp=t1,
    )
    pnl = acc.apply_execution(exec2)
    assert pnl == pytest.approx(2000.0, rel=1e-5)
    assert acc.cash == 93000.0
    assert pos.quantity == 50.0
    assert pos.side == PaperPositionSide.LONG
    assert pos.avg_entry_price == 180.0


# =========================================================================
# 4. Friction Models: Commission, Spread, Slippage, Market Impact, Borrow
# =========================================================================

def test_friction_cost_calculations():
    """K, L, M, N, O. Test cost models and execution adverse price shifts."""
    # BUY with 5 bps slippage, 4 bps spread (2 bps half-spread), 10 bps impact
    # total friction = 0.0005 + 0.0002 + 0.0010 = 0.0017
    exec_price_buy = PaperCostModel.calculate_executed_price(
        side=PaperOrderSide.BUY,
        base_price=100.0,
        slippage_rate=0.0005,
        spread_rate=0.0004,
        market_impact_rate=0.0010,
    )
    assert exec_price_buy == pytest.approx(100.17, rel=1e-5)

    # SELL with same friction
    exec_price_sell = PaperCostModel.calculate_executed_price(
        side=PaperOrderSide.SELL,
        base_price=100.0,
        slippage_rate=0.0005,
        spread_rate=0.0004,
        market_impact_rate=0.0010,
    )
    assert exec_price_sell == pytest.approx(99.83, rel=1e-5)

    # Spread cost calculation
    spread_cost = PaperCostModel.calculate_spread_cost(quantity=100.0, base_price=100.0, spread_rate=0.0004)
    assert spread_cost == pytest.approx(2.0, rel=1e-5)

    # Slippage cost calculation
    slip_cost = PaperCostModel.calculate_slippage_cost(quantity=100.0, base_price=100.0, executed_price=100.17)
    assert slip_cost == pytest.approx(17.0, rel=1e-5)

    # Borrow fee
    borrow_fee = PaperCostModel.calculate_borrow_fee(short_market_value=50000.0, daily_borrow_rate=0.0001)
    assert borrow_fee == pytest.approx(5.0, rel=1e-5)


# =========================================================================
# 5. Corporate Actions: Cash Dividends & Stock Splits
# =========================================================================

def test_corporate_actions_dividends_and_splits():
    """P, Q. Test dividend credit/debit and stock split adjustments."""
    cfg = PaperTradingConfig(initial_capital=100000.0, allow_short=True, commission_rate=0.0, slippage_rate=0.0, bid_ask_spread_rate=0.0)
    acc = PaperAccount(account_id="acc_1", session_id="ses_1", config=cfg)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # Buy 200 AAPL @ $150
    exec1 = PaperExecution(
        execution_id="e1", order_id="o1", client_order_id="c1", symbol="AAPL",
        side=PaperOrderSide.BUY, quantity=200.0, executed_price=150.0, base_price=150.0,
        timestamp=t0,
    )
    acc.apply_execution(exec1)
    assert acc.cash == 70000.0

    # Apply $0.50 dividend
    div = DividendEvent(timestamp=t0, symbol="AAPL", amount_per_share=0.50)
    cash_delta = acc.apply_dividend(div)
    assert cash_delta == 100.0
    assert acc.cash == 70100.0

    # Apply 2-for-1 split (split_ratio=2.0)
    split = SplitEvent(timestamp=t0, symbol="AAPL", split_ratio=2.0)
    acc.apply_split(split)
    pos = acc.get_position("AAPL")
    assert pos.quantity == 400.0
    assert pos.avg_entry_price == 75.0
    assert pos.fifo_lots[0].quantity == 400.0
    assert pos.fifo_lots[0].entry_price == 75.0


# =========================================================================
# 6. Order Lifecycle & State Machine Guards
# =========================================================================

def test_order_lifecycle_and_invalid_state_transitions():
    """R, S. Test valid transitions and rejection of invalid state transitions."""
    manager = PaperOrderManager(session_id="ses_1")
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    order = PaperOrder(
        order_id="ord_1",
        client_order_id="cli_1",
        session_id="ses_1",
        symbol="AAPL",
        side=PaperOrderSide.BUY,
        quantity=50.0,
        status=PaperOrderStatus.CREATED,
    )
    manager.register_order(order)

    # Valid: CREATED -> SUBMITTED -> ACCEPTED -> FILLED
    manager.transition_order("ord_1", PaperOrderStatus.SUBMITTED, t0)
    assert order.status == PaperOrderStatus.SUBMITTED

    manager.transition_order("ord_1", PaperOrderStatus.ACCEPTED, t0)
    assert order.status == PaperOrderStatus.ACCEPTED

    manager.transition_order("ord_1", PaperOrderStatus.FILLED, t0)
    assert order.status == PaperOrderStatus.FILLED

    # Invalid: FILLED -> ACCEPTED (terminal state violation)
    with pytest.raises(ValueError, match="Invalid order state transition"):
        manager.transition_order("ord_1", PaperOrderStatus.ACCEPTED, t0)


# =========================================================================
# 7. Execution Risk Guard Rejections
# =========================================================================

def test_execution_risk_guard_validations():
    """T, U, V, W, X, Y. Test order rejection under invalid parameters."""
    cfg = PaperTradingConfig(min_order_notional=10.0, allow_short=False)

    # Missing symbol
    o1 = PaperOrder(order_id="1", client_order_id="c1", session_id="s", symbol="", side=PaperOrderSide.BUY, quantity=10.0)
    valid, reason = PaperExecutionRiskGuard.validate_order(o1, current_market_price=100.0, config=cfg)
    assert not valid and "symbol" in reason.lower()

    # NaN quantity
    o2 = PaperOrder(order_id="2", client_order_id="c2", session_id="s", symbol="AAPL", side=PaperOrderSide.BUY, quantity=float("nan"))
    valid, reason = PaperExecutionRiskGuard.validate_order(o2, current_market_price=100.0, config=cfg)
    assert not valid and "nan" in reason.lower()

    # Infinite quantity
    o3 = PaperOrder(order_id="3", client_order_id="c3", session_id="s", symbol="AAPL", side=PaperOrderSide.BUY, quantity=float("inf"))
    valid, reason = PaperExecutionRiskGuard.validate_order(o3, current_market_price=100.0, config=cfg)
    assert not valid and "infinite" in reason.lower()

    # Negative quantity
    o4 = PaperOrder(order_id="4", client_order_id="c4", session_id="s", symbol="AAPL", side=PaperOrderSide.BUY, quantity=-5.0)
    valid, reason = PaperExecutionRiskGuard.validate_order(o4, current_market_price=100.0, config=cfg)
    assert not valid and "must be > 0" in reason.lower()

    # Invalid price (<= 0)
    o5 = PaperOrder(order_id="5", client_order_id="c5", session_id="s", symbol="AAPL", side=PaperOrderSide.BUY, quantity=10.0)
    valid, reason = PaperExecutionRiskGuard.validate_order(o5, current_market_price=-10.0, config=cfg)
    assert not valid and "price must be > 0" in reason.lower()

    # Cash limit rejection for Long Buy
    o6 = PaperOrder(order_id="6", client_order_id="c6", session_id="s", symbol="AAPL", side=PaperOrderSide.BUY, quantity=1000.0)
    valid, reason = PaperExecutionRiskGuard.validate_order(o6, current_market_price=150.0, available_cash=1000.0, config=cfg)
    assert not valid and "insufficient cash" in reason.lower()


# =========================================================================
# 8. Idempotency & Reconciliation Tests
# =========================================================================

def test_idempotent_order_submission():
    """Z. Verify duplicate client_order_id returns existing order without creating duplicate."""
    broker = SimulatedPaperBroker(session_id="ses_idem")
    o1 = PaperOrder(order_id="ord_1", client_order_id="cli_dup_1", session_id="ses_idem", symbol="AAPL", side=PaperOrderSide.BUY, quantity=10.0)
    broker.submit_order(o1)

    o2 = PaperOrder(order_id="ord_2", client_order_id="cli_dup_1", session_id="ses_idem", symbol="AAPL", side=PaperOrderSide.BUY, quantity=20.0)
    ret = broker.submit_order(o2)
    assert ret.order_id == "ord_1"
    assert ret.quantity == 10.0


def test_reconciliation_success_and_mismatch():
    """AB, AC. Test reconciliation detection of matched state and deliberate mismatch."""
    broker = SimulatedPaperBroker(session_id="ses_recon")
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # Matched reconciliation
    report_ok = PaperReconciliationEngine.reconcile(
        session_id="ses_recon",
        broker=broker,
        expected_cash=100000.0,
        expected_positions={},
        current_prices={},
        timestamp=t0,
    )
    assert report_ok.status == PaperReconciliationStatus.MATCHED
    assert len(report_ok.positions_mismatches) == 0

    # Mismatch reconciliation (expected cash mismatch)
    report_bad = PaperReconciliationEngine.reconcile(
        session_id="ses_recon",
        broker=broker,
        expected_cash=50000.0,  # Deliberate mismatch
        expected_positions={},
        current_prices={},
        timestamp=t0,
    )
    assert report_bad.status == PaperReconciliationStatus.MISMATCH


# =========================================================================
# 9. Session Lifecycle, Pause/Resume, Persistence/Recovery
# =========================================================================

def test_session_lifecycle_and_pause_resume(tmp_path: Path):
    """AD, AE, AF. Test session transitions, pause/resume, and filesystem persistence recovery."""
    storage = PaperTradingStorage(base_dir=tmp_path)
    service = PaperTradingService(storage=storage)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 2, tzinfo=timezone.utc)

    service.start_session(symbols=["AAPL"])
    assert service.session.status == PaperSessionStatus.RUNNING

    # Step 1
    bars1 = {"AAPL": BarData(symbol="AAPL", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)}
    service.process_step(timestamp=t0, current_bars=bars1, targets={"AAPL": 0.5})
    assert len(service.account_snapshots) == 1

    # Pause
    service.pause_session()
    assert service.session.status == PaperSessionStatus.PAUSED

    # Resume
    service.resume_session()
    assert service.session.status == PaperSessionStatus.RUNNING

    # Step 2
    bars2 = {"AAPL": BarData(symbol="AAPL", timestamp=t1, open=105.0, high=106.0, low=104.0, close=105.0, volume=1000.0)}
    service.process_step(timestamp=t1, current_bars=bars2, targets={"AAPL": 0.5})
    assert len(service.account_snapshots) == 2

    service.stop_session(status=PaperSessionStatus.COMPLETED)
    res = service.get_result()
    saved_dir = storage.save_result(res)
    assert (saved_dir / "paper_result.json").exists()

    # Recovery reload
    loaded_res = storage.load_result(service.session_id)
    assert loaded_res is not None
    assert loaded_res.session.session_id == service.session_id
    assert len(loaded_res.account_snapshots) == 2
    assert loaded_res.provenance_hash == res.provenance_hash


# =========================================================================
# 10. Causality & Future Mutation Invariance Tests
# =========================================================================

def test_causal_future_price_mutation_invariance():
    """AH, AI, AJ, AG. Verify changing future prices does NOT alter past executions or accounting."""
    bars_orig = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0, 103.0, 108.0])
    bars_mutated = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0, 999.0, 999.0])  # Mutate bars 3 & 4

    def run_sim(bars):
        svc = PaperTradingService()
        for i in range(len(bars)):
            b = bars[i]
            tgt = {"AAPL": 0.6} if i == 0 else None
            svc.process_step(timestamp=b.timestamp, current_bars={"AAPL": b}, targets=tgt)
        return svc.get_result()

    res1 = run_sim(bars_orig)
    res2 = run_sim(bars_mutated)

    # Steps 0, 1, 2 must be IDENTICAL in equity and cash
    for idx in range(3):
        assert res1.account_snapshots[idx].equity == pytest.approx(res2.account_snapshots[idx].equity, rel=1e-6)
        assert res1.account_snapshots[idx].cash == pytest.approx(res2.account_snapshots[idx].cash, rel=1e-6)


# =========================================================================
# 11. Empirical Local Historical Paper Simulation
# =========================================================================

def test_empirical_local_historical_paper_trading_simulation():
    """AO. Run empirical paper trading session on real local historical AAPL market data."""
    storage = ProcessedDataStorage()
    aapl_bars = storage.load_bars("AAPL", TimeFrame.DAY_1, format="parquet")
    assert len(aapl_bars) >= 500

    cfg = PaperTradingConfig(
        initial_capital=100000.0,
        commission_rate=0.0005,
        slippage_rate=0.0002,
        bid_ask_spread_rate=0.0002,
    )
    service = PaperTradingService(config=cfg)

    # Run point-in-time SMA strategy simulation over 523 bars
    for i in range(20, len(aapl_bars)):
        bar = aapl_bars[i]
        sma20 = sum(b.close for b in aapl_bars[i-20:i]) / 20.0
        target = {"AAPL": 0.8} if bar.close > sma20 else {"AAPL": 0.0}
        service.process_step(timestamp=bar.timestamp, current_bars={"AAPL": bar}, targets=target)

    service.stop_session(status=PaperSessionStatus.COMPLETED)
    result = service.get_result()

    assert result.session.status == PaperSessionStatus.COMPLETED
    assert len(result.account_snapshots) == len(aapl_bars) - 20
    assert result.session.total_trades > 0
    assert result.session.total_costs > 0.0
    assert len(result.audit_events) > 100
    assert all(r.status == PaperReconciliationStatus.MATCHED for r in result.reconciliation_reports)


# =========================================================================
# 12. Full Pipeline Integration, Symbol Isolation & Determinism Tests
# =========================================================================

def test_pipeline_integration_signals_portfolio_risk_paper_execution():
    """
    Test canonical integration from Signal Engine -> Portfolio Construction -> Risk Engine -> Paper Trading.
    """
    bars = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0, 103.0, 108.0])
    service = PaperTradingService(config=PaperTradingConfig(initial_capital=100000.0))

    def signal_gen(ts, current_bars):
        # Generate BULLISH signal if AAPL close > 101.0
        bar = current_bars.get("AAPL")
        if bar and bar.close > 101.0:
            return [
                SignalCandidate(
                    timestamp=ts,
                    symbol="AAPL",
                    signal=SignalDirection.LONG,
                    signal_strength=0.8,
                    confidence=0.9,
                    reason_codes=[ReasonCode.BULLISH_ENSEMBLE_SCORE.value],
                )
            ]
        return []

    p_service = PortfolioConstructionService()
    r_config = RiskEngineConfig(max_position_weight=0.20)

    res = service.run_integrated_pipeline_session(
        market_data={"AAPL": bars},
        signal_generator=signal_gen,
        portfolio_service=p_service,
        risk_config=r_config,
    )

    assert res.session.status == PaperSessionStatus.COMPLETED
    assert len(res.account_snapshots) == len(bars)
    assert res.session.total_trades > 0
    # Enforce risk limit: max position weight was 0.20
    for snap in res.account_snapshots:
        if snap.positions_count > 0:
            pos = snap.positions["AAPL"]
            weight = pos.market_value / snap.equity if snap.equity > 0 else 0.0
            assert weight <= 0.21  # 20% limit + minor price movement buffer


def test_symbol_isolation_in_paper_trading():
    """AJ. Test symbol isolation in multi-asset paper trading."""
    bars_aapl = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0])
    bars_msft = create_synthetic_bars("MSFT", [200.0, 204.0, 210.0])

    service = PaperTradingService()
    t0 = bars_aapl[0].timestamp

    # Buy AAPL only
    service.process_step(timestamp=t0, current_bars={"AAPL": bars_aapl[0], "MSFT": bars_msft[0]}, targets={"AAPL": 0.5, "MSFT": 0.0})

    pos_aapl = service.broker.account.get_position("AAPL")
    pos_msft = service.broker.account.get_position("MSFT")

    assert pos_aapl.quantity > 0
    assert pos_msft.quantity == 0.0
    assert len(pos_msft.fifo_lots) == 0


def test_paper_trading_determinism_identical_runs():
    """AG. Test bitwise determinism across two identical paper trading runs."""
    bars = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0, 103.0, 108.0])

    def run_sim():
        svc = PaperTradingService(config=PaperTradingConfig(initial_capital=100000.0), session_id="ses_det_fixed")
        for b in bars:
            svc.process_step(timestamp=b.timestamp, current_bars={"AAPL": b}, targets={"AAPL": 0.5})
        svc.stop_session(status=PaperSessionStatus.COMPLETED)
        return svc.get_result()

    res1 = run_sim()
    res2 = run_sim()

    assert res1.session.final_equity == res2.session.final_equity
    assert res1.session.total_costs == res2.session.total_costs
    assert len(res1.orders) == len(res2.orders)
    assert len(res1.executions) == len(res2.executions)
    for s1, s2 in zip(res1.account_snapshots, res2.account_snapshots):
        assert s1.cash == s2.cash
        assert s1.equity == s2.equity
        assert s1.unrealized_pnl == s2.unrealized_pnl
        assert s1.realized_pnl == s2.realized_pnl


def test_no_real_broker_connection_safety():
    """AN. Safety test proving simulated paper broker has zero live broker bindings or endpoints."""
    broker = SimulatedPaperBroker(session_id="safety_check")
    assert isinstance(broker, BasePaperBroker)
    assert not hasattr(broker, "api_key")
    assert not hasattr(broker, "secret_key")
    assert not hasattr(broker, "base_url")
    assert not hasattr(broker, "live_endpoint")

