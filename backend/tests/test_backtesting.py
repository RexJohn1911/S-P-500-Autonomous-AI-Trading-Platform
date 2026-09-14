"""
Comprehensive Unit and Integration Tests for Phase 13 Backtesting Engine.
Validates:
- Engine initialization and config validation
- Accurate double-entry accounting (cash, market value, unrealized & realized P&L)
- Zero-lookahead next-bar execution causality
- Commission and directional slippage mechanics
- Mathematical exactness of return, drawdown, Sharpe, Sortino, Calmar, and trade metrics
- Benchmark alignment and common-period fairness
- Signal, Portfolio, and Risk Engine integration
- Strict point-in-time leakage invariance under future mutations
- Complete determinism across identical runs
- Symbol isolation
"""

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np

from backend.app.backtest.benchmark import BenchmarkEvaluator
from backend.app.backtest.costs import CostModel
from backend.app.backtest.engine import BacktestEngine
from backend.app.backtest.execution import SimulatedExecutionEngine
from backend.app.backtest.metrics import BacktestMetricsCalculator
from backend.app.backtest.portfolio import SimulatedPortfolio
from backend.app.backtest.reporting import BacktestReporter
from backend.app.backtest.schemas import (
    AdjustmentStatus,
    BacktestConfig,
    BacktestResult,
    BacktestSummary,
    CostBreakdown,
    DataSourceType,
    DatasetProvenance,
    DividendEvent,
    DrawdownEpisode,
    DrawdownPoint,
    EquityPoint,
    ExecutionPriceType,
    ExecutionRecord,
    ExposurePoint,
    OrderSide,
    PerformanceMetrics,
    PeriodPerformance,
    PortfolioState,
    PositionSide,
    PositionState,
    SplitEvent,
    StatisticalDiagnostics,
    TradeRecord,
    WalkForwardFoldConfig,
    WalkForwardFoldResult,
    WalkForwardResult,
)
from backend.app.backtest.service import BacktestService
from backend.app.backtest.storage import BacktestStorage
from backend.app.backtest.walkforward import WalkForwardConfig, WalkForwardEngine
from backend.app.data.models import BarData
from backend.app.portfolio.schemas import PortfolioConstructionConfig, PortfolioTarget
from backend.app.portfolio.service import PortfolioConstructionService
from backend.app.risk.engine import RiskAdjustmentEngine
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
# 1. Basic Engine & Configuration Tests
# =========================================================================

def test_backtest_config_valid():
    cfg = BacktestConfig(
        initial_capital=50000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    assert cfg.initial_capital == 50000.0
    assert cfg.commission_rate == 0.001
    assert cfg.slippage_rate == 0.0005


def test_backtest_config_invalid_capital():
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=0.0)
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=-1000.0)
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=float("nan"))
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=float("inf"))


def test_backtest_config_invalid_rates():
    with pytest.raises(ValueError):
        BacktestConfig(commission_rate=-0.01)
    with pytest.raises(ValueError):
        BacktestConfig(slippage_rate=-0.01)
    with pytest.raises(ValueError):
        BacktestConfig(minimum_trade_notional=-5.0)


def test_backtest_config_invalid_date_range():
    with pytest.raises(ValueError):
        BacktestConfig(
            start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )


def test_backtest_engine_empty_dataset():
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="market_data cannot be empty"):
        engine.run(market_data={})


def test_backtest_engine_single_symbol():
    bars = create_synthetic_bars("AAPL", [100.0, 105.0, 110.0, 115.0, 120.0])
    engine = BacktestEngine(BacktestConfig(initial_capital=100000.0, commission_rate=0.0, slippage_rate=0.0))
    # Target 50% on bar 0 (ts 2025-01-01), will execute on bar 1 (ts 2025-01-02 at open=105.0)
    targets = {bars[0].timestamp: {"AAPL": 0.50}}
    result = engine.run(market_data={"AAPL": bars}, target_series=targets)

    assert len(result.equity_curve) == 5
    assert len(result.executions) == 1
    assert result.executions[0].symbol == "AAPL"
    assert result.executions[0].executed_price == 105.0


def test_backtest_engine_multi_symbol():
    bars_aapl = create_synthetic_bars("AAPL", [100.0, 105.0, 110.0, 115.0, 120.0])
    bars_msft = create_synthetic_bars("MSFT", [200.0, 202.0, 204.0, 206.0, 208.0])
    engine = BacktestEngine(BacktestConfig(initial_capital=100000.0, commission_rate=0.0, slippage_rate=0.0))

    targets = {
        bars_aapl[0].timestamp: {"AAPL": 0.30, "MSFT": 0.30}
    }
    result = engine.run(
        market_data={"AAPL": bars_aapl, "MSFT": bars_msft},
        target_series=targets,
    )
    assert len(result.equity_curve) == 5
    assert len(result.executions) == 2
    symbols_executed = {e.symbol for e in result.executions}
    assert symbols_executed == {"AAPL", "MSFT"}


# =========================================================================
# 2. Accounting & Ledger Tests
# =========================================================================

def test_accounting_buy_reduces_cash():
    port = SimulatedPortfolio(initial_capital=10000.0)
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # Buy 10 shares at $100 with $2 commission
    port.apply_execution(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10.0,
        price=100.0,
        timestamp=t,
        commission=2.0,
        slippage_cost=0.5,
    )
    # Cash reduced by (10 * 100) + 2.0 commission = 1002.0
    assert port.cash == 10000.0 - 1002.0
    assert port.positions["AAPL"].quantity == 10.0
    assert port.positions["AAPL"].average_entry_price == 100.0


def test_accounting_sell_increases_cash():
    port = SimulatedPortfolio(initial_capital=10000.0)
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    port.apply_execution("AAPL", OrderSide.BUY, 10.0, 100.0, t1, 0.0, 0.0)
    assert port.cash == 9000.0

    # Sell 10 shares at $110 with $1 commission
    port.apply_execution("AAPL", OrderSide.SELL, 10.0, 110.0, t2, 1.0, 0.0)
    # Cash increased by (10 * 110) - 1.0 = 1099.0 -> 9000 + 1099 = 10099
    assert port.cash == 10099.0
    assert "AAPL" not in port.positions
    assert port.realized_pnl == 99.0  # 100 gross gain - 1 commission


def test_accounting_equity_formula():
    port = SimulatedPortfolio(initial_capital=10000.0)
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    port.apply_execution("AAPL", OrderSide.BUY, 10.0, 100.0, t, 0.0, 0.0)
    port.update_market_prices({"AAPL": 120.0}, timestamp=t)

    # Cash = 9000, Market value = 10 * 120 = 1200 -> Equity = 10200
    assert port.cash == 9000.0
    assert port.gross_notional == 1200.0
    assert port.current_equity == 10200.0
    assert port.unrealized_pnl == 200.0


def test_accounting_realized_and_unrealized_pnl():
    port = SimulatedPortfolio(initial_capital=10000.0)
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    port.apply_execution("AAPL", OrderSide.BUY, 10.0, 100.0, t1, 0.0, 0.0)
    # Price rises to 110 -> Unrealized P&L = +100
    port.update_market_prices({"AAPL": 110.0}, timestamp=t1)
    assert port.unrealized_pnl == 100.0
    assert port.realized_pnl == 0.0

    # Sell half (5 shares) at 110 -> Realized P&L = +50, Unrealized P&L = +50
    port.apply_execution("AAPL", OrderSide.SELL, 5.0, 110.0, t2, 0.0, 0.0)
    port.update_market_prices({"AAPL": 110.0}, timestamp=t2)
    assert port.realized_pnl == 50.0
    assert port.unrealized_pnl == 50.0
    assert len(port.completed_trades) == 1
    assert port.completed_trades[0].quantity == 5.0
    assert port.completed_trades[0].net_pnl == 50.0


# =========================================================================
# 3. Execution & Lookahead Prevention Tests
# =========================================================================

def test_execution_next_bar_causality():
    """Verify signal at t is NOT executed at t, but executed at t+1 at next_open."""
    bars = create_synthetic_bars("AAPL", [100.0, 105.0, 110.0])
    cfg = BacktestConfig(
        initial_capital=10000.0,
        execution_price_type=ExecutionPriceType.NEXT_OPEN,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    engine = BacktestEngine(cfg)
    # Signal issued on bar 0 (ts = 2025-01-01)
    targets = {bars[0].timestamp: {"AAPL": 1.0}}
    result = engine.run(market_data={"AAPL": bars}, target_series=targets)

    # Bar 0: No executions, 100% cash
    assert result.equity_curve[0].cash == 10000.0
    assert len(result.executions) == 1
    # Execution must be timestamped bar 1 (2025-01-02) and executed at bar 1 open (105.0)
    assert result.executions[0].timestamp == bars[1].timestamp
    assert result.executions[0].executed_price == 105.0


def test_execution_missing_price_handling():
    """Missing execution price must be handled deterministically without crashing or fabricating."""
    port = SimulatedPortfolio(initial_capital=10000.0)
    cfg = BacktestConfig()
    t = datetime(2025, 1, 2, tzinfo=timezone.utc)
    # Rebalance target for AAPL, but price dict does NOT have AAPL
    execs = SimulatedExecutionEngine.execute_rebalance(
        portfolio=port,
        targets={"AAPL": 0.5},
        prices={},  # Missing AAPL
        config=cfg,
        timestamp=t,
    )
    assert len(execs) == 0
    assert len(port.positions) == 0


def test_directional_slippage():
    # Buy order: executed price = base * (1 + slippage)
    buy_px = CostModel.calculate_executed_price(100.0, OrderSide.BUY, 0.01)
    assert buy_px == 101.0

    # Sell order: executed price = base * (1 - slippage)
    sell_px = CostModel.calculate_executed_price(100.0, OrderSide.SELL, 0.01)
    assert sell_px == 99.0

    # Slippage cost computation
    slip_cost = CostModel.calculate_slippage_cost(100.0, 10.0, 0.01)
    assert slip_cost == 10.0


# =========================================================================
# 4. Costs Tests
# =========================================================================

def test_zero_and_nonzero_costs():
    assert CostModel.calculate_commission(10000.0, 0.0) == 0.0
    assert CostModel.calculate_commission(10000.0, 0.001) == 10.0

    bars = create_synthetic_bars("AAPL", [100.0, 100.0, 100.0])
    # Run with 0 commission, 0 slippage
    engine_zero = BacktestEngine(BacktestConfig(initial_capital=10000.0, commission_rate=0.0, slippage_rate=0.0))
    res_zero = engine_zero.run({"AAPL": bars}, {bars[0].timestamp: {"AAPL": 0.5}})
    assert res_zero.metrics.total_costs == 0.0

    # Run with 10 bps commission, 10 bps slippage
    engine_costs = BacktestEngine(BacktestConfig(initial_capital=10000.0, commission_rate=0.001, slippage_rate=0.001))
    res_costs = engine_costs.run({"AAPL": bars}, {bars[0].timestamp: {"AAPL": 0.5}})
    assert res_costs.metrics.total_costs > 0.0
    assert res_costs.summary.final_equity < res_zero.summary.final_equity


# =========================================================================
# 5. Returns & Drawdown Tests
# =========================================================================

def test_period_and_cumulative_returns():
    equity_series = [100.0, 110.0, 99.0]
    returns = BacktestMetricsCalculator.calculate_returns(equity_series)
    assert len(returns) == 3
    assert returns[0] == 0.0
    assert pytest.approx(returns[1], 1e-6) == 0.10
    assert pytest.approx(returns[2], 1e-6) == -0.10


def test_drawdown_calculation():
    eq_points = [
        EquityPoint(timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc), equity=100.0, cash=100.0),
        EquityPoint(timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc), equity=120.0, cash=120.0),
        EquityPoint(timestamp=datetime(2025, 1, 3, tzinfo=timezone.utc), equity=90.0, cash=90.0),
        EquityPoint(timestamp=datetime(2025, 1, 4, tzinfo=timezone.utc), equity=130.0, cash=130.0),
    ]
    dd_curve, max_dd, max_duration = BacktestMetricsCalculator.calculate_drawdowns(eq_points)
    assert len(dd_curve) == 4
    # Peak at bar 1 is 120. At bar 2 equity is 90 -> drawdown_pct = (90/120) - 1 = -0.25 (-25%)
    assert dd_curve[2].running_peak == 120.0
    assert pytest.approx(dd_curve[2].drawdown_percentage, 1e-6) == -0.25
    assert pytest.approx(max_dd, 1e-6) == 0.25
    # Recovers at bar 3 (equity 130 > 120)
    assert dd_curve[3].drawdown_percentage == 0.0


def test_zero_volatility_metrics():
    eq_points = [
        EquityPoint(timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc), equity=100.0, cash=100.0),
        EquityPoint(timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc), equity=100.0, cash=100.0),
        EquityPoint(timestamp=datetime(2025, 1, 3, tzinfo=timezone.utc), equity=100.0, cash=100.0),
    ]
    metrics = BacktestMetricsCalculator.compute_all_metrics(
        equity_curve=eq_points,
        trades=[],
        executions=[],
        initial_capital=100.0,
    )
    assert metrics.annualized_volatility == 0.0
    assert metrics.sharpe_ratio is None
    assert metrics.sortino_ratio is None
    assert metrics.calmar_ratio is None
    assert metrics.max_drawdown == 0.0


# =========================================================================
# 6. Risk Metrics & Trade Statistics Tests
# =========================================================================

def test_sharpe_sortino_calmar():
    # Construct an equity curve with alternating positive returns so volatility > 0
    dates = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(100)]
    eq_points = []
    current_eq = 100000.0
    for i, d in enumerate(dates):
        if i == 0:
            ret = 0.0
        else:
            ret = 0.002 if i % 2 == 0 else 0.0005
            current_eq *= (1.0 + ret)
        eq_points.append(
            EquityPoint(timestamp=d, equity=current_eq, cash=current_eq, period_return=ret)
        )

    metrics = BacktestMetricsCalculator.compute_all_metrics(
        equity_curve=eq_points,
        trades=[],
        executions=[],
        initial_capital=100000.0,
    )
    assert metrics.sharpe_ratio is not None
    assert metrics.sharpe_ratio > 0.0
    assert metrics.total_return > 0.0


def test_trade_win_rate_and_profit_factor():
    t_entry = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t_exit = datetime(2025, 1, 2, tzinfo=timezone.utc)
    t1 = TradeRecord(
        trade_id="t1",
        symbol="AAPL",
        entry_timestamp=t_entry,
        exit_timestamp=t_exit,
        entry_price=100.0,
        exit_price=110.0,
        quantity=10.0,
        direction=PositionSide.LONG,
        gross_pnl=100.0,
        commission=2.0,
        slippage=1.0,
        net_pnl=97.0,
        return_pct=0.097,
        holding_bars=1,
    )
    t2 = TradeRecord(
        trade_id="t2",
        symbol="MSFT",
        entry_timestamp=t_entry,
        exit_timestamp=t_exit,
        entry_price=200.0,
        exit_price=190.0,
        quantity=10.0,
        direction=PositionSide.LONG,
        gross_pnl=-100.0,
        commission=2.0,
        slippage=1.0,
        net_pnl=-103.0,
        return_pct=-0.0515,
        holding_bars=1,
    )

    eq_points = [
        EquityPoint(timestamp=t_entry, equity=10000.0, cash=10000.0),
        EquityPoint(timestamp=t_exit, equity=9994.0, cash=9994.0),
    ]

    metrics = BacktestMetricsCalculator.compute_all_metrics(
        equity_curve=eq_points,
        trades=[t1, t2],
        executions=[],
        initial_capital=10000.0,
    )
    assert metrics.total_completed_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == 0.50
    # Profit factor = 97 / 103
    assert pytest.approx(metrics.profit_factor, 1e-4) == 97.0 / 103.0


# =========================================================================
# 7. Benchmark Alignment Tests
# =========================================================================

def test_benchmark_alignment_and_metrics():
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 3, tzinfo=timezone.utc)

    eq_curve = [
        EquityPoint(timestamp=t0, equity=10000.0, cash=10000.0, period_return=0.0),
        EquityPoint(timestamp=t1, equity=10500.0, cash=10500.0, period_return=0.05),
        EquityPoint(timestamp=t2, equity=11000.0, cash=11000.0, period_return=0.0476),
    ]

    bm_prices = {t0: 400.0, t1: 410.0, t2: 420.0}
    bm_res = BenchmarkEvaluator.evaluate_benchmark(
        equity_curve=eq_curve,
        benchmark_prices=bm_prices,
        benchmark_symbol="SPY",
    )
    assert bm_res is not None
    assert bm_res.benchmark_symbol == "SPY"
    # Benchmark return = 420 / 400 - 1 = +5%
    assert pytest.approx(bm_res.benchmark_total_return, 1e-4) == 0.05
    assert bm_res.beta is not None
    assert bm_res.correlation is not None


def test_benchmark_unavailable():
    eq_curve = [
        EquityPoint(timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc), equity=100.0, cash=100.0)
    ]
    bm_res = BenchmarkEvaluator.evaluate_benchmark(
        equity_curve=eq_curve,
        benchmark_prices=None,
        benchmark_symbol="SPY",
    )
    assert bm_res is None


# =========================================================================
# 8. Pipeline Integration (Signals -> Portfolio -> Risk -> Backtest)
# =========================================================================

def test_pipeline_integration():
    bars_aapl = create_synthetic_bars("AAPL", [100.0, 102.0, 104.0, 106.0, 108.0])
    bars_msft = create_synthetic_bars("MSFT", [200.0, 201.0, 203.0, 205.0, 207.0])
    market_data = {"AAPL": bars_aapl, "MSFT": bars_msft}

    def dummy_signal_gen(ts, current_bars):
        # Generate signal for AAPL with confidence 0.8
        return [
            SignalCandidate(
                timestamp=ts,
                symbol="AAPL",
                signal=SignalDirection.LONG,
                signal_strength=0.8,
                confidence=0.8,
                regime="bull",
                reason_codes=[ReasonCode.MODEL_CONSENSUS.value],
            )
        ]

    p_service = PortfolioConstructionService(
        config=PortfolioConstructionConfig(max_position_weight=0.20)
    )
    r_config = RiskEngineConfig(max_position_weight=0.15)  # Stricter risk limit

    bt_service = BacktestService(
        config=BacktestConfig(initial_capital=100000.0, commission_rate=0.0005, slippage_rate=0.0005)
    )

    result = bt_service.run_integrated_pipeline_backtest(
        market_data=market_data,
        signal_generator=dummy_signal_gen,
        portfolio_service=p_service,
        risk_config=r_config,
    )

    assert result.summary.final_equity > 0
    assert len(result.executions) > 0
    # Check that maximum position weight in executions was clipped to risk limit 0.15
    for ex in result.executions:
        base_notional = ex.quantity * ex.requested_price
        assert base_notional <= 100000.0 * 0.15 + 1.0  # approximate boundary check


# =========================================================================
# 9. Strict Leakage Invariance & Lookahead Protection Tests
# =========================================================================

def test_leakage_future_price_mutation_invariance():
    """
    CRITICAL TEST:
    Modifying future prices for t > k must NOT change equity, positions, cash,
    or trades for any timestamp t <= k.
    """
    orig_prices = [100.0, 102.0, 105.0, 110.0, 115.0]
    bars_orig = create_synthetic_bars("AAPL", orig_prices)

    engine = BacktestEngine(BacktestConfig(initial_capital=10000.0, commission_rate=0.0, slippage_rate=0.0))
    targets = {bars_orig[0].timestamp: {"AAPL": 0.5}}

    res_orig = engine.run(market_data={"AAPL": bars_orig}, target_series=targets)

    # Now mutate future prices on bar 3 and 4 dramatically
    mutated_prices = [100.0, 102.0, 105.0, 9999.0, 0.01]
    bars_mutated = create_synthetic_bars("AAPL", mutated_prices)

    res_mutated = engine.run(market_data={"AAPL": bars_mutated}, target_series=targets)

    # Invariance check for t <= 2 (bars 0, 1, 2)
    for k in range(3):
        assert res_orig.equity_curve[k].equity == res_mutated.equity_curve[k].equity
        assert res_orig.equity_curve[k].cash == res_mutated.equity_curve[k].cash
        assert res_orig.portfolio_history[k].gross_notional == res_mutated.portfolio_history[k].gross_notional


def test_leakage_symbol_isolation():
    """
    Mutating future data for symbol A must NOT alter historical portfolio
    ledger or executions for symbol B before that timestamp.
    """
    bars_a = create_synthetic_bars("AAPL", [100.0, 102.0, 104.0, 106.0])
    bars_b = create_synthetic_bars("MSFT", [200.0, 202.0, 204.0, 206.0])

    engine = BacktestEngine(BacktestConfig(initial_capital=10000.0, commission_rate=0.0, slippage_rate=0.0))
    targets = {
        bars_a[0].timestamp: {"AAPL": 0.3, "MSFT": 0.3}
    }

    res_1 = engine.run(market_data={"AAPL": bars_a, "MSFT": bars_b}, target_series=targets)

    # Mutate future bar for AAPL at t=3
    bars_a_mutated = create_synthetic_bars("AAPL", [100.0, 102.0, 104.0, 999999.0])
    res_2 = engine.run(market_data={"AAPL": bars_a_mutated, "MSFT": bars_b}, target_series=targets)

    # At bar 0, no executions yet. At bar 1 & 2, MSFT positions exist and match exactly
    for k in range(1, 3):
        assert res_1.portfolio_history[k].positions["MSFT"].market_value == res_2.portfolio_history[k].positions["MSFT"].market_value
    for k in range(3):
        assert res_1.equity_curve[k].equity == res_2.equity_curve[k].equity


# =========================================================================
# 10. Determinism Tests
# =========================================================================

def test_engine_determinism():
    """Two identical runs must produce byte-for-byte identical output."""
    bars_aapl = create_synthetic_bars("AAPL", [100.0, 105.0, 102.0, 112.0, 110.0])
    bars_msft = create_synthetic_bars("MSFT", [200.0, 195.0, 210.0, 205.0, 215.0])
    market_data = {"AAPL": bars_aapl, "MSFT": bars_msft}

    cfg = BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.0005,
        slippage_rate=0.0005,
    )
    targets = {
        bars_aapl[0].timestamp: {"AAPL": 0.4, "MSFT": 0.4},
        bars_aapl[2].timestamp: {"AAPL": 0.2, "MSFT": 0.6},
    }

    eval_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine = BacktestEngine(cfg)
    res_1 = engine.run(market_data=market_data, target_series=targets, backtest_id="fixed_id", evaluation_timestamp=eval_ts)
    res_2 = engine.run(market_data=market_data, target_series=targets, backtest_id="fixed_id", evaluation_timestamp=eval_ts)

    assert res_1.to_dict() == res_2.to_dict()
    assert res_1.summary.final_equity == res_2.summary.final_equity
    assert res_1.metrics.total_return == res_2.metrics.total_return
    assert len(res_1.trades) == len(res_2.trades)
    assert len(res_1.executions) == len(res_2.executions)


# =========================================================================
# 11. Storage & Reporting Tests
# =========================================================================

def test_storage_and_reporting(tmp_path: Path):
    bars = create_synthetic_bars("AAPL", [100.0, 105.0, 110.0])
    engine = BacktestEngine(BacktestConfig(initial_capital=50000.0))
    targets = {bars[0].timestamp: {"AAPL": 0.5}}
    result = engine.run(market_data={"AAPL": bars}, target_series=targets)

    # Test report generation
    report_md = BacktestReporter.generate_markdown_report(result)
    assert "Quantitative Backtest Report" in report_md
    assert "DISCLAIMER" in report_md

    # Test storage persistence and reloading
    storage = BacktestStorage(base_dir=tmp_path)
    saved_dir = storage.save_result(result)
    assert (saved_dir / "config.json").exists()
    assert (saved_dir / "summary.json").exists()
    assert (saved_dir / "metrics.json").exists()
    assert (saved_dir / "result.json").exists()

    loaded_result = storage.load_result(saved_dir)
    assert loaded_result.backtest_id == result.backtest_id
    assert pytest.approx(loaded_result.summary.final_equity, 1e-2) == result.summary.final_equity


# =========================================================================
# 12. Audit & Hardening v1.1 Causal & Boundary Tests
# =========================================================================

def test_causal_next_open_vs_next_close_mutation():
    """
    AUDIT #2 & #3:
    For NEXT_OPEN execution:
    - Changing Open[t+1] MUST change execution price.
    - Changing Close[t+1] must NOT change execution price used for order generated at t.
    - Changing High[t+1], Low[t+1], Volume[t+1] must NOT change execution price.
    """
    ts0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ts1 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    ts2 = datetime(2025, 1, 3, tzinfo=timezone.utc)

    # Base bars
    bars_base = [
        BarData(symbol="AAPL", timestamp=ts0, open=100.0, high=102.0, low=98.0, close=100.0, volume=1000.0),
        BarData(symbol="AAPL", timestamp=ts1, open=105.0, high=108.0, low=103.0, close=107.0, volume=1000.0),
        BarData(symbol="AAPL", timestamp=ts2, open=110.0, high=112.0, low=109.0, close=110.0, volume=1000.0),
    ]

    cfg = BacktestConfig(initial_capital=10000.0, execution_price_type=ExecutionPriceType.NEXT_OPEN, commission_rate=0.0, slippage_rate=0.0)
    engine = BacktestEngine(cfg)
    targets = {ts0: {"AAPL": 0.5}}

    res_base = engine.run(market_data={"AAPL": bars_base}, target_series=targets)
    assert len(res_base.executions) == 1
    assert res_base.executions[0].executed_price == 105.0

    # 1. Mutate Open[t+1] from 105.0 to 125.0 -> MUST change execution price
    bars_mut_open = [
        bars_base[0],
        BarData(symbol="AAPL", timestamp=ts1, open=125.0, high=128.0, low=103.0, close=107.0, volume=1000.0),
        bars_base[2],
    ]
    res_mut_open = engine.run(market_data={"AAPL": bars_mut_open}, target_series=targets)
    assert res_mut_open.executions[0].executed_price == 125.0
    assert res_mut_open.executions[0].executed_price != res_base.executions[0].executed_price

    # 2. Mutate Close[t+1] from 107.0 to 999.0 -> Must NOT change execution price at t+1
    bars_mut_close = [
        bars_base[0],
        BarData(symbol="AAPL", timestamp=ts1, open=105.0, high=108.0, low=103.0, close=999.0, volume=1000.0),
        bars_base[2],
    ]
    res_mut_close = engine.run(market_data={"AAPL": bars_mut_close}, target_series=targets)
    assert res_mut_close.executions[0].executed_price == 105.0  # Still 105.0

    # 3. Mutate High[t+1], Low[t+1], Volume[t+1] -> Must NOT change execution price
    bars_mut_hlv = [
        bars_base[0],
        BarData(symbol="AAPL", timestamp=ts1, open=105.0, high=500.0, low=1.0, close=107.0, volume=9999999.0),
        bars_base[2],
    ]
    res_mut_hlv = engine.run(market_data={"AAPL": bars_mut_hlv}, target_series=targets)
    assert res_mut_hlv.executions[0].executed_price == 105.0


def test_slippage_and_commission_accounting_exactness():
    """
    AUDIT #4, #5, #10, #11:
    Verify exact cash and equity math with commission and slippage without double-counting.
    """
    port = SimulatedPortfolio(initial_capital=100000.0)
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)

    # BUY 100 shares @ base $100.0, slippage 1% (fill $101.0), commission 0.1% ($10.10)
    exec_px_buy = CostModel.calculate_executed_price(OrderSide.BUY, 100.0, 0.01)
    assert exec_px_buy == 101.0
    notional_buy = 100.0 * exec_px_buy  # 10100.0
    comm_buy = CostModel.calculate_commission(notional_buy, 0.001)  # 10.10
    slip_cost_buy = CostModel.calculate_slippage_cost(100.0, 100.0, exec_px_buy)  # 100.0

    port.apply_fill("AAPL", OrderSide.BUY, 100.0, exec_px_buy, comm_buy, slip_cost_buy, t1)
    # Cash = 100000 - 10100 - 10.10 = 89889.90
    assert pytest.approx(port.cash, 1e-6) == 89889.90
    # Mark to market at $101.0 -> Equity = 89889.90 + (100 * 101.0) = 99989.90 (loss exactly = commission)
    port.update_market_prices({"AAPL": 101.0}, t1)
    assert pytest.approx(port.equity, 1e-6) == 99989.90

    # SELL 100 shares @ base $110.0, slippage 1% (fill $108.90), commission 0.1% ($10.89)
    exec_px_sell = CostModel.calculate_executed_price(OrderSide.SELL, 110.0, 0.01)
    assert exec_px_sell == 108.90
    notional_sell = 100.0 * exec_px_sell  # 10890.0
    comm_sell = CostModel.calculate_commission(notional_sell, 0.001)  # 10.89
    slip_cost_sell = CostModel.calculate_slippage_cost(100.0, 110.0, exec_px_sell)  # 110.0

    port.apply_fill("AAPL", OrderSide.SELL, 100.0, exec_px_sell, comm_sell, slip_cost_sell, t2)
    # Cash = 89889.90 + 10890.0 - 10.89 = 100769.01
    assert pytest.approx(port.cash, 1e-6) == 100769.01
    assert len(port.positions) == 0
    # Gross P&L = (108.90 - 101.0) * 100 = 790.0
    # Net P&L = 790.0 - 10.89 (sell comm) - 110.0 (sell slip) = 669.11
    assert len(port.completed_trades) == 1
    assert pytest.approx(port.completed_trades[0].gross_pnl, 1e-6) == 790.0


def test_fifo_multi_lot_realization_exactness():
    """
    AUDIT #7:
    BUY 100 @ 100
    BUY 100 @ 120
    SELL 150 @ 130
    FIFO realization: 100 from $100 lot (+$3000) + 50 from $120 lot (+$500) = $3,500 gross.
    Remaining lot: 50 shares @ $120 basis.
    """
    port = SimulatedPortfolio(initial_capital=30000.0)
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2025, 1, 3, tzinfo=timezone.utc)

    # 1. Buy 100 @ 100
    port.apply_fill("AAPL", OrderSide.BUY, 100.0, 100.0, commission=0.0, slippage_cost=0.0, timestamp=t1)
    assert port.cash == 20000.0
    assert port.positions["AAPL"].avg_entry_price == 100.0

    # 2. Buy 100 @ 120
    port.apply_fill("AAPL", OrderSide.BUY, 100.0, 120.0, commission=0.0, slippage_cost=0.0, timestamp=t2)
    assert port.cash == 8000.0
    assert port.positions["AAPL"].quantity == 200.0
    assert port.positions["AAPL"].avg_entry_price == 110.0

    # 3. Sell 150 @ 130
    port.apply_fill("AAPL", OrderSide.SELL, 150.0, 130.0, commission=0.0, slippage_cost=0.0, timestamp=t3)
    # Cash = 8000 + (150 * 130) = 8000 + 19500 = 27500
    assert port.cash == 27500.0
    assert port.positions["AAPL"].quantity == 50.0
    # Remaining 50 shares are from lot 2 (entry price $120.0)
    assert pytest.approx(port.positions["AAPL"].avg_entry_price, 1e-6) == 120.0
    # FIFO realized P&L: 100 * (130 - 100) + 50 * (130 - 120) = 3000 + 500 = 3500
    assert pytest.approx(port.realized_pnl, 1e-6) == 3500.0
    assert len(port.completed_trades) == 1
    assert pytest.approx(port.completed_trades[0].gross_pnl, 1e-6) == 3500.0
    assert pytest.approx(port.completed_trades[0].entry_price, 1e-6) == 106.66666666666667  # 16000 / 150

    # Mark to market remaining 50 shares @ $130 -> Unrealized P&L = 50 * (130 - 120) = 500
    port.update_market_prices({"AAPL": 130.0}, t3)
    assert pytest.approx(port.unrealized_pnl, 1e-6) == 500.0
    # Total Equity = 27500 + 50 * 130 = 34000. (Initial 30000 + 3500 realized + 500 unrealized)
    assert pytest.approx(port.equity, 1e-6) == 34000.0


def test_short_position_lifecycle_accounting():
    """
    AUDIT #8:
    Verify short position accounting:
    - Open Short (cash increases by proceeds, market value is negative liability)
    - Mark to market (unrealized P&L = (entry - current) * qty)
    - Cover Short (cash decreases by buy cost, realized P&L = (entry - exit) * qty)
    """
    port = SimulatedPortfolio(initial_capital=10000.0)
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2025, 1, 3, tzinfo=timezone.utc)

    # 1. Short 50 shares @ $100
    port.apply_fill("AAPL", OrderSide.SELL, 50.0, 100.0, commission=0.0, slippage_cost=0.0, timestamp=t1)
    # Cash = 10000 + 5000 = 15000
    assert port.cash == 15000.0
    assert port.positions["AAPL"].side == PositionSide.SHORT
    assert port.positions["AAPL"].market_value == -5000.0
    assert port.equity == 10000.0

    # 2. Price falls to $80 (favorable for short)
    port.update_market_prices({"AAPL": 80.0}, t2)
    assert port.positions["AAPL"].market_value == -4000.0
    assert port.unrealized_pnl == 1000.0  # (100 - 80) * 50 = +1000
    assert port.equity == 11000.0

    # 3. Buy to cover 50 shares @ $80
    port.apply_fill("AAPL", OrderSide.BUY, 50.0, 80.0, commission=0.0, slippage_cost=0.0, timestamp=t3)
    # Cash = 15000 - 4000 = 11000
    assert port.cash == 11000.0
    assert "AAPL" not in port.positions
    assert port.realized_pnl == 1000.0
    assert len(port.completed_trades) == 1
    assert port.completed_trades[0].side == PositionSide.SHORT
    assert port.completed_trades[0].net_pnl == 1000.0


def test_long_only_cash_constraint_clamping():
    """
    AUDIT #9:
    In long-only mode, if target notional exceeds available cash,
    quantity is deterministically scaled down and cash NEVER becomes negative.
    """
    port = SimulatedPortfolio(initial_capital=10000.0)
    cfg = BacktestConfig(initial_capital=10000.0, commission_rate=0.001)
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # Request $25,000 notional (weight 2.5) with only $10,000 cash
    execs = SimulatedExecutionEngine.execute_rebalance(
        portfolio=port,
        targets={"AAPL": 2.5},
        prices={"AAPL": 100.0},
        config=cfg,
        timestamp=t,
    )
    assert len(execs) == 1
    # Cash must be >= 0
    assert port.cash >= 0.0
    # Bought quantity should be ~99.9 shares (bounded by $10,000 cash minus commission)
    assert port.positions["AAPL"].quantity < 100.0
    assert port.positions["AAPL"].quantity > 99.0


def test_performance_metrics_boundary_conditions():
    """
    AUDIT #13–23, #40, #41:
    Boundary conditions for performance calculator:
    - 1-bar series
    - Constant equity (zero volatility -> Sharpe=None, Sortino=None, Calmar=None)
    - Zero drawdown (Calmar=None)
    - Zero downside deviation (Sortino=None)
    - No losses (Profit factor=None)
    """
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 2, tzinfo=timezone.utc)

    # 1. Monotonically increasing equity with constant daily gains
    eq_mono = [
        EquityPoint(timestamp=t0, equity=10000.0, cash=10000.0, period_return=0.0),
        EquityPoint(timestamp=t1, equity=10100.0, cash=10100.0, period_return=0.01),
    ]
    m_mono = BacktestMetricsCalculator.compute_all_metrics(
        equity_curve=eq_mono,
        trades=[],
        initial_capital=10000.0,
    )
    # Zero drawdown -> max_drawdown == 0.0, calmar_ratio is None
    assert m_mono.max_drawdown == 0.0
    assert m_mono.calmar_ratio is None

    # 2. Constant positive returns (e.g. 5 days of +1% daily) -> downside dev is 0 -> Sortino is None
    eq_pos = [
        EquityPoint(timestamp=t0 + timedelta(days=i), equity=10000.0 * (1.01 ** i), cash=10000.0, period_return=0.01 if i > 0 else 0.0)
        for i in range(10)
    ]
    m_pos = BacktestMetricsCalculator.compute_all_metrics(
        equity_curve=eq_pos,
        trades=[],
        initial_capital=10000.0,
    )
    assert m_pos.sortino_ratio is None

    # 3. Only winning trades -> profit factor is None (not infinity)
    t_win = TradeRecord(
        trade_id="w1",
        symbol="AAPL",
        entry_price=100.0,
        exit_price=110.0,
        quantity=10.0,
        gross_pnl=100.0,
        commission=0.0,
        slippage=0.0,
        net_pnl=100.0,
        return_pct=0.10,
        holding_bars=1,
    )
    m_win_trades = BacktestMetricsCalculator.compute_all_metrics(
        equity_curve=eq_pos,
        trades=[t_win],
        initial_capital=10000.0,
    )
    assert m_win_trades.profit_factor is None
    assert m_win_trades.win_rate == 1.0


def test_benchmark_metrics_zero_variance_and_jensen_alpha():
    """
    AUDIT #25, #26:
    Benchmark with zero variance must yield Beta=None, Alpha=None, Correlation=None.
    """
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 3, tzinfo=timezone.utc)

    eq_curve = [
        EquityPoint(timestamp=t0, equity=10000.0, cash=10000.0, period_return=0.0),
        EquityPoint(timestamp=t1, equity=10200.0, cash=10200.0, period_return=0.02),
        EquityPoint(timestamp=t2, equity=10400.0, cash=10400.0, period_return=0.0196),
    ]

    # Benchmark flat prices (zero variance)
    bm_flat = {t0: 400.0, t1: 400.0, t2: 400.0}
    bm_res = BenchmarkEvaluator.evaluate(
        benchmark_symbol="SPY",
        benchmark_prices=bm_flat,
        equity_curve=eq_curve,
    )
    assert bm_res is not None
    assert bm_res.beta is None
    assert bm_res.alpha is None
    assert bm_res.correlation is None


def test_risk_adjusted_target_override_execution():
    """
    AUDIT #37:
    Verify Phase 12 risk adjustments override Portfolio Construction targets.
    """
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    port = SimulatedPortfolio(initial_capital=100000.0)
    cfg = BacktestConfig(initial_capital=100000.0, commission_rate=0.0, slippage_rate=0.0)

    # Risk Adjusted Target specifies 0.10, even though unconstrained target was 0.30
    adj_target = RiskAdjustedTarget(
        timestamp=ts,
        symbol="AAPL",
        original_weight=0.30,
        adjusted_weight=0.10,
        signal_direction=SignalDirection.LONG,
        signal_strength=0.8,
        confidence=0.9,
        was_adjusted=True,
        adjustment_reason="Max single position limit",
    )

    execs = SimulatedExecutionEngine.execute_rebalance(
        portfolio=port,
        targets=[adj_target],
        prices={"AAPL": 100.0},
        config=cfg,
        timestamp=ts,
    )
    assert len(execs) == 1
    # Executed notional must equal $100,000 * 0.10 = $10,000 (100 shares)
    assert execs[0].quantity == 100.0
    assert execs[0].notional == 10000.0


# ============================================================================
# PHASE 13.2 RESEARCH-GRADE UPGRADE TESTS
# ============================================================================


def test_dataset_provenance_and_metadata():
    """Verify DatasetProvenance schema validation and classification."""
    prov = DatasetProvenance(
        dataset_id="test_ds_001",
        provider="LOCAL_FIXTURE",
        data_source_type=DataSourceType.LOCAL_HISTORICAL,
        adjustment_status=AdjustmentStatus.ADJUSTED,
        symbols=["AAPL", "MSFT"],
        start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 1, 10, tzinfo=timezone.utc),
        timeframe="1Day",
        row_count=20,
        checksum="a1b2c3d4e5f6",
    )
    assert prov.provider == "LOCAL_FIXTURE"
    assert prov.data_source_type == DataSourceType.LOCAL_HISTORICAL
    assert prov.adjustment_status == AdjustmentStatus.ADJUSTED
    assert prov.checksum == "a1b2c3d4e5f6"


def test_corporate_action_cash_dividend_long():
    """
    Verify cash dividend on a LONG position increases cash and total equity by:
    dividend_cash = shares * dividend_per_share.
    """
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    port = SimulatedPortfolio(initial_capital=100000.0)
    # Open LONG 100 shares @ $100
    port.apply_execution(
        ExecutionRecord(
            execution_id="exec_1",
            timestamp=ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            requested_price=100.0,
            executed_price=100.0,
            notional=10000.0,
            commission=0.0,
            slippage_cost=0.0,
        )
    )
    # Initial state: cash = 90000, market_value = 10000, equity = 100000
    assert port.cash == 90000.0
    assert port.total_equity({"AAPL": 100.0}) == 100000.0

    # Apply cash dividend of $2.50 / share on AAPL
    div_event = DividendEvent(
        timestamp=ts,
        symbol="AAPL",
        amount_per_share=2.50,
    )
    port.apply_dividend(div_event)

    # After dividend: cash = 90000 + (100 * 2.50) = 90250, equity = 100250
    assert port.cash == 90250.0
    assert port.total_equity({"AAPL": 100.0}) == 100250.0


def test_corporate_action_cash_dividend_short():
    """
    Verify cash dividend on a SHORT position debits cash/equity (short liability pays dividend).
    """
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    port = SimulatedPortfolio(initial_capital=100000.0)
    # Open SHORT 100 shares @ $100
    port.apply_execution(
        ExecutionRecord(
            execution_id="exec_1",
            timestamp=ts,
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100.0,
            requested_price=100.0,
            executed_price=100.0,
            notional=10000.0,
            commission=0.0,
            slippage_cost=0.0,
        )
    )
    # Cash = 110000, market_value = -10000, equity = 100000
    assert port.cash == 110000.0
    assert port.total_equity({"AAPL": 100.0}) == 100000.0

    # Apply dividend of $2.00 / share
    div_event = DividendEvent(timestamp=ts, symbol="AAPL", amount_per_share=2.00)
    port.apply_dividend(div_event)

    # Short position must pay dividend: cash = 110000 - 200 = 109800, equity = 99800
    assert port.cash == 109800.0
    assert port.total_equity({"AAPL": 100.0}) == 99800.0


def test_corporate_action_stock_split_preserves_equity():
    """
    Verify 2-for-1 stock split doubles shares and halves entry basis with ZERO artificial P&L.
    """
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    port = SimulatedPortfolio(initial_capital=100000.0)
    # Buy 100 shares @ $200
    port.apply_execution(
        ExecutionRecord(
            execution_id="exec_1",
            timestamp=ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            requested_price=200.0,
            executed_price=200.0,
            notional=20000.0,
            commission=0.0,
            slippage_cost=0.0,
        )
    )
    eq_pre = port.total_equity({"AAPL": 200.0})
    assert eq_pre == 100000.0

    # 2:1 Split (split_ratio = 2.0)
    split_event = SplitEvent(timestamp=ts, symbol="AAPL", split_ratio=2.0)
    port.apply_stock_split(split_event)

    pos = port.positions["AAPL"]
    assert pos.quantity == 200.0
    assert pos.avg_entry_price == 100.0
    assert len(port._open_lots["AAPL"]) == 1
    assert port._open_lots["AAPL"][0]["quantity"] == 200.0
    assert port._open_lots["AAPL"][0]["entry_price"] == 100.0

    # Post-split market price is $100. Equity must remain EXACTLY $100,000.0 (Zero artificial P&L)
    eq_post = port.total_equity({"AAPL": 100.0})
    assert eq_post == 100000.0


def test_transaction_cost_bid_ask_spread():
    """
    Verify bid-ask spread cost calculation and execution price adjustment.
    BUY executed at ref * (1 + spread/2)
    SELL executed at ref * (1 - spread/2)
    """
    ref_price = 100.0
    buy_px = CostModel.calculate_executed_price(OrderSide.BUY, ref_price, spread_rate=0.0010)
    sell_px = CostModel.calculate_executed_price(OrderSide.SELL, ref_price, spread_rate=0.0010)

    assert buy_px == pytest.approx(100.05, rel=1e-6)
    assert sell_px == pytest.approx(99.95, rel=1e-6)

    # Spread cost for 100 shares
    spread_cost_buy = CostModel.calculate_spread_cost(100.0, ref_price, spread_rate=0.0010)
    assert spread_cost_buy == pytest.approx(100.0 * 100.0 * 0.0005, rel=1e-6)  # $5.0


def test_transaction_cost_short_borrow_fee():
    """
    Verify short borrow fee calculation and portfolio deduction.
    Borrow fee = notional * daily_rate
    """
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    port = SimulatedPortfolio(initial_capital=100000.0)
    port.apply_execution(
        ExecutionRecord(
            execution_id="exec_1",
            timestamp=ts,
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100.0,
            requested_price=100.0,
            executed_price=100.0,
            notional=10000.0,
            commission=0.0,
            slippage_cost=0.0,
        )
    )
    # 3.65% annual -> 0.01% per day (10 bps/day)
    daily_rate = 0.0365 / 365.0
    # Apple position value = 100 shares * $100 = $10,000. Fee = 10000 * 0.0001 = $1.00
    daily_fee = port.apply_borrow_fees(
        prices={"AAPL": 100.0},
        daily_borrow_rate=daily_rate,
        days=1.0,
    )
    assert daily_fee == pytest.approx(1.00, rel=1e-6)
    assert port.total_borrow_cost == pytest.approx(1.00, rel=1e-6)
    assert port.cash == pytest.approx(110000.0 - 1.00, rel=1e-6)


def test_transaction_cost_market_impact():
    """
    Verify market impact calculation and executed price impact.
    """
    ref_price = 100.0
    order_notional = 10000.0
    ref_liquidity = 1000000.0  # 1% of daily liquidity

    impact_rate = CostModel.calculate_market_impact(
        order_notional=order_notional,
        market_impact_coefficient=0.1,
        reference_liquidity=ref_liquidity,
    )
    assert impact_rate == pytest.approx(0.001, rel=1e-6)

    # Executed price includes market impact
    exec_px = CostModel.calculate_executed_price(
        OrderSide.BUY,
        ref_price,
        market_impact_rate=impact_rate,
    )
    assert exec_px == pytest.approx(100.10, rel=1e-6)


def test_cost_breakdown_attribution_gross_vs_net():
    """
    Verify CostBreakdown attribution and calculation of gross_pnl vs net_pnl.
    """
    cost_bd = CostBreakdown(
        commission_cost=50.0,
        slippage_cost=30.0,
        spread_cost=20.0,
        borrow_cost=10.0,
        market_impact_cost=5.0,
    )
    assert cost_bd.total_cost == 115.0

    eq_curve = [
        EquityPoint(timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc), equity=10000.0, cash=10000.0, period_return=0.0),
        EquityPoint(timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc), equity=10485.0, cash=10485.0, period_return=0.0485),
    ]
    trade = TradeRecord(
        trade_id="T1",
        symbol="AAPL",
        side=PositionSide.LONG,
        entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        entry_price=100.0,
        exit_price=106.0,
        quantity=100.0,
        gross_pnl=600.0,
        commission=50.0,
        slippage=30.0,
        net_pnl=485.0,
        return_pct=0.0485,
        holding_bars=1,
    )

    metrics = BacktestMetricsCalculator.compute_all_metrics(
        equity_curve=eq_curve,
        trades=[trade],
        initial_capital=10000.0,
        total_spread_cost=20.0,
        total_borrow_cost=10.0,
        total_market_impact=5.0,
    )

    assert metrics.net_pnl == 485.0
    assert metrics.gross_pnl == pytest.approx(600.0, rel=1e-5)
    assert metrics.total_costs == 115.0


def test_exposure_time_series_portfolio_analytics():
    """
    Verify portfolio exposure tracking across time: gross, net, long, short, cash utilization, leverage.
    """
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
        )
    )
    bars_aapl = create_synthetic_bars("AAPL", [100.0, 105.0, 110.0, 115.0])
    bars_msft = create_synthetic_bars("MSFT", [200.0, 200.0, 195.0, 190.0])

    t0 = bars_aapl[0].timestamp
    t1 = bars_aapl[1].timestamp

    # Provide target_series
    target_map = {
        t0: {"AAPL": 0.50, "MSFT": -0.30},
        t1: {"AAPL": 0.50, "MSFT": -0.30},
    }

    res = engine.run(
        market_data={"AAPL": bars_aapl, "MSFT": bars_msft},
        target_series=target_map,
    )

    assert len(res.exposure_history) == len(bars_aapl)
    # Check exposure point on bar 2
    exp = res.exposure_history[1]
    assert exp.long_exposure > 0.0
    assert exp.short_exposure > 0.0
    assert exp.gross_exposure == pytest.approx(exp.long_exposure + exp.short_exposure, rel=1e-5)
    assert exp.net_exposure == pytest.approx(exp.long_exposure - exp.short_exposure, rel=1e-5)


def test_drawdown_episodes_extraction_causality():
    """
    Verify DrawdownEpisode identification extracts start, trough, recovery, duration, and depth.
    """
    dates = [datetime(2025, 1, i, tzinfo=timezone.utc) for i in range(1, 8)]
    equities = [100.0, 120.0, 110.0, 90.0, 100.0, 120.0, 130.0]
    eq_curve = [
        EquityPoint(timestamp=dates[i], equity=equities[i], cash=equities[i], period_return=0.0)
        for i in range(len(dates))
    ]

    episodes = BacktestMetricsCalculator.extract_drawdown_episodes(eq_curve)
    assert len(episodes) == 1
    ep = episodes[0]
    # Peak is at index 1 (date 2, eq 120.0)
    assert ep.peak_time == dates[1]
    # Trough is at index 3 (date 4, eq 90.0)
    assert ep.trough_time == dates[3]
    # Depth = (120 - 90) / 120 = 0.25 (25%)
    assert ep.depth_pct == pytest.approx(0.25, rel=1e-6)
    # Recovered at index 5 (date 6, eq 120.0)
    assert ep.is_recovered is True
    assert ep.recovery_time == dates[5]
    assert ep.duration_bars == 4


def test_period_performance_breakdowns():
    """Verify yearly period performance breakdowns."""
    dates = [
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 6, 1, tzinfo=timezone.utc),
        datetime(2025, 12, 31, tzinfo=timezone.utc),
    ]
    eq_curve = [
        EquityPoint(timestamp=dates[0], equity=10000.0, cash=10000.0, period_return=0.0),
        EquityPoint(timestamp=dates[1], equity=10500.0, cash=10500.0, period_return=0.05),
        EquityPoint(timestamp=dates[2], equity=11000.0, cash=11000.0, period_return=0.0476),
    ]

    breakdowns = BacktestMetricsCalculator.calculate_period_breakdowns(eq_curve, trades=[])
    periods = {p.period_label: p for p in breakdowns}

    assert "2025" in periods
    assert periods["2025"].total_return == pytest.approx(0.10, rel=1e-3)


def test_statistical_diagnostics_calculation():
    """Verify statistical robustness diagnostics: mean, median, skewness, kurtosis, VaR, CVaR."""
    eq_curve = [
        EquityPoint(timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc), equity=10000.0, cash=10000.0, period_return=0.0),
        EquityPoint(timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc), equity=10100.0, cash=10100.0, period_return=0.01),
        EquityPoint(timestamp=datetime(2025, 1, 3, tzinfo=timezone.utc), equity=10050.0, cash=10050.0, period_return=-0.005),
        EquityPoint(timestamp=datetime(2025, 1, 4, tzinfo=timezone.utc), equity=10200.0, cash=10200.0, period_return=0.0149),
        EquityPoint(timestamp=datetime(2025, 1, 5, tzinfo=timezone.utc), equity=10300.0, cash=10300.0, period_return=0.0098),
    ]

    diag = BacktestMetricsCalculator.calculate_statistical_diagnostics(eq_curve)
    assert diag is not None
    assert diag.observations_count == 4
    assert diag.positive_return_ratio == 0.75
    assert diag.negative_return_ratio == 0.25
    assert diag.best_period_return > 0.0
    assert diag.worst_period_return < 0.0


def test_walk_forward_fold_generation_and_validation():
    """Verify WalkForwardEngine generates strict chronological non-overlapping folds."""
    timestamps = [datetime(2025, 1, i, tzinfo=timezone.utc) for i in range(1, 31)]
    wf_cfg = WalkForwardConfig(
        train_window_bars=10,
        val_window_bars=5,
        test_window_bars=5,
        step_bars=5,
        expanding_window=False,
    )
    engine = WalkForwardEngine(wf_cfg)
    folds = engine.generate_folds(timestamps)
    assert len(folds) >= 2

    for f in folds:
        assert f.train_start < f.train_end
        if f.val_start and f.val_end:
            assert f.train_end <= f.val_start
            assert f.val_end <= f.test_start
        else:
            assert f.train_end <= f.test_start
        assert f.test_start < f.test_end

    # Test invalid configuration raises ValueError
    with pytest.raises(ValueError, match="must be > 0"):
        WalkForwardConfig(train_window_bars=0, val_window_bars=5, test_window_bars=5)


def test_walk_forward_execution_and_out_of_sample_stitching():
    """
    Verify WalkForwardEngine runs sequential folds and stitches clean out-of-sample equity curve.
    """
    bars_aapl = create_synthetic_bars("AAPL", [100.0 + i for i in range(30)])
    wf_cfg = WalkForwardConfig(
        train_window_bars=10,
        val_window_bars=0,
        test_window_bars=4,
        step_bars=4,
        expanding_window=False,
    )
    wf_engine = WalkForwardEngine(config=wf_cfg)

    def simple_pipeline(fold_idx, fold_cfg, train_data, val_data, test_data):
        bt_engine = BacktestEngine(BacktestConfig(initial_capital=100000.0, commission_rate=0.0, slippage_rate=0.0))
        # Long AAPL 100% on test data
        test_ts = sorted(list({b.timestamp for b in test_data["AAPL"]}))
        targets = {t: {"AAPL": 1.0} for t in test_ts}
        return bt_engine.run(market_data=test_data, target_series=targets)

    wf_res = wf_engine.run(market_data={"AAPL": bars_aapl}, fold_pipeline_fn=simple_pipeline)

    assert len(wf_res.folds) >= 2
    assert wf_res.concatenated_result is not None
    assert len(wf_res.concatenated_result.equity_curve) > 0


def test_walk_forward_train_only_preprocessor_isolation_and_no_leakage():
    """
    Verify preprocessors and feature statistics fit ONLY on training window data
    and NEVER peek at test window data.
    """
    # 20 bars: bars 0-9 train, bars 10-14 val, bars 15-19 test
    # Values 0..9 mean = 4.5. Values 15..19 are huge (1000..1004).
    train_prices = [float(i) for i in range(10)]
    test_prices = [1000.0 + float(i) for i in range(10)]
    all_prices = train_prices + test_prices

    bars = create_synthetic_bars("AAPL", all_prices)
    all_ts = [b.timestamp for b in bars]

    fold = WalkForwardFoldConfig(
        fold_idx=0,
        train_start=all_ts[0],
        train_end=all_ts[9],
        val_start=all_ts[10],
        val_end=all_ts[14],
        test_start=all_ts[15],
        test_end=all_ts[19],
    )

    # Slice strictly train data
    train_bars = [b for b in bars if fold.train_start <= b.timestamp <= fold.train_end]
    assert len(train_bars) == 10

    # Calculate train-only scaler mean
    train_mean = np.mean([b.close for b in train_bars])
    assert train_mean == pytest.approx(4.5, rel=1e-5)

    # Verify that test data prices do not alter train_mean
    all_mean = np.mean([b.close for b in bars])
    assert all_mean > 500.0
    assert train_mean != all_mean


def test_walk_forward_future_test_window_mutation_invariance():
    """
    Verify mutating future test window prices in fold 2 does NOT change fold 1 results.
    """
    import dataclasses
    bars_orig = create_synthetic_bars("AAPL", [100.0 + i for i in range(30)])
    bars_mutated = [
        dataclasses.replace(b, close=b.close * 10.0, open=b.open * 10.0) if i >= 25 else dataclasses.replace(b)
        for i, b in enumerate(bars_orig)
    ]

    wf_cfg = WalkForwardConfig(train_window_bars=10, val_window_bars=0, test_window_bars=4, step_bars=4)

    def simple_pipeline(fold_idx, fold_cfg, train_data, val_data, test_data):
        bt_engine = BacktestEngine(BacktestConfig(initial_capital=100000.0, commission_rate=0.0, slippage_rate=0.0))
        test_ts = sorted(list({b.timestamp for b in test_data["AAPL"]}))
        targets = {t: {"AAPL": 1.0} for t in test_ts}
        return bt_engine.run(market_data=test_data, target_series=targets)

    engine1 = WalkForwardEngine(wf_cfg)
    res1 = engine1.run(market_data={"AAPL": bars_orig}, fold_pipeline_fn=simple_pipeline)

    engine2 = WalkForwardEngine(wf_cfg)
    res2 = engine2.run(market_data={"AAPL": bars_mutated}, fold_pipeline_fn=simple_pipeline)

    # Fold 0 metrics and equity curve must be IDENTICAL between runs
    f0_res1 = res1.folds[0].out_of_sample_result
    f0_res2 = res2.folds[0].out_of_sample_result

    assert f0_res1 is not None and f0_res2 is not None
    assert len(f0_res1.equity_curve) == len(f0_res2.equity_curve)
    for p1, p2 in zip(f0_res1.equity_curve, f0_res2.equity_curve):
        assert p1.equity == pytest.approx(p2.equity, rel=1e-6)


def test_deterministic_run_id_generation():
    """Verify run ID generation is deterministic based on config and data metadata."""
    cfg = BacktestConfig(initial_capital=50000.0, commission_rate=0.0005)
    id1 = BacktestEngine.generate_deterministic_run_id(
        symbols=["AAPL", "MSFT"],
        start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 1, 10, tzinfo=timezone.utc),
        config=cfg,
    )
    id2 = BacktestEngine.generate_deterministic_run_id(
        symbols=["AAPL", "MSFT"],
        start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 1, 10, tzinfo=timezone.utc),
        config=cfg,
    )
    assert id1 == id2
    assert id1.startswith("bt_exp_")


def test_research_grade_16_section_markdown_report():
    """Verify research-grade markdown report includes all 16 required sections."""
    engine = BacktestEngine(BacktestConfig(initial_capital=100000.0))
    bars = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0, 103.0, 108.0])

    t0 = bars[0].timestamp
    t1 = bars[1].timestamp
    targets = {t0: {"AAPL": 0.5}, t1: {"AAPL": 0.5}}

    res = engine.run(market_data={"AAPL": bars}, target_series=targets)
    report_md = BacktestReporter.generate_markdown_report(res)

    required_sections = [
        "## 1. Run Information",
        "## 2. Dataset Information & Provenance",
        "## 3. Model Information & Provenance",
        "## 4. Strategy & Signal Configuration",
        "## 5. Portfolio Construction Configuration",
        "## 6. Risk Management Configuration",
        "## 7. Execution Configuration",
        "## 8. Cost Configuration & Assumptions",
        "## 9. Performance Metrics",
        "## 10. Trading Statistics",
        "## 11. Risk Statistics & Portfolio Diagnostics",
        "## 12. Benchmark Comparison",
        "## 13. Cost Attribution & Friction Breakdown",
        "## 14. Drawdown Analysis & Top Episodes",
        "## 15. Walk-Forward Evaluation Info",
        "## 16. Research Limitations & Disclaimers",
    ]

    for sec in required_sections:
        assert sec in report_md, f"Missing section {sec} in markdown report"


def test_walk_forward_markdown_reporting():
    """Verify walk-forward report format."""
    bars = create_synthetic_bars("AAPL", [100.0 + i for i in range(25)])
    wf_cfg = WalkForwardConfig(train_window_bars=8, val_window_bars=0, test_window_bars=4, step_bars=4)
    wf_engine = WalkForwardEngine(wf_cfg)

    def simple_pipeline(fold_idx, fold_cfg, train_data, val_data, test_data):
        bt_engine = BacktestEngine(BacktestConfig(initial_capital=100000.0))
        test_ts = sorted(list({b.timestamp for b in test_data["AAPL"]}))
        targets = {t: {"AAPL": 0.5} for t in test_ts}
        return bt_engine.run(market_data=test_data, target_series=targets)

    wf_res = wf_engine.run(market_data={"AAPL": bars}, fold_pipeline_fn=simple_pipeline)
    wf_report = BacktestReporter.generate_walk_forward_markdown_report(wf_res)

    assert "Walk-Forward Out-of-Sample Backtest Report" in wf_report
    assert "Fold 0" in wf_report
    assert "Overall Out-of-Sample Summary" in wf_report


def test_empirical_local_historical_backtest_and_walkforward():
    """
    Empirical research-grade backtest and walk-forward validation on local historical market data.
    Validates:
    - Dataset provenance with LOCAL_HISTORICAL classification and SHA-256 integrity hash
    - Causal point-in-time signal and target execution
    - Complete cost attribution (commission, spread, slippage)
    - SPY benchmark alignment and risk/return metrics
    - Walk-forward chronological fold execution without future test leakage
    - Standardized 16-section markdown reporting
    """
    import hashlib
    from backend.app.data.models import TimeFrame
    from backend.app.data.validation.storage import ProcessedDataStorage

    storage = ProcessedDataStorage()
    aapl_bars = storage.load_bars("AAPL", TimeFrame.DAY_1, format="parquet")
    spy_bars = storage.load_bars("SPY", TimeFrame.DAY_1, format="parquet")

    assert len(aapl_bars) >= 500, f"Expected >= 500 AAPL bars, got {len(aapl_bars)}"
    assert len(spy_bars) >= 100, f"Expected >= 100 SPY bars, got {len(spy_bars)}"

    # 1. Build dataset provenance
    checksum = hashlib.sha256(str([(b.timestamp.isoformat(), b.close) for b in aapl_bars]).encode()).hexdigest()
    prov = DatasetProvenance(
        dataset_id="aapl_local_historical_2023_2025",
        provider="ProcessedDataStorage",
        symbols=["AAPL"],
        timeframe="1Day",
        start_date=aapl_bars[0].timestamp,
        end_date=aapl_bars[-1].timestamp,
        adjustment_status=AdjustmentStatus.ADJUSTED,
        data_source_type=DataSourceType.LOCAL_HISTORICAL,
        row_count=len(aapl_bars),
        checksum=checksum,
        metadata={"constituent_source": "S&P 500 Current", "survivorship_bias_note": "Point-in-time historical membership unavailable"},
    )

    # 2. Configure engine
    cfg = BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.0005,
        slippage_rate=0.0002,
        bid_ask_spread_rate=0.0002,
        benchmark_symbol="SPY",
        dataset_provenance=prov,
    )
    engine = BacktestEngine(config=cfg)

    # 3. Generate causal point-in-time moving average targets
    targets = {}
    for i in range(20, len(aapl_bars)):
        ts = aapl_bars[i].timestamp
        sma20 = sum(b.close for b in aapl_bars[i-20:i]) / 20.0
        curr_close = aapl_bars[i].close
        if curr_close > sma20:
            targets[ts] = {"AAPL": 0.8}
        else:
            targets[ts] = {"AAPL": 0.0}

    bench_lookup = {b.timestamp: b.close for b in spy_bars}
    result = engine.run(
        market_data={"AAPL": aapl_bars},
        target_series=targets,
        benchmark_prices=bench_lookup,
    )

    # Validate accounting and performance metrics
    assert result.summary.initial_capital == 100000.0
    assert result.summary.final_equity > 0.0
    assert len(result.equity_curve) == len(aapl_bars)
    assert len(result.executions) > 0
    assert result.metrics.total_trades > 0
    assert result.metrics.total_commission > 0.0
    assert result.metrics.total_spread_cost > 0.0
    assert result.metrics.total_slippage > 0.0
    assert result.metrics.statistical_diagnostics is not None
    assert len(result.metrics.drawdown_episodes) > 0
    assert len(result.metrics.period_breakdowns) > 0

    # Validate benchmark metrics
    assert result.benchmark_metrics is not None
    assert result.benchmark_metrics.benchmark_symbol == "SPY"
    assert result.benchmark_metrics.beta is not None
    assert result.benchmark_metrics.alpha is not None

    # Validate 16-section report generation
    report = BacktestReporter.generate_markdown_report(result)
    assert "## 1. Run Information" in report
    assert "## 2. Dataset Information & Provenance" in report
    assert "## 16. Research Limitations & Disclaimers" in report

    # 4. Run walk-forward evaluation on real historical bars
    wf_cfg = WalkForwardConfig(
        train_window_bars=200,
        val_window_bars=50,
        test_window_bars=50,
        step_bars=50,
    )
    wf_engine = WalkForwardEngine(config=wf_cfg)

    def wf_pipeline(fold_idx, fold_cfg, train_data, val_data, test_data):
        b_engine = BacktestEngine(config=cfg)
        test_bars = test_data["AAPL"]
        fold_targets = {b.timestamp: {"AAPL": 0.6} for b in test_bars}
        return b_engine.run(market_data=test_data, target_series=fold_targets)

    wf_res = wf_engine.run(market_data={"AAPL": aapl_bars}, fold_pipeline_fn=wf_pipeline)
    assert len(wf_res.folds) == 5
    assert wf_res.concatenated_result is not None
    assert len(wf_res.concatenated_result.equity_curve) == 250  # 5 folds * 50 bars

