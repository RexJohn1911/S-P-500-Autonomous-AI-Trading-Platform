"""
Backtest Engine Execution Loop (Phase 13 Upgrade v1.2).
Executes bar-by-bar chronological backtest simulation with next-bar execution pricing,
mark-to-market valuations, corporate actions, borrow costs, zero lookahead causality,
and comprehensive research-grade ledger tracking.
"""

from datetime import datetime, timezone
import hashlib
import json
from typing import Callable, Dict, List, Optional, Union
import uuid
import numpy as np

from backend.app.backtest.benchmark import BenchmarkEvaluator
from backend.app.backtest.execution import SimulatedExecutionEngine
from backend.app.backtest.metrics import BacktestMetricsCalculator
from backend.app.backtest.portfolio import SimulatedPortfolio
from backend.app.backtest.schemas import (
    BacktestConfig,
    BacktestResult,
    BacktestSummary,
    DividendEvent,
    EquityPoint,
    ExecutionPriceType,
    ExecutionRecord,
    ExposurePoint,
    PerformanceMetrics,
    PortfolioState,
    SplitEvent,
    TradeRecord,
)
from backend.app.data.models import BarData
from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.risk.schemas import RiskAdjustedTarget


class BacktestEngine:
    """
    Main point-in-time chronological simulation engine for research and production backtesting.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    @classmethod
    def generate_deterministic_run_id(
        cls,
        symbols: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        config: BacktestConfig,
    ) -> str:
        """
        Generate reproducible, deterministic experiment hash ID from configuration and data boundaries.
        """
        id_payload = {
            "symbols": sorted(symbols),
            "start_date": start_date.isoformat() if start_date else "",
            "end_date": end_date.isoformat() if end_date else "",
            "capital": float(config.initial_capital),
            "commission": float(config.commission_rate),
            "slippage": float(config.slippage_rate),
            "spread": float(config.bid_ask_spread_rate),
            "borrow": float(config.daily_borrow_rate),
            "impact": float(config.market_impact_coefficient),
            "version": config.version,
        }
        digest = hashlib.sha256(json.dumps(id_payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"bt_exp_{digest}"

    def run(
        self,
        market_data: Dict[str, List[BarData]],  # symbol -> list of bars sorted chronologically
        target_series: Optional[Dict[datetime, Union[Dict[str, float], List[RiskAdjustedTarget], List[PortfolioTarget]]]] = None,
        target_generator: Optional[Callable[[datetime, Dict[str, BarData], SimulatedPortfolio], Dict[str, float]]] = None,
        corporate_actions: Optional[Dict[datetime, List[Union[DividendEvent, SplitEvent]]]] = None,
        benchmark_prices: Optional[Dict[Union[datetime, str], float]] = None,
        backtest_id: Optional[str] = None,
        evaluation_timestamp: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Execute chronological backtest simulation across all bars.
        """
        if not market_data:
            raise ValueError("market_data cannot be empty")

        eval_time = evaluation_timestamp or datetime.now(timezone.utc)

        # 1. Collect and sort all distinct chronological timestamps
        timestamp_set = set()
        bar_lookup: Dict[datetime, Dict[str, BarData]] = {}

        for sym, bars in market_data.items():
            for b in bars:
                ts = b.timestamp
                # Filter by start_date and end_date if configured
                if self.config.start_date and ts < self.config.start_date:
                    continue
                if self.config.end_date and ts > self.config.end_date:
                    continue

                timestamp_set.add(ts)
                if ts not in bar_lookup:
                    bar_lookup[ts] = {}
                bar_lookup[ts][sym] = b

        sorted_timestamps = sorted(timestamp_set)
        if not sorted_timestamps:
            raise ValueError("No market data bars available within configured date range")

        symbols_list = sorted(list(market_data.keys()))
        bt_id = backtest_id or self.generate_deterministic_run_id(
            symbols=symbols_list,
            start_date=sorted_timestamps[0],
            end_date=sorted_timestamps[-1],
            config=self.config,
        )

        # 2. Initialize portfolio ledger
        portfolio = SimulatedPortfolio(
            initial_capital=self.config.initial_capital,
            base_currency=self.config.base_currency,
        )

        equity_curve: List[EquityPoint] = []
        portfolio_history: List[PortfolioState] = []
        exposure_history: List[ExposurePoint] = []
        all_executions: List[ExecutionRecord] = []
        target_history_records: List[Dict] = []

        running_peak = self.config.initial_capital
        prev_equity = self.config.initial_capital
        pending_targets: Optional[Union[Dict[str, float], List[RiskAdjustedTarget], List[PortfolioTarget]]] = None
        total_turnover = 0.0

        # 3. Bar-by-bar chronological execution loop
        for bar_idx, ts in enumerate(sorted_timestamps):
            portfolio.set_bar_index(bar_idx)
            current_bars = bar_lookup.get(ts, {})

            # Step A: Execute pending targets at point-in-time execution prices for this bar
            if pending_targets is not None:
                if self.config.execution_price_type == ExecutionPriceType.NEXT_OPEN:
                    exec_prices = {s: b.open for s, b in current_bars.items() if b.open > 0}
                else:
                    exec_prices = {s: b.close for s, b in current_bars.items() if b.close > 0}

                executions = SimulatedExecutionEngine.execute_rebalance(
                    portfolio=portfolio,
                    targets=pending_targets,
                    prices=exec_prices,
                    config=self.config,
                    timestamp=ts,
                )
                all_executions.extend(executions)
                pending_targets = None

            # Step B: Process Corporate Actions (Dividends & Splits) at timestamp if present
            if corporate_actions and ts in corporate_actions:
                for action in corporate_actions[ts]:
                    if isinstance(action, DividendEvent):
                        portfolio.apply_dividend(action)
                    elif isinstance(action, SplitEvent):
                        portfolio.apply_stock_split(action)

            # Step C: Apply daily borrow financing fees on short positions if configured
            if self.config.daily_borrow_rate > 0:
                portfolio.apply_borrow_fees(self.config.daily_borrow_rate)

            # Step D: Mark-to-market portfolio valuation at bar Close
            close_prices = {s: b.close for s, b in current_bars.items() if b.close > 0}
            portfolio.update_market_prices(close_prices, timestamp=ts)

            # Step E: Record EquityPoint, Drawdown, and ExposurePoint
            curr_equity = portfolio.equity
            period_ret = (curr_equity / prev_equity) - 1.0 if (bar_idx > 0 and prev_equity > 0) else 0.0
            running_peak = max(running_peak, curr_equity)
            dd_pct = (curr_equity / running_peak) - 1.0 if running_peak > 0 else 0.0

            eq_pt = EquityPoint(
                timestamp=ts,
                equity=curr_equity,
                cash=portfolio.cash,
                gross_exposure=(portfolio.gross_notional / curr_equity) if curr_equity > 0 else 0.0,
                net_exposure=(portfolio.net_notional / curr_equity) if curr_equity > 0 else 0.0,
                long_exposure=(portfolio.long_notional / curr_equity) if curr_equity > 0 else 0.0,
                short_exposure=(portfolio.short_notional / curr_equity) if curr_equity > 0 else 0.0,
                period_return=period_ret,
                cumulative_return=(curr_equity / self.config.initial_capital) - 1.0,
                running_peak=running_peak,
                drawdown_pct=dd_pct,
            )
            equity_curve.append(eq_pt)
            exposure_history.append(portfolio.get_exposure_point(ts))
            prev_equity = curr_equity

            # Step F: Determine strategy target allocations for NEXT bar execution
            new_targets = None
            if target_generator is not None:
                new_targets = target_generator(ts, current_bars, portfolio)
            elif target_series is not None and ts in target_series:
                new_targets = target_series[ts]

            # If not exact match, try matching date string
            if new_targets is None and target_series is not None:
                ts_str = ts.strftime("%Y-%m-%d")
                for k_dt, v_tgt in target_series.items():
                    if isinstance(k_dt, datetime) and k_dt.strftime("%Y-%m-%d") == ts_str:
                        new_targets = v_tgt
                        break

            turnover_at_t = 0.0
            if new_targets is not None:
                pending_targets = new_targets
                # Compute turnover relative to current positions
                curr_weights = {s: pos.weight for s, pos in portfolio.positions.items()}
                tgt_dict = {}
                if isinstance(new_targets, dict):
                    tgt_dict = new_targets
                elif isinstance(new_targets, list):
                    for t in new_targets:
                        w = getattr(t, "adjusted_weight", getattr(t, "target_weight", 0.0))
                        tgt_dict[t.symbol] = w

                all_syms = set(curr_weights.keys()) | set(tgt_dict.keys())
                turnover_at_t = sum(abs(tgt_dict.get(s, 0.0) - curr_weights.get(s, 0.0)) for s in all_syms)
                total_turnover += turnover_at_t

                target_history_records.append({
                    "timestamp": ts.isoformat(),
                    "targets": tgt_dict,
                })

            # Step G: Record PortfolioState snapshot
            p_state = portfolio.get_state(timestamp=ts, turnover=turnover_at_t)
            portfolio_history.append(p_state)

        # 4. Compute comprehensive performance and benchmark metrics
        metrics = BacktestMetricsCalculator.compute_all_metrics(
            equity_curve=equity_curve,
            trades=portfolio.completed_trades,
            executions=all_executions,
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
            total_turnover=total_turnover,
            total_spread_cost=portfolio.total_spread_cost,
            total_borrow_cost=portfolio.total_borrow_cost,
            total_market_impact=portfolio.total_market_impact,
        )

        benchmark_metrics = None
        if benchmark_prices and self.config.benchmark_symbol:
            benchmark_metrics = BenchmarkEvaluator.evaluate(
                benchmark_symbol=self.config.benchmark_symbol,
                benchmark_prices=benchmark_prices,
                equity_curve=equity_curve,
                risk_free_rate=self.config.risk_free_rate,
            )

        summary = BacktestSummary(
            backtest_id=bt_id,
            timestamp=eval_time,
            start_date=sorted_timestamps[0],
            end_date=sorted_timestamps[-1],
            total_bars=len(sorted_timestamps),
            symbols=list(market_data.keys()),
            initial_capital=self.config.initial_capital,
            final_equity=portfolio.equity,
            total_return=metrics.total_return,
            annualized_return=metrics.annualized_return,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            total_trades=metrics.total_trades,
            win_rate=metrics.win_rate,
            benchmark_symbol=self.config.benchmark_symbol,
            benchmark_return=benchmark_metrics.total_return if benchmark_metrics else None,
        )

        return BacktestResult(
            backtest_id=bt_id,
            timestamp=eval_time,
            config=self.config,
            summary=summary,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=portfolio.trades,
            executions=all_executions,
            portfolio_history=portfolio_history,
            benchmark_metrics=benchmark_metrics,
            target_history=target_history_records,
            cost_breakdown=portfolio.get_cost_breakdown(),
            exposure_history=exposure_history,
            dataset_provenance=self.config.dataset_provenance,
            dividends_collected=portfolio.dividends_collected,
            splits_applied_count=portfolio.splits_applied_count,
            metadata={
                "engine_version": self.config.version,
                "execution_price_type": self.config.execution_price_type.value,
                "symbols_count": len(market_data),
            },
        )
