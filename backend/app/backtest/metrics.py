"""
Backtest Performance Metrics Calculator (Phase 13 Upgrade v1.2).
Computes standard quantitative metrics including returns, annualized statistics, Sharpe ratio,
Sortino ratio, Calmar ratio, maximum drawdown, win rates, turnover, period breakdowns,
drawdown episodes, statistical diagnostics, and granular friction attribution.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.app.backtest.schemas import (
    DrawdownEpisode,
    DrawdownPoint,
    EquityPoint,
    ExecutionRecord,
    PerformanceMetrics,
    PeriodPerformance,
    StatisticalDiagnostics,
    TradeRecord,
)


class BacktestMetricsCalculator:
    """
    Computes deterministic research-grade quantitative performance statistics.
    """

    @classmethod
    def calculate_returns(cls, equity_series: List[float]) -> List[float]:
        """
        Calculate period-by-period returns from equity values.
        Initial period return is defined as 0.0.
        """
        if not equity_series:
            return []
        returns = [0.0]
        for i in range(1, len(equity_series)):
            prev = equity_series[i - 1]
            curr = equity_series[i]
            ret = (curr / prev - 1.0) if prev > 0 else 0.0
            returns.append(ret)
        return returns

    @classmethod
    def calculate_drawdowns(
        cls, equity_curve: List[EquityPoint]
    ) -> Tuple[List[DrawdownPoint], float, int]:
        """
        Calculate point-in-time drawdowns, maximum drawdown, and max drawdown duration in bars.
        """
        if not equity_curve:
            return [], 0.0, 0

        dd_points: List[DrawdownPoint] = []
        running_peak = float(equity_curve[0].equity)
        max_dd_pct = 0.0
        max_duration = 0
        current_duration = 0

        for ep in equity_curve:
            eq = float(ep.equity)
            if eq > running_peak:
                running_peak = eq

            dd_abs = eq - running_peak
            dd_pct = (eq / running_peak - 1.0) if running_peak > 0 else 0.0

            if abs(dd_pct) > max_dd_pct and dd_pct < 0:
                max_dd_pct = abs(dd_pct)

            if dd_pct < -1e-6:
                current_duration += 1
                if current_duration > max_duration:
                    max_duration = current_duration
            else:
                current_duration = 0

            dd_points.append(
                DrawdownPoint(
                    timestamp=ep.timestamp,
                    equity=eq,
                    running_peak=running_peak,
                    drawdown=dd_abs,
                    drawdown_percentage=dd_pct,
                )
            )

        return dd_points, max_dd_pct, max_duration

    @classmethod
    def extract_drawdown_episodes(
        cls, equity_curve: List[EquityPoint]
    ) -> List[DrawdownEpisode]:
        """
        Extract discrete historical drawdown episodes with start, peak, trough, and recovery dates.
        """
        if len(equity_curve) < 2:
            return []

        episodes: List[DrawdownEpisode] = []
        in_drawdown = False
        peak_idx = 0
        peak_val = float(equity_curve[0].equity)
        trough_idx = 0
        trough_val = peak_val
        trough_depth = 0.0

        for i, ep in enumerate(equity_curve):
            eq = float(ep.equity)
            if eq >= peak_val:
                if in_drawdown:
                    # Episode ended/recovered
                    episodes.append(
                        DrawdownEpisode(
                            start_time=equity_curve[peak_idx].timestamp,
                            peak_time=equity_curve[peak_idx].timestamp,
                            trough_time=equity_curve[trough_idx].timestamp,
                            recovery_time=ep.timestamp,
                            depth_pct=trough_depth,
                            duration_bars=i - peak_idx,
                            is_recovered=True,
                        )
                    )
                    in_drawdown = False
                peak_val = eq
                peak_idx = i
                trough_val = eq
                trough_idx = i
                trough_depth = 0.0
            else:
                in_drawdown = True
                curr_depth = (eq / peak_val - 1.0) if peak_val > 0 else 0.0
                if eq < trough_val:
                    trough_val = eq
                    trough_idx = i
                    trough_depth = abs(curr_depth)

        # Unrecovered ongoing drawdown at end of series
        if in_drawdown and trough_depth > 1e-4:
            episodes.append(
                DrawdownEpisode(
                    start_time=equity_curve[peak_idx].timestamp,
                    peak_time=equity_curve[peak_idx].timestamp,
                    trough_time=equity_curve[trough_idx].timestamp,
                    recovery_time=None,
                    depth_pct=trough_depth,
                    duration_bars=len(equity_curve) - 1 - peak_idx,
                    is_recovered=False,
                )
            )

        return episodes

    @classmethod
    def calculate_period_breakdowns(
        cls,
        equity_curve: List[EquityPoint],
        trades: List[TradeRecord],
        periods_per_year: int = 252,
    ) -> List[PeriodPerformance]:
        """
        Break down performance by calendar year.
        """
        if len(equity_curve) < 2:
            return []

        # Group equity points by year
        year_groups: Dict[str, List[EquityPoint]] = defaultdict(list)
        for ep in equity_curve:
            year_groups[ep.timestamp.strftime("%Y")].append(ep)

        breakdowns: List[PeriodPerformance] = []
        for y_str in sorted(year_groups.keys()):
            pts = year_groups[y_str]
            if len(pts) < 2:
                continue

            y_start_eq = pts[0].equity
            y_end_eq = pts[-1].equity
            y_tot_ret = (y_end_eq / y_start_eq - 1.0) if y_start_eq > 0 else 0.0
            y_bars = len(pts)

            # Annualized return for year
            y_ann_ret = float(((1.0 + y_tot_ret) ** (periods_per_year / y_bars)) - 1.0) if (1.0 + y_tot_ret) > 0 else y_tot_ret

            y_rets = [p.period_return for p in pts[1:]]
            y_vol = float(np.std(y_rets, ddof=1) * np.sqrt(periods_per_year)) if len(y_rets) >= 2 else 0.0

            _, y_max_dd, _ = cls.calculate_drawdowns(pts)
            y_trades = sum(1 for t in trades if t.exit_time and t.exit_time.strftime("%Y") == y_str)

            breakdowns.append(
                PeriodPerformance(
                    period_label=y_str,
                    total_return=round(y_tot_ret, 6),
                    annualized_return=round(y_ann_ret, 6),
                    annualized_volatility=round(y_vol, 6),
                    max_drawdown=round(y_max_dd, 6),
                    trade_count=y_trades,
                )
            )

        return breakdowns

    @classmethod
    def calculate_statistical_diagnostics(
        cls, equity_curve: List[EquityPoint]
    ) -> Optional[StatisticalDiagnostics]:
        """
        Compute distribution diagnostics over return series.
        """
        if len(equity_curve) < 2:
            return None

        rets = np.array([e.period_return for e in equity_curve[1:]], dtype=float)
        n = len(rets)
        if n == 0:
            return None

        pos_count = int(np.sum(rets > 0))
        neg_count = int(np.sum(rets < 0))

        pos_ratio = float(pos_count / n) if n > 0 else None
        neg_ratio = float(neg_count / n) if n > 0 else None
        best_ret = float(np.max(rets))
        worst_ret = float(np.min(rets))
        mean_ret = float(np.mean(rets))
        median_ret = float(np.median(rets))

        return StatisticalDiagnostics(
            observations_count=n,
            trading_days_count=len(equity_curve),
            positive_return_ratio=pos_ratio,
            negative_return_ratio=neg_ratio,
            best_period_return=best_ret,
            worst_period_return=worst_ret,
            mean_period_return=mean_ret,
            median_period_return=median_ret,
        )

    @classmethod
    def compute_all_metrics(
        cls,
        equity_curve: List[EquityPoint],
        trades: List[TradeRecord],
        executions: Optional[List[ExecutionRecord]] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        total_turnover: float = 0.0,
        total_spread_cost: float = 0.0,
        total_borrow_cost: float = 0.0,
        total_market_impact: float = 0.0,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance statistics from simulated equity points and trade records.
        """
        if not equity_curve:
            return PerformanceMetrics(
                start_equity=float(initial_capital),
                end_equity=float(initial_capital),
                total_return=0.0,
                annualized_return=0.0,
                annualized_volatility=0.0,
                sharpe_ratio=None,
                sortino_ratio=None,
                calmar_ratio=None,
                max_drawdown=0.0,
                max_drawdown_duration_bars=0,
                win_rate=None,
                profit_factor=None,
                avg_trade_return=None,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_turnover=0.0,
                total_commission=0.0,
                total_slippage=0.0,
                total_spread_cost=0.0,
                total_borrow_cost=0.0,
                total_market_impact_cost=0.0,
                gross_pnl=0.0,
                net_pnl=0.0,
            )

        start_equity = float(equity_curve[0].equity)
        end_equity = float(equity_curve[-1].equity)
        n_bars = len(equity_curve)

        total_return = (end_equity / start_equity) - 1.0 if start_equity > 0 else 0.0

        # Annualized return (CAGR)
        if n_bars > 1 and start_equity > 0 and (1.0 + total_return) > 0:
            ann_return = float(((1.0 + total_return) ** (periods_per_year / n_bars)) - 1.0)
        else:
            ann_return = total_return

        # Periodic returns series
        period_returns = np.array([e.period_return for e in equity_curve[1:]], dtype=float) if n_bars > 1 else np.array([], dtype=float)

        ann_vol: Optional[float] = 0.0
        sharpe: Optional[float] = None
        sortino: Optional[float] = None

        if len(period_returns) >= 2:
            vol = float(np.std(period_returns, ddof=1))
            ann_vol = float(vol * np.sqrt(periods_per_year))
            if vol > 1e-8:
                periodic_rf = risk_free_rate / periods_per_year
                excess_returns = period_returns - periodic_rf
                mean_excess = float(np.mean(excess_returns))
                sharpe = float((mean_excess / vol) * np.sqrt(periods_per_year))

                # Sortino downside deviation
                downside = np.minimum(excess_returns, 0.0)
                downside_std = float(np.sqrt(np.mean(downside ** 2)))
                if downside_std > 1e-8:
                    sortino = float((mean_excess / downside_std) * np.sqrt(periods_per_year))
        elif len(period_returns) == 1:
            ann_vol = 0.0

        # Drawdown analysis
        _, max_dd, max_dd_duration = cls.calculate_drawdowns(equity_curve)

        # Calmar Ratio
        calmar: Optional[float] = None
        if abs(max_dd) > 1e-6:
            calmar = float(ann_return / abs(max_dd))

        # Trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        losing_trades = sum(1 for t in trades if t.net_pnl < 0)

        win_rate: Optional[float] = None
        if total_trades > 0:
            win_rate = float(winning_trades / total_trades)

        profit_factor: Optional[float] = None
        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = sum(abs(t.net_pnl) for t in trades if t.net_pnl < 0)
        if gross_loss > 1e-6:
            profit_factor = float(gross_profit / gross_loss)
        elif gross_profit > 1e-6:
            profit_factor = None  # Undefined (zero losses)

        avg_trade_return: Optional[float] = None
        if total_trades > 0:
            avg_trade_return = float(np.mean([t.return_pct for t in trades]))

        total_commission = float(sum(t.commission for t in trades))
        total_slippage = float(sum(t.slippage for t in trades))
        if executions:
            total_commission = float(sum(e.commission for e in executions))
            total_slippage = float(sum(e.slippage_cost for e in executions))
            total_spread_cost = float(sum(e.spread_cost for e in executions))
            total_market_impact = float(sum(e.market_impact_cost for e in executions))

        net_pnl = float(end_equity - start_equity)
        gross_pnl = float(net_pnl + total_commission + total_slippage + total_spread_cost + total_borrow_cost + total_market_impact)

        # Advanced research analytics
        episodes = cls.extract_drawdown_episodes(equity_curve)
        period_breakdowns = cls.calculate_period_breakdowns(equity_curve, trades, periods_per_year)
        stat_diag = cls.calculate_statistical_diagnostics(equity_curve)

        return PerformanceMetrics(
            start_equity=round(start_equity, 2),
            end_equity=round(end_equity, 2),
            total_return=round(total_return, 6),
            annualized_return=round(ann_return, 6),
            annualized_volatility=round(ann_vol, 6) if ann_vol is not None else 0.0,
            sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
            sortino_ratio=round(sortino, 4) if sortino is not None else None,
            calmar_ratio=round(calmar, 4) if calmar is not None else None,
            max_drawdown=round(max_dd, 6),
            max_drawdown_duration_bars=max_dd_duration,
            win_rate=round(win_rate, 4) if win_rate is not None else None,
            profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
            avg_trade_return=round(avg_trade_return, 6) if avg_trade_return is not None else None,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_turnover=round(float(total_turnover), 6),
            total_commission=round(total_commission, 2),
            total_slippage=round(total_slippage, 2),
            total_spread_cost=round(total_spread_cost, 2),
            total_borrow_cost=round(total_borrow_cost, 2),
            total_market_impact_cost=round(total_market_impact, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            period_breakdowns=period_breakdowns,
            drawdown_episodes=episodes,
            statistical_diagnostics=stat_diag,
        )

    @classmethod
    def calculate(
        cls,
        equity_curve: List[EquityPoint],
        trades: List[TradeRecord],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> PerformanceMetrics:
        """Alias for compute_all_metrics for backwards compatibility."""
        return cls.compute_all_metrics(
            equity_curve=equity_curve,
            trades=trades,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
