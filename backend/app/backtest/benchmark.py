"""
Benchmark Performance Evaluator (Phase 13).
Aligns strategy equity curve with a benchmark asset (e.g., SPY) and computes comparative metrics
including Beta, Alpha, and Correlation over common valid evaluation periods.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np

from backend.app.backtest.schemas import BenchmarkMetrics, EquityPoint


class BenchmarkEvaluator:
    """
    Computes standalone benchmark performance and strategy-vs-benchmark relative statistics.
    """

    @classmethod
    def evaluate(
        cls,
        benchmark_symbol: str,
        benchmark_prices: Optional[Dict[Union[datetime, str], float]],  # date / iso / dt -> close price
        equity_curve: List[EquityPoint],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> Optional[BenchmarkMetrics]:
        """
        Evaluate benchmark performance aligned with equity curve timestamps.
        """
        if not benchmark_prices or len(equity_curve) < 2:
            return None

        # Normalize benchmark_prices keys
        norm_bm: Dict[str, float] = {}
        for k, v in benchmark_prices.items():
            if isinstance(k, datetime):
                norm_bm[k.strftime("%Y-%m-%d")] = float(v)
            else:
                norm_bm[str(k)[:10]] = float(v)

        # Align timestamps
        aligned_strat_rets: List[float] = []
        aligned_bm_prices: List[float] = []

        for eq_pt in equity_curve:
            ts_key = eq_pt.timestamp.strftime("%Y-%m-%d")
            matched_px = norm_bm.get(ts_key)

            if matched_px is not None and matched_px > 0:
                aligned_bm_prices.append(float(matched_px))
                aligned_strat_rets.append(float(eq_pt.period_return))

        if len(aligned_bm_prices) < 2:
            return None

        start_px = aligned_bm_prices[0]
        end_px = aligned_bm_prices[-1]
        n_bars = len(aligned_bm_prices)

        total_return = (end_px / start_px) - 1.0 if start_px > 0 else 0.0

        # Annualized return
        if n_bars > 1 and (1.0 + total_return) > 0:
            ann_return = float(((1.0 + total_return) ** (periods_per_year / n_bars)) - 1.0)
        else:
            ann_return = total_return

        # Periodic benchmark returns
        bm_arr = np.array(aligned_bm_prices, dtype=float)
        bm_rets = (bm_arr[1:] / bm_arr[:-1]) - 1.0

        ann_vol: Optional[float] = None
        sharpe: Optional[float] = None

        if len(bm_rets) >= 2:
            vol = float(np.std(bm_rets, ddof=1))
            if vol > 1e-8:
                ann_vol = float(vol * np.sqrt(periods_per_year))
                periodic_rf = risk_free_rate / periods_per_year
                excess = bm_rets - periodic_rf
                sharpe = float((np.mean(excess) / vol) * np.sqrt(periods_per_year))

        # Benchmark Max Drawdown
        peaks = np.maximum.accumulate(bm_arr)
        dds = (bm_arr / peaks) - 1.0
        max_dd = abs(float(np.min(dds)))

        # Relative stats: Beta, Alpha, Correlation
        strat_arr = np.array(aligned_strat_rets[1:], dtype=float) if len(aligned_strat_rets) > 1 else np.array([])
        beta: Optional[float] = None
        alpha: Optional[float] = None
        corr: Optional[float] = None

        if len(strat_arr) == len(bm_rets) and len(bm_rets) >= 2:
            var_bm = float(np.var(bm_rets, ddof=1))
            if var_bm > 1e-8:
                cov = float(np.cov(strat_arr, bm_rets)[0, 1])
                beta = float(cov / var_bm)

                strat_total_ret = (equity_curve[-1].equity / equity_curve[0].equity) - 1.0
                strat_ann_ret = float(((1.0 + strat_total_ret) ** (periods_per_year / n_bars)) - 1.0) if (1.0 + strat_total_ret) > 0 else strat_total_ret
                alpha = float(strat_ann_ret - (beta * ann_return))

                std_strat = float(np.std(strat_arr, ddof=1))
                std_bm = float(np.std(bm_rets, ddof=1))
                if std_strat > 1e-8 and std_bm > 1e-8:
                    corr = float(cov / (std_strat * std_bm))

        return BenchmarkMetrics(
            benchmark_symbol=benchmark_symbol,
            total_return=round(total_return, 6),
            annualized_return=round(ann_return, 6),
            annualized_volatility=round(ann_vol, 6) if ann_vol is not None else 0.0,
            sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
            max_drawdown=round(max_dd, 6),
            alpha=round(alpha, 6) if alpha is not None else None,
            beta=round(beta, 4) if beta is not None else None,
            correlation=round(corr, 4) if corr is not None else None,
        )

    @classmethod
    def evaluate_benchmark(
        cls,
        equity_curve: List[EquityPoint],
        benchmark_prices: Optional[Dict[Union[datetime, str], float]],
        benchmark_symbol: str = "SPY",
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> Optional[BenchmarkMetrics]:
        """Convenience alias accepting equity_curve as first argument."""
        return cls.evaluate(
            benchmark_symbol=benchmark_symbol,
            benchmark_prices=benchmark_prices,
            equity_curve=equity_curve,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
