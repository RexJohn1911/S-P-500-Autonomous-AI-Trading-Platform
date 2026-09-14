"""
Market-Relative Strength Feature Generator
Computes asset performance relative to broader market benchmark (e.g. SPY).
"""

from typing import Any, Dict, List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.models import FeatureMetadata


class MarketRelativeFeatureGenerator(BaseFeatureGenerator):
    """
    Calculates stock relative strength vs S&P 500 benchmark (SPY).
    Formula: relative_strength_Nd = return_stock_Nd - return_market_Nd
    """

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def name(self) -> str:
        return "market_relative_features"

    @property
    def description(self) -> str:
        return f"Calculates market return and relative strength vs benchmark over {self.window} periods."

    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        n = len(bars)
        is_daily = bars[0].timeframe == TimeFrame.DAY_1 if bars else True
        mkt_ret_col = f"market_return_{self.window}d" if is_daily else f"market_return_{self.window}"
        rel_str_col = f"relative_strength_{self.window}d" if is_daily else f"relative_strength_{self.window}"

        mkt_ret_vals: List[Optional[float]] = [None] * n
        rel_str_vals: List[Optional[float]] = [None] * n

        if n == 0:
            return {mkt_ret_col: mkt_ret_vals, rel_str_col: rel_str_vals}

        # Check if benchmark bars are available in context
        benchmark_bars: Optional[List[BarData]] = context.get("benchmark_bars") if context else None

        if benchmark_bars:
            # Map benchmark timestamp -> close price
            bench_map: Dict[Any, float] = {}
            for b in benchmark_bars:
                key = b.timestamp.date() if is_daily else b.timestamp
                bench_map[key] = b.close

            # Compute stock N-period returns
            stock_returns: List[Optional[float]] = [None] * n
            for i in range(self.window, n):
                prev_c = bars[i - self.window].close
                curr_c = bars[i].close
                if prev_c > 0:
                    stock_returns[i] = (curr_c - prev_c) / prev_c

            # Align and compute market return and relative strength
            for i in range(self.window, n):
                curr_key = bars[i].timestamp.date() if is_daily else bars[i].timestamp
                prev_key = bars[i - self.window].timestamp.date() if is_daily else bars[i - self.window].timestamp

                if curr_key in bench_map and prev_key in bench_map:
                    mkt_prev = bench_map[prev_key]
                    mkt_curr = bench_map[curr_key]
                    if mkt_prev > 0:
                        mkt_ret = (mkt_curr - mkt_prev) / mkt_prev
                        mkt_ret_vals[i] = mkt_ret

                        stk_ret = stock_returns[i]
                        if stk_ret is not None:
                            rel_str_vals[i] = stk_ret - mkt_ret

        return {
            mkt_ret_col: mkt_ret_vals,
            rel_str_col: rel_str_vals,
        }

    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        is_daily = timeframe == TimeFrame.DAY_1
        mkt_ret_col = f"market_return_{self.window}d" if is_daily else f"market_return_{self.window}"
        rel_str_col = f"relative_strength_{self.window}d" if is_daily else f"relative_strength_{self.window}"

        return [
            FeatureMetadata(
                feature_name=mkt_ret_col,
                description=f"{self.window}-period return of market benchmark (SPY)",
                formula=f"(SPY_Close[t] - SPY_Close[t-{self.window}]) / SPY_Close[t-{self.window}]",
                window=self.window,
                source_columns=["benchmark_close"],
                timeframe=timeframe,
            ),
            FeatureMetadata(
                feature_name=rel_str_col,
                description=f"{self.window}-period stock excess return relative to market benchmark",
                formula=f"stock_return_{self.window} - market_return_{self.window}",
                window=self.window,
                source_columns=["close", "benchmark_close"],
                timeframe=timeframe,
            ),
        ]
