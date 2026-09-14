"""
Rolling Historical Beta Feature Generator
Computes asset systematic risk (Beta) relative to market benchmark.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.models import FeatureMetadata


class BetaFeatureGenerator(BaseFeatureGenerator):
    """
    Calculates rolling stock Beta relative to market benchmark over a configured window (e.g. 60 periods).
    Formula: Beta = Cov(r_stock, r_market) / Var(r_market)
    """

    def __init__(self, window: int = 60):
        self.window = window

    @property
    def name(self) -> str:
        return "beta_features"

    @property
    def description(self) -> str:
        return f"Calculates rolling {self.window}-period historical Beta vs benchmark."

    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        n = len(bars)
        is_daily = bars[0].timeframe == TimeFrame.DAY_1 if bars else True
        beta_col = f"beta_{self.window}d" if is_daily else f"beta_{self.window}"
        beta_vals: List[Optional[float]] = [None] * n

        if n < self.window or not context or "benchmark_bars" not in context:
            return {beta_col: beta_vals}

        benchmark_bars: List[BarData] = context["benchmark_bars"]
        bench_map: Dict[Any, float] = {}
        for b in benchmark_bars:
            key = b.timestamp.date() if is_daily else b.timestamp
            bench_map[key] = b.close

        # Compute 1-period aligned returns
        stock_rets: List[Optional[float]] = [None] * n
        mkt_rets: List[Optional[float]] = [None] * n

        for i in range(1, n):
            prev_stk_c = bars[i - 1].close
            curr_stk_c = bars[i].close
            if prev_stk_c > 0:
                stock_rets[i] = (curr_stk_c - prev_stk_c) / prev_stk_c

            curr_key = bars[i].timestamp.date() if is_daily else bars[i].timestamp
            prev_key = bars[i - 1].timestamp.date() if is_daily else bars[i - 1].timestamp

            if curr_key in bench_map and prev_key in bench_map:
                prev_mkt_c = bench_map[prev_key]
                curr_mkt_c = bench_map[curr_key]
                if prev_mkt_c > 0:
                    mkt_rets[i] = (curr_mkt_c - prev_mkt_c) / prev_mkt_c

        # Rolling Covariance / Variance
        for i in range(self.window, n):
            pair_stk = []
            pair_mkt = []
            for j in range(i - self.window + 1, i + 1):
                sr = stock_rets[j]
                mr = mkt_rets[j]
                if sr is not None and mr is not None:
                    pair_stk.append(sr)
                    pair_mkt.append(mr)

            if len(pair_stk) >= int(self.window * 0.8):  # Require at least 80% valid pairs
                var_mkt = float(np.var(pair_mkt, ddof=1))
                if var_mkt > 1e-7:
                    cov = float(np.cov(pair_stk, pair_mkt, ddof=1)[0, 1])
                    beta_vals[i] = cov / var_mkt

        return {beta_col: beta_vals}

    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        is_daily = timeframe == TimeFrame.DAY_1
        beta_col = f"beta_{self.window}d" if is_daily else f"beta_{self.window}"

        return [
            FeatureMetadata(
                feature_name=beta_col,
                description=f"{self.window}-period rolling systematic risk Beta relative to market benchmark",
                formula=f"Cov(return_1d_stock, return_1d_market, {self.window}) / Var(return_1d_market, {self.window})",
                window=self.window,
                source_columns=["close", "benchmark_close"],
                timeframe=timeframe,
            )
        ]
