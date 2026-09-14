"""
Sector-Relative Strength Feature Generator
Computes asset performance relative to sector ETF benchmark (e.g. XLK, XLF, XLE).
"""

from typing import Any, Dict, List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.models import FeatureMetadata


class SectorFeatureGenerator(BaseFeatureGenerator):
    """
    Calculates stock relative strength vs sector benchmark (e.g. XLK for Tech).
    Formula: sector_relative_strength_Nd = return_stock_Nd - return_sector_Nd
    """

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def name(self) -> str:
        return "sector_features"

    @property
    def description(self) -> str:
        return f"Calculates sector return and sector relative strength over {self.window} periods."

    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        n = len(bars)
        is_daily = bars[0].timeframe == TimeFrame.DAY_1 if bars else True
        sec_ret_col = f"sector_return_{self.window}d" if is_daily else f"sector_return_{self.window}"
        sec_rel_str_col = f"sector_relative_strength_{self.window}d" if is_daily else f"sector_relative_strength_{self.window}"

        sec_ret_vals: List[Optional[float]] = [None] * n
        sec_rel_str_vals: List[Optional[float]] = [None] * n

        if n == 0 or not context or "sector_bars" not in context:
            return {sec_ret_col: sec_ret_vals, sec_rel_str_col: sec_rel_str_vals}

        sector_bars: List[BarData] = context["sector_bars"]
        sec_map: Dict[Any, float] = {}
        for b in sector_bars:
            key = b.timestamp.date() if is_daily else b.timestamp
            sec_map[key] = b.close

        # Compute stock N-period returns
        stock_returns: List[Optional[float]] = [None] * n
        for i in range(self.window, n):
            prev_c = bars[i - self.window].close
            curr_c = bars[i].close
            if prev_c > 0:
                stock_returns[i] = (curr_c - prev_c) / prev_c

        # Align and compute sector return and relative strength
        for i in range(self.window, n):
            curr_key = bars[i].timestamp.date() if is_daily else bars[i].timestamp
            prev_key = bars[i - self.window].timestamp.date() if is_daily else bars[i - self.window].timestamp

            if curr_key in sec_map and prev_key in sec_map:
                sec_prev = sec_map[prev_key]
                sec_curr = sec_map[curr_key]
                if sec_prev > 0:
                    sec_ret = (sec_curr - sec_prev) / sec_prev
                    sec_ret_vals[i] = sec_ret

                    stk_ret = stock_returns[i]
                    if stk_ret is not None:
                        sec_rel_str_vals[i] = stk_ret - sec_ret

        return {
            sec_ret_col: sec_ret_vals,
            sec_rel_str_col: sec_rel_str_vals,
        }

    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        is_daily = timeframe == TimeFrame.DAY_1
        sec_ret_col = f"sector_return_{self.window}d" if is_daily else f"sector_return_{self.window}"
        sec_rel_str_col = f"sector_relative_strength_{self.window}d" if is_daily else f"sector_relative_strength_{self.window}"

        return [
            FeatureMetadata(
                feature_name=sec_ret_col,
                description=f"{self.window}-period return of sector benchmark ETF",
                formula=f"(Sector_Close[t] - Sector_Close[t-{self.window}]) / Sector_Close[t-{self.window}]",
                window=self.window,
                source_columns=["sector_close"],
                timeframe=timeframe,
            ),
            FeatureMetadata(
                feature_name=sec_rel_str_col,
                description=f"{self.window}-period stock excess return relative to sector benchmark ETF",
                formula=f"stock_return_{self.window} - sector_return_{self.window}",
                window=self.window,
                source_columns=["close", "sector_close"],
                timeframe=timeframe,
            ),
        ]
