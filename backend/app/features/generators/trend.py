"""
Trend and Moving Average Feature Generator
Computes Simple Moving Averages (SMA), Exponential Moving Averages (EMA), and price-to-SMA ratios.
"""

from typing import Any, Dict, List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.models import FeatureMetadata


class TrendFeatureGenerator(BaseFeatureGenerator):
    """
    Calculates SMA, EMA, and Price-vs-SMA ratios for configured windows (e.g. 20, 50, 200).
    """

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or [20, 50, 200]

    @property
    def name(self) -> str:
        return "trend_features"

    @property
    def description(self) -> str:
        return f"Calculates SMA, EMA, and Price-vs-SMA indicators for windows {self.windows}."

    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        n = len(bars)
        results: Dict[str, List[Optional[float]]] = {}
        closes = [b.close for b in bars]

        for w in self.windows:
            sma_col = f"sma_{w}"
            ema_col = f"ema_{w}"
            p_vs_sma_col = f"price_vs_sma_{w}"

            sma_vals: List[Optional[float]] = [None] * n
            ema_vals: List[Optional[float]] = [None] * n
            p_vs_sma_vals: List[Optional[float]] = [None] * n

            # 1. Simple Moving Average (SMA) & Price vs SMA
            if n >= w:
                rolling_sum = sum(closes[:w])
                sma_vals[w - 1] = rolling_sum / w
                if sma_vals[w - 1] > 0:
                    p_vs_sma_vals[w - 1] = (closes[w - 1] - sma_vals[w - 1]) / sma_vals[w - 1]

                for i in range(w, n):
                    rolling_sum += closes[i] - closes[i - w]
                    curr_sma = rolling_sum / w
                    sma_vals[i] = curr_sma
                    if curr_sma > 0:
                        p_vs_sma_vals[i] = (closes[i] - curr_sma) / curr_sma

            # 2. Exponential Moving Average (EMA)
            # Alpha multiplier = 2 / (window + 1)
            # Initial seed is SMA of first w points
            if n >= w:
                alpha = 2.0 / (w + 1)
                curr_ema = sum(closes[:w]) / w
                ema_vals[w - 1] = curr_ema

                for i in range(w, n):
                    curr_ema = (closes[i] * alpha) + (curr_ema * (1.0 - alpha))
                    ema_vals[i] = curr_ema

            results[sma_col] = sma_vals
            results[ema_col] = ema_vals
            results[p_vs_sma_col] = p_vs_sma_vals

        return results

    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        meta = []
        for w in self.windows:
            meta.extend([
                FeatureMetadata(
                    feature_name=f"sma_{w}",
                    description=f"{w}-period Simple Moving Average of closing price",
                    formula=f"Mean(close, window={w})",
                    window=w,
                    source_columns=["close"],
                    timeframe=timeframe,
                ),
                FeatureMetadata(
                    feature_name=f"ema_{w}",
                    description=f"{w}-period Exponential Moving Average of closing price",
                    formula=f"EMA(close, alpha=2/({w}+1))",
                    window=w,
                    source_columns=["close"],
                    timeframe=timeframe,
                ),
                FeatureMetadata(
                    feature_name=f"price_vs_sma_{w}",
                    description=f"Normalized percentage distance between close and {w}-period SMA",
                    formula=f"(close[t] - sma_{w}[t]) / sma_{w}[t]",
                    window=w,
                    source_columns=["close"],
                    timeframe=timeframe,
                ),
            ])
        return meta
