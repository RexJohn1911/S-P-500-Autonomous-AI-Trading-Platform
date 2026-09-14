"""
Volatility and Average True Range (ATR) Feature Generator
Computes rolling return volatility, ATR, and optional macroeconomic VIX level/change features.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
from backend.app.data.models import BarData, TimeFrame, MacroData
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.models import FeatureMetadata


class VolatilityFeatureGenerator(BaseFeatureGenerator):
    """
    Computes rolling return volatility, Average True Range (ATR), and VIX context features.
    """

    def __init__(self, vol_window: int = 20, atr_window: int = 14, annualize_daily: bool = True):
        self.vol_window = vol_window
        self.atr_window = atr_window
        self.annualize_daily = annualize_daily

    @property
    def name(self) -> str:
        return "volatility_features"

    @property
    def description(self) -> str:
        return f"Calculates rolling return volatility ({self.vol_window}), ATR ({self.atr_window}), and VIX features."

    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        n = len(bars)
        is_daily = bars[0].timeframe == TimeFrame.DAY_1 if bars else True
        vol_col = f"rolling_volatility_{self.vol_window}d" if is_daily else f"rolling_volatility_{self.vol_window}"
        atr_col = f"atr_{self.atr_window}"

        vol_values: List[Optional[float]] = [None] * n
        atr_values: List[Optional[float]] = [None] * n
        vix_level_values: List[Optional[float]] = [None] * n
        vix_change_values: List[Optional[float]] = [None] * n

        if n == 0:
            return {
                vol_col: vol_values,
                atr_col: atr_values,
                "vix_level": vix_level_values,
                "vix_change_5d": vix_change_values,
            }

        # 1. 1-Period Returns for Rolling Volatility
        one_period_returns: List[Optional[float]] = [None] * n
        for i in range(1, n):
            prev_c = bars[i - 1].close
            curr_c = bars[i].close
            if prev_c > 0:
                one_period_returns[i] = (curr_c - prev_c) / prev_c

        annual_multiplier = math.sqrt(252) if (self.annualize_daily and is_daily) else 1.0

        for i in range(self.vol_window, n):
            window_returns = [r for r in one_period_returns[i - self.vol_window + 1 : i + 1] if r is not None]
            if len(window_returns) == self.vol_window:
                sample_std = float(np.std(window_returns, ddof=1))
                vol_values[i] = sample_std * annual_multiplier

        # 2. Average True Range (ATR)
        true_ranges: List[float] = [0.0] * n
        for i in range(n):
            h, l = bars[i].high, bars[i].low
            if i == 0:
                true_ranges[i] = h - l
            else:
                prev_c = bars[i - 1].close
                true_ranges[i] = max(h - l, abs(h - prev_c), abs(l - prev_c))

        if n >= self.atr_window:
            # Initial simple average
            initial_atr = sum(true_ranges[: self.atr_window]) / self.atr_window
            atr_values[self.atr_window - 1] = initial_atr
            running_atr = initial_atr

            # Wilder smoothing: ATR[t] = (ATR[t-1] * (N - 1) + TR[t]) / N
            for i in range(self.atr_window, n):
                running_atr = (running_atr * (self.atr_window - 1) + true_ranges[i]) / self.atr_window
                atr_values[i] = running_atr

        # 3. Macro VIX Integration (if supplied via context)
        vix_input = None
        if context:
            vix_input = context.get("vix_data") or context.get("vix_bars")

        if vix_input:
            vix_data = vix_input
            # Map timestamp (date or datetime) to VIX value
            vix_map: Dict[Any, float] = {}
            if isinstance(vix_data, list):
                for item in vix_data:
                    if isinstance(item, MacroData):
                        vix_map[item.timestamp.date() if is_daily else item.timestamp] = item.value
                    elif isinstance(item, BarData):
                        vix_map[item.timestamp.date() if is_daily else item.timestamp] = item.close

            for i, b in enumerate(bars):
                ts_key = b.timestamp.date() if is_daily else b.timestamp
                if ts_key in vix_map:
                    vix_level_values[i] = vix_map[ts_key]

            # 5-period VIX Change
            for i in range(5, n):
                curr_v = vix_level_values[i]
                prev_v = vix_level_values[i - 5]
                if curr_v is not None and prev_v is not None and prev_v > 0:
                    vix_change_values[i] = (curr_v - prev_v) / prev_v

        return {
            vol_col: vol_values,
            atr_col: atr_values,
            "vix_level": vix_level_values,
            "vix_change_5d": vix_change_values,
        }

    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        is_daily = timeframe == TimeFrame.DAY_1
        vol_col = f"rolling_volatility_{self.vol_window}d" if is_daily else f"rolling_volatility_{self.vol_window}"
        atr_col = f"atr_{self.atr_window}"

        return [
            FeatureMetadata(
                feature_name=vol_col,
                description=f"{self.vol_window}-period rolling return sample volatility (annualized for daily)",
                formula=f"StdDev(return_1d, window={self.vol_window}) * sqrt(252)",
                window=self.vol_window,
                source_columns=["close"],
                timeframe=timeframe,
            ),
            FeatureMetadata(
                feature_name=atr_col,
                description=f"{self.atr_window}-period Average True Range using Wilder smoothing",
                formula=f"Wilder_EMA(max(H-L, |H-C_prev|, |L-C_prev|), window={self.atr_window})",
                window=self.atr_window,
                source_columns=["high", "low", "close"],
                timeframe=timeframe,
            ),
            FeatureMetadata(
                feature_name="vix_level",
                description="Macro VIX index closing level aligned to bar timestamp",
                formula="VIX_Close[t]",
                window=1,
                source_columns=["vix_close"],
                timeframe=timeframe,
            ),
            FeatureMetadata(
                feature_name="vix_change_5d",
                description="5-period percentage change in VIX index level",
                formula="(VIX[t] - VIX[t-5]) / VIX[t-5]",
                window=5,
                source_columns=["vix_close"],
                timeframe=timeframe,
            ),
        ]
