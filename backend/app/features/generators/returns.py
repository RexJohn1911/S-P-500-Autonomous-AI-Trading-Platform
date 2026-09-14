"""
Return and Momentum Feature Generator
Computes multi-horizon returns strictly using historical prices.
"""

from typing import Any, Dict, List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.models import FeatureMetadata


class ReturnFeatureGenerator(BaseFeatureGenerator):
    """
    Calculates 1-period, 5-period, and 20-period price returns.
    Formula: Return(N) = (Close[t] - Close[t-N]) / Close[t-N]
    """

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or [1, 5, 20]

    @property
    def name(self) -> str:
        return "return_features"

    @property
    def description(self) -> str:
        return f"Calculates multi-horizon returns over windows {self.windows}."

    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        n = len(bars)
        results: Dict[str, List[Optional[float]]] = {}

        for w in self.windows:
            col_name = f"return_{w}d" if bars and bars[0].timeframe == TimeFrame.DAY_1 else f"return_{w}"
            col_values: List[Optional[float]] = [None] * n

            for i in range(w, n):
                prev_close = bars[i - w].close
                curr_close = bars[i].close
                if prev_close > 0:
                    col_values[i] = (curr_close - prev_close) / prev_close
                else:
                    col_values[i] = None

            results[col_name] = col_values

        return results

    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        meta = []
        for w in self.windows:
            col_name = f"return_{w}d" if timeframe == TimeFrame.DAY_1 else f"return_{w}"
            meta.append(
                FeatureMetadata(
                    feature_name=col_name,
                    description=f"{w}-period historical price return",
                    formula=f"(close[t] - close[t-{w}]) / close[t-{w}]",
                    window=w,
                    source_columns=["close"],
                    timeframe=timeframe,
                )
            )
        return meta
