"""
Volume Activity & Standardized Score Feature Generator
Computes volume percentage changes and rolling volume z-scores.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.models import FeatureMetadata


class VolumeFeatureGenerator(BaseFeatureGenerator):
    """
    Calculates volume changes and rolling volume Z-scores.
    """

    def __init__(self, zscore_window: int = 20):
        self.zscore_window = zscore_window

    @property
    def name(self) -> str:
        return "volume_features"

    @property
    def description(self) -> str:
        return f"Calculates 1-period volume change and rolling volume Z-score ({self.zscore_window})."

    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        n = len(bars)
        is_daily = bars[0].timeframe == TimeFrame.DAY_1 if bars else True
        chg_col = "volume_change_1d" if is_daily else "volume_change_1"
        zscore_col = f"volume_zscore_{self.zscore_window}d" if is_daily else f"volume_zscore_{self.zscore_window}"

        chg_vals: List[Optional[float]] = [None] * n
        zscore_vals: List[Optional[float]] = [None] * n

        volumes = [b.volume for b in bars]

        # 1. 1-Period Volume Change
        for i in range(1, n):
            prev_v = volumes[i - 1]
            curr_v = volumes[i]
            if prev_v is not None and prev_v > 0 and curr_v is not None:
                chg_vals[i] = (curr_v - prev_v) / prev_v
            else:
                chg_vals[i] = None

        # 2. Rolling Volume Z-Score
        for i in range(self.zscore_window - 1, n):
            win_v = volumes[i - self.zscore_window + 1 : i + 1]
            mean_v = float(np.mean(win_v))
            std_v = float(np.std(win_v, ddof=1)) if len(win_v) > 1 else 0.0

            if std_v > 1e-4:
                zscore_vals[i] = (volumes[i] - mean_v) / std_v
            else:
                zscore_vals[i] = 0.0

        return {
            chg_col: chg_vals,
            zscore_col: zscore_vals,
        }

    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        is_daily = timeframe == TimeFrame.DAY_1
        chg_col = "volume_change_1d" if is_daily else "volume_change_1"
        zscore_col = f"volume_zscore_{self.zscore_window}d" if is_daily else f"volume_zscore_{self.zscore_window}"

        return [
            FeatureMetadata(
                feature_name=chg_col,
                description="1-period percentage change in trading volume",
                formula="(volume[t] - volume[t-1]) / volume[t-1]",
                window=1,
                source_columns=["volume"],
                timeframe=timeframe,
            ),
            FeatureMetadata(
                feature_name=zscore_col,
                description=f"{self.zscore_window}-period rolling standardized volume score",
                formula=f"(volume[t] - Mean(volume, {self.zscore_window})) / StdDev(volume, {self.zscore_window})",
                window=self.zscore_window,
                source_columns=["volume"],
                timeframe=timeframe,
            ),
        ]
