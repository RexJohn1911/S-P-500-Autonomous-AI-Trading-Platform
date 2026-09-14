"""
Market Regime Feature Placeholder
Formal architectural placeholder for future Phase 09 (Market Regime Detection Engine).
Explicitly marks regime detection as not implemented in Phase 06.
"""

from typing import Any, Dict, List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.models import FeatureMetadata


class RegimeFeaturePlaceholder(BaseFeatureGenerator):
    """
    Architectural placeholder for future Market Regime feature integration (Phase 09).
    Does NOT calculate or infer market regimes in Phase 06.
    """

    @property
    def name(self) -> str:
        return "regime_feature_placeholder"

    @property
    def description(self) -> str:
        return "Architectural placeholder for future Phase 09 regime classification integration."

    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        # Strictly returns empty dictionary in Phase 06
        return {}

    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        return []
