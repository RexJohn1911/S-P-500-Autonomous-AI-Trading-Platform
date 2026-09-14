"""
Modular Feature Generators.
"""

from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.generators.returns import ReturnFeatureGenerator
from backend.app.features.generators.volatility import VolatilityFeatureGenerator
from backend.app.features.generators.trend import TrendFeatureGenerator
from backend.app.features.generators.volume import VolumeFeatureGenerator
from backend.app.features.generators.market_relative import MarketRelativeFeatureGenerator
from backend.app.features.generators.beta import BetaFeatureGenerator
from backend.app.features.generators.sector import SectorFeatureGenerator
from backend.app.features.generators.regime import RegimeFeaturePlaceholder

__all__ = [
    "BaseFeatureGenerator",
    "ReturnFeatureGenerator",
    "VolatilityFeatureGenerator",
    "TrendFeatureGenerator",
    "VolumeFeatureGenerator",
    "MarketRelativeFeatureGenerator",
    "BetaFeatureGenerator",
    "SectorFeatureGenerator",
    "RegimeFeaturePlaceholder",
]
