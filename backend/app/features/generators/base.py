"""
Base Feature Generator Interface
Defines the abstract contract for all modular, single-purpose feature generators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.models import FeatureMetadata


class BaseFeatureGenerator(ABC):
    """Abstract base class for all feature generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the feature generator."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the feature group."""
        pass

    @abstractmethod
    def generate(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        """
        Compute features strictly using historical information at or before index i.
        Returns a dictionary mapping feature_name -> list of values aligned 1-to-1 with input bars.
        """
        pass

    @abstractmethod
    def get_metadata(self, timeframe: TimeFrame) -> List[FeatureMetadata]:
        """Return list of metadata objects for all features produced by this generator."""
        pass
