"""
Market Data Provider Abstraction Layer
Defines abstract contracts for historical and live real-time market data providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional, Sequence
import logging
from backend.app.data.models import BarData, QuoteData, TickData, MacroData, TimeFrame

logger = logging.getLogger(__name__)


class BaseMarketDataProvider(ABC):
    """
    Abstract Base Class for Market Data Providers.
    Encapsulates network communication, parsing, retries, and data normalization.
    """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the unique name/identifier of this data provider."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check provider connection status and API reachability."""
        pass

    @abstractmethod
    async def fetch_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> List[BarData]:
        """
        Fetch historical OHLCV bars for a given symbol and time window.
        Must return normalized, sorted BarData instances in UTC.
        """
        pass

    @abstractmethod
    async def fetch_latest_bar(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> BarData:
        """Fetch the single most recent completed or forming bar."""
        pass

    @abstractmethod
    async def fetch_latest_quote(self, symbol: str) -> QuoteData:
        """Fetch current top-of-book bid/ask quote for a symbol."""
        pass

    @abstractmethod
    async def fetch_macro_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> List[MacroData]:
        """Fetch macroeconomic / benchmark series (e.g. VIX, TNX, SPY)."""
        pass

    # Live Streaming Subscriptions
    @abstractmethod
    async def subscribe_live_quotes(
        self,
        symbols: Sequence[str],
        callback: Callable[[QuoteData], None],
    ) -> None:
        """Subscribe to real-time quote updates for specified symbols."""
        pass

    @abstractmethod
    async def subscribe_live_bars(
        self,
        symbols: Sequence[str],
        timeframe: TimeFrame,
        callback: Callable[[BarData], None],
    ) -> None:
        """Subscribe to real-time bar completion updates for specified symbols."""
        pass

    @abstractmethod
    async def unsubscribe_live(self, symbols: Sequence[str]) -> None:
        """Unsubscribe from live data feeds for specified symbols."""
        pass
