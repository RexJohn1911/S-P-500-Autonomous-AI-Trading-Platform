"""
Market Data Service Orchestrator
Coordinates multi-provider market data fetching, raw storage persistence, retries,
and subscription management for equities, benchmark indices, and macro indicators.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence
import logging
from backend.app.config.settings import get_settings
from backend.app.data.base import BaseMarketDataProvider
from backend.app.data.models import (
    BarData,
    QuoteData,
    MacroData,
    TimeFrame,
    ensure_utc,
)
from backend.app.data.providers.mock_provider import MockMarketDataProvider
from backend.app.data.providers.alpaca_provider import AlpacaMarketDataProvider
from backend.app.data.providers.yfinance_provider import YahooFinanceDataProvider
from backend.app.data.storage import RawDataStorage

logger = logging.getLogger(__name__)

# Key Market Macro & Benchmark Symbols
BENCHMARK_SYMBOLS = {
    "SPY": "S&P 500 ETF Trust",
    "^GSPC": "S&P 500 Index",
    "^VIX": "CBOE Volatility Index",
    "^TNX": "10-Year Treasury Yield Note",
}

SECTOR_ETFS = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Healthcare
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLU",  # Utilities
    "XLRE", # Real Estate
    "XLC",  # Communication Services
]


class MarketDataService:
    """
    High-level Market Data Service coordinating data provider access, caching,
    persistence, and resilient ingestion.
    """

    def __init__(
        self,
        provider: Optional[BaseMarketDataProvider] = None,
        storage: Optional[RawDataStorage] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        settings = get_settings()
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._storage = storage or RawDataStorage()

        if provider is not None:
            self._provider = provider
        else:
            # Configure provider based on settings
            provider_type = settings.DATA_PROVIDER.lower()
            if provider_type == "alpaca" and settings.DATA_PROVIDER_API_KEY != "dev_data_provider_key":
                self._provider = AlpacaMarketDataProvider(
                    api_key=settings.DATA_PROVIDER_API_KEY,
                    secret_key=settings.BROKER_SECRET_KEY,
                )
            elif provider_type == "yahoo":
                self._provider = YahooFinanceDataProvider()
            else:
                # Default to Mock Synthetic Provider for local development & testing
                self._provider = MockMarketDataProvider()

        logger.info(f"MarketDataService initialized with provider: {self._provider.get_provider_name()}")

    @property
    def provider(self) -> BaseMarketDataProvider:
        return self._provider

    @property
    def storage(self) -> RawDataStorage:
        return self._storage

    async def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        persist_raw: bool = True,
    ) -> List[BarData]:
        """
        Fetch historical bars with automatic retries and raw data persistence.
        """
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)

        last_exception = None
        for attempt in range(1, self._max_retries + 1):
            try:
                bars = await self._provider.fetch_historical_bars(
                    symbol=symbol,
                    start=start_utc,
                    end=end_utc,
                    timeframe=timeframe,
                )
                if bars and persist_raw:
                    self._storage.save_bars(bars, format="json")
                return bars
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt}/{self._max_retries} failed fetching bars for {symbol}: {e}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay * (2 ** (attempt - 1)))

        logger.error(f"Failed to fetch historical bars for {symbol} after {self._max_retries} attempts")
        if last_exception:
            raise last_exception
        return []

    async def get_latest_quote(self, symbol: str) -> QuoteData:
        """Fetch latest quote with retry resilience."""
        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._provider.fetch_latest_quote(symbol)
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{self._max_retries} failed fetching quote for {symbol}: {e}")
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay)
        raise RuntimeError(f"Could not retrieve quote for {symbol}")

    async def get_macro_market_context(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, List[MacroData]]:
        """
        Fetch core macroeconomic and market context series (SPY, VIX, 10Y Yield).
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=30)

        results: Dict[str, List[MacroData]] = {}
        for symbol in ["SPY", "^VIX", "^TNX"]:
            try:
                macro_series = await self._provider.fetch_macro_series(symbol, start, end)
                results[symbol] = macro_series
            except Exception as e:
                logger.error(f"Failed to fetch macro series for {symbol}: {e}")
                results[symbol] = []

        return results

    async def batch_fetch_historical_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> Dict[str, List[BarData]]:
        """
        Fetch historical bars concurrently for multiple symbols.
        """
        tasks = [
            self.get_historical_bars(sym, start, end, timeframe, persist_raw=True)
            for sym in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_map: Dict[str, List[BarData]] = {}
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.error(f"Error in batch fetch for {sym}: {res}")
                data_map[sym] = []
            else:
                data_map[sym] = res
        return data_map

    async def subscribe_realtime_quotes(
        self,
        symbols: Sequence[str],
        callback: Callable[[QuoteData], None],
    ) -> None:
        """Subscribe to real-time quotes."""
        await self._provider.subscribe_live_quotes(symbols, callback)

    async def check_health(self) -> bool:
        """Check status of active data provider."""
        return await self._provider.is_healthy()
