"""
Mock / Synthetic Market Data Provider
Generates realistic, deterministic multi-asset historical bars, live quotes, and macro series
for offline simulation, backtesting validation, and automated test suites.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Set
import logging
from backend.app.data.base import BaseMarketDataProvider
from backend.app.data.models import (
    BarData,
    QuoteData,
    TickData,
    MacroData,
    TimeFrame,
    AssetClass,
    ensure_utc,
)

logger = logging.getLogger(__name__)

# Base reference prices for key symbols
BASE_PRICES = {
    "SPY": 500.0,
    "^GSPC": 5000.0,
    "^VIX": 15.0,
    "^TNX": 4.25,
    "AAPL": 185.0,
    "MSFT": 420.0,
    "NVDA": 120.0,
    "AMZN": 180.0,
    "GOOGL": 175.0,
    "META": 490.0,
    "TSLA": 220.0,
    "XLK": 210.0,
    "XLF": 42.0,
    "XLE": 90.0,
}


class MockMarketDataProvider(BaseMarketDataProvider):
    """
    Deterministic Synthetic Market Data Provider utilizing Geometric Brownian Motion.
    """

    def __init__(self, seed: int = 42, drift: float = 0.0002, volatility: float = 0.015):
        self._seed = seed
        self._drift = drift
        self._volatility = volatility
        self._subscribed_quotes: Set[str] = set()
        self._subscribed_bars: Set[str] = set()
        self._streaming_active: bool = False
        self._quote_callbacks: List[Callable[[QuoteData], None]] = []
        self._bar_callbacks: List[Callable[[BarData], None]] = []
        self._latest_prices: Dict[str, float] = dict(BASE_PRICES)

    def get_provider_name(self) -> str:
        return "mock_synthetic_provider"

    async def is_healthy(self) -> bool:
        return True

    def _get_timeframe_delta(self, timeframe: TimeFrame) -> timedelta:
        mapping = {
            TimeFrame.MINUTE_1: timedelta(minutes=1),
            TimeFrame.MINUTE_5: timedelta(minutes=5),
            TimeFrame.MINUTE_15: timedelta(minutes=15),
            TimeFrame.HOUR_1: timedelta(hours=1),
            TimeFrame.DAY_1: timedelta(days=1),
        }
        return mapping.get(timeframe, timedelta(days=1))

    async def fetch_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> List[BarData]:
        """Generate deterministic synthetic historical OHLCV bars."""
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if start_utc >= end_utc:
            return []

        rng = random.Random(f"{self._seed}_{symbol}_{timeframe.value}")
        delta = self._get_timeframe_delta(timeframe)
        current_dt = start_utc
        price = self._latest_prices.get(symbol, 100.0)

        bars: List[BarData] = []
        
        while current_dt <= end_utc:
            # Skip weekends for daily bars
            if timeframe == TimeFrame.DAY_1 and current_dt.weekday() >= 5:
                current_dt += delta
                continue

            # Geometric Brownian step
            shock = rng.gauss(0, 1)
            ret = self._drift + self._volatility * shock
            open_p = price
            close_p = round(max(1.0, open_p * (1.0 + ret)), 2)
            
            # Intra-bar excursion
            intra_vol = self._volatility * abs(rng.gauss(0, 1))
            high_p = round(max(open_p, close_p) * (1.0 + intra_vol * 0.75), 2)
            low_p = round(max(0.5, min(open_p, close_p) * (1.0 - intra_vol * 0.75)), 2)
            volume = round(max(1000.0, rng.lognormvariate(12, 1.2)), 0)
            vwap = round((open_p + high_p + low_p + close_p) / 4.0, 2)

            asset_class = AssetClass.VOLATILITY if "VIX" in symbol else (
                AssetClass.RATES if "TNX" in symbol else (
                    AssetClass.ETF if symbol.startswith("XL") or symbol == "SPY" else AssetClass.EQUITY
                )
            )

            bar = BarData(
                symbol=symbol,
                timestamp=current_dt,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=volume,
                vwap=vwap,
                trade_count=int(volume / 100),
                timeframe=timeframe,
                asset_class=asset_class,
            )
            bars.append(bar)
            price = close_p
            current_dt += delta

        if bars:
            self._latest_prices[symbol] = bars[-1].close

        return bars

    async def fetch_latest_bar(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> BarData:
        now = datetime.now(timezone.utc)
        delta = self._get_timeframe_delta(timeframe)
        bars = await self.fetch_historical_bars(
            symbol=symbol,
            start=now - delta * 2,
            end=now,
            timeframe=timeframe,
        )
        if bars:
            return bars[-1]
        
        price = self._latest_prices.get(symbol, 100.0)
        return BarData(
            symbol=symbol,
            timestamp=now,
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=50000.0,
            vwap=price,
            timeframe=timeframe,
        )

    async def fetch_latest_quote(self, symbol: str) -> QuoteData:
        now = datetime.now(timezone.utc)
        price = self._latest_prices.get(symbol, 100.0)
        spread = round(max(0.01, price * 0.0005), 2)
        bid = round(price - spread / 2.0, 2)
        ask = round(price + spread / 2.0, 2)
        return QuoteData(
            symbol=symbol,
            timestamp=now,
            bid_price=bid,
            ask_price=ask,
            bid_size=500.0,
            ask_size=500.0,
            exchange="MOCK_EXCHANGE",
        )

    async def fetch_macro_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> List[MacroData]:
        bars = await self.fetch_historical_bars(series_id, start, end, TimeFrame.DAY_1)
        macro_series: List[MacroData] = []
        for b in bars:
            macro_series.append(
                MacroData(
                    series_id=series_id,
                    timestamp=b.timestamp,
                    value=b.close,
                    description=f"Synthetic macro series for {series_id}",
                    asset_class=b.asset_class,
                )
            )
        return macro_series

    async def subscribe_live_quotes(
        self,
        symbols: Sequence[str],
        callback: Callable[[QuoteData], None],
    ) -> None:
        self._subscribed_quotes.update(symbols)
        self._quote_callbacks.append(callback)
        logger.info(f"MockProvider: Subscribed to live quotes for {list(symbols)}")

    async def subscribe_live_bars(
        self,
        symbols: Sequence[str],
        timeframe: TimeFrame,
        callback: Callable[[BarData], None],
    ) -> None:
        self._subscribed_bars.update(symbols)
        self._bar_callbacks.append(callback)
        logger.info(f"MockProvider: Subscribed to live {timeframe.value} bars for {list(symbols)}")

    async def unsubscribe_live(self, symbols: Sequence[str]) -> None:
        for s in symbols:
            self._subscribed_quotes.discard(s)
            self._subscribed_bars.discard(s)
        logger.info(f"MockProvider: Unsubscribed from {list(symbols)}")
