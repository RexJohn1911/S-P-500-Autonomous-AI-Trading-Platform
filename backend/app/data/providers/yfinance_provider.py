"""
Yahoo Finance Public Market Data Provider
Fetches market data, benchmark indices, Treasury yields (^TNX), and VIX (^VIX) via Yahoo Finance API.
"""

from datetime import datetime
from typing import Callable, List, Optional, Sequence
import logging
import httpx
from backend.app.data.base import BaseMarketDataProvider
from backend.app.data.models import (
    BarData,
    QuoteData,
    MacroData,
    TimeFrame,
    AssetClass,
    ensure_utc,
)

logger = logging.getLogger(__name__)


class YahooFinanceDataProvider(BaseMarketDataProvider):
    """
    Yahoo Finance Public API Provider for historical quotes, macro indicators, and ETFs.
    """

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout
        self._base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

    def get_provider_name(self) -> str:
        return "yahoo_finance_public"

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}/SPY?interval=1d&range=1d", headers=self._headers)
                return resp.status_code == 200
        except Exception as e:
            logger.warning(f"YahooFinance provider health check failed: {e}")
            return False

    def _map_interval(self, timeframe: TimeFrame) -> str:
        mapping = {
            TimeFrame.MINUTE_1: "1m",
            TimeFrame.MINUTE_5: "5m",
            TimeFrame.MINUTE_15: "15m",
            TimeFrame.HOUR_1: "60m",
            TimeFrame.DAY_1: "1d",
        }
        return mapping.get(timeframe, "1d")

    async def fetch_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> List[BarData]:
        start_ts = int(ensure_utc(start).timestamp())
        end_ts = int(ensure_utc(end).timestamp())
        interval = self._map_interval(timeframe)

        url = f"{self._base_url}/{symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers, params=params)
                if resp.status_code != 200:
                    logger.error(f"Yahoo Finance error ({resp.status_code}) for {symbol}")
                    return []

                data = resp.json()
                result = data.get("chart", {}).get("result")
                if not result:
                    return []

                chart_data = result[0]
                timestamps = chart_data.get("timestamp", [])
                indicators = chart_data.get("indicators", {}).get("quote", [{}])[0]

                opens = indicators.get("open", [])
                highs = indicators.get("high", [])
                lows = indicators.get("low", [])
                closes = indicators.get("close", [])
                volumes = indicators.get("volume", [])

                bars: List[BarData] = []
                for i in range(len(timestamps)):
                    # Check for null values
                    if (
                        opens[i] is None
                        or highs[i] is None
                        or lows[i] is None
                        or closes[i] is None
                    ):
                        continue

                    dt = datetime.fromtimestamp(timestamps[i], tz=datetime.now().astimezone().tzinfo)
                    
                    bar = BarData(
                        symbol=symbol,
                        timestamp=ensure_utc(dt),
                        open=float(opens[i]),
                        high=float(highs[i]),
                        low=float(lows[i]),
                        close=float(closes[i]),
                        volume=float(volumes[i] or 0.0),
                        timeframe=timeframe,
                    )
                    bars.append(bar)

                return bars

        except Exception as e:
            logger.error(f"YahooFinance fetch error for {symbol}: {e}")
            return []

    async def fetch_latest_bar(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> BarData:
        bars = await self.fetch_historical_bars(
            symbol=symbol,
            start=datetime.now() - datetime.timedelta(days=5),
            end=datetime.now(),
            timeframe=timeframe,
        )
        if bars:
            return bars[-1]
        raise ValueError(f"No bars returned for {symbol}")

    async def fetch_latest_quote(self, symbol: str) -> QuoteData:
        bar = await self.fetch_latest_bar(symbol, TimeFrame.DAY_1)
        return QuoteData(
            symbol=symbol,
            timestamp=bar.timestamp,
            bid_price=bar.close,
            ask_price=bar.close,
            bid_size=100.0,
            ask_size=100.0,
            exchange="YAHOO",
        )

    async def fetch_macro_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> List[MacroData]:
        bars = await self.fetch_historical_bars(series_id, start, end, TimeFrame.DAY_1)
        return [
            MacroData(
                series_id=series_id,
                timestamp=b.timestamp,
                value=b.close,
                description=f"Yahoo Finance {series_id}",
            )
            for b in bars
        ]

    async def subscribe_live_quotes(
        self,
        symbols: Sequence[str],
        callback: Callable[[QuoteData], None],
    ) -> None:
        logger.info(f"YahooFinanceProvider: Real-time subscription not supported natively (polling fallback).")

    async def subscribe_live_bars(
        self,
        symbols: Sequence[str],
        timeframe: TimeFrame,
        callback: Callable[[BarData], None],
    ) -> None:
        logger.info(f"YahooFinanceProvider: Real-time bar subscription not supported natively.")

    async def unsubscribe_live(self, symbols: Sequence[str]) -> None:
        pass
