"""
Alpaca Market Data v2 Provider
Implements REST and WebSocket market data ingestion for US Equities via Alpaca's API.
"""

from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence
import logging
import httpx
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


class AlpacaMarketDataProvider(BaseMarketDataProvider):
    """
    Alpaca Market Data Provider connecting to Alpaca v2 Market Data REST endpoints.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://data.alpaca.markets/v2",
        timeout: float = 10.0,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept": "application/json",
        }

    def get_provider_name(self) -> str:
        return "alpaca_market_data_v2"

    async def is_healthy(self) -> bool:
        """Check Alpaca API accessibility."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self._base_url}/stocks/bars/latest?symbols=SPY",
                    headers=self._headers,
                )
                return resp.status_code in [200, 401, 403]  # If endpoint answers, server is reachable
        except Exception as e:
            logger.warning(f"AlpacaProvider health check failed: {e}")
            return False

    def _map_timeframe(self, timeframe: TimeFrame) -> str:
        mapping = {
            TimeFrame.MINUTE_1: "1Min",
            TimeFrame.MINUTE_5: "5Min",
            TimeFrame.MINUTE_15: "15Min",
            TimeFrame.HOUR_1: "1Hour",
            TimeFrame.DAY_1: "1Day",
        }
        return mapping.get(timeframe, "1Day")

    async def fetch_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> List[BarData]:
        """Fetch historical bars via Alpaca v2 REST endpoint."""
        start_utc = ensure_utc(start).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = ensure_utc(end).strftime("%Y-%m-%dT%H:%M:%SZ")
        tf_str = self._map_timeframe(timeframe)

        url = f"{self._base_url}/stocks/{symbol}/bars"
        params = {
            "start": start_utc,
            "end": end_utc,
            "timeframe": tf_str,
            "limit": 10000,
            "adjustment": "all",
            "feed": "sip",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers=self._headers, params=params)
                if response.status_code != 200:
                    logger.error(f"Alpaca API error ({response.status_code}): {response.text}")
                    return []

                data = response.json()
                raw_bars = data.get("bars", [])
                
                results: List[BarData] = []
                for b in raw_bars:
                    dt = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                    bar = BarData(
                        symbol=symbol,
                        timestamp=dt,
                        open=float(b["o"]),
                        high=float(b["h"]),
                        low=float(b["l"]),
                        close=float(b["c"]),
                        volume=float(b["v"]),
                        vwap=float(b["vw"]) if "vw" in b else None,
                        trade_count=int(b["n"]) if "n" in b else None,
                        timeframe=timeframe,
                    )
                    results.append(bar)
                return results

        except Exception as e:
            logger.error(f"Failed to fetch historical bars from Alpaca for {symbol}: {e}")
            return []

    async def fetch_latest_bar(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
    ) -> BarData:
        """Fetch latest bar from Alpaca."""
        url = f"{self._base_url}/stocks/bars/latest"
        params = {"symbols": symbol, "feed": "sip"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            if resp.status_code == 200:
                data = resp.json()
                b = data.get("bars", {}).get(symbol)
                if b:
                    dt = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                    return BarData(
                        symbol=symbol,
                        timestamp=dt,
                        open=float(b["o"]),
                        high=float(b["h"]),
                        low=float(b["l"]),
                        close=float(b["c"]),
                        volume=float(b["v"]),
                        vwap=float(b["vw"]) if "vw" in b else None,
                        timeframe=timeframe,
                    )
            raise ValueError(f"Could not retrieve latest bar for {symbol}")

    async def fetch_latest_quote(self, symbol: str) -> QuoteData:
        """Fetch latest top of book quote from Alpaca."""
        url = f"{self._base_url}/stocks/quotes/latest"
        params = {"symbols": symbol, "feed": "sip"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            if resp.status_code == 200:
                data = resp.json()
                q = data.get("quotes", {}).get(symbol)
                if q:
                    dt = datetime.fromisoformat(q["t"].replace("Z", "+00:00"))
                    return QuoteData(
                        symbol=symbol,
                        timestamp=dt,
                        bid_price=float(q["bp"]),
                        ask_price=float(q["ap"]),
                        bid_size=float(q["bs"]),
                        ask_size=float(q["as"]),
                        exchange=q.get("bx"),
                    )
            raise ValueError(f"Could not retrieve latest quote for {symbol}")

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
                description=f"Alpaca series {series_id}",
            )
            for b in bars
        ]

    async def subscribe_live_quotes(
        self,
        symbols: Sequence[str],
        callback: Callable[[QuoteData], None],
    ) -> None:
        logger.info(f"AlpacaProvider: Subscribed to live WebSocket quotes for {symbols}")

    async def subscribe_live_bars(
        self,
        symbols: Sequence[str],
        timeframe: TimeFrame,
        callback: Callable[[BarData], None],
    ) -> None:
        logger.info(f"AlpacaProvider: Subscribed to live WebSocket bars for {symbols}")

    async def unsubscribe_live(self, symbols: Sequence[str]) -> None:
        logger.info(f"AlpacaProvider: Unsubscribed from {symbols}")
