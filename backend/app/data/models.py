"""
Market Data Domain Models & Schemas
Provides normalized, immutable, timezone-aware domain entities for all market data types.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
import zoneinfo

# Standard Market Timezones
UTC = timezone.utc
NY_TZ = zoneinfo.ZoneInfo("America/New_York")


class TimeFrame(str, Enum):
    MINUTE_1 = "1Min"
    MINUTE_5 = "5Min"
    MINUTE_15 = "15Min"
    HOUR_1 = "1Hour"
    DAY_1 = "1Day"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    INDEX = "INDEX"
    ETF = "ETF"
    VOLATILITY = "VOLATILITY"
    RATES = "RATES"


class MarketDataType(str, Enum):
    BAR = "BAR"
    QUOTE = "QUOTE"
    TICK = "TICK"
    MACRO = "MACRO"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    ECONOMIC_EVENT = "ECONOMIC_EVENT"
    NEWS = "NEWS"


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime object is timezone-aware and normalized to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class BarData:
    """Normalized OHLCV Bar data for an asset over a discrete timeframe."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    timeframe: TimeFrame = TimeFrame.DAY_1
    asset_class: AssetClass = AssetClass.EQUITY

    def __post_init__(self):
        # Validate prices
        if self.low > self.high:
            raise ValueError(f"BarData low price ({self.low}) cannot exceed high price ({self.high}) for {self.symbol}")
        if self.open < 0 or self.high < 0 or self.low < 0 or self.close < 0:
            raise ValueError(f"BarData prices must be non-negative for {self.symbol}")
        if self.volume < 0:
            raise ValueError(f"BarData volume must be non-negative for {self.symbol}")
        
        # Ensure UTC normalization
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def ny_timestamp(self) -> datetime:
        """Return the timestamp in America/New_York market time."""
        return self.timestamp.astimezone(NY_TZ)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bar to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
            "timeframe": self.timeframe.value,
            "asset_class": self.asset_class.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarData":
        """Construct BarData from a dictionary."""
        import math
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        
        vwap_val = data.get("vwap")
        if vwap_val is not None:
            try:
                if math.isnan(float(vwap_val)):
                    vwap_val = None
                else:
                    vwap_val = float(vwap_val)
            except (ValueError, TypeError):
                vwap_val = None

        trade_count_val = data.get("trade_count")
        if trade_count_val is not None:
            try:
                if math.isnan(float(trade_count_val)):
                    trade_count_val = None
                else:
                    trade_count_val = int(trade_count_val)
            except (ValueError, TypeError):
                trade_count_val = None

        return cls(
            symbol=data["symbol"],
            timestamp=ensure_utc(ts),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            vwap=vwap_val,
            trade_count=trade_count_val,
            timeframe=TimeFrame(data.get("timeframe", TimeFrame.DAY_1.value)),
            asset_class=AssetClass(data.get("asset_class", AssetClass.EQUITY.value)),
        )


@dataclass(frozen=True)
class QuoteData:
    """Normalized Top-of-Book Bid/Ask Quote."""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    exchange: Optional[str] = None

    def __post_init__(self):
        if self.bid_price < 0 or self.ask_price < 0:
            raise ValueError(f"Quote prices must be non-negative for {self.symbol}")
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def mid_price(self) -> float:
        """Calculate midpoint price."""
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return max(0.0, self.ask_price - self.bid_price)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "exchange": self.exchange,
            "mid_price": self.mid_price,
            "spread": self.spread,
        }


@dataclass(frozen=True)
class TickData:
    """Normalized Individual Trade Tick."""
    symbol: str
    timestamp: datetime
    price: float
    size: float
    exchange: Optional[str] = None
    conditions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.price < 0 or self.size < 0:
            raise ValueError(f"Tick price and size must be non-negative for {self.symbol}")
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "size": self.size,
            "exchange": self.exchange,
            "conditions": self.conditions,
        }


@dataclass(frozen=True)
class MacroData:
    """Normalized Macroeconomic / Market Breadth Series (VIX, 10Y Yield, SPX Breadth)."""
    series_id: str
    timestamp: datetime
    value: float
    description: Optional[str] = None
    asset_class: AssetClass = AssetClass.VOLATILITY

    def __post_init__(self):
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "description": self.description,
            "asset_class": self.asset_class.value,
        }


# Metadata schemas for future phases (Corporate actions, Earnings, News)
@dataclass(frozen=True)
class CorporateActionData:
    symbol: str
    timestamp: datetime
    action_type: str  # DIVIDEND, SPLIT, MERGER, SPINOFF
    details: Dict[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))


@dataclass(frozen=True)
class EconomicEventData:
    event_id: str
    name: str
    timestamp: datetime
    actual: Optional[float] = None
    consensus: Optional[float] = None
    previous: Optional[float] = None
    impact: str = "MEDIUM"  # LOW, MEDIUM, HIGH

    def __post_init__(self):
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))


@dataclass(frozen=True)
class NewsArticleData:
    headline: str
    source: str
    timestamp: datetime
    symbols: List[str]
    url: Optional[str] = None
    sentiment_score: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
