"""
Automated unit tests for Market Data Layer (Phase 04).
Verifies provider abstractions, normalized models, timezone handling, raw storage, and retry logic.
"""

from datetime import datetime, timedelta, timezone
import shutil
import tempfile
from pathlib import Path
import pytest

from backend.app.data.models import (
    BarData,
    QuoteData,
    TickData,
    MacroData,
    TimeFrame,
    AssetClass,
    ensure_utc,
    UTC,
    NY_TZ,
)
from backend.app.data.base import BaseMarketDataProvider
from backend.app.data.providers.mock_provider import MockMarketDataProvider
from backend.app.data.storage import RawDataStorage
from backend.app.data.service import MarketDataService, BENCHMARK_SYMBOLS, SECTOR_ETFS


# ==========================================
# 1. Models & Timezone Normalization Tests
# ==========================================

def test_bar_data_creation_and_validation():
    """Verify BarData validation and UTC conversion."""
    now_naive = datetime(2026, 9, 1, 9, 30)
    bar = BarData(
        symbol="AAPL",
        timestamp=now_naive,
        open=150.0,
        high=155.0,
        low=149.0,
        close=154.0,
        volume=1000000.0,
        vwap=152.5,
        timeframe=TimeFrame.DAY_1,
    )

    assert bar.timestamp.tzinfo == UTC
    assert bar.open == 150.0
    assert bar.high == 155.0
    assert bar.low == 149.0
    assert bar.close == 154.0
    assert bar.volume == 1000000.0
    assert bar.vwap == 152.5

    # Check NY timestamp conversion
    ny_time = bar.ny_timestamp
    assert ny_time.tzinfo == NY_TZ

    # Check serialization roundtrip
    d = bar.to_dict()
    reconstructed = BarData.from_dict(d)
    assert reconstructed.symbol == bar.symbol
    assert reconstructed.timestamp == bar.timestamp
    assert reconstructed.close == bar.close


def test_bar_data_invalid_constraints():
    """Verify BarData raises ValueError for invalid price or volume relationships."""
    now = datetime.now(UTC)
    
    # Low price exceeds high price
    with pytest.raises(ValueError, match="cannot exceed high price"):
        BarData(
            symbol="AAPL",
            timestamp=now,
            open=150.0,
            high=140.0,
            low=145.0,
            close=142.0,
            volume=1000.0,
        )

    # Negative price
    with pytest.raises(ValueError, match="must be non-negative"):
        BarData(
            symbol="AAPL",
            timestamp=now,
            open=-10.0,
            high=10.0,
            low=5.0,
            close=8.0,
            volume=1000.0,
        )

    # Negative volume
    with pytest.raises(ValueError, match="volume must be non-negative"):
        BarData(
            symbol="AAPL",
            timestamp=now,
            open=10.0,
            high=15.0,
            low=9.0,
            close=12.0,
            volume=-50.0,
        )


def test_quote_data_midpoint_and_spread():
    """Verify QuoteData calculates spread and mid price accurately."""
    now = datetime.now(UTC)
    quote = QuoteData(
        symbol="MSFT",
        timestamp=now,
        bid_price=419.50,
        ask_price=420.50,
        bid_size=100.0,
        ask_size=200.0,
    )
    assert quote.mid_price == 420.00
    assert round(quote.spread, 2) == 1.00
    assert quote.timestamp.tzinfo == UTC


# ==========================================
# 2. Mock Provider & Abstraction Tests
# ==========================================

@pytest.mark.asyncio
async def test_mock_provider_historical_generation():
    """Verify MockMarketDataProvider generates deterministic multi-day bars."""
    provider = MockMarketDataProvider(seed=123)
    assert isinstance(provider, BaseMarketDataProvider)
    assert await provider.is_healthy() is True

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 15, tzinfo=UTC)

    bars = await provider.fetch_historical_bars("SPY", start, end, TimeFrame.DAY_1)
    assert len(bars) > 0
    assert bars[0].symbol == "SPY"
    for b in bars:
        assert b.high >= b.low
        assert b.volume > 0
        assert b.timestamp >= start
        assert b.timestamp <= end

    # Test latest bar and latest quote
    latest_bar = await provider.fetch_latest_bar("NVDA")
    assert latest_bar.symbol == "NVDA"
    assert latest_bar.close > 0

    latest_quote = await provider.fetch_latest_quote("NVDA")
    assert latest_quote.symbol == "NVDA"
    assert latest_quote.ask_price >= latest_quote.bid_price


@pytest.mark.asyncio
async def test_mock_provider_macro_series():
    """Verify macro series generation for VIX and Treasury yield."""
    provider = MockMarketDataProvider(seed=456)
    start = datetime(2026, 5, 1, tzinfo=UTC)
    end = datetime(2026, 5, 10, tzinfo=UTC)

    vix_series = await provider.fetch_macro_series("^VIX", start, end)
    assert len(vix_series) > 0
    assert vix_series[0].series_id == "^VIX"
    assert vix_series[0].value > 0


# ==========================================
# 3. Raw Data Storage Persistence Tests
# ==========================================

def test_raw_data_storage_json_and_parquet():
    """Verify saving and loading raw bar data in JSON, CSV, and Parquet."""
    temp_dir = tempfile.mkdtemp()
    try:
        storage = RawDataStorage(base_storage_dir=Path(temp_dir))
        
        now = datetime(2026, 8, 1, tzinfo=UTC)
        bars = [
            BarData(
                symbol="GOOGL",
                timestamp=now + timedelta(days=i),
                open=170.0 + i,
                high=175.0 + i,
                low=169.0 + i,
                close=173.0 + i,
                volume=50000.0,
                timeframe=TimeFrame.DAY_1,
            )
            for i in range(5)
        ]

        # Test JSON storage
        json_path = storage.save_bars(bars, format="json")
        assert json_path.exists()
        loaded_json = storage.load_bars("GOOGL", TimeFrame.DAY_1, format="json")
        assert len(loaded_json) == 5
        assert loaded_json[0].symbol == "GOOGL"
        assert loaded_json[0].close == 173.0

        # Test CSV storage
        csv_path = storage.save_bars(bars, format="csv")
        assert csv_path.exists()
        loaded_csv = storage.load_bars("GOOGL", TimeFrame.DAY_1, format="csv")
        assert len(loaded_csv) == 5

        # Test Parquet storage
        parquet_path = storage.save_bars(bars, format="parquet")
        assert parquet_path.exists()
        loaded_parquet = storage.load_bars("GOOGL", TimeFrame.DAY_1, format="parquet")
        assert len(loaded_parquet) == 5

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==========================================
# 4. Market Data Service Orchestration Tests
# ==========================================

@pytest.mark.asyncio
async def test_market_data_service_historical_and_macro():
    """Verify MarketDataService orchestration, persistence, and batch fetching."""
    temp_dir = tempfile.mkdtemp()
    try:
        storage = RawDataStorage(base_storage_dir=Path(temp_dir))
        provider = MockMarketDataProvider(seed=999)
        service = MarketDataService(provider=provider, storage=storage)

        start = datetime(2026, 7, 1, tzinfo=UTC)
        end = datetime(2026, 7, 10, tzinfo=UTC)

        # Ingest SPY
        bars = await service.get_historical_bars("SPY", start, end, TimeFrame.DAY_1, persist_raw=True)
        assert len(bars) > 0
        
        # Verify persistence on disk
        stored_bars = storage.load_bars("SPY", TimeFrame.DAY_1, format="json")
        assert len(stored_bars) == len(bars)

        # Batch ingestion
        batch = await service.batch_fetch_historical_bars(["AAPL", "MSFT", "XLK"], start, end)
        assert "AAPL" in batch
        assert "MSFT" in batch
        assert "XLK" in batch
        assert len(batch["AAPL"]) > 0

        # Macro market context
        macro_ctx = await service.get_macro_market_context(start, end)
        assert "SPY" in macro_ctx
        assert "^VIX" in macro_ctx
        assert "^TNX" in macro_ctx

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_market_data_service_retry_resilience():
    """Verify that MarketDataService handles intermittent provider exceptions with retries."""
    class FlakyProvider(MockMarketDataProvider):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        async def fetch_historical_bars(self, symbol, start, end, timeframe=TimeFrame.DAY_1):
            self.attempts += 1
            if self.attempts < 3:
                raise ConnectionError("Simulated temporary network glitch")
            return await super().fetch_historical_bars(symbol, start, end, timeframe)

    flaky = FlakyProvider()
    service = MarketDataService(
        provider=flaky,
        max_retries=3,
        retry_delay_seconds=0.05,
    )

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 5, tzinfo=UTC)

    bars = await service.get_historical_bars("TSLA", start, end)
    assert flaky.attempts == 3
    assert len(bars) > 0
