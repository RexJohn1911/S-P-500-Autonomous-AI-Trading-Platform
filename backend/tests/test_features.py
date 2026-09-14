"""
Comprehensive Feature Engineering Test Suite (PHASE 06)
Tests return, volatility, trend, volume, relative strength, beta, sector features,
leakage prevention, non-finite handling, determinism, storage round-tripping,
and end-to-end integration with Phase 04 & 05.
"""

import dataclasses
from datetime import datetime, timedelta, timezone
import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from backend.app.data import MockMarketDataProvider
from backend.app.data.models import BarData, TimeFrame, AssetClass
from backend.app.data.service import MarketDataService
from backend.app.data.storage import RawDataStorage
from backend.app.data.validation.service import DataValidationService
from backend.app.data.validation.storage import ProcessedDataStorage
from backend.app.features.engine import FeatureEngine
from backend.app.features.generators.returns import ReturnFeatureGenerator
from backend.app.features.generators.volatility import VolatilityFeatureGenerator
from backend.app.features.generators.trend import TrendFeatureGenerator
from backend.app.features.generators.volume import VolumeFeatureGenerator
from backend.app.features.generators.market_relative import MarketRelativeFeatureGenerator
from backend.app.features.generators.beta import BetaFeatureGenerator
from backend.app.features.generators.sector import SectorFeatureGenerator
from backend.app.features.generators.regime import RegimeFeaturePlaceholder
from backend.app.features.models import FeatureDataset, FeatureMetadata, FeatureRecord
from backend.app.features.storage import FeatureStorage


def generate_synthetic_bars(
    symbol: str = "AAPL",
    count: int = 100,
    start_price: float = 100.0,
    daily_trend: float = 0.001,
    timeframe: TimeFrame = TimeFrame.DAY_1,
    start_dt: datetime = None,
) -> list[BarData]:
    """Helper to generate deterministic, realistic BarData sequences."""
    if start_dt is None:
        start_dt = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)

    bars = []
    current_price = start_price
    for i in range(count):
        ts = start_dt + timedelta(days=i)
        # Deterministic pseudo-random variation
        variation = math.sin(i * 0.5) * 2.0 + (i * daily_trend * start_price)
        close = round(start_price + variation, 4)
        high = round(close + 1.5, 4)
        low = round(close - 1.5, 4)
        open_price = round(close - 0.2, 4)
        volume = 1000000.0 + (i % 10) * 50000.0

        bar = BarData(
            symbol=symbol,
            timestamp=ts,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timeframe=timeframe,
            asset_class=AssetClass.EQUITY,
        )
        bars.append(bar)
    return bars


# =====================================================================
# 1. Return & Momentum Feature Tests
# =====================================================================

def test_return_features_calculation():
    """Verify return_1d, return_5d, return_20d calculations."""
    bars = generate_synthetic_bars(count=30, start_price=100.0)
    gen = ReturnFeatureGenerator(windows=[1, 5, 20])
    features = gen.generate(bars)

    assert "return_1d" in features
    assert "return_5d" in features
    assert "return_20d" in features

    # Index 0: warm-up period, should be None
    assert features["return_1d"][0] is None
    assert features["return_5d"][0] is None
    assert features["return_20d"][0] is None

    # Index 1: return_1d should match (close[1] / close[0] - 1)
    expected_ret_1d = (bars[1].close / bars[0].close) - 1.0
    assert pytest.approx(features["return_1d"][1], rel=1e-5) == expected_ret_1d

    # Index 5: return_5d should match (close[5] / close[0] - 1)
    expected_ret_5d = (bars[5].close / bars[0].close) - 1.0
    assert pytest.approx(features["return_5d"][5], rel=1e-5) == expected_ret_5d

    # Index 20: return_20d should match (close[20] / close[0] - 1)
    expected_ret_20d = (bars[20].close / bars[0].close) - 1.0
    assert pytest.approx(features["return_20d"][20], rel=1e-5) == expected_ret_20d


def test_return_insufficient_bars():
    """Verify return generator handles sequence shorter than window gracefully."""
    bars = generate_synthetic_bars(count=3)
    gen = ReturnFeatureGenerator(windows=[5, 20])
    features = gen.generate(bars)
    assert all(v is None for v in features["return_5d"])
    assert all(v is None for v in features["return_20d"])


# =====================================================================
# 2. Volatility & ATR Tests
# =====================================================================

def test_volatility_and_atr_calculation():
    """Verify rolling volatility and ATR calculations."""
    bars = generate_synthetic_bars(count=50)
    gen = VolatilityFeatureGenerator(vol_window=20, atr_window=14)
    features = gen.generate(bars)

    assert "rolling_volatility_20d" in features
    assert "atr_14" in features
    assert "vix_level" in features

    # Check warm-up
    assert features["rolling_volatility_20d"][15] is None
    assert features["rolling_volatility_20d"][25] is not None
    assert features["rolling_volatility_20d"][25] >= 0.0

    # ATR at index 13+ (14th element) should be positive
    assert features["atr_14"][12] is None
    assert features["atr_14"][13] is not None
    assert features["atr_14"][13] > 0.0

    # VIX without context should be None
    assert all(v is None for v in features["vix_level"])


def test_vix_integration_with_context():
    """Verify VIX feature integration when VIX market context is supplied."""
    bars = generate_synthetic_bars(count=30)
    vix_bars = generate_synthetic_bars(symbol="^VIX", count=30, start_price=18.0)

    gen = VolatilityFeatureGenerator(vol_window=20, atr_window=14)
    features = gen.generate(bars, context={"vix_bars": vix_bars})

    assert features["vix_level"][0] is not None
    assert pytest.approx(features["vix_level"][0], rel=1e-4) == vix_bars[0].close
    assert features["vix_change_5d"][5] is not None


# =====================================================================
# 3. Trend Feature Tests (SMA, EMA, Price vs SMA)
# =====================================================================

def test_trend_features():
    """Verify SMA, EMA, and Price vs SMA across windows."""
    bars = generate_synthetic_bars(count=210)
    gen = TrendFeatureGenerator(windows=[20, 50, 200])
    features = gen.generate(bars)

    assert "sma_20" in features
    assert "sma_50" in features
    assert "sma_200" in features
    assert "ema_20" in features
    assert "price_vs_sma_20" in features

    # SMA 20 warm-up
    assert features["sma_20"][18] is None
    assert features["sma_20"][19] is not None

    # Verify SMA 20 exact value
    expected_sma_20 = sum(b.close for b in bars[:20]) / 20.0
    assert pytest.approx(features["sma_20"][19], rel=1e-5) == expected_sma_20

    # Price vs SMA: (close - SMA) / SMA
    expected_pvs = (bars[19].close - expected_sma_20) / expected_sma_20
    assert pytest.approx(features["price_vs_sma_20"][19], rel=1e-5) == expected_pvs

    # SMA 200 warm-up
    assert features["sma_200"][198] is None
    assert features["sma_200"][199] is not None


# =====================================================================
# 4. Volume Feature Tests
# =====================================================================

def test_volume_features():
    """Verify volume change and volume z-score."""
    bars = generate_synthetic_bars(count=40)
    gen = VolumeFeatureGenerator(zscore_window=20)
    features = gen.generate(bars)

    assert "volume_change_1d" in features
    assert "volume_zscore_20d" in features

    # Index 0 volume change is None
    assert features["volume_change_1d"][0] is None

    # Index 1 volume change
    expected_vol_chg = (bars[1].volume / bars[0].volume) - 1.0
    assert pytest.approx(features["volume_change_1d"][1], rel=1e-5) == expected_vol_chg

    # Volume z-score warm-up
    assert features["volume_zscore_20d"][18] is None
    assert features["volume_zscore_20d"][19] is not None


def test_volume_zero_previous():
    """Verify volume change handles zero previous volume cleanly without ZeroDivisionError."""
    bars = generate_synthetic_bars(count=5)
    bars[0] = dataclasses.replace(bars[0], volume=0.0)
    gen = VolumeFeatureGenerator(zscore_window=20)
    features = gen.generate(bars)
    assert features["volume_change_1d"][1] is None


# =====================================================================
# 5. Market-Relative & Beta Feature Tests
# =====================================================================

def test_market_relative_features():
    """Verify relative strength vs S&P 500 market benchmark."""
    stock_bars = generate_synthetic_bars(symbol="AAPL", count=40, start_price=150.0, daily_trend=0.002)
    spy_bars = generate_synthetic_bars(symbol="SPY", count=40, start_price=500.0, daily_trend=0.001)

    gen = MarketRelativeFeatureGenerator(window=20)

    # Without context
    feat_no_ctx = gen.generate(stock_bars)
    assert all(v is None for v in feat_no_ctx["relative_strength_20d"])

    # With benchmark context
    feat_with_ctx = gen.generate(stock_bars, context={"benchmark_bars": spy_bars})
    assert feat_with_ctx["relative_strength_20d"][25] is not None
    assert feat_with_ctx["market_return_20d"][25] is not None


def test_beta_calculation():
    """Verify historical Beta calculation relative to benchmark."""
    # Synthetic correlated returns
    stock_bars = generate_synthetic_bars(symbol="MSFT", count=100, start_price=200.0)
    spy_bars = generate_synthetic_bars(symbol="SPY", count=100, start_price=500.0)

    gen = BetaFeatureGenerator(window=60)
    features = gen.generate(stock_bars, context={"benchmark_bars": spy_bars})

    assert "beta_60d" in features
    # First 60 bars should be None
    assert features["beta_60d"][59] is None
    # Bar 61+ should be calculated
    assert features["beta_60d"][61] is not None
    assert isinstance(features["beta_60d"][61], float)


def test_beta_zero_market_variance():
    """Verify Beta handles zero market variance (flat market) without ZeroDivisionError."""
    stock_bars = generate_synthetic_bars(symbol="AAPL", count=70, start_price=100.0)
    flat_spy = [
        BarData(
            symbol="SPY",
            timestamp=b.timestamp,
            open=500.0,
            high=500.0,
            low=500.0,
            close=500.0,
            volume=1000.0,
            timeframe=TimeFrame.DAY_1,
            asset_class=AssetClass.EQUITY,
        )
        for b in stock_bars
    ]

    gen = BetaFeatureGenerator(window=60)
    features = gen.generate(stock_bars, context={"benchmark_bars": flat_spy})
    assert features["beta_60d"][65] is None


# =====================================================================
# 6. Sector Relative Feature Tests
# =====================================================================

def test_sector_relative_features():
    """Verify sector relative strength calculations."""
    stock_bars = generate_synthetic_bars(symbol="NVDA", count=35, start_price=120.0)
    sector_bars = generate_synthetic_bars(symbol="XLK", count=35, start_price=220.0)

    gen = SectorFeatureGenerator(window=20)

    # Missing sector context
    feat_missing = gen.generate(stock_bars)
    assert all(v is None for v in feat_missing["sector_relative_strength_20d"])

    # With sector context
    feat_valid = gen.generate(stock_bars, context={"sector_bars": sector_bars})
    assert feat_valid["sector_relative_strength_20d"][25] is not None
    assert feat_valid["sector_return_20d"][25] is not None


# =====================================================================
# 7. Regime Placeholders Test
# =====================================================================

def test_regime_placeholder():
    """Ensure regime placeholders produce None and do not implement Phase 09 prematurely."""
    bars = generate_synthetic_bars(count=20)
    gen = RegimeFeaturePlaceholder()
    features = gen.generate(bars)
    assert features == {}
    assert gen.get_metadata(TimeFrame.DAY_1) == []


# =====================================================================
# 8. Time-Series Alignment Test
# =====================================================================

def test_timestamp_alignment_mismatch():
    """Ensure benchmark alignment handles differing dates gracefully."""
    stock_bars = generate_synthetic_bars(symbol="AAPL", count=30, start_dt=datetime(2025, 1, 1, tzinfo=timezone.utc))
    # Benchmark offset by 10 days
    spy_bars = generate_synthetic_bars(symbol="SPY", count=30, start_dt=datetime(2025, 1, 11, tzinfo=timezone.utc))

    gen = MarketRelativeFeatureGenerator(window=10)
    features = gen.generate(stock_bars, context={"benchmark_bars": spy_bars})
    # The first 10 stock bars have no corresponding SPY bar, so relative strength must be None
    assert features["market_return_10d"][5] is None


# =====================================================================
# 9. CRITICAL REQUIREMENT — ZERO FUTURE DATA LEAKAGE TEST
# =====================================================================

def test_zero_future_leakage_guarantee():
    """
    Critical verification: features computed at timestamp t must NOT change
    when subsequent future data points (t+1, t+2, ...) are altered.
    """
    engine = FeatureEngine()

    base_bars = generate_synthetic_bars(symbol="AAPL", count=60, start_price=100.0)
    spy_bars = generate_synthetic_bars(symbol="SPY", count=60, start_price=500.0)

    # 1. Compute features on standard dataset
    dataset_1 = engine.generate_features(base_bars, context={"benchmark_bars": spy_bars})

    # 2. Modify the future bars from index 40 onwards radically
    modified_bars = list(base_bars[:40])
    for i in range(40, len(base_bars)):
        b = base_bars[i]
        new_close = b.close * 5.0
        new_bar = dataclasses.replace(
            b,
            close=new_close,
            high=new_close + 5.0,
            low=max(0.1, new_close - 5.0),
            volume=b.volume * 100.0,
        )
        modified_bars.append(new_bar)

    dataset_2 = engine.generate_features(modified_bars, context={"benchmark_bars": spy_bars})

    # 3. Assert that all feature values for bars 0 to 39 are strictly IDENTICAL
    for i in range(40):
        rec1 = dataset_1.records[i].features
        rec2 = dataset_2.records[i].features
        for feat_name, val1 in rec1.items():
            val2 = rec2[feat_name]
            if val1 is None:
                assert val2 is None, f"Leakage detected in {feat_name} at index {i}"
            else:
                assert math.isclose(val1, val2, rel_tol=1e-7, abs_tol=1e-7), (
                    f"Leakage detected in {feat_name} at index {i}: {val1} != {val2}"
                )


# =====================================================================
# 10. Non-Finite Value Sanitization & Warm-Up Handling
# =====================================================================

def test_non_finite_handling():
    """Verify NaN and Inf values are safely converted to None."""
    engine = FeatureEngine()
    bars = generate_synthetic_bars(count=50)

    dataset = engine.generate_features(bars)
    for rec in dataset.records:
        for fname, val in rec.features.items():
            if val is not None:
                assert not math.isnan(val), f"NaN found in {fname}"
                assert not math.isinf(val), f"Inf found in {fname}"


# =====================================================================
# 11. Determinism Test
# =====================================================================

def test_feature_generation_determinism():
    """Verify that running feature generation twice on identical input produces identical output."""
    engine = FeatureEngine()
    bars = generate_synthetic_bars(count=50)
    spy = generate_synthetic_bars(symbol="SPY", count=50)

    ds1 = engine.generate_features(bars, context={"benchmark_bars": spy})
    ds2 = engine.generate_features(bars, context={"benchmark_bars": spy})

    assert len(ds1) == len(ds2)
    assert ds1.feature_names == ds2.feature_names

    df1 = ds1.to_dataframe()
    df2 = ds2.to_dataframe()
    assert df1.equals(df2)


# =====================================================================
# 12. Storage & Persistence Tests
# =====================================================================

def test_feature_storage_roundtrip(tmp_path):
    """Verify Parquet, CSV, and metadata storage round-trip."""
    storage = FeatureStorage(base_dir=tmp_path / "features")
    engine = FeatureEngine(storage=storage)

    bars = generate_synthetic_bars(symbol="AAPL", count=40, timeframe=TimeFrame.DAY_1)
    dataset = engine.generate_features(bars)

    # Save to Parquet
    parquet_path = storage.save_dataset(dataset, format="parquet", save_metadata_json=True)
    assert parquet_path.exists()

    # Verify metadata JSON exists
    meta_path = storage.get_metadata_path("AAPL", TimeFrame.DAY_1)
    assert meta_path.exists()

    # Load DataFrame back and verify
    df_loaded = storage.load_dataframe("AAPL", TimeFrame.DAY_1, format="parquet")
    assert df_loaded is not None
    assert len(df_loaded) == 40
    assert "return_1d" in df_loaded.columns
    assert "timestamp" in df_loaded.columns

    # Save to CSV
    csv_path = storage.save_dataset(dataset, format="csv", save_metadata_json=False)
    assert csv_path.exists()

    df_csv = storage.load_dataframe("AAPL", TimeFrame.DAY_1, format="csv")
    assert df_csv is not None
    assert len(df_csv) == 40


# =====================================================================
# 13. Multi-Timeframe Feature Generation Test
# =====================================================================

def test_multi_timeframe_generation():
    """Verify feature naming and calculations respect intraday vs daily timeframes."""
    engine = FeatureEngine()
    intraday_bars = generate_synthetic_bars(symbol="NVDA", count=50, timeframe=TimeFrame.MINUTE_1)
    dataset = engine.generate_features(intraday_bars)

    assert dataset.timeframe == TimeFrame.MINUTE_1
    assert "return_1" in dataset.feature_names
    assert "return_5" in dataset.feature_names
    assert "rolling_volatility_20" in dataset.feature_names
    assert "volume_zscore_20" in dataset.feature_names
    assert len(dataset) == 50


# =====================================================================
# 14. End-to-End Integration Test (Phase 04 -> Phase 05 -> Phase 06)
# =====================================================================

@pytest.mark.asyncio
async def test_full_pipeline_ingest_validate_features(tmp_path):
    """
    Verify complete pipeline flow:
    Phase 04 Ingestion -> Phase 05 Cleaning/Validation -> Phase 06 Feature Engineering.
    """
    raw_storage = RawDataStorage(base_storage_dir=tmp_path / "raw")
    proc_storage = ProcessedDataStorage(base_storage_dir=tmp_path / "processed")
    feat_storage = FeatureStorage(base_storage_dir=tmp_path / "features")

    provider = MockMarketDataProvider()
    market_service = MarketDataService(provider=provider, storage=raw_storage)
    validation_service = DataValidationService(storage=proc_storage)
    feature_engine = FeatureEngine(storage=feat_storage)

    # Step 1: Ingest Raw Data (PHASE 04)
    raw_bars = await market_service.get_historical_bars(
        symbol="AAPL",
        timeframe=TimeFrame.DAY_1,
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 4, 1, tzinfo=timezone.utc),
    )
    assert len(raw_bars) > 50

    # Step 2: Clean & Validate (PHASE 05)
    cleaned_bars, report = validation_service.validate_and_clean(
        bars=raw_bars,
        expected_symbol="AAPL",
        expected_timeframe=TimeFrame.DAY_1,
        persist_processed=True,
    )
    assert report.is_valid is True
    assert len(cleaned_bars) > 50

    # Step 3: Compute Features (PHASE 06)
    feature_dataset = feature_engine.generate_and_save(
        bars=cleaned_bars,
        format="parquet",
        save_metadata=True,
    )

    assert feature_dataset.symbol == "AAPL"
    assert len(feature_dataset) == len(cleaned_bars)
    assert len(feature_dataset.feature_names) >= 15

    # Verify features saved properly in data/features/
    assert feat_storage.feature_exists("AAPL", TimeFrame.DAY_1, format="parquet")
    meta = feat_storage.load_metadata("AAPL", TimeFrame.DAY_1)
    assert "return_1d" in meta
    assert meta["return_1d"]["window"] == 1
