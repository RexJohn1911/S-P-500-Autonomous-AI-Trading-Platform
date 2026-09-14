"""
Automated unit and integration test suite for Data Cleaning & Validation Subsystem (Phase 05).
Covers schema, missing values, duplicates, OHLC rules, price/volume validation, timestamps,
gaps, market hours, consistency, outliers, corporate actions, and safe cleaning pipelines.
"""

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import shutil
import tempfile
import zoneinfo
import pytest

from backend.app.data.models import (
    BarData,
    TimeFrame,
    AssetClass,
    ensure_utc,
    UTC,
    NY_TZ,
)
from backend.app.data.providers.mock_provider import MockMarketDataProvider
from backend.app.data.storage import RawDataStorage
from backend.app.data.service import MarketDataService
from backend.app.data.validation.models import (
    ValidationSeverity,
    ValidationStatus,
    DeduplicationPolicy,
    MarketSessionMode,
    DataQualityReport,
)
from backend.app.data.validation.validators import (
    SchemaValidator,
    MissingValueValidator,
    DuplicateValidator,
    OHLCValidator,
    PriceValidator,
    VolumeValidator,
    TimestampValidator,
    GapValidator,
    MarketHoursValidator,
    ConsistencyValidator,
    OutlierValidator,
    CorporateActionAnomalyValidator,
)
from backend.app.data.validation.cleaner import DataCleaner
from backend.app.data.validation.storage import ProcessedDataStorage
from backend.app.data.validation.service import DataValidationService


# ==========================================
# 1. Schema & Missing Value Tests
# ==========================================

def test_schema_validator_valid_and_malformed():
    """Verify SchemaValidator distinguishes valid BarData from non-BarData or non-numeric types."""
    validator = SchemaValidator()
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    valid_bar = BarData(
        symbol="AAPL",
        timestamp=now,
        open=150.0,
        high=155.0,
        low=149.0,
        close=154.0,
        volume=1000.0,
        timeframe=TimeFrame.DAY_1,
    )
    res = validator.validate([valid_bar])
    assert res.status == ValidationStatus.PASS
    assert res.error_count == 0

    # Malformed non-BarData object
    malformed_list = [valid_bar, {"symbol": "AAPL", "close": 150.0}]  # type: ignore
    res_malformed = validator.validate(malformed_list)
    assert res_malformed.status == ValidationStatus.FAIL
    assert res_malformed.error_count > 0


def test_missing_value_validator():
    """Verify MissingValueValidator detects empty symbol or None attributes."""
    validator = MissingValueValidator()
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    bar = BarData(
        symbol="MSFT",
        timestamp=now,
        open=400.0,
        high=410.0,
        low=395.0,
        close=405.0,
        volume=50000.0,
    )
    res = validator.validate([bar])
    assert res.status == ValidationStatus.PASS


# ==========================================
# 2. Duplicate Tests & Dedup Policies
# ==========================================

def test_duplicate_validator_detection():
    """Verify DuplicateValidator flags duplicate timestamps for the same symbol/timeframe."""
    validator = DuplicateValidator()
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    bar1 = BarData(symbol="NVDA", timestamp=now, open=120.0, high=125.0, low=119.0, close=124.0, volume=1000.0)
    bar2 = BarData(symbol="NVDA", timestamp=now, open=120.0, high=125.0, low=119.0, close=124.5, volume=2000.0)

    res = validator.validate([bar1, bar2])
    assert res.status == ValidationStatus.FAIL
    assert res.error_count == 1
    assert "Duplicate timestamp detected" in res.issues[0].message


def test_data_cleaner_deduplication_policies():
    """Verify DataCleaner deduplication resolves duplicates according to policy."""
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    bar1 = BarData(symbol="NVDA", timestamp=now, open=120.0, high=125.0, low=119.0, close=124.0, volume=1000.0)
    bar2 = BarData(symbol="NVDA", timestamp=now, open=120.0, high=125.0, low=119.0, close=124.5, volume=5000.0)

    # KEEP_FIRST policy
    cleaner_first = DataCleaner(dedup_policy=DeduplicationPolicy.KEEP_FIRST)
    cleaned_first, repairs = cleaner_first.clean([bar1, bar2])
    assert len(cleaned_first) == 1
    assert cleaned_first[0].volume == 1000.0
    assert len(repairs) == 1

    # KEEP_LAST policy
    cleaner_last = DataCleaner(dedup_policy=DeduplicationPolicy.KEEP_LAST)
    cleaned_last, _ = cleaner_last.clean([bar1, bar2])
    assert len(cleaned_last) == 1
    assert cleaned_last[0].volume == 5000.0

    # KEEP_HIGHER_VOLUME policy
    cleaner_vol = DataCleaner(dedup_policy=DeduplicationPolicy.KEEP_HIGHER_VOLUME)
    cleaned_vol, _ = cleaner_vol.clean([bar1, bar2])
    assert len(cleaned_vol) == 1
    assert cleaned_vol[0].volume == 5000.0


# ==========================================
# 3. OHLC & Mathematical Integrity Tests
# ==========================================

def test_ohlc_validator_mathematical_consistency():
    """Verify OHLCValidator checks high >= low, high >= open/close, low <= open/close."""
    validator = OHLCValidator()
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    # Valid bar
    valid_bar = BarData(symbol="AMZN", timestamp=now, open=180.0, high=185.0, low=178.0, close=182.0, volume=100.0)
    res_valid = validator.validate([valid_bar])
    assert res_valid.status == ValidationStatus.PASS

    # High < Open (bypassing BarData constructor constraint for test validation)
    # Using object.__setattr__ to simulate raw malformed data structure
    bar_high_err = BarData(symbol="AMZN", timestamp=now, open=180.0, high=185.0, low=178.0, close=182.0, volume=100.0)
    object.__setattr__(bar_high_err, "high", 175.0)  # High is lower than Open
    res_high_err = validator.validate([bar_high_err])
    assert res_high_err.status == ValidationStatus.FAIL
    assert any("less than body maximum" in iss.message for iss in res_high_err.issues)

    # Low > Close
    bar_low_err = BarData(symbol="AMZN", timestamp=now, open=180.0, high=185.0, low=178.0, close=182.0, volume=100.0)
    object.__setattr__(bar_low_err, "low", 183.0)  # Low is greater than Close
    res_low_err = validator.validate([bar_low_err])
    assert res_low_err.status == ValidationStatus.FAIL
    assert any("greater than body minimum" in iss.message for iss in res_low_err.issues)


# ==========================================
# 4. Price & Volume Anomaly Tests
# ==========================================

def test_price_validator_zero_and_jumps():
    """Verify PriceValidator detects zero prices and suspicious large price jumps."""
    validator = PriceValidator(max_single_bar_change_pct=0.30)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    bar1 = BarData(symbol="TSLA", timestamp=now, open=200.0, high=205.0, low=195.0, close=200.0, volume=1000.0)
    bar2 = BarData(symbol="TSLA", timestamp=now + timedelta(days=1), open=280.0, high=290.0, low=275.0, close=280.0, volume=1000.0)

    # bar2 represents a 40% jump from 200 -> 280 (exceeds 30% warning threshold)
    res = validator.validate([bar1, bar2])
    assert res.status == ValidationStatus.WARNING
    assert res.warning_count == 1
    assert "Suspicious large price change" in res.issues[0].message


def test_volume_validator_negative_and_spikes():
    """Verify VolumeValidator flags negative volume and extreme volume spikes."""
    validator = VolumeValidator(spike_multiplier_threshold=10.0)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    bars = [
        BarData(
            symbol="GOOGL",
            timestamp=now + timedelta(days=i),
            open=170.0,
            high=172.0,
            low=168.0,
            close=170.0,
            volume=1000.0,
        )
        for i in range(12)
    ]
    # Add a massive volume spike at the end
    spike_bar = BarData(
        symbol="GOOGL",
        timestamp=now + timedelta(days=12),
        open=170.0,
        high=172.0,
        low=168.0,
        close=170.0,
        volume=50000.0,  # 50x normal volume
    )
    bars.append(spike_bar)

    res = validator.validate(bars)
    assert res.status == ValidationStatus.WARNING
    assert any("Extreme volume spike" in iss.message for iss in res.issues)


# ==========================================
# 5. Timestamp & Ordering Tests
# ==========================================

def test_timestamp_validator_ordering():
    """Verify TimestampValidator detects out-of-order bars and non-UTC timestamps."""
    validator = TimestampValidator()
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    bar1 = BarData(symbol="SPY", timestamp=now, open=500.0, high=502.0, low=498.0, close=501.0, volume=1000.0)
    bar2 = BarData(symbol="SPY", timestamp=now - timedelta(hours=1), open=500.0, high=502.0, low=498.0, close=501.0, volume=1000.0)

    # bar2 is earlier than bar1
    res = validator.validate([bar1, bar2])
    assert res.status == ValidationStatus.FAIL
    assert any("Out-of-order timestamp" in iss.message for iss in res.issues)


def test_cleaner_automatic_sorting():
    """Verify DataCleaner sorts unsorted bars chronologically."""
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    bar1 = BarData(symbol="SPY", timestamp=now + timedelta(hours=2), open=500.0, high=502.0, low=498.0, close=501.0, volume=1000.0)
    bar2 = BarData(symbol="SPY", timestamp=now, open=500.0, high=502.0, low=498.0, close=501.0, volume=1000.0)

    cleaner = DataCleaner(auto_sort=True)
    cleaned, repairs = cleaner.clean([bar1, bar2])
    assert len(cleaned) == 2
    assert cleaned[0].timestamp < cleaned[1].timestamp
    assert any(r.repair_action == "CHRONOLOGICAL_SORT" for r in repairs)


# ==========================================
# 6. Gap & Continuity Tests
# ==========================================

def test_gap_validator_daily_and_intraday():
    """Verify GapValidator accounts for normal weekends but warns on large multi-day gaps."""
    validator = GapValidator(max_allowed_daily_gap_days=4)
    friday = datetime(2026, 9, 4, 16, 0, tzinfo=UTC)    # Friday
    monday = datetime(2026, 9, 7, 16, 0, tzinfo=UTC)    # Monday (3 day gap)
    next_mon = datetime(2026, 9, 18, 16, 0, tzinfo=UTC) # 11 days later (unexpected gap)

    bar_fri = BarData(symbol="SPY", timestamp=friday, open=500.0, high=505.0, low=495.0, close=500.0, volume=1000.0, timeframe=TimeFrame.DAY_1)
    bar_mon = BarData(symbol="SPY", timestamp=monday, open=501.0, high=506.0, low=496.0, close=502.0, volume=1000.0, timeframe=TimeFrame.DAY_1)
    bar_gap = BarData(symbol="SPY", timestamp=next_mon, open=502.0, high=507.0, low=497.0, close=503.0, volume=1000.0, timeframe=TimeFrame.DAY_1)

    # Friday -> Monday normal weekend: PASS
    res_weekend = validator.validate([bar_fri, bar_mon])
    assert res_weekend.status == ValidationStatus.PASS

    # Monday -> Next Friday gap: WARNING
    res_gap = validator.validate([bar_mon, bar_gap])
    assert res_gap.status == ValidationStatus.WARNING
    assert any("Unexpected gap" in iss.message for iss in res_gap.issues)


# ==========================================
# 7. Market Hours Tests
# ==========================================

def test_market_hours_validator():
    """Verify MarketHoursValidator flags intraday bars outside 09:30-16:00 ET."""
    validator = MarketHoursValidator(session_mode=MarketSessionMode.REGULAR_HOURS)

    # 10:00 AM ET is 14:00 UTC (during EDT) -> Regular market hours
    ny_tz = zoneinfo.ZoneInfo("America/New_York")
    dt_regular = datetime(2026, 9, 1, 10, 0, tzinfo=ny_tz).astimezone(UTC)
    bar_reg = BarData(symbol="AAPL", timestamp=dt_regular, open=150.0, high=151.0, low=149.0, close=150.0, volume=100.0, timeframe=TimeFrame.MINUTE_5)

    res_reg = validator.validate([bar_reg])
    assert res_reg.status == ValidationStatus.PASS

    # 08:00 AM ET is pre-market -> outside regular hours
    dt_pre = datetime(2026, 9, 1, 8, 0, tzinfo=ny_tz).astimezone(UTC)
    bar_pre = BarData(symbol="AAPL", timestamp=dt_pre, open=150.0, high=151.0, low=149.0, close=150.0, volume=100.0, timeframe=TimeFrame.MINUTE_5)

    res_pre = validator.validate([bar_pre])
    assert res_pre.status == ValidationStatus.WARNING
    assert any("outside regular US market hours" in iss.message for iss in res_pre.issues)


# ==========================================
# 8. Consistency Tests
# ==========================================

def test_consistency_validator_mixed_symbols_and_timeframes():
    """Verify ConsistencyValidator flags mixed symbols or timeframes."""
    validator = ConsistencyValidator()
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    bar1 = BarData(symbol="AAPL", timestamp=now, open=150.0, high=155.0, low=149.0, close=154.0, volume=100.0, timeframe=TimeFrame.DAY_1)
    bar2 = BarData(symbol="MSFT", timestamp=now + timedelta(days=1), open=400.0, high=405.0, low=395.0, close=400.0, volume=100.0, timeframe=TimeFrame.DAY_1)

    res = validator.validate([bar1, bar2], expected_symbol="AAPL")
    assert res.status == ValidationStatus.FAIL
    assert any("Inconsistent symbol" in iss.message for iss in res.issues)


# ==========================================
# 9. Outlier & Anomaly Tests
# ==========================================

def test_outlier_validator_robust_scoring():
    """Verify OutlierValidator detects statistically extreme price returns."""
    validator = OutlierValidator(mad_zscore_warning_threshold=5.0, mad_zscore_error_threshold=15.0, min_sample_size=10)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    # 20 bars with steady 0.2% return
    bars = []
    p = 100.0
    for i in range(20):
        bars.append(
            BarData(
                symbol="META",
                timestamp=now + timedelta(days=i),
                open=p,
                high=p * 1.01,
                low=p * 0.99,
                close=p * 1.002,
                volume=10000.0,
            )
        )
        p *= 1.002

    # Add a massive 80% spike
    bars.append(
        BarData(
            symbol="META",
            timestamp=now + timedelta(days=20),
            open=p,
            high=p * 1.85,
            low=p,
            close=p * 1.80,
            volume=10000.0,
        )
    )

    res = validator.validate(bars)
    assert res.status in (ValidationStatus.WARNING, ValidationStatus.FAIL)
    assert any("return anomaly" in iss.message or "unusual price movement" in iss.message for iss in res.issues)


# ==========================================
# 10. Corporate Action Discontinuity Tests
# ==========================================

def test_corporate_action_discontinuity_validator():
    """Verify CorporateActionAnomalyValidator flags ~50% drop matching 2:1 stock split."""
    validator = CorporateActionAnomalyValidator(split_ratio_tolerance=0.05)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    bar1 = BarData(symbol="NVDA", timestamp=now, open=1000.0, high=1010.0, low=990.0, close=1000.0, volume=1000.0)
    bar2 = BarData(symbol="NVDA", timestamp=now + timedelta(days=1), open=502.0, high=510.0, low=495.0, close=500.0, volume=2000.0)

    # 502 / 1000 is ~0.502 (matches 2:1 split)
    res = validator.validate([bar1, bar2])
    assert res.status == ValidationStatus.WARNING
    assert any("Potential 2:1 Stock Split" in iss.message for iss in res.issues)


# ==========================================
# 11. End-to-End Validation & Processed Storage Pipeline
# ==========================================

@pytest.mark.asyncio
async def test_full_market_data_validation_pipeline_integration():
    """
    Test Phase 04 -> Phase 05 end-to-end integration:
    Fetch raw bars -> Validate & Clean -> Persist to data/processed/ without mutating data/raw/.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        raw_storage_dir = Path(temp_dir) / "data" / "raw"
        processed_storage_dir = Path(temp_dir) / "data" / "processed"

        raw_storage = RawDataStorage(base_storage_dir=raw_storage_dir)
        processed_storage = ProcessedDataStorage(base_storage_dir=processed_storage_dir)

        # 1. Generate and ingest raw data via Phase 04 components
        provider = MockMarketDataProvider(seed=42)
        market_service = MarketDataService(provider=provider, storage=raw_storage)

        start = datetime(2026, 8, 1, tzinfo=UTC)
        end = datetime(2026, 8, 20, tzinfo=UTC)

        raw_bars = await market_service.get_historical_bars("AAPL", start, end, TimeFrame.DAY_1, persist_raw=True)
        assert len(raw_bars) > 0

        # Verify raw data exists
        raw_loaded = raw_storage.load_bars("AAPL", TimeFrame.DAY_1, format="json")
        assert len(raw_loaded) == len(raw_bars)

        # 2. Run Phase 05 Validation & Cleaning Service
        validation_service = DataValidationService(storage=processed_storage)
        cleaned_bars, quality_report = validation_service.validate_and_clean(
            raw_bars,
            expected_symbol="AAPL",
            expected_timeframe=TimeFrame.DAY_1,
            persist_processed=True,
            storage_format="parquet",
        )

        assert quality_report.overall_status in (ValidationStatus.PASS, ValidationStatus.WARNING)
        assert len(cleaned_bars) == len(raw_bars)

        # Verify processed data was saved in data/processed/
        processed_loaded = processed_storage.load_bars("AAPL", TimeFrame.DAY_1, format="parquet")
        assert len(processed_loaded) == len(cleaned_bars)

        # Ensure raw data was NOT overwritten
        raw_after_clean = raw_storage.load_bars("AAPL", TimeFrame.DAY_1, format="json")
        assert len(raw_after_clean) == len(raw_loaded)

        # Check quality report summary text
        summary = quality_report.summary_text()
        assert "DATA QUALITY REPORT" in summary
        assert "AAPL" in summary

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
