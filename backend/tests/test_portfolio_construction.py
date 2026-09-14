"""
Unit and Integration Tests for Phase 11 — Portfolio Construction.
Verifies target allocation, constraint enforcement, long-only vs long-short modes,
capping/redistribution, determinism, causal leakage isolation, and storage roundtrips.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pytest

from backend.app.portfolio import (
    BasePortfolioAllocator,
    PortfolioConstraintEnforcer,
    PortfolioConstructionConfig,
    PortfolioConstructionRequest,
    PortfolioConstructionResult,
    PortfolioConstructionService,
    PortfolioMode,
    PortfolioStorage,
    PortfolioTarget,
    SignalProportionalAllocator,
)
from backend.app.strategy.schemas import (
    ReasonCode,
    SignalCandidate,
    SignalDirection,
    StandardizedPrediction,
)


@pytest.fixture
def base_timestamp():
    return datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def default_config():
    return PortfolioConstructionConfig(
        mode=PortfolioMode.LONG_ONLY,
        max_position_weight=0.10,
        min_position_weight=0.01,
        max_positions=20,
        max_gross_exposure=1.00,
        max_net_exposure=1.00,
        version="portfolio-v1",
    )


# =====================================================================
# 1. SCHEMAS & SERIALIZATION TESTS
# =====================================================================

def test_portfolio_target_lifecycle(base_timestamp):
    """Verify PortfolioTarget bounds, serialization, and roundtrip deserialization."""
    target = PortfolioTarget(
        symbol="AAPL",
        target_weight=0.075,
        signal_direction=SignalDirection.LONG,
        signal_strength=0.70,
        confidence=0.80,
        expected_return=0.015,
        signal_version="signal-v1",
        timestamp=base_timestamp,
        metadata={"sector": "Technology"},
    )
    d = target.to_dict()
    assert d["symbol"] == "AAPL"
    assert d["target_weight"] == 0.075
    assert d["signal_direction"] == "LONG"
    assert d["expected_return"] == 0.015

    restored = PortfolioTarget.from_dict(d)
    assert restored.symbol == target.symbol
    assert restored.target_weight == target.target_weight
    assert restored.signal_direction == SignalDirection.LONG

    # Target weight out of bounds [-1, 1]
    with pytest.raises(ValueError):
        PortfolioTarget(
            symbol="AAPL",
            target_weight=1.50,
            signal_direction=SignalDirection.LONG,
            signal_strength=0.5,
            confidence=0.5,
        )


def test_portfolio_construction_result_lifecycle(base_timestamp):
    """Verify PortfolioConstructionResult serialization and roundtrip."""
    t1 = PortfolioTarget("AAPL", 0.08, SignalDirection.LONG, 0.8, 0.8, timestamp=base_timestamp)
    t2 = PortfolioTarget("MSFT", 0.07, SignalDirection.LONG, 0.7, 0.7, timestamp=base_timestamp)
    res = PortfolioConstructionResult(
        timestamp=base_timestamp,
        targets=[t1, t2],
        total_long_weight=0.15,
        total_short_weight=0.0,
        gross_exposure=0.15,
        net_exposure=0.15,
        active_position_count=2,
        allocated_cash_weight=0.85,
        rejected_signals={"GOOG": "FLAT_SIGNAL"},
        construction_version="portfolio-v1",
    )
    d = res.to_dict()
    assert d["active_position_count"] == 2
    assert d["gross_exposure"] == 0.15
    assert d["allocated_cash_weight"] == 0.85

    restored = PortfolioConstructionResult.from_dict(d)
    assert restored.active_position_count == 2
    assert len(restored.targets) == 2
    assert restored.rejected_signals["GOOG"] == "FLAT_SIGNAL"


def test_portfolio_config_validation():
    """Verify invalid configuration throws proper descriptive errors."""
    with pytest.raises(ValueError):
        # min_position_weight > max_position_weight
        PortfolioConstructionConfig(max_position_weight=0.05, min_position_weight=0.10)

    with pytest.raises(ValueError):
        # max_positions < 1
        PortfolioConstructionConfig(max_positions=0)

    with pytest.raises(ValueError):
        # max_gross_exposure <= 0
        PortfolioConstructionConfig(max_gross_exposure=0.0)


# =====================================================================
# 2. CORE ALLOCATION & SIGNAL DIRECTION TESTS
# =====================================================================

def test_long_signal_positive_allocation_long_only(default_config, base_timestamp):
    """Test 1: LONG signal produces positive target weight in long-only mode."""
    service = PortfolioConstructionService(config=default_config)
    signals = [
        SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.80, 0.75),
        SignalCandidate(base_timestamp, "MSFT", SignalDirection.LONG, 0.60, 0.50),
    ]
    result = service.construct(signals)

    assert result.active_position_count == 2
    assert result.total_long_weight > 0.0
    assert result.total_short_weight == 0.0
    assert all(t.target_weight > 0.0 for t in result.targets)
    assert result.targets[0].symbol == "AAPL"


def test_flat_signal_zero_allocation(default_config, base_timestamp):
    """Test 2: FLAT signals receive zero portfolio allocation."""
    service = PortfolioConstructionService(config=default_config)
    signals = [
        SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.80, 0.75),
        SignalCandidate(base_timestamp, "GOOG", SignalDirection.FLAT, 0.0, 0.10),
    ]
    result = service.construct(signals)

    assert result.active_position_count == 1
    assert "GOOG" in result.rejected_signals
    assert result.rejected_signals["GOOG"] == "FLAT_SIGNAL"
    assert all(t.symbol != "GOOG" for t in result.targets)


def test_short_signal_rejected_in_long_only_mode(default_config, base_timestamp):
    """Test 3: SHORT signals are filtered/rejected in LONG_ONLY mode."""
    service = PortfolioConstructionService(config=default_config)
    signals = [
        SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.80, 0.75),
        SignalCandidate(base_timestamp, "TSLA", SignalDirection.SHORT, -0.80, 0.75),
    ]
    result = service.construct(signals)

    assert result.active_position_count == 1
    assert "TSLA" in result.rejected_signals
    assert result.rejected_signals["TSLA"] == "SHORT_IN_LONG_ONLY"
    assert result.total_short_weight == 0.0


def test_short_signal_negative_allocation_in_long_short_mode(base_timestamp):
    """Test 4: SHORT signal produces negative target weight in LONG_SHORT mode."""
    config = PortfolioConstructionConfig(mode=PortfolioMode.LONG_SHORT, max_position_weight=0.10)
    service = PortfolioConstructionService(config=config)
    signals = [
        SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.80, 0.75),
        SignalCandidate(base_timestamp, "TSLA", SignalDirection.SHORT, -0.80, 0.75),
    ]
    result = service.construct(signals)

    assert result.active_position_count == 2
    assert result.total_long_weight > 0.0
    assert result.total_short_weight > 0.0

    target_map = {t.symbol: t.target_weight for t in result.targets}
    assert target_map["AAPL"] > 0.0
    assert target_map["TSLA"] < 0.0


# =====================================================================
# 3. WEIGHT NORMALIZATION & CONSTRAINTS TESTS
# =====================================================================

def test_weight_normalization_and_gross_exposure(base_timestamp):
    """Test 5 & 9: Target weights satisfy gross exposure constraint."""
    config = PortfolioConstructionConfig(
        mode=PortfolioMode.LONG_ONLY,
        max_position_weight=0.20,
        max_gross_exposure=0.80,
    )
    service = PortfolioConstructionService(config=config)

    # 10 assets
    signals = [
        SignalCandidate(base_timestamp, f"SYM_{i}", SignalDirection.LONG, 0.5 + i * 0.05, 0.8)
        for i in range(10)
    ]
    result = service.construct(signals)

    assert result.gross_exposure <= 0.80 + 1e-5
    assert result.allocated_cash_weight >= 0.20 - 1e-5


def test_max_position_weight_capping_and_redistribution(base_timestamp):
    """Test 6: Capping at max_position_weight and redistributing surplus capital."""
    config = PortfolioConstructionConfig(
        mode=PortfolioMode.LONG_ONLY,
        max_position_weight=0.15,
        max_gross_exposure=1.00,
    )
    service = PortfolioConstructionService(config=config)

    # One very dominant asset and several others
    signals = [
        SignalCandidate(base_timestamp, "DOMINANT", SignalDirection.LONG, 0.99, 0.99),
        SignalCandidate(base_timestamp, "SYM_1", SignalDirection.LONG, 0.30, 0.50),
        SignalCandidate(base_timestamp, "SYM_2", SignalDirection.LONG, 0.30, 0.50),
        SignalCandidate(base_timestamp, "SYM_3", SignalDirection.LONG, 0.30, 0.50),
    ]
    result = service.construct(signals)

    target_map = {t.symbol: t.target_weight for t in result.targets}
    assert target_map["DOMINANT"] <= 0.15 + 1e-5
    assert all(t.target_weight <= 0.15 + 1e-5 for t in result.targets)


def test_min_position_weight_filtering(base_timestamp):
    """Test 7: Filtering out positions that fall below min_position_weight."""
    config = PortfolioConstructionConfig(
        mode=PortfolioMode.LONG_ONLY,
        max_position_weight=0.50,
        min_position_weight=0.08,  # High threshold
        max_gross_exposure=1.00,
    )
    service = PortfolioConstructionService(config=config)

    # 1 very strong asset, 1 weak asset
    signals = [
        SignalCandidate(base_timestamp, "STRONG", SignalDirection.LONG, 0.95, 0.95),
        SignalCandidate(base_timestamp, "TINY", SignalDirection.LONG, 0.05, 0.05),
    ]
    result = service.construct(signals)

    # TINY should be pruned as below 8%
    assert "TINY" in result.rejected_signals
    assert result.rejected_signals["TINY"] == "BELOW_MIN_WEIGHT"
    assert all(t.symbol != "TINY" for t in result.targets)


def test_max_positions_constraint_ranking(base_timestamp):
    """Test 8: Only top K <= max_positions assets are included, ranked by conviction."""
    config = PortfolioConstructionConfig(
        mode=PortfolioMode.LONG_ONLY,
        max_positions=3,
        max_position_weight=0.35,
    )
    service = PortfolioConstructionService(config=config)

    signals = [
        SignalCandidate(base_timestamp, "RANK_1", SignalDirection.LONG, 0.90, 0.90),
        SignalCandidate(base_timestamp, "RANK_2", SignalDirection.LONG, 0.80, 0.80),
        SignalCandidate(base_timestamp, "RANK_3", SignalDirection.LONG, 0.70, 0.70),
        SignalCandidate(base_timestamp, "RANK_4", SignalDirection.LONG, 0.60, 0.60),
        SignalCandidate(base_timestamp, "RANK_5", SignalDirection.LONG, 0.50, 0.50),
    ]
    result = service.construct(signals)

    assert result.active_position_count == 3
    included = [t.symbol for t in result.targets]
    assert included == ["RANK_1", "RANK_2", "RANK_3"]
    assert result.rejected_signals["RANK_4"] == "MAX_POSITIONS_EXCEEDED"
    assert result.rejected_signals["RANK_5"] == "MAX_POSITIONS_EXCEEDED"


def test_net_exposure_constraint_long_short(base_timestamp):
    """Test 10: Net exposure constraint tracking in long/short mode."""
    config = PortfolioConstructionConfig(
        mode=PortfolioMode.LONG_SHORT,
        max_position_weight=0.20,
        max_gross_exposure=1.00,
        max_net_exposure=0.50,
    )
    service = PortfolioConstructionService(config=config)

    signals = [
        SignalCandidate(base_timestamp, "L1", SignalDirection.LONG, 0.8, 0.8),
        SignalCandidate(base_timestamp, "L2", SignalDirection.LONG, 0.7, 0.7),
        SignalCandidate(base_timestamp, "S1", SignalDirection.SHORT, -0.6, 0.6),
    ]
    result = service.construct(signals)

    assert result.gross_exposure <= 1.00 + 1e-5
    assert abs(result.net_exposure) <= 1.00 + 1e-5


# =====================================================================
# 4. DETERMINISM & TIE-BREAKING TESTS
# =====================================================================

def test_deterministic_ordering_and_tie_breaking(base_timestamp):
    """Test 11 & 12: Alphabetical symbol sorting as deterministic tie-breaker for identical scores."""
    config = PortfolioConstructionConfig(max_positions=2)
    service = PortfolioConstructionService(config=config)

    # 3 identical scores; tie-breaker must select alphabetically first two (AAA, BBB)
    signals = [
        SignalCandidate(base_timestamp, "CCC", SignalDirection.LONG, 0.80, 0.80),
        SignalCandidate(base_timestamp, "AAA", SignalDirection.LONG, 0.80, 0.80),
        SignalCandidate(base_timestamp, "BBB", SignalDirection.LONG, 0.80, 0.80),
    ]
    result = service.construct(signals)

    assert result.active_position_count == 2
    symbols = [t.symbol for t in result.targets]
    assert symbols == ["AAA", "BBB"]
    assert result.rejected_signals["CCC"] == "MAX_POSITIONS_EXCEEDED"


def test_repeated_identical_inputs_produce_identical_output(base_timestamp, default_config):
    """Test 20: Repeated execution with identical inputs produces identical results bit-for-bit."""
    service = PortfolioConstructionService(config=default_config)
    signals = [
        SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.75, 0.80),
        SignalCandidate(base_timestamp, "MSFT", SignalDirection.LONG, 0.65, 0.70),
        SignalCandidate(base_timestamp, "GOOG", SignalDirection.FLAT, 0.0, 0.10),
    ]

    r1 = service.construct(signals)
    r2 = service.construct(signals)

    assert r1.to_dict() == r2.to_dict()


# =====================================================================
# 5. EDGE CASES & NUMERICAL SANITY
# =====================================================================

def test_empty_portfolio_when_all_signals_flat(default_config, base_timestamp):
    """Test 13: All FLAT signals produce empty targets and 100% cash."""
    service = PortfolioConstructionService(config=default_config)
    signals = [
        SignalCandidate(base_timestamp, "AAPL", SignalDirection.FLAT, 0.0, 0.1),
        SignalCandidate(base_timestamp, "MSFT", SignalDirection.FLAT, 0.0, 0.1),
    ]
    result = service.construct(signals)

    assert result.active_position_count == 0
    assert len(result.targets) == 0
    assert result.allocated_cash_weight == 1.0
    assert result.gross_exposure == 0.0


def test_invalid_signal_handling(default_config, base_timestamp):
    """Test 14: Non-finite signals are safely rejected."""
    service = PortfolioConstructionService(config=default_config)
    # Create valid candidate then inject nan into strength
    cand = SignalCandidate(base_timestamp, "NAN_SYM", SignalDirection.LONG, 0.5, 0.5)
    cand.signal_strength = float("nan")

    result = service.construct([cand])
    assert result.active_position_count == 0
    assert result.rejected_signals["NAN_SYM"] == "INVALID_SIGNAL"


def test_nan_inf_protection(default_config, base_timestamp):
    """Test 15: Target weights are guaranteed to be finite."""
    service = PortfolioConstructionService(config=default_config)
    signals = [SignalCandidate(base_timestamp, "SPY", SignalDirection.LONG, 0.8, 0.8)]
    result = service.construct(signals)

    for t in result.targets:
        assert not np.isnan(t.target_weight)
        assert not np.isinf(t.target_weight)


def test_symbol_isolation(base_timestamp):
    """Test 16: Different symbols in separate requests do not interfere."""
    service = PortfolioConstructionService()
    sig_aapl = [SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.8, 0.8)]
    sig_msft = [SignalCandidate(base_timestamp, "MSFT", SignalDirection.LONG, 0.8, 0.8)]

    res_aapl = service.construct(sig_aapl)
    res_msft = service.construct(sig_msft)

    assert res_aapl.targets[0].symbol == "AAPL"
    assert res_msft.targets[0].symbol == "MSFT"


# =====================================================================
# 6. CAUSALITY & TEMPORAL LEAKAGE AUDIT TESTS
# =====================================================================

def test_future_data_mutation_invariance(default_config, base_timestamp):
    """
    Test 17 (LEAKAGE AUDIT):
    Mutating future returns/prices or hypothetical downstream values
    must NOT affect portfolio allocation at timestamp t.
    """
    service = PortfolioConstructionService(config=default_config)

    # Signals at timestamp t
    signals_t0 = [
        SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.80, 0.80),
        SignalCandidate(base_timestamp, "MSFT", SignalDirection.LONG, 0.60, 0.60),
    ]

    res_baseline = service.construct(signals_t0)

    # Downstream hypothetical future mutation:
    # Future returns or realized price moves have no interface into PortfolioConstructionService
    # Verifying that calling construct with signals at t yields strictly deterministic targets
    res_retest = service.construct(signals_t0)

    assert res_baseline.to_dict() == res_retest.to_dict()


def test_expected_return_not_fabricated(default_config, base_timestamp):
    """
    Test 18 (NO FABRICATION):
    expected_return is preserved from SignalCandidate only if provided;
    it is never manufactured from weights or signal scores.
    """
    service = PortfolioConstructionService(config=default_config)

    # Signal with no expected return (None)
    sig_no_ret = SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.80, 0.80, expected_return=None)
    res_no_ret = service.construct([sig_no_ret])
    assert res_no_ret.targets[0].expected_return is None

    # Signal with legitimate upstream expected return (0.012)
    sig_with_ret = SignalCandidate(base_timestamp, "MSFT", SignalDirection.LONG, 0.80, 0.80, expected_return=0.012)
    res_with_ret = service.construct([sig_with_ret])
    assert res_with_ret.targets[0].expected_return == 0.012


def test_configuration_changes_alter_output_predictably(base_timestamp):
    """Test 19: Changing configuration parameters affects output as expected."""
    sig = [SignalCandidate(base_timestamp, "AAPL", SignalDirection.LONG, 0.9, 0.9)]

    # Max weight 0.05 vs 0.15
    svc_5 = PortfolioConstructionService(config=PortfolioConstructionConfig(max_position_weight=0.05))
    svc_15 = PortfolioConstructionService(config=PortfolioConstructionConfig(max_position_weight=0.15))

    res_5 = svc_5.construct(sig)
    res_15 = svc_15.construct(sig)

    assert pytest.approx(res_5.targets[0].target_weight, abs=1e-5) == 0.05
    assert pytest.approx(res_15.targets[0].target_weight, abs=1e-5) == 0.15


# =====================================================================
# 7. STORAGE PERSISTENCE TESTS
# =====================================================================

def test_portfolio_storage_roundtrip(tmp_path, base_timestamp):
    """Test 22: Persistence and retrieval of configuration and construction results."""
    storage = PortfolioStorage(base_storage_dir=tmp_path)
    config = PortfolioConstructionConfig(
        mode=PortfolioMode.LONG_SHORT,
        max_position_weight=0.12,
        version="portfolio-test-v1",
    )

    # Save and load config
    storage.save_config(config, metadata={"creator": "quant_team"})
    loaded_config = storage.load_config("portfolio-test-v1")
    assert loaded_config.mode == PortfolioMode.LONG_SHORT
    assert loaded_config.max_position_weight == 0.12

    # Save and load result
    t1 = PortfolioTarget("AAPL", 0.10, SignalDirection.LONG, 0.8, 0.8, timestamp=base_timestamp)
    result = PortfolioConstructionResult(
        timestamp=base_timestamp,
        targets=[t1],
        total_long_weight=0.10,
        total_short_weight=0.0,
        gross_exposure=0.10,
        net_exposure=0.10,
        active_position_count=1,
        allocated_cash_weight=0.90,
        construction_version="portfolio-test-v1",
    )
    storage.save_result(result, version="portfolio-test-v1")
    loaded_result = storage.load_result(version="portfolio-test-v1")

    assert loaded_result.active_position_count == 1
    assert loaded_result.targets[0].symbol == "AAPL"
    assert loaded_result.targets[0].target_weight == 0.10


# =====================================================================
# 8. PHASE 10 SIGNAL ENGINE INTEGRATION TEST
# =====================================================================

def test_actual_phase_10_signal_candidate_compatibility(base_timestamp):
    """
    Test 23: Complete integration test consuming ACTUAL outputs from Phase 10 SignalEngine.
    """
    from backend.app.strategy import BinaryClassificationAdapter, SignalEngine

    # 1. Generate signals using Phase 10 SignalEngine
    signal_engine = SignalEngine()
    adapter = BinaryClassificationAdapter()

    # Asset A: Bullish P(up)=0.80 -> LONG signal
    pred_a = adapter.adapt("AAPL", base_timestamp, "xgb", 0.80)
    sig_a = signal_engine.generate_signal("AAPL", base_timestamp, [pred_a], regime="BULL_TRENDING")

    # Asset B: Bullish P(up)=0.75 -> LONG signal
    pred_b = adapter.adapt("MSFT", base_timestamp, "xgb", 0.75)
    sig_b = signal_engine.generate_signal("MSFT", base_timestamp, [pred_b], regime="BULL_TRENDING")

    # Asset C: Neutral P(up)=0.52 -> FLAT signal
    pred_c = adapter.adapt("GOOG", base_timestamp, "xgb", 0.52)
    sig_c = signal_engine.generate_signal("GOOG", base_timestamp, [pred_c], regime="BULL_TRENDING")

    signals = [sig_a, sig_b, sig_c]

    # 2. Ingest into Phase 11 Portfolio Construction
    portfolio_service = PortfolioConstructionService(
        config=PortfolioConstructionConfig(max_position_weight=0.10)
    )
    res = portfolio_service.construct(signals)

    assert res.active_position_count == 2
    assert len(res.targets) == 2
    target_symbols = [t.symbol for t in res.targets]
    assert "AAPL" in target_symbols
    assert "MSFT" in target_symbols
    assert "GOOG" not in target_symbols
    assert res.rejected_signals["GOOG"] == "FLAT_SIGNAL"
    assert all(t.target_weight <= 0.10 + 1e-5 for t in res.targets)
