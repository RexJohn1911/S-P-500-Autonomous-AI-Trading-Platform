"""
Unit and Integration Tests for Phase 10 — Signal Engine.
Verifies schemas, prediction normalization, ensemble aggregation, regime gating, explainability, causal integrity, and storage persistence.
Includes dedicated regression tests for Phase 10 Audit Pass.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pytest

from backend.app.regime.schemas import MarketRegimeState, MarketRegimeType, TrendState, VolatilityState
from backend.app.strategy import (
    BinaryClassificationAdapter,
    DirectPredictionAdapter,
    ReasonCode,
    RegimeGate,
    SignalCandidate,
    SignalDirection,
    SignalEngine,
    SignalEngineConfig,
    SignalEnsemble,
    SignalEvaluationReport,
    SignalEvaluator,
    SignalStorage,
    StandardizedPrediction,
)


@pytest.fixture
def base_timestamp():
    return datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def default_config():
    return SignalEngineConfig(
        long_threshold=0.20,
        short_threshold=-0.20,
        min_confidence=0.25,
        min_active_models=1,
        min_agreement=0.50,
        default_horizon=5,
        model_weights={
            "xgboost": 0.30,
            "lightgbm": 0.30,
            "random_forest": 0.20,
            "mlp": 0.20,
        },
        regime_gating_enabled=True,
        counter_trend_policy="filter",
        sideways_confidence_multiplier=1.30,
        version="signal-v1",
    )


# =====================================================================
# 1. SCHEMAS & SERIALIZATION TESTS
# =====================================================================

def test_standardized_prediction_lifecycle(base_timestamp):
    """Verify StandardizedPrediction validation, bounds, and round-trip serialization."""
    pred = StandardizedPrediction(
        symbol="SPY",
        timestamp=base_timestamp,
        model_name="xgboost",
        probability_up=0.75,
        directional_score=0.50,
        confidence=0.50,
        expected_return=0.012,
        model_type="gradient_boosting",
        weight=1.5,
    )
    d = pred.to_dict()
    assert d["symbol"] == "SPY"
    assert d["probability_up"] == 0.75
    assert d["directional_score"] == 0.50

    restored = StandardizedPrediction.from_dict(d)
    assert restored.symbol == pred.symbol
    assert restored.probability_up == pred.probability_up
    assert restored.directional_score == pred.directional_score

    # Out of bounds validation
    with pytest.raises(ValueError):
        StandardizedPrediction(
            symbol="SPY",
            timestamp=base_timestamp,
            model_name="bad",
            probability_up=1.5,
            directional_score=0.0,
            confidence=0.0,
        )


def test_signal_candidate_lifecycle(base_timestamp):
    """Verify SignalCandidate schema, round-trip serialization, and immutability of constraints."""
    cand = SignalCandidate(
        timestamp=base_timestamp,
        symbol="SPY",
        signal=SignalDirection.LONG,
        signal_strength=0.65,
        confidence=0.72,
        direction_probability=0.825,
        expected_return=0.008,
        regime="BULL_TRENDING",
        model_contributions={"xgboost": 0.4, "lightgbm": 0.25},
        model_agreement=1.0,
        bullish_votes=2,
        bearish_votes=0,
        neutral_votes=0,
        active_model_count=2,
        reason_codes=[ReasonCode.BULLISH_ENSEMBLE_SCORE.value, ReasonCode.MODEL_CONSENSUS.value],
        signal_horizon=5,
        signal_version="signal-v1",
    )
    d = cand.to_dict()
    assert d["signal"] == "LONG"
    assert d["signal_strength"] == 0.65
    assert d["confidence"] == 0.72

    restored = SignalCandidate.from_dict(d)
    assert restored.signal == SignalDirection.LONG
    assert restored.symbol == "SPY"
    assert restored.active_model_count == 2
    assert ReasonCode.MODEL_CONSENSUS.value in restored.reason_codes


def test_signal_engine_config_validation():
    """Verify invalid configuration throws proper descriptive errors."""
    with pytest.raises(ValueError):
        # short_threshold >= long_threshold
        SignalEngineConfig(long_threshold=0.10, short_threshold=0.20)

    with pytest.raises(ValueError):
        # min_active_models < 1
        SignalEngineConfig(min_active_models=0)

    with pytest.raises(ValueError):
        # negative model weight
        SignalEngineConfig(model_weights={"xgb": -0.5})


# =====================================================================
# 2. ADAPTERS & PREDICTION NORMALIZATION
# =====================================================================

def test_binary_classification_adapter(base_timestamp):
    """Verify binary probability mapping to [-1, 1] score and [0, 1] confidence."""
    adapter = BinaryClassificationAdapter()

    # 1. Strong Bullish P(up) = 0.85 -> score = 0.70, confidence = 0.70
    p1 = adapter.adapt(
        symbol="SPY",
        timestamp=base_timestamp,
        model_name="rf",
        raw_output=0.85,
    )
    assert pytest.approx(p1.directional_score, abs=1e-4) == 0.70
    assert pytest.approx(p1.confidence, abs=1e-4) == 0.70
    assert p1.expected_return is None  # Classification models never invent expected returns

    # 2. Neutral P(up) = 0.50 -> score = 0.0, confidence = 0.0
    p2 = adapter.adapt(
        symbol="SPY",
        timestamp=base_timestamp,
        model_name="rf",
        raw_output=0.50,
    )
    assert pytest.approx(p2.directional_score, abs=1e-4) == 0.0
    assert pytest.approx(p2.confidence, abs=1e-4) == 0.0

    # 3. Strong Bearish P(up) = 0.10 -> score = -0.80, confidence = 0.80
    p3 = adapter.adapt(
        symbol="SPY",
        timestamp=base_timestamp,
        model_name="rf",
        raw_output=np.array([0.90, 0.10]),  # [P(0), P(1)]
    )
    assert pytest.approx(p3.directional_score, abs=1e-4) == -0.80
    assert pytest.approx(p3.confidence, abs=1e-4) == 0.80


# =====================================================================
# 3. ENSEMBLE COMBINATION & MODEL AGREEMENT
# =====================================================================

def test_ensemble_weight_normalization_and_aggregation(base_timestamp):
    """Verify ensemble dynamically normalizes weights for available models."""
    weights = {"m1": 0.40, "m2": 0.60, "m3": 0.20}
    ensemble = SignalEnsemble(configured_weights=weights)

    # Only m1 and m2 are active (m3 is missing)
    preds = [
        StandardizedPrediction("SPY", base_timestamp, "m1", 0.80, 0.60, 0.60),
        StandardizedPrediction("SPY", base_timestamp, "m2", 0.70, 0.40, 0.40),
    ]
    res = ensemble.aggregate(preds)

    # Re-normalized weights: m1 = 0.4/1.0 = 0.4, m2 = 0.6/1.0 = 0.6
    # Expected score = 0.4 * 0.60 + 0.6 * 0.40 = 0.24 + 0.24 = 0.48
    assert pytest.approx(res["ensemble_score"], abs=1e-4) == 0.48
    assert res["active_model_count"] == 2
    assert res["bullish_votes"] == 2
    assert res["bearish_votes"] == 0
    assert res["neutral_votes"] == 0
    assert res["bullish_votes"] + res["bearish_votes"] + res["neutral_votes"] == res["active_model_count"]
    assert res["model_agreement"] == 1.0


def test_ensemble_model_disagreement(base_timestamp):
    """Verify conflicting models reduce agreement ratio and dampen confidence."""
    ensemble = SignalEnsemble()
    preds = [
        StandardizedPrediction("SPY", base_timestamp, "m1", 0.90, 0.80, 0.80),
        StandardizedPrediction("SPY", base_timestamp, "m2", 0.10, -0.80, 0.80),
    ]
    res = ensemble.aggregate(preds)
    assert pytest.approx(res["ensemble_score"], abs=1e-4) == 0.0
    assert res["model_agreement"] == 0.50
    assert res["bullish_votes"] == 1
    assert res["bearish_votes"] == 1
    assert res["neutral_votes"] == 0
    assert res["bullish_votes"] + res["bearish_votes"] + res["neutral_votes"] == res["active_model_count"]


# =====================================================================
# 4. EXPECTED RETURN AUDIT TESTS (FIX #5)
# =====================================================================

def test_expected_return_not_fabricated_from_probability(base_timestamp):
    """
    AUDIT FIX #5 CASE 1 & 3:
    When only classification probabilities / directional scores are available,
    expected_return must remain None and NEVER be fabricated from probability or score.
    """
    adapter = BinaryClassificationAdapter()
    pred1 = adapter.adapt("SPY", base_timestamp, "m1", 0.85)  # P(up) = 0.85, score = 0.70
    pred2 = adapter.adapt("SPY", base_timestamp, "m2", 0.75)  # P(up) = 0.75, score = 0.50

    assert pred1.expected_return is None
    assert pred2.expected_return is None

    engine = SignalEngine()
    sig = engine.generate_signal("SPY", base_timestamp, [pred1, pred2])

    assert sig.signal == SignalDirection.LONG
    assert sig.signal_strength > 0.50
    assert sig.expected_return is None  # MUST be None, never 0.60 or 0.80!


def test_expected_return_legitimately_propagated_when_supplied(base_timestamp):
    """
    AUDIT FIX #5 CASE 2:
    When upstream models legitimately supply an expected return forecast,
    it is correctly weighted and propagated.
    """
    direct_adapter = DirectPredictionAdapter()
    p1 = direct_adapter.adapt(
        "SPY",
        base_timestamp,
        "reg_model_1",
        {"probability_up": 0.80, "directional_score": 0.60, "expected_return": 0.012},
        weight=1.0,
    )
    p2 = direct_adapter.adapt(
        "SPY",
        base_timestamp,
        "reg_model_2",
        {"probability_up": 0.70, "directional_score": 0.40, "expected_return": 0.008},
        weight=1.0,
    )

    engine = SignalEngine()
    sig = engine.generate_signal("SPY", base_timestamp, [p1, p2])

    assert sig.signal == SignalDirection.LONG
    assert sig.expected_return is not None
    assert pytest.approx(sig.expected_return, abs=1e-5) == 0.010  # Weighted average (0.012 + 0.008)/2


# =====================================================================
# 5. THRESHOLDS, CONFIDENCE & ABSTENTION (FLAT)
# =====================================================================

def test_signal_generation_thresholds(default_config, base_timestamp):
    """Verify LONG, SHORT, and FLAT outcomes according to score thresholds."""
    engine = SignalEngine(config=default_config)

    # 1. Bullish above threshold -> LONG
    p_long = [
        StandardizedPrediction("SPY", base_timestamp, "xgboost", 0.75, 0.50, 0.50),
        StandardizedPrediction("SPY", base_timestamp, "lightgbm", 0.70, 0.40, 0.40),
    ]
    sig_long = engine.generate_signal("SPY", base_timestamp, p_long)
    assert sig_long.signal == SignalDirection.LONG
    assert sig_long.signal_strength > 0.20
    assert ReasonCode.BULLISH_ENSEMBLE_SCORE.value in sig_long.reason_codes

    # 2. Bearish below threshold -> SHORT
    p_short = [
        StandardizedPrediction("SPY", base_timestamp, "xgboost", 0.20, -0.60, 0.60),
        StandardizedPrediction("SPY", base_timestamp, "lightgbm", 0.25, -0.50, 0.50),
    ]
    sig_short = engine.generate_signal("SPY", base_timestamp, p_short)
    assert sig_short.signal == SignalDirection.SHORT
    assert sig_short.signal_strength < -0.20
    assert ReasonCode.BEARISH_ENSEMBLE_SCORE.value in sig_short.reason_codes

    # 3. Neutral between thresholds -> FLAT
    p_neutral = [
        StandardizedPrediction("SPY", base_timestamp, "xgboost", 0.53, 0.06, 0.06),
        StandardizedPrediction("SPY", base_timestamp, "lightgbm", 0.52, 0.04, 0.04),
    ]
    sig_flat = engine.generate_signal("SPY", base_timestamp, p_neutral)
    assert sig_flat.signal == SignalDirection.FLAT
    assert ReasonCode.NEUTRAL_SCORE.value in sig_flat.reason_codes


def test_signal_generation_low_confidence_abstention(default_config, base_timestamp):
    """Verify low confidence forces FLAT even if directional score is high."""
    config = SignalEngineConfig(
        long_threshold=0.20,
        short_threshold=-0.20,
        min_confidence=0.80,
    )
    engine = SignalEngine(config=config)

    preds = [
        StandardizedPrediction("SPY", base_timestamp, "m1", 0.65, 0.30, 0.30),
    ]
    sig = engine.generate_signal("SPY", base_timestamp, preds)
    assert sig.signal == SignalDirection.FLAT
    assert ReasonCode.LOW_CONFIDENCE.value in sig.reason_codes


def test_signal_generation_insufficient_models_abstention(base_timestamp):
    """Verify insufficient active models triggers FLAT with reason code."""
    config = SignalEngineConfig(min_active_models=3)
    engine = SignalEngine(config=config)

    # Only 2 models provided
    preds = [
        StandardizedPrediction("SPY", base_timestamp, "m1", 0.90, 0.80, 0.80),
        StandardizedPrediction("SPY", base_timestamp, "m2", 0.90, 0.80, 0.80),
    ]
    sig = engine.generate_signal("SPY", base_timestamp, preds)
    assert sig.signal == SignalDirection.FLAT
    assert ReasonCode.INSUFFICIENT_ACTIVE_MODELS.value in sig.reason_codes


def test_signal_generation_disagreement_abstention(base_timestamp):
    """Verify agreement below threshold triggers FLAT."""
    config = SignalEngineConfig(min_agreement=0.70)
    engine = SignalEngine(config=config)

    # 1 bullish, 1 bearish -> agreement = 0.50 < 0.70
    preds = [
        StandardizedPrediction("SPY", base_timestamp, "m1", 0.80, 0.60, 0.60),
        StandardizedPrediction("SPY", base_timestamp, "m2", 0.20, -0.60, 0.60),
    ]
    sig = engine.generate_signal("SPY", base_timestamp, preds)
    assert sig.signal == SignalDirection.FLAT
    assert ReasonCode.MODEL_DISAGREEMENT.value in sig.reason_codes


def test_signal_generation_zero_predictions(base_timestamp):
    """Verify graceful FLAT return when zero predictions are available."""
    engine = SignalEngine()
    sig = engine.generate_signal("SPY", base_timestamp, [])
    assert sig.signal == SignalDirection.FLAT
    assert ReasonCode.NO_PREDICTIONS_AVAILABLE.value in sig.reason_codes


# =====================================================================
# 6. REGIME-AWARE GATING TESTS (PHASE 09 INTEGRATION AUDIT)
# =====================================================================

def test_regime_gate_bull_trend(base_timestamp):
    """Verify BULL regime allows LONG and filters counter-trend SHORT under filter policy."""
    config = SignalEngineConfig(counter_trend_policy="filter", regime_gating_enabled=True)
    engine = SignalEngine(config=config)

    # Bullish signal in BULL regime -> LONG
    p_long = [StandardizedPrediction("SPY", base_timestamp, "m1", 0.80, 0.60, 0.60)]
    sig_long = engine.generate_signal("SPY", base_timestamp, p_long, regime=MarketRegimeType.BULL_TRENDING)
    assert sig_long.signal == SignalDirection.LONG
    assert ReasonCode.REGIME_PERMITTED_BULL.value in sig_long.reason_codes

    # Bearish signal in BULL regime -> FLAT (blocked counter-trend)
    p_short = [StandardizedPrediction("SPY", base_timestamp, "m1", 0.20, -0.60, 0.60)]
    sig_short = engine.generate_signal("SPY", base_timestamp, p_short, regime=MarketRegimeType.BULL_TRENDING)
    assert sig_short.signal == SignalDirection.FLAT
    assert ReasonCode.REGIME_BLOCKED_COUNTER_TREND.value in sig_short.reason_codes


def test_regime_gate_bear_trend(base_timestamp):
    """Verify BEAR regime allows SHORT and filters counter-trend LONG under filter policy."""
    config = SignalEngineConfig(counter_trend_policy="filter", regime_gating_enabled=True)
    engine = SignalEngine(config=config)

    # Bearish signal in BEAR regime -> SHORT
    p_short = [StandardizedPrediction("SPY", base_timestamp, "m1", 0.20, -0.60, 0.60)]
    sig_short = engine.generate_signal("SPY", base_timestamp, p_short, regime=MarketRegimeType.BEAR_TRENDING)
    assert sig_short.signal == SignalDirection.SHORT
    assert ReasonCode.REGIME_PERMITTED_BEAR.value in sig_short.reason_codes

    # Bullish signal in BEAR regime -> FLAT (blocked counter-trend)
    p_long = [StandardizedPrediction("SPY", base_timestamp, "m1", 0.80, 0.60, 0.60)]
    sig_long = engine.generate_signal("SPY", base_timestamp, p_long, regime=MarketRegimeType.BEAR_TRENDING)
    assert sig_long.signal == SignalDirection.FLAT
    assert ReasonCode.REGIME_BLOCKED_COUNTER_TREND.value in sig_long.reason_codes


def test_regime_gate_sideways_filter(base_timestamp):
    """Verify SIDEWAYS regime enforces elevated confidence multiplier."""
    config = SignalEngineConfig(
        min_confidence=0.30,
        sideways_confidence_multiplier=1.50,  # requires 0.45 confidence
    )
    engine = SignalEngine(config=config)

    # Moderate confidence = 0.36 (passes normal 0.30, fails sideways 0.45)
    preds = [StandardizedPrediction("SPY", base_timestamp, "m1", 0.68, 0.36, 0.36)]
    sig = engine.generate_signal("SPY", base_timestamp, preds, regime=MarketRegimeType.SIDEWAYS_NEUTRAL)
    assert sig.signal == SignalDirection.FLAT
    assert ReasonCode.REGIME_SIDEWAYS_FILTER.value in sig.reason_codes


def test_regime_gate_raw_numeric_cluster_id_safety(base_timestamp):
    """
    AUDIT FIX #3:
    Passing a raw numeric cluster ID (e.g. 0, 1) instead of semantic mapping
    must NOT accidentally trigger BULL or BEAR; it must safely fall back to REGIME_UNKNOWN_DEFAULT.
    """
    engine = SignalEngine()
    preds = [StandardizedPrediction("SPY", base_timestamp, "m1", 0.80, 0.60, 0.60)]

    # Pass raw integer 0 (which is an unmapped cluster ID)
    sig = engine.generate_signal("SPY", base_timestamp, preds, regime=0)
    assert sig.signal == SignalDirection.LONG
    assert ReasonCode.REGIME_UNKNOWN_DEFAULT.value in sig.reason_codes

    # Pass unmapped cluster string "CLUSTER_0"
    sig_clust = engine.generate_signal("SPY", base_timestamp, preds, regime="CLUSTER_0")
    assert sig_clust.signal == SignalDirection.LONG
    assert ReasonCode.REGIME_UNKNOWN_DEFAULT.value in sig_clust.reason_codes


# =====================================================================
# 7. TEMPORAL LEAKAGE & CAUSAL INTEGRITY TESTS
# =====================================================================

def test_causal_signal_immutability_under_future_mutation():
    """
    LEAKAGE AUDIT TEST:
    Modifying future predictions at t+1 must NOT change signal generated at t.
    """
    t0 = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 15, 11, 0, 0, tzinfo=timezone.utc)

    engine = SignalEngine()

    preds_t0 = [
        StandardizedPrediction("SPY", t0, "m1", 0.80, 0.60, 0.60),
        StandardizedPrediction("SPY", t0, "m2", 0.75, 0.50, 0.50),
    ]

    # Baseline signal at t0
    sig_t0_baseline = engine.generate_signal("SPY", t0, preds_t0)

    # Now simulate future t1 with wild mutations
    preds_t1_v1 = [StandardizedPrediction("SPY", t1, "m1", 0.99, 0.98, 0.98)]
    preds_t1_v2 = [StandardizedPrediction("SPY", t1, "m1", 0.01, -0.98, 0.98)]

    # Generate sequence with future v1
    seq_v1 = engine.generate_signals_sequence("SPY", [t0, t1], [preds_t0, preds_t1_v1])
    # Generate sequence with future v2
    seq_v2 = engine.generate_signals_sequence("SPY", [t0, t1], [preds_t0, preds_t1_v2])

    # Assert t0 output is 100% identical regardless of future values
    assert seq_v1[0].to_dict() == sig_t0_baseline.to_dict()
    assert seq_v2[0].to_dict() == sig_t0_baseline.to_dict()
    assert seq_v1[0].signal == sig_t0_baseline.signal
    assert seq_v1[0].signal_strength == sig_t0_baseline.signal_strength


def test_future_regime_mutation_invariance():
    """
    LEAKAGE AUDIT TEST:
    Modifying the future regime at t+1 must have zero effect on signal generated at t.
    """
    t0 = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 15, 11, 0, 0, tzinfo=timezone.utc)

    engine = SignalEngine()
    preds_t0 = [StandardizedPrediction("SPY", t0, "m1", 0.80, 0.60, 0.60)]
    preds_t1 = [StandardizedPrediction("SPY", t1, "m1", 0.80, 0.60, 0.60)]

    # Sequence with future regime BEAR
    seq1 = engine.generate_signals_sequence(
        "SPY",
        [t0, t1],
        [preds_t0, preds_t1],
        regimes_by_time=["BULL_TRENDING", "BEAR_TRENDING"],
    )

    # Sequence with future regime BULL
    seq2 = engine.generate_signals_sequence(
        "SPY",
        [t0, t1],
        [preds_t0, preds_t1],
        regimes_by_time=["BULL_TRENDING", "BULL_TRENDING"],
    )

    # Signal at t0 must be identical in both sequences
    assert seq1[0].to_dict() == seq2[0].to_dict()


def test_future_target_mutation_invariance():
    """
    LEAKAGE AUDIT TEST:
    Realized forward returns or labels only evaluate signals post-facto;
    they cannot leak into or alter signal generation.
    """
    t0 = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    engine = SignalEngine()
    preds = [StandardizedPrediction("SPY", t0, "m1", 0.80, 0.60, 0.60)]

    sig_baseline = engine.generate_signal("SPY", t0, preds)

    # Evaluating with positive returns vs negative returns
    rep_pos = engine.evaluate([sig_baseline], future_returns=[0.05])
    rep_neg = engine.evaluate([sig_baseline], future_returns=[-0.05])

    # The generated signal object itself is completely unaffected
    assert sig_baseline.signal == SignalDirection.LONG
    assert sig_baseline.signal_strength > 0.0
    assert rep_pos.directional_accuracy == 1.0
    assert rep_neg.directional_accuracy == 0.0


def test_symbol_isolation(base_timestamp):
    """Verify symbol signals remain completely isolated without cross-talk."""
    engine = SignalEngine()

    p_spy = [StandardizedPrediction("SPY", base_timestamp, "m1", 0.85, 0.70, 0.70)]
    p_qqq = [StandardizedPrediction("QQQ", base_timestamp, "m1", 0.15, -0.70, 0.70)]

    sig_spy = engine.generate_signal("SPY", base_timestamp, p_spy)
    sig_qqq = engine.generate_signal("QQQ", base_timestamp, p_qqq)

    assert sig_spy.symbol == "SPY"
    assert sig_spy.signal == SignalDirection.LONG
    assert sig_qqq.symbol == "QQQ"
    assert sig_qqq.signal == SignalDirection.SHORT


def test_signal_engine_determinism(base_timestamp):
    """Verify identical inputs yield identical outputs bit-for-bit."""
    engine = SignalEngine()
    preds = [
        StandardizedPrediction("SPY", base_timestamp, "xgb", 0.78, 0.56, 0.56),
        StandardizedPrediction("SPY", base_timestamp, "lgb", 0.72, 0.44, 0.44),
    ]

    out1 = engine.generate_signal("SPY", base_timestamp, preds, regime="BULL_TRENDING")
    out2 = engine.generate_signal("SPY", base_timestamp, preds, regime="BULL_TRENDING")

    assert out1.to_dict() == out2.to_dict()


# =====================================================================
# 8. SIGNAL EVALUATION DIAGNOSTICS TESTS
# =====================================================================

def test_signal_evaluator_metrics(base_timestamp):
    """Verify diagnostic evaluation report calculation."""
    t0 = base_timestamp
    t1 = datetime(2025, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    signals = [
        SignalCandidate(t0, "SPY", SignalDirection.LONG, 0.6, 0.6, 0.8, reason_codes=["BULL"]),
        SignalCandidate(t1, "SPY", SignalDirection.SHORT, -0.6, 0.6, 0.2, reason_codes=["BEAR"]),
        SignalCandidate(t2, "SPY", SignalDirection.FLAT, 0.0, 0.1, 0.5, reason_codes=["NEUTRAL"]),
    ]
    future_returns = [0.02, -0.015, 0.001]
    realized_dirs = [1, 0, 1]

    report = SignalEvaluator.evaluate(
        signals=signals,
        future_returns=future_returns,
        realized_directions=realized_dirs,
    )

    assert report.total_signals == 3
    assert report.long_count == 1
    assert report.short_count == 1
    assert report.flat_count == 1
    assert report.directional_accuracy == 1.0  # Both active signals were correct
    assert report.mean_forward_return_long == 0.02
    assert report.mean_forward_return_short == -0.015
    assert report.return_spread == 0.035
    assert report.metadata["type"] == "diagnostic_signal_evaluation"


# =====================================================================
# 9. STORAGE PERSISTENCE TESTS
# =====================================================================

def test_signal_storage_roundtrip(tmp_path):
    """Verify persistence and retrieval of SignalEngineConfig and evaluation reports."""
    storage = SignalStorage(base_storage_dir=tmp_path)
    config = SignalEngineConfig(
        long_threshold=0.25,
        short_threshold=-0.25,
        min_confidence=0.35,
        version="signal-test-v1",
    )

    # Save config
    storage.save_config(config, metadata={"author": "quant_researcher"})

    # Load config
    loaded_config = storage.load_config(version="signal-test-v1")
    assert loaded_config.long_threshold == 0.25
    assert loaded_config.short_threshold == -0.25
    assert loaded_config.min_confidence == 0.35

    # Save evaluation report
    report = SignalEvaluationReport(
        total_signals=100,
        long_count=40,
        short_count=30,
        flat_count=30,
        long_percentage=0.40,
        short_percentage=0.30,
        flat_percentage=0.30,
        directional_accuracy=0.62,
    )
    storage.save_evaluation_report(report, version="signal-test-v1")

    loaded_report = storage.load_evaluation_report(version="signal-test-v1")
    assert loaded_report.total_signals == 100
    assert loaded_report.directional_accuracy == 0.62


# =====================================================================
# 10. FULL PIPELINE INTEGRATION TEST (PHASE 06 -> 07/08 -> 09 -> 10)
# =====================================================================

def test_full_pipeline_features_models_regime_signal_synthesis():
    """
    End-to-End test synthesizing:
    - Phase 06 Feature Engineering
    - Phase 07 Baseline ML Models
    - Phase 09 Market Regime Detector
    - Phase 10 Signal Engine
    """
    import pandas as pd
    from backend.app.data.models import BarData, TimeFrame
    from backend.app.features.engine import FeatureEngine
    from backend.app.models.dataset import MLDatasetBuilder, TimeSeriesSplitter
    from backend.app.models.random_forest import RandomForestModel
    from backend.app.models.schemas import TargetConfig
    from backend.app.regime.kmeans_detector import KMeansRegimeDetector
    from backend.app.regime.preprocessor import RegimeFeaturePreprocessor
    from backend.app.regime.service import MarketRegimeService

    # 1. Generate multi-regime synthetic bars
    np.random.seed(42)
    n_bars = 200
    bars = []
    price = 100.0
    start_dt = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)

    for i in range(n_bars):
        ret = np.random.normal(0.001, 0.01) if i < 100 else np.random.normal(-0.002, 0.02)
        price *= (1.0 + ret)
        bars.append(
            BarData(
                symbol="SPY",
                timeframe=TimeFrame.DAY_1,
                timestamp=start_dt + pd.Timedelta(days=i),
                open=price * 0.998,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=1_000_000,
            )
        )

    # 2. Extract features
    feature_engine = FeatureEngine()
    feature_dataset = feature_engine.generate_features(bars)

    # 3. Build ML dataset split and train model
    target_config = TargetConfig(forward_horizon=5)
    dataset_builder = MLDatasetBuilder(target_config=target_config)
    X, y, ts, feat_names, fut_rets = dataset_builder.build_dataset(feature_dataset, bars)
    splitter = TimeSeriesSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    split = splitter.split(
        X=X,
        y=y,
        timestamps=ts,
        feature_names=feat_names,
        target_config=target_config,
        symbol="SPY",
        timeframe=TimeFrame.DAY_1,
        future_returns=fut_rets,
    )

    rf_model = RandomForestModel(feature_names=split.feature_names)
    rf_model.fit(split.X_train, split.y_train)

    # 4. Prepare regime detection on train
    regime_service = MarketRegimeService(splitter=splitter)
    reg_split = regime_service.prepare_regime_split(feature_dataset, bars)

    preprocessor = RegimeFeaturePreprocessor(feature_names=reg_split.feature_names)
    X_tr_reg = preprocessor.fit_transform(reg_split.X_train_raw)
    X_te_reg = preprocessor.transform(reg_split.X_test_raw)

    regime_detector = KMeansRegimeDetector(n_clusters=2, random_state=42, feature_names=reg_split.feature_names)
    regime_detector.fit(X_tr_reg, returns=reg_split.train_returns, raw_features_df=reg_split.X_train_raw)

    test_regimes = regime_detector.detect(X_te_reg, reg_split.test_timestamps, symbol="SPY")

    # 5. Ingest into Phase 10 Signal Engine on test set
    engine = SignalEngine()
    test_signals = []
    adapter = BinaryClassificationAdapter()

    for idx, t in enumerate(split.test_timestamps):
        x_obs = split.X_test[idx:idx+1]
        prob = rf_model.predict_proba(x_obs)[0]

        pred = adapter.adapt(
            symbol="SPY",
            timestamp=t,
            model_name="random_forest",
            raw_output=prob,
        )

        reg_state = test_regimes[idx] if idx < len(test_regimes) else None
        sig = engine.generate_signal("SPY", t, [pred], regime=reg_state)
        test_signals.append(sig)

    assert len(test_signals) == split.test_size
    assert all(isinstance(s, SignalCandidate) for s in test_signals)
    assert all(s.signal in (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT) for s in test_signals)

    # Run evaluation on test signals
    report = engine.evaluate(test_signals, future_returns=split.future_returns_test)
    assert report.total_signals == split.test_size
    assert 0.0 <= report.long_percentage <= 1.0
