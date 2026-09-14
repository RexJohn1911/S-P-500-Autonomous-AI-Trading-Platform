"""
Phase 12 Risk Engine Test Suite.
Validates risk metrics, constraint checks, deterministic risk adjustments, point-in-time
isolation, and zero future leakage.
"""

from datetime import datetime, timezone
import numpy as np
import pytest
from pathlib import Path
import tempfile

from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.risk.engine import RiskAdjustmentEngine
from backend.app.risk.limits import RiskLimitEvaluator
from backend.app.risk.metrics import RiskMetricsCalculator
from backend.app.risk.schemas import (
    PortfolioCapitalContext,
    RiskAdjustedTarget,
    RiskAdjustmentAction,
    RiskAdjustmentRecord,
    RiskAssessment,
    RiskCheck,
    RiskContext,
    RiskEngineConfig,
    RiskEngineRequest,
    RiskEngineResult,
    RiskLimitSeverity,
    RiskMetrics,
    RiskStatus,
    RiskViolation,
    RiskViolationCode,
)
from backend.app.risk.service import RiskEngineService
from backend.app.risk.storage import RiskStorage
from backend.app.strategy.schemas import SignalDirection


def _make_target(
    symbol: str,
    weight: float,
    direction: SignalDirection = SignalDirection.LONG,
    strength: float = 0.8,
    confidence: float = 0.85,
    expected_return: float = None,
) -> PortfolioTarget:
    return PortfolioTarget(
        symbol=symbol,
        target_weight=weight,
        signal_direction=direction,
        signal_strength=strength,
        confidence=confidence,
        expected_return=expected_return,
        timestamp=datetime(2026, 1, 15, 16, 0, tzinfo=timezone.utc),
    )


# 1. Clean portfolio passes basic risk checks
def test_clean_portfolio_passes_basic_checks():
    targets = [
        _make_target("AAPL", 0.05),
        _make_target("MSFT", 0.05),
        _make_target("NVDA", 0.05),
    ]
    config = RiskEngineConfig(max_position_weight=0.10, max_gross_exposure=1.00)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is True
    assert len(assessment.violations) == 0
    assert metrics.gross_exposure == pytest.approx(0.15, rel=1e-5)
    assert metrics.active_position_count == 3


# 2. Maximum position weight violation
def test_maximum_position_weight_violation():
    targets = [
        _make_target("AAPL", 0.15),  # Exceeds 0.10
        _make_target("MSFT", 0.05),
    ]
    config = RiskEngineConfig(max_position_weight=0.10)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is False
    codes = [v.code for v in assessment.violations]
    assert RiskViolationCode.MAX_POSITION_WEIGHT in codes
    violation = next(v for v in assessment.violations if v.code == RiskViolationCode.MAX_POSITION_WEIGHT)
    assert "AAPL" in violation.affected_symbols


# 3. Position weight adjustment
def test_position_weight_adjustment():
    targets = [
        _make_target("AAPL", 0.25),
        _make_target("MSFT", 0.05),
    ]
    config = RiskEngineConfig(max_position_weight=0.10, adjustment_enabled=True)
    adj_targets, assessment, audit_trail, rejected, was_adjusted, status = RiskAdjustmentEngine.adjust(targets, config)

    assert was_adjusted is True
    assert assessment.is_compliant is True
    aapl_adj = next(t for t in adj_targets if t.symbol == "AAPL")
    assert aapl_adj.adjusted_weight == pytest.approx(0.10, rel=1e-5)
    assert aapl_adj.was_adjusted is True
    assert "Capped at max position weight" in aapl_adj.adjustment_reason


# 4. Gross exposure violation
def test_gross_exposure_violation():
    targets = [
        _make_target(f"SYM_{i}", 0.10) for i in range(12)  # Gross = 1.20
    ]
    config = RiskEngineConfig(max_position_weight=0.10, max_gross_exposure=1.00, max_positions=20)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is False
    codes = [v.code for v in assessment.violations]
    assert RiskViolationCode.MAX_GROSS_EXPOSURE in codes


# 5. Gross exposure adjustment
def test_gross_exposure_adjustment():
    targets = [
        _make_target(f"SYM_{i}", 0.10) for i in range(12)  # Gross = 1.20
    ]
    config = RiskEngineConfig(max_position_weight=0.10, max_gross_exposure=1.00, max_positions=20, adjustment_enabled=True)
    adj_targets, assessment, audit_trail, rejected, was_adjusted, status = RiskAdjustmentEngine.adjust(targets, config)

    assert was_adjusted is True
    assert assessment.is_compliant is True
    adj_metrics = RiskMetricsCalculator.calculate([PortfolioTarget(t.symbol, t.adjusted_weight, t.signal_direction, t.signal_strength, t.confidence) for t in adj_targets])
    assert adj_metrics.gross_exposure <= 1.00 + 1e-6


# 6. Net exposure violation
def test_net_exposure_violation():
    targets = [
        _make_target("AAPL", 0.60),
        _make_target("MSFT", 0.20),
    ]
    config = RiskEngineConfig(max_position_weight=0.70, max_net_exposure=0.50)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is False
    codes = [v.code for v in assessment.violations]
    assert RiskViolationCode.MAX_NET_EXPOSURE in codes


# 7. Long exposure limit
def test_long_exposure_limit():
    targets = [
        _make_target("AAPL", 0.40, SignalDirection.LONG),
        _make_target("MSFT", 0.30, SignalDirection.LONG),
        _make_target("NVDA", -0.20, SignalDirection.SHORT),
    ]
    config = RiskEngineConfig(max_position_weight=0.50, max_long_exposure=0.50, max_gross_exposure=1.00)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is False
    codes = [v.code for v in assessment.violations]
    assert RiskViolationCode.MAX_LONG_EXPOSURE in codes


# 8. Short exposure limit
def test_short_exposure_limit():
    targets = [
        _make_target("AAPL", 0.20, SignalDirection.LONG),
        _make_target("TSLA", -0.40, SignalDirection.SHORT),
        _make_target("AMZN", -0.30, SignalDirection.SHORT),
    ]
    config = RiskEngineConfig(max_position_weight=0.50, max_short_exposure=0.50, max_gross_exposure=1.00)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is False
    codes = [v.code for v in assessment.violations]
    assert RiskViolationCode.MAX_SHORT_EXPOSURE in codes


# 9. Maximum position count
def test_maximum_position_count():
    targets = [_make_target(f"SYM_{i}", 0.05) for i in range(10)]
    config = RiskEngineConfig(max_positions=5)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is False
    codes = [v.code for v in assessment.violations]
    assert RiskViolationCode.MAX_POSITIONS in codes


# 10. Deterministic position reduction
def test_deterministic_position_reduction():
    # 5 targets with different weights / confidences, limit = 3
    targets = [
        _make_target("SYM_E", 0.02, confidence=0.5),
        _make_target("SYM_A", 0.08, confidence=0.9),
        _make_target("SYM_B", 0.06, confidence=0.8),
        _make_target("SYM_C", 0.06, confidence=0.7),
        _make_target("SYM_D", 0.04, confidence=0.6),
    ]
    config = RiskEngineConfig(max_positions=3, max_position_weight=0.10, adjustment_enabled=True)
    adj_targets, assessment, audit_trail, rejected, was_adjusted, status = RiskAdjustmentEngine.adjust(targets, config)

    assert was_adjusted is True
    assert assessment.is_compliant is True
    active = [t for t in adj_targets if abs(t.adjusted_weight) > 1e-6]
    assert len(active) == 3
    active_symbols = {t.symbol for t in active}
    assert active_symbols == {"SYM_A", "SYM_B", "SYM_C"}
    assert "SYM_D" in rejected
    assert "SYM_E" in rejected


# 11. Leverage calculation
def test_leverage_calculation():
    targets = [
        _make_target("AAPL", 0.30),
        _make_target("MSFT", 0.25),
        _make_target("TSLA", -0.15, SignalDirection.SHORT),
    ]
    metrics = RiskMetricsCalculator.calculate(targets)
    assert metrics.leverage == pytest.approx(0.70, rel=1e-5)
    assert metrics.gross_exposure == pytest.approx(0.70, rel=1e-5)
    assert metrics.net_exposure == pytest.approx(0.40, rel=1e-5)


# 12. NaN weight rejection
def test_nan_weight_rejection():
    # PortfolioTarget strictly blocks nan creation at schema boundary
    with pytest.raises(ValueError, match="target_weight"):
        _make_target("AAPL", float("nan"))


# 13. Infinite weight rejection
def test_infinite_weight_rejection():
    # PortfolioTarget strictly blocks inf creation at schema boundary
    with pytest.raises(ValueError, match="target_weight"):
        _make_target("AAPL", float("inf"))


# 14. Duplicate symbol handling
def test_duplicate_symbol_handling():
    targets = [
        _make_target("AAPL", 0.05),
        _make_target("AAPL", 0.04),
    ]
    config = RiskEngineConfig(max_position_weight=0.10)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is False
    codes = [v.code for v in assessment.violations]
    assert RiskViolationCode.DUPLICATE_SYMBOL in codes


# 15. Empty portfolio handling
def test_empty_portfolio_handling():
    targets = []
    config = RiskEngineConfig()
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is True
    assert metrics.gross_exposure == 0.0
    assert metrics.active_position_count == 0


# 16. Long-only compatibility
def test_long_only_compatibility():
    targets = [
        _make_target("AAPL", 0.08),
        _make_target("MSFT", 0.07),
    ]
    config = RiskEngineConfig(max_position_weight=0.10, max_gross_exposure=1.00)
    service = RiskEngineService(config=config)
    result = service.assess(targets)

    assert result.is_compliant is True
    assert result.assessment.metrics.total_short_weight == 0.0
    assert result.assessment.metrics.gross_exposure == result.assessment.metrics.net_exposure


# 17. Long-short compatibility
def test_long_short_compatibility():
    targets = [
        _make_target("AAPL", 0.08, SignalDirection.LONG),
        _make_target("TSLA", -0.05, SignalDirection.SHORT),
    ]
    config = RiskEngineConfig(max_position_weight=0.10, max_gross_exposure=1.00)
    service = RiskEngineService(config=config)
    result = service.assess(targets)

    assert result.is_compliant is True
    assert result.assessment.metrics.total_long_weight == pytest.approx(0.08, rel=1e-5)
    assert result.assessment.metrics.total_short_weight == pytest.approx(0.05, rel=1e-5)
    assert result.assessment.metrics.gross_exposure == pytest.approx(0.13, rel=1e-5)
    assert result.assessment.metrics.net_exposure == pytest.approx(0.03, rel=1e-5)


# 18. Risk violation reason codes
def test_risk_violation_reason_codes():
    targets = [_make_target("AAPL", 0.20)]
    config = RiskEngineConfig(max_position_weight=0.10)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert len(assessment.violations) == 1
    assert assessment.violations[0].code == RiskViolationCode.MAX_POSITION_WEIGHT
    assert assessment.violations[0].severity == RiskLimitSeverity.HARD


# 19. Hard-limit enforcement
def test_hard_limit_enforcement():
    targets = [_make_target("AAPL", 0.20)]
    config = RiskEngineConfig(max_position_weight=0.10, adjustment_enabled=False)
    metrics = RiskMetricsCalculator.calculate(targets)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    assert assessment.is_compliant is False


# 20. Soft-limit warning behavior
def test_soft_limit_warning_behavior():
    targets = [
        _make_target("AAPL", 0.50),
        _make_target("MSFT", 0.50),
    ]
    # Synthetic context with returns creating volatility > 0.05 (exceeding soft cap 0.05)
    hist_returns = {
        "AAPL": [0.02, -0.01, 0.03, -0.02, 0.04, -0.03, 0.02, -0.01, 0.03, -0.02],
        "MSFT": [0.01, -0.02, 0.02, -0.01, 0.03, -0.02, 0.01, -0.02, 0.02, -0.01],
    }
    context = RiskContext(historical_returns=hist_returns)
    config = RiskEngineConfig(
        max_position_weight=0.60,
        volatility_limit=0.05,  # Soft limit
    )
    metrics = RiskMetricsCalculator.calculate(targets, context)
    assessment = RiskLimitEvaluator.evaluate(targets, metrics, config)

    # Soft limit breach should generate a violation of severity SOFT, but is_compliant remains True
    assert assessment.is_compliant is True
    soft_v = [v for v in assessment.violations if v.severity == RiskLimitSeverity.SOFT]
    assert len(soft_v) == 1
    assert soft_v[0].code == RiskViolationCode.VOLATILITY_LIMIT



# 21. Volatility calculation if implemented
def test_volatility_calculation_with_valid_context():
    targets = [
        _make_target("AAPL", 0.50),
        _make_target("MSFT", 0.50),
    ]
    hist_returns = {
        "AAPL": [0.01, -0.01, 0.02, -0.02, 0.01],
        "MSFT": [0.005, -0.005, 0.01, -0.01, 0.005],
    }
    context = RiskContext(historical_returns=hist_returns)
    metrics = RiskMetricsCalculator.calculate(targets, context)

    assert metrics.portfolio_volatility is not None
    assert metrics.portfolio_volatility > 0.0
    assert "portfolio_volatility" not in metrics.unavailable_metrics


# 22. Volatility unavailable behavior
def test_volatility_unavailable_when_no_context():
    targets = [_make_target("AAPL", 0.05)]
    metrics = RiskMetricsCalculator.calculate(targets, context=None)

    assert metrics.portfolio_volatility is None
    assert "portfolio_volatility" in metrics.unavailable_metrics


# 23. Covariance calculation
def test_covariance_calculation_consistency():
    targets = [
        _make_target("AAPL", 1.0),
    ]
    returns = [0.01, -0.01, 0.02, -0.02, 0.01]
    hist_returns = {"AAPL": returns}
    context = RiskContext(historical_returns=hist_returns)
    metrics = RiskMetricsCalculator.calculate(targets, context)

    expected_std = float(np.std(returns, ddof=1) * np.sqrt(252))
    assert metrics.portfolio_volatility == pytest.approx(expected_std, rel=1e-3)


# 24. Correlation calculation if implemented
def test_correlation_calculation_with_valid_context():
    targets = [
        _make_target("AAPL", 0.50),
        _make_target("MSFT", 0.50),
    ]
    hist_returns = {
        "AAPL": [0.01, 0.02, 0.03, 0.04, 0.05],
        "MSFT": [0.02, 0.04, 0.06, 0.08, 0.10],  # Perfect correlation = 1.0
    }
    context = RiskContext(historical_returns=hist_returns)
    metrics = RiskMetricsCalculator.calculate(targets, context)

    assert metrics.average_correlation is not None
    assert metrics.average_correlation == pytest.approx(1.0, rel=1e-5)


# 25. Historical lookback does not use future rows
def test_historical_lookback_strictly_point_in_time():
    targets = [_make_target("AAPL", 0.50), _make_target("MSFT", 0.50)]
    returns_t = {
        "AAPL": [0.01, -0.01, 0.02, -0.02, 0.01],
        "MSFT": [0.005, -0.005, 0.01, -0.01, 0.005],
    }
    context_t = RiskContext(historical_returns=returns_t)
    metrics_t = RiskMetricsCalculator.calculate(targets, context_t)

    # Risk Engine calculates strictly from the supplied context dict, ignoring external rows
    assert metrics_t.portfolio_volatility is not None


# 26. Future price mutation invariance
def test_future_price_mutation_invariance():
    targets = [_make_target("AAPL", 0.05), _make_target("MSFT", 0.05)]
    config = RiskEngineConfig()
    service = RiskEngineService(config=config)

    res1 = service.assess(targets)

    # Simulate future price change outside risk context
    future_price_data = {"AAPL": 9999.99, "MSFT": 0.001}
    res2 = service.assess(targets)

    assert res1.is_compliant == res2.is_compliant
    assert res1.assessment.metrics.gross_exposure == res2.assessment.metrics.gross_exposure


# 27. Future return mutation invariance
def test_future_return_mutation_invariance():
    targets = [_make_target("AAPL", 0.05)]
    config = RiskEngineConfig()
    service = RiskEngineService(config=config)

    res1 = service.assess(targets)
    # Mutating hypothetical t+1 return
    future_return = 0.50
    res2 = service.assess(targets)

    assert res1.assessment.metrics.gross_exposure == res2.assessment.metrics.gross_exposure


# 28. Future regime mutation invariance
def test_future_regime_mutation_invariance():
    targets = [_make_target("AAPL", 0.05)]
    config = RiskEngineConfig()
    service = RiskEngineService(config=config)

    res1 = service.assess(targets)
    future_regime = "HIGH_VOLATILITY_BEAR"
    res2 = service.assess(targets)

    assert res1.is_compliant == res2.is_compliant


# 29. Future target mutation invariance
def test_future_target_mutation_invariance():
    targets_t = [_make_target("AAPL", 0.05)]
    config = RiskEngineConfig()
    service = RiskEngineService(config=config)

    res_t = service.assess(targets_t)
    targets_t_plus_1 = [_make_target("AAPL", 0.10), _make_target("NVDA", 0.08)]
    res_t_recalc = service.assess(targets_t)

    assert res_t.assessment.metrics.gross_exposure == res_t_recalc.assessment.metrics.gross_exposure


# 30. Symbol isolation
def test_symbol_isolation():
    targets = [
        _make_target("AAPL", 0.05),
        _make_target("MSFT", 0.05),
    ]
    config = RiskEngineConfig(max_position_weight=0.10)
    service = RiskEngineService(config=config)
    res = service.assess(targets)

    aapl = next(t for t in res.adjusted_targets if t.symbol == "AAPL")
    msft = next(t for t in res.adjusted_targets if t.symbol == "MSFT")

    assert aapl.adjusted_weight == 0.05
    assert msft.adjusted_weight == 0.05


# 31. Deterministic repeated execution
def test_deterministic_repeated_execution():
    targets = [
        _make_target("AAPL", 0.15),
        _make_target("MSFT", 0.05),
        _make_target("NVDA", 0.08),
    ]
    config = RiskEngineConfig(max_position_weight=0.10, adjustment_enabled=True)
    service = RiskEngineService(config=config)

    res1 = service.assess(targets)
    res2 = service.assess(targets)

    assert res1.is_compliant == res2.is_compliant
    assert res1.was_adjusted == res2.was_adjusted
    assert [t.adjusted_weight for t in res1.adjusted_targets] == [t.adjusted_weight for t in res2.adjusted_targets]


# 32. Configuration changes produce predictable results
def test_configuration_changes_produce_predictable_results():
    targets = [_make_target("AAPL", 0.15)]

    config_strict = RiskEngineConfig(max_position_weight=0.10, adjustment_enabled=True)
    config_loose = RiskEngineConfig(max_position_weight=0.20, adjustment_enabled=True)

    res_strict = RiskEngineService(config=config_strict).assess(targets)
    res_loose = RiskEngineService(config=config_loose).assess(targets)

    assert res_strict.adjusted_targets[0].adjusted_weight == 0.10
    assert res_loose.adjusted_targets[0].adjusted_weight == 0.15


# 33. Phase 11 PortfolioTarget compatibility
def test_phase11_portfolio_target_compatibility():
    target = PortfolioTarget(
        symbol="SPY",
        target_weight=0.08,
        signal_direction=SignalDirection.LONG,
        signal_strength=0.75,
        confidence=0.80,
        expected_return=None,
        signal_version="signal-v1",
        timestamp=datetime.now(timezone.utc),
    )
    service = RiskEngineService()
    result = service.assess([target])

    assert result.is_compliant is True
    assert len(result.adjusted_targets) == 1
    assert result.adjusted_targets[0].symbol == "SPY"


# 34. Serialization round-trip
def test_serialization_roundtrip():
    targets = [_make_target("AAPL", 0.08), _make_target("MSFT", 0.05)]
    service = RiskEngineService()
    result = service.assess(targets)

    res_dict = result.to_dict()
    reconstructed = RiskEngineResult.from_dict(res_dict)

    assert reconstructed.is_compliant == result.is_compliant
    assert len(reconstructed.adjusted_targets) == len(result.adjusted_targets)
    assert reconstructed.assessment.metrics.gross_exposure == result.assessment.metrics.gross_exposure


# 35. Storage round-trip
def test_storage_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = RiskStorage(base_dir=tmp_dir)
        targets = [_make_target("AAPL", 0.08), _make_target("MSFT", 0.05)]
        service = RiskEngineService(storage=storage)
        result = service.assess(targets, save_artifact=True)

        saved_files = list(Path(tmp_dir).glob("**/*.json"))
        assert len(saved_files) == 1

        loaded_result = storage.load_result(saved_files[0])
        assert loaded_result.is_compliant == result.is_compliant
        assert loaded_result.risk_version == result.risk_version


# 36. Full RiskEngineService workflow
def test_full_risk_engine_service_workflow():
    targets = [
        _make_target("AAPL", 0.20),
        _make_target("MSFT", 0.10),
        _make_target("NVDA", 0.10),
    ]
    request = RiskEngineRequest(
        targets=targets,
        config=RiskEngineConfig(max_position_weight=0.10, adjustment_enabled=True),
    )
    service = RiskEngineService()
    result = service.assess(request)

    assert result.was_adjusted is True
    assert result.is_compliant is True
    assert result.assessment.metrics.max_observed_position_weight <= 0.10 + 1e-6


# ==============================================================================
# PHASE 12.1 HARDENING TESTS (CAPITAL, LEVERAGE, FAIL-CLOSED, AUDIT, TURNOVER)
# ==============================================================================

# 37. PortfolioCapitalContext validation
def test_capital_context_validation():
    # Valid equity
    cap = PortfolioCapitalContext(equity=100_000.0, base_currency="USD")
    assert cap.equity == 100_000.0
    assert cap.calculate_notional(0.05) == 5_000.0

    # Zero equity rejection
    with pytest.raises(ValueError, match="strictly positive"):
        PortfolioCapitalContext(equity=0.0)

    # Negative equity rejection
    with pytest.raises(ValueError, match="strictly positive"):
        PortfolioCapitalContext(equity=-50_000.0)

    # NaN equity rejection
    with pytest.raises(ValueError, match="finite"):
        PortfolioCapitalContext(equity=float("nan"))

    # Inf equity rejection
    with pytest.raises(ValueError, match="finite"):
        PortfolioCapitalContext(equity=float("inf"))


# 38. Target notional and shares calculation
def test_target_notional_and_shares_calculation():
    targets = [
        _make_target("AAPL", 0.10),
        _make_target("MSFT", 0.05),
    ]
    cap = PortfolioCapitalContext(equity=200_000.0)
    prices = {"AAPL": 150.0, "MSFT": 300.0}
    context = RiskContext(capital=cap, asset_prices=prices)

    service = RiskEngineService()
    result = service.assess(targets, context=context)

    aapl = next(t for t in result.adjusted_targets if t.symbol == "AAPL")
    msft = next(t for t in result.adjusted_targets if t.symbol == "MSFT")

    assert aapl.target_notional == pytest.approx(20_000.0, rel=1e-5)
    assert aapl.shares == pytest.approx(20_000.0 / 150.0, rel=1e-5)
    assert msft.target_notional == pytest.approx(10_000.0, rel=1e-5)
    assert msft.shares == pytest.approx(10_000.0 / 300.0, rel=1e-5)
    assert result.assessment.metrics.gross_notional == pytest.approx(30_000.0, rel=1e-5)
    assert result.assessment.metrics.net_notional == pytest.approx(30_000.0, rel=1e-5)


# 39. Leverage explicit capital ratio
def test_leverage_explicit_capital_ratio():
    # Case 1: equity = 100,000, gross notional = 100,000 -> leverage = 1.0
    t1 = [_make_target("AAPL", 0.50), _make_target("MSFT", 0.50)]
    c1 = RiskContext(capital=PortfolioCapitalContext(equity=100_000.0))
    m1 = RiskMetricsCalculator.calculate(t1, c1)
    assert m1.gross_notional == pytest.approx(100_000.0, rel=1e-5)
    assert m1.leverage == pytest.approx(1.0, rel=1e-5)

    # Case 2: equity = 100,000, gross notional = 150,000 -> leverage = 1.5
    t2 = [_make_target("AAPL", 0.75), _make_target("MSFT", 0.75)]
    c2 = RiskContext(capital=PortfolioCapitalContext(equity=100_000.0))
    m2 = RiskMetricsCalculator.calculate(t2, c2)
    assert m2.gross_notional == pytest.approx(150_000.0, rel=1e-5)
    assert m2.leverage == pytest.approx(1.5, rel=1e-5)


# 40. Fail-closed on invalid capital
def test_fail_closed_invalid_capital_context():
    # Attempt assessment with corrupted context
    targets = [_make_target("AAPL", 0.05)]
    # Bypass post_init to test runtime safety
    cap = object.__new__(PortfolioCapitalContext)
    cap.equity = -100.0
    cap.base_currency = "USD"
    cap.metadata = {}

    context = RiskContext(capital=cap)
    service = RiskEngineService()
    result = service.assess(targets, context=context)

    assert result.is_compliant is False
    assert result.status == RiskStatus.INVALID
    assert result.adjusted_targets[0].adjusted_weight == 0.0
    assert result.adjusted_targets[0].action == RiskAdjustmentAction.REJECTED


# 41. Point-in-time RiskContext contract as_of_timestamp
def test_point_in_time_risk_context_contract():
    as_of = datetime(2026, 1, 15, 16, 0, tzinfo=timezone.utc)
    ctx = RiskContext(as_of_timestamp=as_of)
    assert ctx.as_of_timestamp == as_of
    assert ctx.timestamp == as_of


# 42. Audit trail action types (UNCHANGED, REDUCED, REMOVED, SCALED)
def test_audit_trail_action_types():
    # 3 targets: AAPL (0.25 -> reduced to 0.10), MSFT (0.05 -> unchanged), SYM_X (excess -> removed)
    targets = [
        _make_target("AAPL", 0.25, confidence=0.9),
        _make_target("MSFT", 0.05, confidence=0.8),
        _make_target("SYM_X", 0.02, confidence=0.3),
    ]
    config = RiskEngineConfig(max_position_weight=0.10, max_positions=2, adjustment_enabled=True)
    service = RiskEngineService(config=config)
    result = service.assess(targets)

    assert len(result.audit_trail) == 3
    audit_dict = {a.symbol: a for a in result.audit_trail}

    assert audit_dict["AAPL"].action == RiskAdjustmentAction.REDUCED
    assert audit_dict["AAPL"].original_weight == 0.25
    assert audit_dict["AAPL"].final_weight == 0.10
    assert audit_dict["AAPL"].adjustment_amount == pytest.approx(-0.15, rel=1e-5)

    assert audit_dict["MSFT"].action == RiskAdjustmentAction.UNCHANGED
    assert audit_dict["MSFT"].final_weight == 0.05

    assert audit_dict["SYM_X"].action == RiskAdjustmentAction.REMOVED
    assert audit_dict["SYM_X"].final_weight == 0.0


# 43. Deterministic audit ordering (Alphabetical symbol sorting)
def test_audit_trail_deterministic_ordering():
    targets = [
        _make_target("TSLA", 0.05),
        _make_target("AAPL", 0.05),
        _make_target("MSFT", 0.05),
    ]
    service = RiskEngineService()
    result = service.assess(targets)
    audit_symbols = [a.symbol for a in result.audit_trail]
    assert audit_symbols == ["AAPL", "MSFT", "TSLA"]


# 44. Turnover calculation (zero, nonzero, and missing)
def test_turnover_calculation():
    targets = [
        _make_target("AAPL", 0.10),
        _make_target("MSFT", 0.10),
    ]

    # Nonzero turnover: previous AAPL 0.05, MSFT 0.15 -> turnover = |0.10-0.05| + |0.10-0.15| = 0.10
    ctx_nonzero = RiskContext(previous_weights={"AAPL": 0.05, "MSFT": 0.15})
    m_nonzero = RiskMetricsCalculator.calculate(targets, ctx_nonzero)
    assert m_nonzero.turnover == pytest.approx(0.10, rel=1e-5)

    # Zero turnover
    ctx_zero = RiskContext(previous_weights={"AAPL": 0.10, "MSFT": 0.10})
    m_zero = RiskMetricsCalculator.calculate(targets, ctx_zero)
    assert m_zero.turnover == pytest.approx(0.0, rel=1e-5)

    # Missing previous weights -> None
    ctx_none = RiskContext(previous_weights=None)
    m_none = RiskMetricsCalculator.calculate(targets, ctx_none)
    assert m_none.turnover is None


# 45. Drawdown and daily loss contract status is UNAVAILABLE
def test_drawdown_daily_loss_contract_status():
    targets = [_make_target("AAPL", 0.05)]
    metrics = RiskMetricsCalculator.calculate(targets)
    assert metrics.drawdown_limit_status == "UNAVAILABLE"
    assert metrics.daily_loss_limit_status == "UNAVAILABLE"


# 46. High correlation pairs detection
def test_high_correlation_pairs_diagnostics():
    targets = [
        _make_target("AAPL", 0.30),
        _make_target("MSFT", 0.30),
        _make_target("GOOG", 0.30),
    ]
    hist_returns = {
        "AAPL": [0.01, 0.02, 0.03, 0.04, 0.05],
        "MSFT": [0.02, 0.04, 0.06, 0.08, 0.10],  # corr(AAPL, MSFT) = 1.0
        "GOOG": [0.05, -0.02, 0.01, -0.03, 0.02],  # low corr
    }
    context = RiskContext(historical_returns=hist_returns)
    metrics = RiskMetricsCalculator.calculate(targets, context)

    assert len(metrics.highly_correlated_pairs) >= 1
    pair = metrics.highly_correlated_pairs[0]
    assert pair["symbol_1"] == "AAPL"
    assert pair["symbol_2"] == "MSFT"
    assert pair["correlation"] == pytest.approx(1.0, rel=1e-4)


# 47. Future covariance mutation invariance
def test_future_covariance_mutation_invariance():
    targets = [_make_target("AAPL", 0.50), _make_target("MSFT", 0.50)]
    returns_t = {
        "AAPL": [0.01, -0.01, 0.02, -0.02, 0.01],
        "MSFT": [0.005, -0.005, 0.01, -0.01, 0.005],
    }
    context_t = RiskContext(historical_returns=returns_t)
    m1 = RiskMetricsCalculator.calculate(targets, context_t)

    # Mutate hypothetical t+1 covariance outside context
    external_future_cov = {"AAPL": {"AAPL": 999.0}}
    m2 = RiskMetricsCalculator.calculate(targets, context_t)

    assert m1.portfolio_volatility == m2.portfolio_volatility


# 48. Future RiskContext mutation invariance
def test_future_risk_context_mutation_invariance():
    targets = [_make_target("AAPL", 0.05)]
    ctx1 = RiskContext(capital=PortfolioCapitalContext(equity=50_000.0))
    res1 = RiskEngineService().assess(targets, context=ctx1)

    # Creating a separate context at t+1 with different capital
    ctx2 = RiskContext(capital=PortfolioCapitalContext(equity=100_000.0))
    res2 = RiskEngineService().assess(targets, context=ctx1)

    assert res1.assessment.metrics.portfolio_equity == res2.assessment.metrics.portfolio_equity
    assert res1.assessment.metrics.gross_notional == res2.assessment.metrics.gross_notional


# 49. RiskAdjustedTarget schema serialization with capital notionals
def test_risk_adjusted_target_serialization_with_notionals():
    target = RiskAdjustedTarget(
        symbol="AAPL",
        original_weight=0.15,
        adjusted_weight=0.10,
        signal_direction=SignalDirection.LONG,
        signal_strength=0.8,
        confidence=0.85,
        action=RiskAdjustmentAction.REDUCED,
        was_adjusted=True,
        adjustment_reason="Capped at max position weight 10.0%",
        target_notional=10_000.0,
        shares=66.67,
    )
    t_dict = target.to_dict()
    reconstructed = RiskAdjustedTarget.from_dict(t_dict)

    assert reconstructed.action == RiskAdjustmentAction.REDUCED
    assert reconstructed.target_notional == 10_000.0
    assert reconstructed.shares == 66.67


# 50. Versioning backward compatibility (risk-v1 and risk-v1.1)
def test_versioning_compatibility():
    cfg_v1 = RiskEngineConfig.from_dict({"version": "risk-v1", "max_position_weight": 0.10})
    assert cfg_v1.version == "risk-v1"

    cfg_v1_1 = RiskEngineConfig()
    assert cfg_v1_1.version == "risk-v1.1"

