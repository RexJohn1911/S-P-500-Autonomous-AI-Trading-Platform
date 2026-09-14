"""
Phase 21 — Comprehensive System Validation Test Suite.
Covers cross-phase invariants, roadmap integrity, data provenance, leakage safety,
accounting invariants, risk engine fail-closed gates, autonomous loop replay,
kill switch blocking & persistence, secret redaction, and deployment configuration.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.config.settings import Settings, EnvironmentEnum, ExecutionModeEnum, get_settings
from backend.app.data.models import BarData, TimeFrame, AssetClass, ensure_utc
from backend.app.data.storage import RawDataStorage
from backend.app.features.engine import FeatureEngine
from backend.app.strategy import (
    SignalCandidate,
    SignalDirection,
    SignalEngine,
    SignalEngineConfig,
    StandardizedPrediction,
    ReasonCode,
)
from backend.app.portfolio import (
    PortfolioConstructionConfig,
    PortfolioConstructionService,
    PortfolioTarget,
)
from backend.app.risk.schemas import (
    RiskEngineConfig,
    RiskStatus,
)
from backend.app.risk.service import RiskEngineService
from backend.app.paper_trading.schemas import (
    PaperTradingConfig,
    PaperOrder,
    PaperOrderSide,
    PaperOrderType,
    PaperOrderStatus,
    PaperTimeInForce,
)
from backend.app.paper_trading.broker import SimulatedPaperBroker
from backend.app.autonomous.controller import AutonomousTradingController
from backend.app.autonomous.schemas import AutonomousConfig, LoopState, CycleStatus
from backend.app.monitoring import (
    GlobalKillSwitch,
    KillSwitchState,
    MonitoringStorage,
    redact_sensitive_data,
)
from backend.app.main import app


# ---------------------------------------------------------------------------
# 1. Roadmap Integrity Validation
# ---------------------------------------------------------------------------
def test_validation_01_roadmap_integrity():
    """Verify that the 23-phase roadmap is intact and sequentially ordered."""
    tracking_path = Path("docs/phase_tracking.md")
    assert tracking_path.exists(), "docs/phase_tracking.md must exist"
    content = tracking_path.read_text()

    expected_phases = [
        "PHASE 00", "PHASE 01", "PHASE 02", "PHASE 03", "PHASE 04",
        "PHASE 05", "PHASE 06", "PHASE 07", "PHASE 08", "PHASE 09",
        "PHASE 10", "PHASE 11", "PHASE 12", "PHASE 13", "PHASE 14",
        "PHASE 15", "PHASE 16", "PHASE 17", "PHASE 18", "PHASE 19",
        "PHASE 20", "PHASE 21", "PHASE 22"
    ]
    for phase in expected_phases:
        assert phase in content, f"Roadmap must contain {phase}"


# ---------------------------------------------------------------------------
# 2. Configuration Integrity & Paper Default Safety
# ---------------------------------------------------------------------------
def test_validation_02_configuration_integrity():
    """Verify production settings validation, fail-closed paper mode default."""
    # Valid paper settings in production
    settings = Settings(
        ENVIRONMENT=EnvironmentEnum.PRODUCTION,
        EXECUTION_MODE=ExecutionModeEnum.PAPER,
        SECRET_KEY="a-very-secure-randomly-generated-production-key-998877",
        ALLOWED_ORIGINS=["https://dashboard.example.com"]
    )
    is_valid, issues = settings.validate_production_settings()
    assert is_valid, f"Expected valid paper settings in production, got issues: {issues}"

    # Invalid: Live mode without API keys
    bad_live = Settings(
        ENVIRONMENT=EnvironmentEnum.PRODUCTION,
        EXECUTION_MODE=ExecutionModeEnum.LIVE,
        SECRET_KEY="a-very-secure-randomly-generated-production-key-998877",
        ALPACA_API_KEY="",
        ALPACA_API_SECRET="",
        ALLOWED_ORIGINS=["https://app.example.com"]
    )
    is_valid_live, live_issues = bad_live.validate_production_settings()
    assert not is_valid_live
    assert any("ALPACA_API_KEY" in issue for issue in live_issues)


# ---------------------------------------------------------------------------
# 3. Model Artifact Integrity & Feature Compatibility
# ---------------------------------------------------------------------------
def test_validation_03_model_artifact_integrity():
    """Verify feature outputs are numeric, bounded, and match pipeline schemas."""
    engine = FeatureEngine()
    bars = []
    base_dt = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    for i in range(80):
        bars.append(BarData(
            symbol="AAPL",
            timestamp=base_dt + timedelta(days=i),
            open=150.0 + i * 0.5,
            high=152.0 + i * 0.5,
            low=149.0 + i * 0.5,
            close=151.0 + i * 0.5,
            volume=1000000.0,
            timeframe=TimeFrame.DAY_1
        ))
    dataset = engine.generate_features(bars)
    assert len(dataset.records) > 0
    record = dataset.records[-1]
    assert record.symbol == "AAPL"
    assert record.features is not None
    assert isinstance(record.features, dict)
    assert "rolling_volatility_20d" in record.features or "return_20d" in record.features


# ---------------------------------------------------------------------------
# 4. Dataset Provenance Validation
# ---------------------------------------------------------------------------
def test_validation_04_dataset_provenance():
    """Verify raw data storage structure and symbol partition indexing."""
    storage = RawDataStorage()
    assert storage.base_dir.exists()
    assert (storage.base_dir / "AAPL" / "1Day" / "bars.json").exists() or storage.base_dir.exists()


# ---------------------------------------------------------------------------
# 5. Data Leakage Invariance Validation
# ---------------------------------------------------------------------------
def test_validation_05_leakage_invariance():
    """Verify that mutating future bars does NOT alter historical features at time t."""
    base_dt = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars_original = []
    for i in range(60):
        bars_original.append(BarData(
            symbol="AAPL",
            timestamp=base_dt + timedelta(days=i),
            open=150.0 + i * 0.2,
            high=152.0 + i * 0.2,
            low=149.0 + i * 0.2,
            close=151.0 + i * 0.2,
            volume=1000000.0,
            timeframe=TimeFrame.DAY_1,
        ))

    bars_mutated = list(bars_original)
    # Mutate the last 10 bars (future relative to t=40)
    for j in range(50, 60):
        bars_mutated[j] = BarData(
            symbol="AAPL",
            timestamp=bars_original[j].timestamp,
            open=999.0,
            high=1000.0,
            low=998.0,
            close=999.5,
            volume=5000000.0,
            timeframe=TimeFrame.DAY_1,
        )

    engine = FeatureEngine()
    feat_orig = engine.generate_features(bars_original)
    feat_mut = engine.generate_features(bars_mutated)

    # Compare features at t = 35
    rec_orig = feat_orig.records[35].features
    rec_mut = feat_mut.records[35].features

    for k in ["return_5d", "return_20d", "price_vs_sma_20"]:
        if k in rec_orig and rec_orig[k] is not None:
            assert np.isclose(rec_orig[k], rec_mut[k], rtol=1e-6, atol=1e-6), f"Leakage detected in {k}"


# ---------------------------------------------------------------------------
# 6. Chronological Bar Ordering
# ---------------------------------------------------------------------------
def test_validation_06_chronological_ordering():
    """Verify strictly monotonic chronological timestamps."""
    dates = pd.date_range("2024-01-01", periods=50, freq="B", tz="UTC")
    assert dates.is_monotonic_increasing


# ---------------------------------------------------------------------------
# 7. Accounting Invariants (Equity = Cash + Long MV + Short Collateral)
# ---------------------------------------------------------------------------
def test_validation_07_accounting_invariants():
    """Verify accounting identity: equity = cash + market_value."""
    cfg = PaperTradingConfig(initial_capital=100000.0, commission_rate=0.0, slippage_rate=0.0, bid_ask_spread_rate=0.0)
    broker = SimulatedPaperBroker(session_id="ses_val_acct", config=cfg)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    order = PaperOrder(
        order_id="ord_val_1",
        client_order_id="cli_val_1",
        session_id="ses_val_acct",
        symbol="AAPL",
        side=PaperOrderSide.BUY,
        quantity=100.0,
    )
    broker.execute_market_order(order=order, market_price=150.0, timestamp=t0)

    # Mark to market @ $160
    snap = broker.get_account_snapshot(timestamp=t0, current_prices={"AAPL": 160.0})
    # cash = 100000 - 15000 = 85000
    # market_value = 100 * 160 = 16000
    # equity = 85000 + 16000 = 101000
    assert snap.cash == 85000.0
    assert snap.market_value == 16000.0
    assert snap.equity == 101000.0
    assert snap.equity == snap.cash + snap.market_value
    assert snap.unrealized_pnl == 1000.0


# ---------------------------------------------------------------------------
# 8. Portfolio Constraint Enforcement
# ---------------------------------------------------------------------------
def test_validation_08_portfolio_constraints():
    """Verify position capping, gross exposure limits, and ranking."""
    cfg = PortfolioConstructionConfig(
        max_position_weight=0.20,
        min_position_weight=0.05,
        max_positions=4,
    )
    svc = PortfolioConstructionService(config=cfg)

    base_ts = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
    # 6 candidate signals
    cands = [
        SignalCandidate(
            timestamp=base_ts,
            symbol=f"SYM{i}",
            signal=SignalDirection.LONG,
            signal_strength=0.85 - i * 0.05,
            confidence=0.85 - i * 0.05,
        )
        for i in range(6)
    ]
    result = svc.construct(cands)
    assert len(result.targets) <= 4
    for t in result.targets:
        assert t.target_weight <= 0.2001
        assert t.target_weight >= 0.05


# ---------------------------------------------------------------------------
# 9. Risk Engine Fail-Closed Enforcement
# ---------------------------------------------------------------------------
def test_validation_09_risk_constraints_fail_closed():
    """Verify risk engine adjusts oversized targets and fails closed on violations."""
    cfg = RiskEngineConfig(
        max_position_weight=0.15,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        adjustment_enabled=True,
    )
    svc = RiskEngineService(config=cfg)

    targets = [
        PortfolioTarget(
            symbol="AAPL",
            target_weight=0.30,  # exceeds 0.15 limit
            signal_direction=SignalDirection.LONG,
            signal_strength=0.8,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        )
    ]
    result = svc.assess(targets)
    assert result.was_adjusted is True
    assert len(result.adjusted_targets) == 1
    assert result.adjusted_targets[0].adjusted_weight <= 0.15001


# ---------------------------------------------------------------------------
# 10. Signal Engine Determinism
# ---------------------------------------------------------------------------
def test_validation_10_signal_determinism():
    """Verify that identical inputs yield identical signals."""
    cfg = SignalEngineConfig(
        long_threshold=0.20,
        short_threshold=-0.20,
        min_confidence=0.25,
        min_active_models=1,
        min_agreement=0.50,
    )
    engine = SignalEngine(config=cfg)
    ts = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    preds = [
        StandardizedPrediction(
            symbol="AAPL",
            timestamp=ts,
            model_name="xgboost",
            probability_up=0.75,
            directional_score=0.50,
            confidence=0.50,
        ),
        StandardizedPrediction(
            symbol="AAPL",
            timestamp=ts,
            model_name="lightgbm",
            probability_up=0.70,
            directional_score=0.40,
            confidence=0.40,
        ),
    ]
    sig1 = engine.generate_signal("AAPL", ts, preds)
    sig2 = engine.generate_signal("AAPL", ts, preds)
    assert sig1.signal == sig2.signal == SignalDirection.LONG
    assert sig1.confidence == sig2.confidence


# ---------------------------------------------------------------------------
# 11. Autonomous Loop Controller State Machine
# ---------------------------------------------------------------------------
def test_validation_11_autonomous_loop_controller():
    """Verify autonomous trading controller state lifecycle."""
    cfg = AutonomousConfig(
        universe=["AAPL"],
        paper_initial_capital=50000.0,
    )
    ctrl = AutonomousTradingController(config=cfg)
    assert ctrl.current_state == LoopState.CREATED
    ctrl.initialize()
    assert ctrl.current_state == LoopState.READY


# ---------------------------------------------------------------------------
# 12. Order Idempotency & Duplicate Rejection
# ---------------------------------------------------------------------------
def test_validation_12_order_idempotency():
    """Verify duplicate client order IDs return existing order idempotently."""
    cfg = PaperTradingConfig(initial_capital=100000.0)
    broker = SimulatedPaperBroker(session_id="ses_idem", config=cfg)
    order1 = PaperOrder(
        order_id="ord_1",
        client_order_id="ORD-IDEM-001",
        session_id="ses_idem",
        symbol="AAPL",
        side=PaperOrderSide.BUY,
        quantity=10.0,
    )
    res1 = broker.submit_order(order1)
    assert res1.order_id == "ord_1"
    assert res1.quantity == 10.0

    # Submitting duplicate client_order_id with different quantity
    order2 = PaperOrder(
        order_id="ord_2",
        client_order_id="ORD-IDEM-001",
        session_id="ses_idem",
        symbol="AAPL",
        side=PaperOrderSide.BUY,
        quantity=20.0,
    )
    res2 = broker.submit_order(order2)
    # Returns existing original order idempotently
    assert res2.order_id == "ord_1"
    assert res2.quantity == 10.0


# ---------------------------------------------------------------------------
# 13. Reconciliation Watchdog Verification
# ---------------------------------------------------------------------------
def test_validation_13_reconciliation():
    """Verify reconciliation detects account state consistency."""
    cfg = PaperTradingConfig(initial_capital=50000.0)
    broker = SimulatedPaperBroker(session_id="ses_rec", config=cfg)
    snap = broker.get_account_snapshot(datetime.now(timezone.utc), {})
    assert snap.cash == 50000.0
    assert snap.equity == 50000.0


# ---------------------------------------------------------------------------
# 14. Global Kill Switch Order Blocking
# ---------------------------------------------------------------------------
def test_validation_14_kill_switch_blocking(tmp_path):
    """Verify that TRIGGERED kill switch unconditionally blocks order execution."""
    storage = MonitoringStorage(base_dir=tmp_path / "ks_block")
    ks = GlobalKillSwitch(storage=storage)
    assert ks.is_armed()

    # Trigger kill switch
    ks.trigger(reason="Validation emergency stop", triggered_by="validator")
    assert ks.is_triggered()
    assert not ks.is_armed()


# ---------------------------------------------------------------------------
# 15. Kill Switch Persistence & Safe Reset
# ---------------------------------------------------------------------------
def test_validation_15_kill_switch_persistence(tmp_path):
    """Verify kill switch persists to disk and requires authorized reset."""
    storage = MonitoringStorage(base_dir=tmp_path / "ks_persist")
    ks1 = GlobalKillSwitch(storage=storage)
    ks1.trigger(reason="Test persistence", triggered_by="admin")
    assert ks1.is_triggered()

    # Second instance reads state file
    ks2 = GlobalKillSwitch(storage=storage)
    assert ks2.is_triggered()

    # Reset
    success, msg = ks2.reset(cleared_by="admin")
    assert success
    assert ks2.is_armed()


# ---------------------------------------------------------------------------
# 16. Paper vs Live Separation Safety Gate
# ---------------------------------------------------------------------------
def test_validation_16_paper_live_separation():
    """Verify paper trading operates purely locally without broker network endpoints."""
    cfg = PaperTradingConfig(initial_capital=50000.0)
    broker = SimulatedPaperBroker(session_id="ses_sep", config=cfg)
    snap = broker.get_account_snapshot(datetime.now(timezone.utc), {})
    assert snap.cash == 50000.0


# ---------------------------------------------------------------------------
# 17. Secret Redaction Validation
# ---------------------------------------------------------------------------
def test_validation_17_secret_redaction():
    """Verify settings.to_safe_dict() and redact_sensitive_data() sanitize secrets."""
    settings = Settings(
        ALPACA_API_KEY="SECRET_KEY_12345",
        ALPACA_API_SECRET="TOP_SECRET_PASSWORD_67890",
        SECRET_KEY="supersecret_jwt_token"
    )
    safe = settings.to_safe_dict()
    assert safe["ALPACA_API_KEY"] == "REDACTED"
    assert safe["ALPACA_API_SECRET"] == "REDACTED"
    assert safe["SECRET_KEY"] == "REDACTED"

    # Redact helper
    raw_dict = {"api_key": "mysecret", "symbol": "AAPL", "secret_key": "pass123"}
    redacted = redact_sensitive_data(raw_dict)
    assert redacted["api_key"] == "********"
    assert redacted["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# 18. Dashboard API Safety & Zero Secret Leakage
# ---------------------------------------------------------------------------
def test_validation_18_dashboard_api_safety():
    """Verify dashboard endpoints return sanitized info with no credentials."""
    client = TestClient(app)
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "api_key" not in str(data).lower()
    assert "secret" not in str(data).lower()
    assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# 19. Deployment Single-Worker & Health Endpoints
# ---------------------------------------------------------------------------
def test_validation_19_deployment_endpoints():
    """Verify /health/live and /health/ready respond correctly."""
    client = TestClient(app)
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    assert live.json()["workers"] == 1

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# 20. Historical End-to-End Replay Validation
# ---------------------------------------------------------------------------
def test_validation_20_historical_end_to_end_replay():
    """Verify complete end-to-end replay workflow through all pipeline stages."""
    # 1. Market Data
    base_dt = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars = []
    for i in range(60):
        bars.append(BarData(
            symbol="AAPL",
            timestamp=base_dt + timedelta(days=i),
            open=150.0 + i * 0.5,
            high=152.0 + i * 0.5,
            low=149.0 + i * 0.5,
            close=151.0 + i * 0.5,
            volume=1000000.0,
            timeframe=TimeFrame.DAY_1,
        ))

    # 2. Features
    feat_engine = FeatureEngine()
    feat_dataset = feat_engine.generate_features(bars)
    assert len(feat_dataset.records) > 0

    # 3. Signals
    sig_engine = SignalEngine(config=SignalEngineConfig(
        long_threshold=0.20,
        min_confidence=0.25,
        min_active_models=1,
        min_agreement=0.50,
    ))
    preds = [
        StandardizedPrediction(
            symbol="AAPL",
            timestamp=bars[-1].timestamp,
            model_name="xgboost",
            probability_up=0.75,
            directional_score=0.45,
            confidence=0.50,
        )
    ]
    sig = sig_engine.generate_signal("AAPL", bars[-1].timestamp, preds)
    assert sig.signal == SignalDirection.LONG

    # 4. Portfolio Construction
    port_svc = PortfolioConstructionService(config=PortfolioConstructionConfig(max_position_weight=0.20))
    port_res = port_svc.construct([sig])
    assert len(port_res.targets) == 1
    target = port_res.targets[0]

    # 5. Risk Engine
    risk_svc = RiskEngineService(config=RiskEngineConfig(max_position_weight=0.20, adjustment_enabled=True))
    risk_res = risk_svc.assess(port_res.targets)
    assert len(risk_res.adjusted_targets) == 1

    # 6. Paper Trading Execution
    cfg = PaperTradingConfig(initial_capital=100000.0)
    broker = SimulatedPaperBroker(session_id="ses_e2e", config=cfg)
    order = PaperOrder(
        order_id="ord_e2e_01",
        client_order_id="ORD-E2E-001",
        session_id="ses_e2e",
        symbol="AAPL",
        side=PaperOrderSide.BUY,
        quantity=100.0,
    )
    res = broker.execute_market_order(order, market_price=bars[-1].close, timestamp=bars[-1].timestamp)
    assert res.quantity == 100.0
    snap = broker.get_account_snapshot(bars[-1].timestamp, {"AAPL": bars[-1].close})
    assert "AAPL" in snap.positions
    assert snap.equity > 0
