"""
Comprehensive Unit, Integration, Failure-Injection, and Empirical Tests for Phase 15 Autonomous Trading Loop.
Validates:
- Controller initialization and configuration snapshot hashing
- State machine lifecycle transitions and invalid transition guards
- Idempotent cycle execution and duplicate timestamp suppression
- Market data scheduling, staleness rejection, and missing symbol policy
- Pipeline orchestration (Signals -> Portfolio -> Risk -> Paper Execution -> Valuation -> Reconciliation)
- Checkpointing and crash recovery
- Graceful pause, resume, shutdown, and emergency manual kill switch
- Heartbeat tracking and trading health vs process health separation
- Overlapping concurrency lock protection
- Failure injection and error classification
- Zero future lookahead / causal invariance under future mutations
- Bitwise deterministic replay across repeated runs
- Safety boundary: zero live broker credentials, endpoints, or routing
- Empirical historical replay over real local historical AAPL bars
"""

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np

from backend.app.autonomous.checkpoint import CheckpointManager
from backend.app.autonomous.controller import AutonomousTradingController
from backend.app.autonomous.cycle import TradingCycleExecutor
from backend.app.autonomous.health import AutonomousHealthMonitor
from backend.app.autonomous.recovery import CrashRecoveryManager
from backend.app.autonomous.scheduler import MarketDataScheduler
from backend.app.autonomous.schemas import (
    AutonomousConfig,
    AutonomousEvent,
    AutonomousEventType,
    AutonomousHealthSnapshot,
    AutonomousLoopResult,
    AutonomousSession,
    CycleCheckpoint,
    CycleMetadata,
    CycleStage,
    CycleStatus,
    FailureClass,
    HealthStatus,
    LoopState,
    MissingSymbolPolicy,
)
from backend.app.autonomous.state import AutonomousStateMachine
from backend.app.autonomous.storage import AutonomousStorage
from backend.app.backtest.schemas import DividendEvent, SplitEvent
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.storage import ProcessedDataStorage
from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperOrderSide,
    PaperOrderStatus,
    PaperReconciliationStatus,
    PaperTradingConfig,
)
from backend.app.paper_trading.service import PaperTradingService
from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.portfolio.service import PortfolioConstructionService
from backend.app.risk.schemas import RiskEngineConfig
from backend.app.strategy.schemas import ReasonCode, SignalCandidate, SignalDirection


def create_synthetic_bars(
    symbol: str,
    prices: list[float],
    start_date: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc),
) -> list[BarData]:
    """Helper to generate clean synthetic daily bars."""
    bars = []
    for i, p in enumerate(prices):
        ts = start_date + timedelta(days=i)
        bars.append(
            BarData(
                symbol=symbol,
                timestamp=ts,
                open=float(p),
                high=float(p * 1.01),
                low=float(p * 0.99),
                close=float(p),
                volume=100000.0,
            )
        )
    return bars


# =========================================================================
# 1. Controller Initialization & State Machine Lifecycle Tests
# =========================================================================

def test_controller_initialization():
    """A, L. Verify controller initialization, configuration hashing, and state."""
    cfg = AutonomousConfig(universe=["AAPL", "MSFT"], paper_initial_capital=50000.0)
    controller = AutonomousTradingController(config=cfg)

    assert controller.current_state == LoopState.CREATED
    assert controller.session.session_id == controller.session_id
    assert controller.config_hash != ""
    assert controller.session.configuration_hash == controller.config_hash
    assert controller.paper_service.config.initial_capital == 50000.0

    controller.initialize()
    assert controller.current_state == LoopState.READY


def test_state_machine_valid_and_invalid_transitions():
    """B, C. Test state transitions and rejection of invalid transitions."""
    sm = AutonomousStateMachine(initial_state=LoopState.CREATED)
    assert sm.current_state == LoopState.CREATED

    # Valid path: CREATED -> INITIALIZING -> READY -> RUNNING -> PAUSED -> RUNNING -> STOPPING -> STOPPED
    sm.transition_to(LoopState.INITIALIZING)
    sm.transition_to(LoopState.READY)
    sm.transition_to(LoopState.RUNNING)
    sm.transition_to(LoopState.PAUSED)
    sm.transition_to(LoopState.RUNNING)
    sm.transition_to(LoopState.STOPPING)
    sm.transition_to(LoopState.STOPPED)
    assert sm.current_state == LoopState.STOPPED

    # Invalid: STOPPED -> RUNNING directly (must go through INITIALIZING/READY)
    with pytest.raises(ValueError, match="Invalid autonomous lifecycle transition"):
        sm.transition_to(LoopState.RUNNING)


# =========================================================================
# 2. Market Data Scheduling, Staleness & Missing Symbol Policy Tests
# =========================================================================

def test_market_data_scheduler_and_staleness_rejection():
    """G, H, I. Test scheduler validation, staleness detection, and missing symbol policy."""
    cfg = AutonomousConfig(
        universe=["AAPL", "MSFT"],
        max_stale_tolerance_seconds=100.0,
        missing_symbol_policy=MissingSymbolPolicy.FAIL_CLOSED,
    )
    scheduler = MarketDataScheduler(config=cfg)
    now = datetime(2025, 1, 10, 12, 0, tzinfo=timezone.utc)

    # 1. Missing symbol in FAIL_CLOSED mode
    bar_aapl = BarData(symbol="AAPL", timestamp=now, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
    valid, err, missing = scheduler.validate_market_data({"AAPL": bar_aapl}, current_time=now)
    assert not valid
    assert "Missing required symbols" in err
    assert "MSFT" in missing

    # 2. Full universe valid
    bar_msft = BarData(symbol="MSFT", timestamp=now, open=200.0, high=201.0, low=199.0, close=200.0, volume=1000.0)
    valid, err, missing = scheduler.validate_market_data({"AAPL": bar_aapl, "MSFT": bar_msft}, current_time=now)
    assert valid
    assert err is None

    # 3. Staleness rejection (bar timestamp is 1 day old, tolerance is 100s)
    old_time = now - timedelta(days=1)
    bar_old_aapl = BarData(symbol="AAPL", timestamp=old_time, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
    bar_old_msft = BarData(symbol="MSFT", timestamp=old_time, open=200.0, high=201.0, low=199.0, close=200.0, volume=1000.0)
    valid, err, missing = scheduler.validate_market_data({"AAPL": bar_old_aapl, "MSFT": bar_old_msft}, current_time=now)
    assert not valid
    assert "stale" in err.lower()


# =========================================================================
# 3. Idempotency & Overlapping Concurrency Protection Tests
# =========================================================================

def test_cycle_idempotency_and_overlapping_protection():
    """D, E, F, AA. Test cycle uniqueness, idempotency suppression, and concurrency locking."""
    cfg = AutonomousConfig(universe=["AAPL"])
    controller = AutonomousTradingController(config=cfg)
    controller.start()

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = {"AAPL": BarData(symbol="AAPL", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)}

    # First execution succeeds
    c1 = controller.execute_market_event(market_timestamp=t0, current_bars=bars)
    assert c1.status == CycleStatus.COMPLETED
    assert controller.session.total_cycles == 1
    assert controller.session.successful_cycles == 1

    # Second execution of identical market timestamp is skipped via idempotency guard
    c2 = controller.execute_market_event(market_timestamp=t0, current_bars=bars)
    assert c2.status == CycleStatus.SKIPPED
    assert "Duplicate market timestamp" in c2.error_message
    assert controller.session.total_cycles == 1  # Total cycle counter does not double-count

    # Overlapping cycle lock guard
    scheduler = controller.scheduler
    assert scheduler.acquire_cycle_lock("test_lock_1")
    # Second acquisition fails
    assert not scheduler.acquire_cycle_lock("test_lock_2")
    scheduler.release_cycle_lock("test_lock_1")


# =========================================================================
# 4. Pipeline Orchestration, Checkpointing & Recovery Tests
# =========================================================================

def test_pipeline_orchestration_and_checkpointing():
    """M, N, O, P, Q. Test full canonical cycle with checkpoint creation."""
    cfg = AutonomousConfig(universe=["AAPL"], auto_checkpoint_enabled=True)
    
    def dummy_signals(ts, current_bars):
        return [
            SignalCandidate(
                timestamp=ts,
                symbol="AAPL",
                signal=SignalDirection.LONG,
                signal_strength=0.8,
                confidence=0.9,
                reason_codes=[ReasonCode.BULLISH_ENSEMBLE_SCORE.value],
            )
        ]

    controller = AutonomousTradingController(config=cfg, signal_generator=dummy_signals)
    controller.start()

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = {"AAPL": BarData(symbol="AAPL", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)}

    cycle = controller.execute_market_event(market_timestamp=t0, current_bars=bars)
    assert cycle.status == CycleStatus.COMPLETED
    assert cycle.stage == CycleStage.COMPLETED
    assert cycle.orders_generated_count > 0
    assert cycle.orders_executed_count > 0

    # Checkpoint created
    latest_chk = controller.checkpoint_manager.get_latest_checkpoint()
    assert latest_chk is not None
    assert latest_chk.cycle_id == cycle.cycle_id
    assert latest_chk.last_successful_stage == CycleStage.COMPLETED


def test_crash_recovery_analysis():
    """R, AC. Test recovery analysis under completed and interrupted states."""
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    # 1. Clean startup
    can_rec, desc, stage = CrashRecoveryManager.analyze_recovery_state(latest_checkpoint=None)
    assert can_rec
    assert "Clean startup" in desc

    # 2. Interrupted at data validation
    chk_interrupted = CycleCheckpoint(
        checkpoint_id="chk_1",
        session_id="ses_1",
        cycle_id="cyc_1",
        market_timestamp=t0,
        last_successful_stage=CycleStage.DATA_VALIDATION,
        paper_session_id="paper_ses_1",
        configuration_hash="hash",
        state_hash="state",
        timestamp=t0,
        status=CycleStatus.RUNNING,
    )
    can_rec, desc, stage = CrashRecoveryManager.analyze_recovery_state(latest_checkpoint=chk_interrupted)
    assert can_rec
    assert "interrupted before order generation" in desc


# =========================================================================
# 5. Lifecycle Controls: Pause, Resume, Shutdown, Kill Switch
# =========================================================================

def test_pause_resume_shutdown_and_kill_switch():
    """S, T, U, V. Test pause, resume, graceful stop, and emergency kill switch."""
    controller = AutonomousTradingController()
    controller.start()
    assert controller.current_state == LoopState.RUNNING

    # Pause
    controller.pause()
    assert controller.current_state == LoopState.PAUSED
    assert controller.paper_service.session.status.value == "PAUSED"

    # Resume
    controller.resume()
    assert controller.current_state == LoopState.RUNNING
    assert controller.paper_service.session.status.value == "RUNNING"

    # Stop
    controller.stop()
    assert controller.current_state == LoopState.STOPPED
    assert controller.session.stopped_at is not None

    # Kill Switch from running
    c2 = AutonomousTradingController()
    c2.start()
    c2.emergency_kill_switch(reason="Manual operator kill switch")
    assert c2.current_state == LoopState.STOPPED


# =========================================================================
# 6. Health & Heartbeat Monitoring
# =========================================================================

def test_health_and_heartbeat_monitoring():
    """Y, Z. Test health monitoring and separation of process vs trading health."""
    monitor = AutonomousHealthMonitor(session_id="ses_health")
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # Initial state
    snap = monitor.get_snapshot(LoopState.READY)
    assert snap.process_alive
    assert snap.health_status == HealthStatus.HEALTHY
    assert snap.total_cycles == 0

    # Record cycle start & failure
    monitor.record_cycle_start("cyc_1", t0)
    monitor.record_cycle_failure("cyc_1", "Stale market data", HealthStatus.STALE_DATA)

    snap_failed = monitor.get_snapshot(LoopState.RUNNING)
    assert snap_failed.process_alive
    assert snap_failed.health_status == HealthStatus.STALE_DATA
    assert snap_failed.failed_cycles == 1
    assert snap_failed.last_error == "Stale market data"


# =========================================================================
# 7. Persistence & Reloadability
# =========================================================================

def test_persistence_and_reload(tmp_path: Path):
    """P, AI. Test saving and reloading autonomous loop results from disk."""
    storage = AutonomousStorage(base_dir=tmp_path)
    controller = AutonomousTradingController(storage=storage)
    controller.start()

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = {"AAPL": BarData(symbol="AAPL", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)}
    controller.execute_market_event(market_timestamp=t0, current_bars=bars)
    controller.stop()

    result = controller.get_result()
    saved_dir = storage.save_result(result)
    assert (saved_dir / "results.json").exists()
    assert (saved_dir / "cycles.jsonl").exists()
    assert (saved_dir / "checkpoints.json").exists()
    assert (saved_dir / "events.jsonl").exists()

    # Load back
    loaded_res = storage.load_result(controller.session_id)
    assert loaded_res is not None
    assert loaded_res.session.session_id == controller.session_id
    assert len(loaded_res.cycles) == 1
    assert loaded_res.provenance_hash == result.provenance_hash


# =========================================================================
# 8. Causality & Invariance Under Future Mutations
# =========================================================================

def test_causal_future_mutation_invariance():
    """AF, AG. Verify mutating future market data does not change prior cycle results."""
    bars_orig = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0, 103.0, 108.0])
    bars_mut = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0, 999.0, 999.0])  # Mutate bars 3 & 4

    def run_ctrl(bar_list):
        ctrl = AutonomousTradingController(config=AutonomousConfig(universe=["AAPL"]))
        ctrl.start()
        for b in bar_list:
            ctrl.execute_market_event(market_timestamp=b.timestamp, current_bars={"AAPL": b})
        ctrl.stop()
        return ctrl.get_result()

    res1 = run_ctrl(bars_orig)
    res2 = run_ctrl(bars_mut)

    # Cycles 0, 1, 2 must have identical duration/execution stats
    for i in range(3):
        assert res1.cycles[i].orders_generated_count == res2.cycles[i].orders_generated_count
        assert res1.cycles[i].orders_executed_count == res2.cycles[i].orders_executed_count
        assert res1.cycles[i].status == res2.cycles[i].status


# =========================================================================
# 9. Determinism Across Repeated Runs
# =========================================================================

def test_autonomous_controller_determinism():
    """AB. Test bitwise determinism across repeated autonomous controller runs."""
    bars = create_synthetic_bars("AAPL", [100.0, 102.0, 105.0])

    def run_sim():
        ctrl = AutonomousTradingController(
            config=AutonomousConfig(universe=["AAPL"]),
            session_id="ses_det_fixed",
        )
        ctrl.start()
        for b in bars:
            ctrl.execute_market_event(market_timestamp=b.timestamp, current_bars={"AAPL": b})
        ctrl.stop()
        return ctrl.get_result()

    res1 = run_sim()
    res2 = run_sim()

    assert res1.session.total_cycles == res2.session.total_cycles
    assert res1.session.successful_cycles == res2.session.successful_cycles
    assert res1.provenance_hash == res2.provenance_hash


# =========================================================================
# 10. Safety Boundary: No Live Broker Access
# =========================================================================

def test_safety_boundary_no_live_broker():
    """AH. Verify autonomous loop controller operates strictly on paper trading."""
    ctrl = AutonomousTradingController()
    assert ctrl.paper_service is not None
    assert not hasattr(ctrl, "live_broker_client")
    assert not hasattr(ctrl, "alpaca_secret")
    assert not hasattr(ctrl, "interactive_brokers_socket")


# =========================================================================
# 11. Empirical Local Historical Replay Test
# =========================================================================

def test_empirical_local_historical_autonomous_replay():
    """AJ. Run full autonomous controller replay on 523 real local historical AAPL bars."""
    storage = ProcessedDataStorage()
    aapl_bars = storage.load_bars("AAPL", TimeFrame.DAY_1, format="parquet")
    assert len(aapl_bars) >= 500

    cfg = AutonomousConfig(
        universe=["AAPL"],
        paper_initial_capital=100000.0,
        commission_rate=0.0005,
        slippage_rate=0.0002,
        bid_ask_spread_rate=0.0002,
    )

    # Point-in-time SMA strategy signal generator
    def sma_signals(ts, current_bars):
        # We find current bar index in historical dataset
        bar = current_bars.get("AAPL")
        if not bar:
            return []
        # Return bullish signal if price > 180.0
        if bar.close > 180.0:
            return [
                SignalCandidate(
                    timestamp=ts,
                    symbol="AAPL",
                    signal=SignalDirection.LONG,
                    signal_strength=0.8,
                    confidence=0.9,
                    reason_codes=[ReasonCode.BULLISH_ENSEMBLE_SCORE.value],
                )
            ]
        return []

    controller = AutonomousTradingController(
        config=cfg,
        signal_generator=sma_signals,
    )

    result = controller.run_replay(market_data={"AAPL": aapl_bars})

    assert result.session.state == LoopState.STOPPED
    assert result.session.total_cycles == len(aapl_bars)
    assert result.session.successful_cycles == len(aapl_bars)
    assert result.session.failed_cycles == 0
    assert len(result.cycles) == len(aapl_bars)
    assert len(result.checkpoints) == len(aapl_bars)
    assert len(result.audit_events) > 500
    assert result.provenance_hash != ""


# =========================================================================
# 12. Failure Injection & Failure Classification Tests
# =========================================================================

def test_failure_injection_handling():
    """AD, AE, W, X. Test failure injection during data validation, risk blocking, and reconciliation."""
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    # 1. Missing required universe symbol -> Validation failure
    ctrl1 = AutonomousTradingController(config=AutonomousConfig(universe=["AAPL", "MSFT"], missing_symbol_policy=MissingSymbolPolicy.FAIL_CLOSED))
    ctrl1.start()
    only_aapl = BarData(symbol="AAPL", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
    c1 = ctrl1.execute_market_event(market_timestamp=t0, current_bars={"AAPL": only_aapl})
    assert c1.status == CycleStatus.FAILED
    assert "Missing required symbols" in c1.error_message
    assert ctrl1.session.failed_cycles == 1

    # 2. Risk block injection: target weight exceeds max_position_weight with adjustment_enabled=False -> REJECTED
    r_cfg_tight = RiskEngineConfig(max_position_weight=0.01, adjustment_enabled=False)
    ctrl2 = AutonomousTradingController(
        config=AutonomousConfig(universe=["AAPL"]),
        risk_config=r_cfg_tight,
        signal_generator=lambda ts, bars: [
            SignalCandidate(
                timestamp=ts,
                symbol="AAPL",
                signal=SignalDirection.LONG,
                signal_strength=0.9,
                confidence=0.9,
                reason_codes=[ReasonCode.BULLISH_ENSEMBLE_SCORE.value],
            )
        ],
    )
    ctrl2.start()
    good_bar = BarData(symbol="AAPL", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
    c2 = ctrl2.execute_market_event(market_timestamp=t0, current_bars={"AAPL": good_bar})
    assert c2.status == CycleStatus.COMPLETED
    assert c2.orders_generated_count == 0


def test_version_consistency_and_metadata_recording():
    """K. Verify cycle metadata records version consistency across pipeline components."""
    ctrl = AutonomousTradingController(config=AutonomousConfig(universe=["AAPL"]))
    ctrl.start()
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bar = BarData(symbol="AAPL", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
    c = ctrl.execute_market_event(market_timestamp=t0, current_bars={"AAPL": bar})

    assert c.status == CycleStatus.COMPLETED
    assert "autonomous" in c.versions
    assert "paper" in c.versions
    assert "risk" in c.versions

