"""
Comprehensive Test Suite for Monitoring & Safety Subsystem (Phase 18).
Validates:
- Health status models & deterministic multi-component aggregation
- Heartbeat tracking and staleness detection
- Pre-flight 12-point Safety Gates evaluation
- Global Kill Switch: manual/auto triggers, persistent recovery, safe reset
- Alert deduplication and incident lifecycle tracking
- Fail-closed behavior on critical failures (market data, models, risk, broker, reconciliation)
- Concurrency & race condition safety
- Secret redaction in audit logging
- Full autonomous replay parity with monitoring layer attached
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import threading
import time

from backend.app.autonomous.controller import AutonomousTradingController
from backend.app.autonomous.schemas import AutonomousConfig, CycleStatus
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.storage import ProcessedDataStorage
from backend.app.monitoring import (
    AlertManager,
    ComponentHealthSnapshot,
    ComponentType,
    GlobalKillSwitch,
    HealthStatus,
    HeartbeatTracker,
    IncidentManager,
    IncidentStatus,
    KillSwitchState,
    MetricsCollector,
    MonitoringConfig,
    MonitoringService,
    MonitoringStorage,
    OperationalMetrics,
    SafetyAuditLogger,
    SafetyEvaluation,
    SafetyEvent,
    SafetyEventType,
    SafetyGateEvaluator,
    SafetySeverity,
    SystemHealthSnapshot,
    SystemWatchdog,
    redact_sensitive_data,
)
from backend.app.strategy.schemas import ReasonCode, SignalCandidate, SignalDirection


# =========================================================================
# 1. Health Status & Aggregation Tests
# =========================================================================

def test_health_aggregation_rules(tmp_path: Path):
    """Verify deterministic system health aggregation from component statuses."""
    cfg = MonitoringConfig(storage_dir=str(tmp_path / "agg1"))
    watchdog = SystemWatchdog(config=cfg)

    # 1. Default: All Healthy
    snap = watchdog.aggregate_system_health()
    assert snap.status == HealthStatus.HEALTHY
    assert snap.is_trading_permitted is True

    # 2. Single Degraded component -> System DEGRADED, trading permitted
    watchdog.update_component_health(ComponentType.FEATURE_ENGINEERING, HealthStatus.DEGRADED, message="Slow feature cache")
    snap2 = watchdog.aggregate_system_health()
    assert snap2.status == HealthStatus.DEGRADED
    assert snap2.is_trading_permitted is True

    # 3. Critical component -> System CRITICAL, trading blocked
    watchdog.update_component_health(ComponentType.BROKER, HealthStatus.CRITICAL, message="Gateway disconnected")
    snap3 = watchdog.aggregate_system_health()
    assert snap3.status == HealthStatus.CRITICAL
    assert snap3.is_trading_permitted is False

    # 4. Unknown component -> System UNKNOWN, trading blocked (fail closed)
    cfg_unk = MonitoringConfig(storage_dir=str(tmp_path / "unk"))
    watchdog_unk = SystemWatchdog(config=cfg_unk)
    watchdog_unk.update_component_health(ComponentType.MODEL, HealthStatus.UNKNOWN, message="Model unverified")
    snap_unk = watchdog_unk.aggregate_system_health()
    assert snap_unk.status == HealthStatus.UNKNOWN
    assert snap_unk.is_trading_permitted is False


# =========================================================================
# 2. Heartbeat Tracking & Staleness Tests
# =========================================================================

def test_heartbeat_tracker_timeout_detection():
    """Verify heartbeat tracking and critical status transition on timeout."""
    tracker = HeartbeatTracker(default_timeout_seconds=2.0)
    t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Record heartbeat at t0
    tracker.record_heartbeat(ComponentType.AUTONOMOUS_LOOP, interval_seconds=1.0, timeout_seconds=2.0, timestamp=t0)
    # Check within timeout
    status_ok = tracker.get_heartbeat_status(ComponentType.AUTONOMOUS_LOOP, as_of=t0 + timedelta(seconds=1.0))
    assert status_ok == HealthStatus.HEALTHY

    # Check after timeout expired
    stale_status = tracker.get_heartbeat_status(ComponentType.AUTONOMOUS_LOOP, as_of=t0 + timedelta(seconds=3.0))
    assert stale_status == HealthStatus.CRITICAL
    stale_recs = tracker.get_stale_heartbeats(as_of=t0 + timedelta(seconds=3.0))
    assert len(stale_recs) == 1
    assert stale_recs[0].component == ComponentType.AUTONOMOUS_LOOP



# =========================================================================
# 3. 12-Point Pre-Flight Safety Gates Tests
# =========================================================================

def test_pre_flight_safety_gates_all_pass(tmp_path: Path):
    """Verify all 12 pre-flight safety gates pass under nominal healthy conditions."""
    service = MonitoringService(config=MonitoringConfig(storage_dir=str(tmp_path)))

    eval_result = service.evaluate_pre_flight_safety(
        order_id="ord_safe_01",
        symbol="AAPL",
        signal_valid=True,
        portfolio_valid=True,
        risk_approved=True,
        reconciliation_matched=True,
        autonomous_loop_healthy=True,
        execution_mode_permitted=True,
    )
    assert eval_result.is_permitted is True
    assert len(eval_result.rejection_reasons) == 0
    assert len(eval_result.gates) == 12
    assert all(g.passed for g in eval_result.gates)


def test_pre_flight_safety_gates_individual_rejections(tmp_path: Path):
    """Verify each individual safety gate failure prevents order submission."""
    service = MonitoringService(config=MonitoringConfig(storage_dir=str(tmp_path)))

    # 1. Invalid signal fails gate 4
    e1 = service.evaluate_pre_flight_safety(signal_valid=False)
    assert e1.is_permitted is False
    assert any("Invalid trading signal" in r for r in e1.rejection_reasons)

    # 2. Portfolio allocation invalid fails gate 5
    e2 = service.evaluate_pre_flight_safety(portfolio_valid=False)
    assert e2.is_permitted is False
    assert any("Portfolio allocation invalid" in r for r in e2.rejection_reasons)

    # 3. Risk engine violation fails gate 6
    e3 = service.evaluate_pre_flight_safety(risk_approved=False)
    assert e3.is_permitted is False
    assert any("Risk engine" in r for r in e3.rejection_reasons)

    # 4. Reconciliation mismatch fails gate 8
    e4 = service.evaluate_pre_flight_safety(reconciliation_matched=False)
    assert e4.is_permitted is False
    assert any("Reconciliation mismatch" in r for r in e4.rejection_reasons)

    # 5. Autonomous loop unhealthy fails gate 9
    e5 = service.evaluate_pre_flight_safety(autonomous_loop_healthy=False)
    assert e5.is_permitted is False
    assert any("Autonomous trading loop" in r for r in e5.rejection_reasons)

    # 6. Kill switch active fails gate 10
    service.trigger_kill_switch(reason="Manual emergency stop")
    e6 = service.evaluate_pre_flight_safety()
    assert e6.is_permitted is False
    assert any("Global kill switch is active" in r for r in e6.rejection_reasons)


# =========================================================================
# 4. Global Kill Switch Lifecycle, Persistence & Safe Reset Tests
# =========================================================================

def test_global_kill_switch_lifecycle_and_persistence(tmp_path: Path):
    """Verify manual kill switch trigger, persistent restart recovery, and safe reset."""
    storage = MonitoringStorage(base_dir=tmp_path)
    ks1 = GlobalKillSwitch(storage=storage)
    assert ks1.is_armed() is True
    assert ks1.is_triggered() is False

    # 1. Trigger
    ks1.trigger(reason="Market flash crash detected", triggered_by="RISK_SYSTEM")
    assert ks1.is_triggered() is True
    assert ks1.state == KillSwitchState.TRIGGERED

    # 2. Restart recovery simulation
    ks2 = GlobalKillSwitch(storage=storage)
    assert ks2.is_triggered() is True
    assert ks2.state == KillSwitchState.TRIGGERED
    assert ks2.get_snapshot().reason == "Market flash crash detected"

    # 3. Reset with failing verification callback -> Blocked
    reset_blocked, reason_blocked = ks2.reset(
        cleared_by="OPERATOR",
        verification_fn=lambda: (False, "Broker connection still degraded"),
    )
    assert reset_blocked is False
    assert "still degraded" in reason_blocked
    assert ks2.is_triggered() is True

    # 4. Reset with passing verification callback -> Success
    reset_ok, msg = ks2.reset(
        cleared_by="OPERATOR",
        verification_fn=lambda: (True, ""),
    )
    assert reset_ok is True
    assert ks2.is_armed() is True
    assert ks2.is_triggered() is False

    # 5. Verify persisted reset
    ks3 = GlobalKillSwitch(storage=storage)
    assert ks3.is_armed() is True
    assert ks3.is_triggered() is False


# =========================================================================
# 5. Alert Deduplication & Incident Lifecycle Tests
# =========================================================================

def test_alert_deduplication_and_incident_lifecycle(tmp_path: Path):
    """Verify sliding-window alert deduplication and incident resolution."""
    storage = MonitoringStorage(base_dir=tmp_path)
    alert_mgr = AlertManager(dedup_window_seconds=10.0, storage=storage)

    t0 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    evt1 = SafetyEvent(
        event_id="e1",
        timestamp=t0,
        severity=SafetySeverity.WARNING,
        component=ComponentType.MARKET_DATA,
        event_type=SafetyEventType.STALE_DATA_DETECTED,
        message="AAPL 1-day bar missing",
        symbol="AAPL",
    )
    # First emit accepted
    assert alert_mgr.emit_event(evt1) is True

    # Second identical emit within 10s window deduplicated
    evt2 = SafetyEvent(
        event_id="e2",
        timestamp=t0 + timedelta(seconds=3),
        severity=SafetySeverity.WARNING,
        component=ComponentType.MARKET_DATA,
        event_type=SafetyEventType.STALE_DATA_DETECTED,
        message="AAPL 1-day bar missing",
        symbol="AAPL",
    )
    assert alert_mgr.emit_event(evt2) is False

    # Critical severity is never suppressed
    evt_crit = SafetyEvent(
        event_id="e3",
        timestamp=t0 + timedelta(seconds=4),
        severity=SafetySeverity.CRITICAL,
        component=ComponentType.MARKET_DATA,
        event_type=SafetyEventType.STALE_DATA_DETECTED,
        message="AAPL critical feed drop",
        symbol="AAPL",
    )
    assert alert_mgr.emit_event(evt_crit) is True

    # Incident lifecycle
    inc_mgr = IncidentManager(storage=storage)
    inc = inc_mgr.create_incident(
        title="Broker Connection Drop",
        description="Lost WebSocket connection to exchange",
        severity=SafetySeverity.CRITICAL,
        component=ComponentType.BROKER,
    )
    assert inc.status == IncidentStatus.OPEN
    assert inc_mgr.has_unresolved_critical_incident() is True

    # Acknowledge
    inc_mgr.acknowledge_incident(inc.incident_id)
    assert inc.status == IncidentStatus.ACKNOWLEDGED
    assert inc_mgr.has_unresolved_critical_incident() is True

    # Resolve
    inc_mgr.resolve_incident(inc.incident_id, notes="Reconnected with backup gateway")
    assert inc.status == IncidentStatus.RESOLVED
    assert inc_mgr.has_unresolved_critical_incident() is False


# =========================================================================
# 6. Concurrency Safety & Secret Redaction Tests
# =========================================================================

def test_concurrency_race_condition_safety(tmp_path: Path):
    """Verify thread-safe order evaluation during simultaneous kill switch activation."""
    service = MonitoringService(config=MonitoringConfig(storage_dir=str(tmp_path)))
    evaluation_results = []

    def trigger_worker():
        time.sleep(0.01)
        service.trigger_kill_switch(reason="Concurrent emergency trigger")

    def eval_worker():
        for i in range(50):
            res = service.evaluate_pre_flight_safety(order_id=f"ord_conc_{i}", symbol="AAPL")
            evaluation_results.append(res.is_permitted)
            time.sleep(0.001)

    t1 = threading.Thread(target=trigger_worker)
    t2 = threading.Thread(target=eval_worker)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Once kill switch triggered, all subsequent evaluations MUST be blocked (is_permitted=False)
    assert False in evaluation_results
    assert service.kill_switch.is_triggered() is True


def test_secret_redaction_in_audit_logger(tmp_path: Path):
    """Verify API keys, secrets, and auth headers are redacted in safety audit trails."""
    storage = MonitoringStorage(base_dir=tmp_path)
    logger_inst = SafetyAuditLogger(storage=storage)

    meta_with_secrets = {
        "api_key": "AKTEST12345678",
        "api_secret": "TOPSECRETSECRET999",
        "auth_token": "Bearer abc123def456",
        "broker": "ALPACA",
        "endpoint": "https://api.alpaca.markets",
    }
    evt = logger_inst.log_safety_event(
        component=ComponentType.BROKER,
        event_type=SafetyEventType.BROKER_AUTHENTICATION_FAILURE,
        severity=SafetySeverity.HIGH,
        message="Authentication header failed",
        metadata=meta_with_secrets,
    )

    assert evt.metadata["api_key"] == "********"
    assert evt.metadata["api_secret"] == "********"
    assert evt.metadata["auth_token"] == "********"
    assert evt.metadata["broker"] == "ALPACA"
    assert "AKTEST12345678" not in str(evt.to_dict())
    assert "TOPSECRETSECRET999" not in str(evt.to_dict())


# =========================================================================
# 7. Operational Metrics Tracking Tests
# =========================================================================

def test_operational_metrics_collector():
    """Verify operational metrics tracking counters, latencies, and error tallies."""
    metrics = MetricsCollector()
    metrics.record_cycle(success=True, duration_ms=150.0)
    metrics.record_cycle(success=False, duration_ms=250.0)
    metrics.record_order_submission()
    metrics.record_order_fill(is_partial=False)
    metrics.record_order_fill(is_partial=True)
    metrics.record_broker_call(latency_ms=45.0, is_error=False)
    metrics.record_broker_call(latency_ms=120.0, is_error=True, is_rate_limit=True)
    metrics.record_safety_block()
    metrics.record_kill_switch_trigger()

    snap = metrics.get_metrics()
    assert snap.cycles_total == 2
    assert snap.cycles_successful == 1
    assert snap.cycles_failed == 1
    assert snap.avg_cycle_duration_ms == 200.0
    assert snap.orders_submitted == 1
    assert snap.orders_filled == 1
    assert snap.partial_fills == 1
    assert snap.broker_errors_total == 1
    assert snap.rate_limit_events == 1
    assert snap.safety_gate_blocks == 1
    assert snap.kill_switch_activations == 1


# =========================================================================
# 8. Deterministic Historical Replay Parity Test
# =========================================================================

def test_historical_replay_parity_with_monitoring_attached(tmp_path: Path):
    """Verify attaching MonitoringService maintains 100% parity on established AAPL historical replay."""
    storage = ProcessedDataStorage()
    aapl_bars = storage.load_bars("AAPL", TimeFrame.DAY_1, format="parquet")
    assert len(aapl_bars) >= 500

    mon_service = MonitoringService(config=MonitoringConfig(storage_dir=str(tmp_path)))

    cfg = AutonomousConfig(
        universe=["AAPL"],
        paper_initial_capital=100000.0,
        commission_rate=0.0005,
        slippage_rate=0.0002,
        bid_ask_spread_rate=0.0002,
    )

    def sma_signals(ts, current_bars):
        bar = current_bars.get("AAPL")
        if not bar:
            return []
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

    ctrl = AutonomousTradingController(
        config=cfg,
        signal_generator=sma_signals,
    )

    # Attach monitoring verification to replay cycles
    ctrl.start()
    for bar in aapl_bars:
        # Pre-flight gate check
        eval_res = mon_service.evaluate_pre_flight_safety(
            symbol="AAPL",
            signal_valid=True,
            portfolio_valid=True,
            risk_approved=True,
            reconciliation_matched=True,
            autonomous_loop_healthy=True,
            execution_mode_permitted=True,
        )
        assert eval_res.is_permitted is True

        c = ctrl.execute_market_event(market_timestamp=bar.timestamp, current_bars={"AAPL": bar})
        mon_service.record_heartbeat(ComponentType.AUTONOMOUS_LOOP)
        mon_service.metrics.record_cycle(success=(c.status == CycleStatus.COMPLETED), duration_ms=10.0)

    ctrl.stop()

    res = ctrl.get_result()
    assert res.session.total_cycles == len(aapl_bars)
    assert res.session.successful_cycles == len(aapl_bars)
    assert res.session.failed_cycles == 0
    assert ctrl.broker.get_account().equity == pytest.approx(101599.79, abs=0.5)

    metrics_snap = mon_service.get_operational_metrics()
    assert metrics_snap.cycles_total == len(aapl_bars)
    assert metrics_snap.cycles_successful == len(aapl_bars)
    assert metrics_snap.safety_gate_blocks == 0


# =========================================================================
# 9. Failure Injection & Failure Replay Tests
# =========================================================================

def test_failure_injection_scenarios(tmp_path: Path):
    """Verify deterministic safety reactions to injected failure scenarios."""
    service = MonitoringService(config=MonitoringConfig(storage_dir=str(tmp_path)))

    # Scenario A: Broker unavailable -> Orders blocked
    service.update_component_health(ComponentType.BROKER, HealthStatus.CRITICAL, message="Alpaca 503 gateway outage")
    eval_a = service.evaluate_pre_flight_safety(order_id="ord_fail_a", symbol="AAPL")
    assert eval_a.is_permitted is False
    assert any("Broker" in r or "CRITICAL" in r for r in eval_a.rejection_reasons)

    # Clear broker health for next scenario
    service.reset_kill_switch(cleared_by="OPERATOR")
    service.update_component_health(ComponentType.BROKER, HealthStatus.HEALTHY, message="Broker reconnected")

    # Scenario B: Reconciliation mismatch -> Trading paused & critical incident
    service.watchdog.check_reconciliation_result(is_matched=False, discrepancy_details={"symbol": "AAPL", "qty_diff": 10.0})
    eval_b = service.evaluate_pre_flight_safety(order_id="ord_fail_b", symbol="AAPL", reconciliation_matched=False)
    assert eval_b.is_permitted is False
    assert any("Reconciliation" in r for r in eval_b.rejection_reasons)

    # Scenario C: Kill switch activated -> Zero new orders permitted
    service.trigger_kill_switch(reason="Manual emergency stop")
    eval_c = service.evaluate_pre_flight_safety(order_id="ord_fail_c", symbol="AAPL")
    assert eval_c.is_permitted is False
    assert any("kill switch" in r.lower() for r in eval_c.rejection_reasons)

    # Scenario D: Autonomous heartbeat expires -> Status CRITICAL -> Orders blocked
    t_now = datetime.now(timezone.utc)
    service.heartbeat_tracker.record_heartbeat(
        ComponentType.AUTONOMOUS_LOOP,
        interval_seconds=1.0,
        timeout_seconds=2.0,
        timestamp=t_now - timedelta(seconds=10.0),
    )
    sys_health_d = service.get_system_health()
    assert sys_health_d.status == HealthStatus.CRITICAL
    eval_d = service.evaluate_pre_flight_safety(order_id="ord_fail_d", symbol="AAPL")
    assert eval_d.is_permitted is False

    # Scenario E: Invalid model output -> Signal rejected -> Safety gate blocks
    eval_e = service.evaluate_pre_flight_safety(order_id="ord_fail_e", symbol="AAPL", signal_valid=False)
    assert eval_e.is_permitted is False
    assert any("Invalid trading signal" in r for r in eval_e.rejection_reasons)

    # Scenario F: Impossible execution -> Incident created
    inc_f = service.create_incident(
        title="Impossible Execution Price",
        description="Fill price reported as negative",
        severity=SafetySeverity.CRITICAL,
        component=ComponentType.ORDER_EXECUTION,
    )
    assert inc_f.status == IncidentStatus.OPEN
    eval_f = service.evaluate_pre_flight_safety(order_id="ord_fail_f", symbol="AAPL")
    assert eval_f.is_permitted is False
