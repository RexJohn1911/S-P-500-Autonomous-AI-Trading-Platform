"""
Unified Monitoring & Safety Service (Phase 18).
Provides a comprehensive high-level interface combining health aggregation,
incident lifecycles, global kill switch, pre-flight safety gates, heartbeats,
operational metrics, and audit logging.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from backend.app.monitoring.alerts import AlertManager
from backend.app.monitoring.audit import SafetyAuditLogger
from backend.app.monitoring.heartbeat import HeartbeatTracker
from backend.app.monitoring.incident import IncidentManager
from backend.app.monitoring.kill_switch import GlobalKillSwitch
from backend.app.monitoring.metrics import MetricsCollector
from backend.app.monitoring.safety import SafetyGateEvaluator
from backend.app.monitoring.schemas import (
    ComponentHealthSnapshot,
    ComponentType,
    HealthStatus,
    KillSwitchSnapshot,
    OperationalMetrics,
    SafetyEvaluation,
    SafetyEvent,
    SafetyIncident,
    SafetySeverity,
    SystemHealthSnapshot,
)
from backend.app.monitoring.storage import MonitoringStorage
from backend.app.monitoring.thresholds import MonitoringConfig
from backend.app.monitoring.watchdog import SystemWatchdog

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Unified entry point for the Monitoring & Safety subsystem.
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        storage: Optional[MonitoringStorage] = None,
    ):
        self.config = config or MonitoringConfig()
        self.config.validate()
        self.storage = storage or MonitoringStorage(base_dir=self.config.storage_dir)

        self.kill_switch = GlobalKillSwitch(storage=self.storage)
        self.incident_manager = IncidentManager(storage=self.storage)
        self.alert_manager = AlertManager(
            dedup_window_seconds=self.config.alert_dedup_window_seconds,
            storage=self.storage,
        )
        self.heartbeat_tracker = HeartbeatTracker(
            default_timeout_seconds=self.config.heartbeat_timeout_seconds,
        )
        self.watchdog = SystemWatchdog(
            config=self.config,
            kill_switch=self.kill_switch,
            incident_manager=self.incident_manager,
            alert_manager=self.alert_manager,
            heartbeat_tracker=self.heartbeat_tracker,
        )
        self.safety_evaluator = SafetyGateEvaluator(
            kill_switch=self.kill_switch,
            incident_manager=self.incident_manager,
        )
        self.metrics = MetricsCollector()
        self.audit_logger = SafetyAuditLogger(storage=self.storage)

    # =========================================================================
    # Health & Heartbeat API
    # =========================================================================

    def record_heartbeat(self, component: ComponentType, interval_seconds: float = 60.0) -> None:
        """Record heartbeat for a subsystem."""
        self.heartbeat_tracker.record_heartbeat(component=component, interval_seconds=interval_seconds)

    def update_component_health(
        self,
        component: ComponentType,
        status: HealthStatus,
        message: str = "OK",
        latency_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        cycle_id: Optional[str] = None,
    ) -> ComponentHealthSnapshot:
        """Record health update for an individual component."""
        return self.watchdog.update_component_health(
            component=component,
            status=status,
            message=message,
            latency_ms=latency_ms,
            details=details,
            cycle_id=cycle_id,
        )

    def get_system_health(self) -> SystemHealthSnapshot:
        """Retrieve aggregated system health snapshot."""
        snap = self.watchdog.aggregate_system_health()
        self.metrics.update_incident_counts(
            active=snap.active_incident_count,
            critical=len(self.incident_manager.get_critical_incidents()),
        )
        return snap

    # =========================================================================
    # Safety Gates & Order Evaluation
    # =========================================================================

    def evaluate_pre_flight_safety(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        signal_valid: bool = True,
        portfolio_valid: bool = True,
        risk_approved: bool = True,
        reconciliation_matched: bool = True,
        autonomous_loop_healthy: bool = True,
        execution_mode_permitted: bool = True,
    ) -> SafetyEvaluation:
        """Evaluate all 12 pre-flight safety gates before order submission."""
        sys_health = self.get_system_health()
        comps = self.watchdog._component_snapshots

        evaluation = self.safety_evaluator.evaluate_order(
            order_id=order_id,
            symbol=symbol,
            system_health=sys_health,
            component_healths=comps,
            signal_valid=signal_valid,
            portfolio_valid=portfolio_valid,
            risk_approved=risk_approved,
            reconciliation_matched=reconciliation_matched,
            autonomous_loop_healthy=autonomous_loop_healthy,
            execution_mode_permitted=execution_mode_permitted,
        )

        if not evaluation.is_permitted:
            self.metrics.record_safety_block()

        return evaluation

    # =========================================================================
    # Kill Switch Management
    # =========================================================================

    def trigger_kill_switch(
        self,
        reason: str,
        triggered_by: str = "OPERATOR",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KillSwitchSnapshot:
        """Manually or programmatically trigger the global kill switch."""
        snap = self.kill_switch.trigger(reason=reason, triggered_by=triggered_by, metadata=metadata)
        self.metrics.record_kill_switch_trigger()
        return snap

    def reset_kill_switch(self, cleared_by: str = "OPERATOR") -> tuple[bool, str]:
        """
        Attempt to reset the kill switch back to ARMED state.
        Fails closed if critical components or active critical incidents exist.
        """
        def _verification_check() -> tuple[bool, str]:
            critical_comps = [
                c.value for c, s in self.watchdog._component_snapshots.items()
                if s.status == HealthStatus.CRITICAL
            ]
            if critical_comps:
                return False, f"Critical component failures exist: {critical_comps}"
            if self.incident_manager.has_unresolved_critical_incident():
                return False, "Unresolved critical operational incidents exist"
            return True, ""

        return self.kill_switch.reset(cleared_by=cleared_by, verification_fn=_verification_check)

    # =========================================================================
    # Incident & Metrics API
    # =========================================================================

    def create_incident(
        self,
        title: str,
        description: str,
        severity: SafetySeverity,
        component: ComponentType,
        metadata: Optional[Dict] = None,
    ) -> SafetyIncident:
        """Create a new tracked operational incident."""
        return self.incident_manager.create_incident(
            title=title,
            description=description,
            severity=severity,
            component=component,
            metadata=metadata,
        )

    def resolve_incident(self, incident_id: str, notes: str) -> Optional[SafetyIncident]:
        """Resolve an active operational incident."""
        return self.incident_manager.resolve_incident(incident_id, notes)

    def get_operational_metrics(self) -> OperationalMetrics:
        """Retrieve aggregated operational performance and error metrics."""
        return self.metrics.get_metrics()
