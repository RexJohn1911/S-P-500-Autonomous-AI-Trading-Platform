"""
System Watchdog & Health Aggregator (Phase 18).
Observes all subsystems, performs deterministic health aggregation,
and detects anomalous conditions across market data, models, broker, and loop.
"""

from datetime import datetime, timezone
import logging
from threading import RLock
from typing import Any, Dict, List, Optional

from backend.app.monitoring.alerts import AlertManager
from backend.app.monitoring.heartbeat import HeartbeatTracker
from backend.app.monitoring.incident import IncidentManager
from backend.app.monitoring.kill_switch import GlobalKillSwitch
from backend.app.monitoring.schemas import (
    ComponentHealthSnapshot,
    ComponentType,
    HealthStatus,
    SafetyEvent,
    SafetyEventType,
    SafetySeverity,
    SystemHealthSnapshot,
)
from backend.app.monitoring.storage import MonitoringStorage
from backend.app.monitoring.thresholds import MonitoringConfig

logger = logging.getLogger(__name__)


class SystemWatchdog:
    """
    Subsystem observer and deterministic health aggregation engine.
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        kill_switch: Optional[GlobalKillSwitch] = None,
        incident_manager: Optional[IncidentManager] = None,
        alert_manager: Optional[AlertManager] = None,
        heartbeat_tracker: Optional[HeartbeatTracker] = None,
        storage: Optional[MonitoringStorage] = None,
    ):
        self._lock = RLock()
        self.config = config or MonitoringConfig()
        self.config.validate()
        self.storage = storage or MonitoringStorage(base_dir=self.config.storage_dir)
        self.kill_switch = kill_switch or GlobalKillSwitch(storage=self.storage)
        self.incident_manager = incident_manager or IncidentManager(storage=self.storage)
        self.alert_manager = alert_manager or AlertManager(
            dedup_window_seconds=self.config.alert_dedup_window_seconds,
            storage=self.storage,
        )
        self.heartbeat_tracker = heartbeat_tracker or HeartbeatTracker(
            default_timeout_seconds=self.config.heartbeat_timeout_seconds,
        )

        self._component_snapshots: Dict[ComponentType, ComponentHealthSnapshot] = {}
        # Initialize default healthy snapshots for standard components
        for c in ComponentType:
            self._component_snapshots[c] = ComponentHealthSnapshot(
                component=c,
                status=HealthStatus.HEALTHY,
                message=f"{c.value} initialized",
            )

    def update_component_health(
        self,
        component: ComponentType,
        status: HealthStatus,
        message: str = "OK",
        latency_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        cycle_id: Optional[str] = None,
    ) -> ComponentHealthSnapshot:
        """Update and record a component's health state."""
        with self._lock:
            prev_snapshot = self._component_snapshots.get(component)
            prev_status = prev_snapshot.status if prev_snapshot else HealthStatus.UNKNOWN

            snapshot = ComponentHealthSnapshot(
                component=component,
                status=status,
                timestamp=datetime.now(timezone.utc),
                message=message,
                latency_ms=latency_ms,
                details=details or {},
                cycle_id=cycle_id,
            )
            self._component_snapshots[component] = snapshot

            # Check for critical degradation or transition
            if status != prev_status:
                severity = SafetySeverity.CRITICAL if status == HealthStatus.CRITICAL else (
                    SafetySeverity.WARNING if status in (HealthStatus.WARNING, HealthStatus.DEGRADED) else SafetySeverity.INFO
                )
                evt = SafetyEvent(
                    event_id=f"evt_health_{datetime.now(timezone.utc).strftime('%H%M%S%f')}",
                    severity=severity,
                    component=component,
                    event_type=SafetyEventType.HEALTH_STATUS_CHANGE,
                    message=f"{component.value} transitioned from {prev_status.value} to {status.value}: {message}",
                    reason=message,
                    cycle_id=cycle_id,
                    previous_state=prev_status.value,
                    new_state=status.value,
                    metadata=details or {},
                )
                self.alert_manager.emit_event(evt)

            # Auto-trigger kill switch on critical failures if configured
            if status == HealthStatus.CRITICAL and self.config.auto_kill_switch_enabled:
                if component in (ComponentType.BROKER, ComponentType.RECONCILIATION, ComponentType.RISK_ENGINE, ComponentType.SYSTEM):
                    self.kill_switch.trigger(
                        reason=f"Auto kill switch triggered by critical {component.value} failure: {message}",
                        triggered_by=f"WATCHDOG_{component.value}",
                        metadata=details,
                    )
                    self.incident_manager.create_incident(
                        title=f"Critical {component.value} Failure",
                        description=message,
                        severity=SafetySeverity.CRITICAL,
                        component=component,
                        metadata=details,
                    )

            return snapshot

    def aggregate_system_health(self) -> SystemHealthSnapshot:
        """
        Deterministically aggregate all component snapshots into overall SystemHealthSnapshot.
        """
        with self._lock:
            # Check stale heartbeats
            stale_records = self.heartbeat_tracker.get_stale_heartbeats()
            for stale in stale_records:
                self._component_snapshots[stale.component] = ComponentHealthSnapshot(
                    component=stale.component,
                    status=HealthStatus.CRITICAL,
                    message=f"Heartbeat timeout expired (elapsed > {stale.timeout_seconds}s)",
                )

            # Health rules
            critical_comps = [c for c, s in self._component_snapshots.items() if s.status == HealthStatus.CRITICAL]
            unknown_comps = [c for c, s in self._component_snapshots.items() if s.status == HealthStatus.UNKNOWN]
            degraded_comps = [c for c, s in self._component_snapshots.items() if s.status in (HealthStatus.DEGRADED, HealthStatus.WARNING)]

            kill_active = self.kill_switch.is_triggered()
            active_incidents = len(self.incident_manager.get_active_incidents())
            has_crit_incidents = self.incident_manager.has_unresolved_critical_incident()

            if critical_comps or kill_active or has_crit_incidents:
                overall_status = HealthStatus.CRITICAL
                summary = f"CRITICAL health: kill_active={kill_active}, critical_components={[c.value for c in critical_comps]}"
                trading_permitted = False
            elif unknown_comps:
                overall_status = HealthStatus.UNKNOWN
                summary = f"UNKNOWN health: unverified_components={[c.value for c in unknown_comps]}"
                trading_permitted = False
            elif degraded_comps:
                overall_status = HealthStatus.DEGRADED
                summary = f"DEGRADED health: components={[c.value for c in degraded_comps]}"
                trading_permitted = True
            else:
                overall_status = HealthStatus.HEALTHY
                summary = "All monitored components are operational"
                trading_permitted = True

            return SystemHealthSnapshot(
                status=overall_status,
                timestamp=datetime.now(timezone.utc),
                components={c.value: s for c, s in self._component_snapshots.items()},
                summary=summary,
                is_trading_permitted=trading_permitted,
                active_incident_count=active_incidents,
                kill_switch_triggered=kill_active,
            )

    def check_reconciliation_result(self, is_matched: bool, discrepancy_details: Optional[Dict] = None) -> None:
        """Process a reconciliation event from paper/broker layer."""
        if is_matched:
            self.update_component_health(ComponentType.RECONCILIATION, HealthStatus.HEALTHY, message="Ledgers matched")
        else:
            self.update_component_health(
                ComponentType.RECONCILIATION,
                HealthStatus.CRITICAL,
                message="Ledger reconciliation mismatch detected",
                details=discrepancy_details,
            )

    def check_market_data_staleness(self, latest_bar_timestamp: datetime, max_staleness_seconds: Optional[float] = None) -> bool:
        """Check market data timestamp against staleness threshold."""
        max_age = max_staleness_seconds or self.config.max_data_staleness_seconds
        now = datetime.now(timezone.utc)
        if latest_bar_timestamp.tzinfo is None:
            latest_bar_timestamp = latest_bar_timestamp.replace(tzinfo=timezone.utc)
        elapsed = (now - latest_bar_timestamp).total_seconds()
        is_stale = elapsed > max_age
        if is_stale:
            self.update_component_health(
                ComponentType.MARKET_DATA,
                HealthStatus.WARNING,
                message=f"Market data stale: latest={latest_bar_timestamp.isoformat()}, elapsed={elapsed:.1f}s > {max_age}s",
            )
        else:
            self.update_component_health(ComponentType.MARKET_DATA, HealthStatus.HEALTHY, message="Market data fresh")
        return not is_stale
