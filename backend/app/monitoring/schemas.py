"""
Normalized Monitoring & Safety Schemas (Phase 18).
Defines comprehensive domain contracts for health checks, safety events,
incident lifecycles, safety gates, global kill switch, heartbeats, and metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class HealthStatus(str, Enum):
    """Normalized health status for components and overall system."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

    def is_healthy(self) -> bool:
        return self == HealthStatus.HEALTHY

    def allows_trading(self) -> bool:
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.WARNING)

    def is_critical(self) -> bool:
        return self == HealthStatus.CRITICAL


class ComponentType(str, Enum):
    """System components observed by the monitoring subsystem."""
    MARKET_DATA = "MARKET_DATA"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    MODEL = "MODEL"
    SIGNAL_ENGINE = "SIGNAL_ENGINE"
    PORTFOLIO = "PORTFOLIO"
    RISK_ENGINE = "RISK_ENGINE"
    BROKER = "BROKER"
    ORDER_EXECUTION = "ORDER_EXECUTION"
    RECONCILIATION = "RECONCILIATION"
    AUTONOMOUS_LOOP = "AUTONOMOUS_LOOP"
    ACCOUNTING = "ACCOUNTING"
    STORAGE = "STORAGE"
    SYSTEM = "SYSTEM"


@dataclass
class ComponentHealthSnapshot:
    """Point-in-time health report for an individual subsystem."""
    component: ComponentType
    status: HealthStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = "Component operating normally"
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    cycle_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "version": self.version,
            "cycle_id": self.cycle_id,
        }


@dataclass
class SystemHealthSnapshot:
    """Aggregated system-wide health status across all active components."""
    status: HealthStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    components: Dict[str, ComponentHealthSnapshot] = field(default_factory=dict)
    summary: str = "All systems operational"
    is_trading_permitted: bool = True
    active_incident_count: int = 0
    kill_switch_triggered: bool = False

    def __post_init__(self):
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "is_trading_permitted": self.is_trading_permitted,
            "active_incident_count": self.active_incident_count,
            "kill_switch_triggered": self.kill_switch_triggered,
            "components": {k: v.to_dict() for k, v in self.components.items()},
        }


class SafetySeverity(str, Enum):
    """Severity classification for safety events and incidents."""
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SafetyEventType(str, Enum):
    """Categorized safety event triggers."""
    HEALTH_STATUS_CHANGE = "HEALTH_STATUS_CHANGE"
    STALE_DATA_DETECTED = "STALE_DATA_DETECTED"
    DATA_INTEGRITY_VIOLATION = "DATA_INTEGRITY_VIOLATION"
    MODEL_INFERENCE_FAILURE = "MODEL_INFERENCE_FAILURE"
    MODEL_VERSION_MISMATCH = "MODEL_VERSION_MISMATCH"
    INVALID_SIGNAL_DETECTED = "INVALID_SIGNAL_DETECTED"
    RISK_LIMIT_VIOLATION = "RISK_LIMIT_VIOLATION"
    BROKER_CONNECTIVITY_LOST = "BROKER_CONNECTIVITY_LOST"
    BROKER_AUTHENTICATION_FAILURE = "BROKER_AUTHENTICATION_FAILURE"
    BROKER_RATE_LIMIT_EXCEEDED = "BROKER_RATE_LIMIT_EXCEEDED"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    UNEXPECTED_FILL_DETECTED = "UNEXPECTED_FILL_DETECTED"
    IMPOSSIBLE_EXECUTION = "IMPOSSIBLE_EXECUTION"
    AMBIGUOUS_ORDER_SUBMISSION = "AMBIGUOUS_ORDER_SUBMISSION"
    HEARTBEAT_TIMEOUT = "HEARTBEAT_TIMEOUT"
    LOOP_STUCK_DETECTED = "LOOP_STUCK_DETECTED"
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"
    KILL_SWITCH_CLEARED = "KILL_SWITCH_CLEARED"
    SAFETY_GATE_REJECTION = "SAFETY_GATE_REJECTION"
    OPERATOR_ACTION = "OPERATOR_ACTION"


@dataclass
class SafetyEvent:
    """Structured, credential-safe safety audit event."""
    event_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: SafetySeverity = SafetySeverity.INFO
    component: ComponentType = ComponentType.SYSTEM
    event_type: SafetyEventType = SafetyEventType.HEALTH_STATUS_CHANGE
    message: str = ""
    reason: Optional[str] = None
    cycle_id: Optional[str] = None
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    previous_state: Optional[str] = None
    new_state: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    @property
    def fingerprint(self) -> str:
        """Deterministic fingerprint for alert deduplication."""
        return f"{self.component.value}:{self.event_type.value}:{self.symbol or 'GLOBAL'}:{self.severity.value}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "component": self.component.value,
            "event_type": self.event_type.value,
            "message": self.message,
            "reason": self.reason,
            "cycle_id": self.cycle_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "metadata": self.metadata,
        }


class IncidentStatus(str, Enum):
    """Lifecycle statuses for operational incidents."""
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    MITIGATING = "MITIGATING"
    RESOLVED = "RESOLVED"

    def is_active(self) -> bool:
        return self != IncidentStatus.RESOLVED


@dataclass
class SafetyIncident:
    """Operational incident requiring acknowledgment or mitigation."""
    incident_id: str
    title: str
    description: str
    severity: SafetySeverity
    component: ComponentType
    status: IncidentStatus = IncidentStatus.OPEN
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trigger_event_id: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
        if self.resolved_at and self.resolved_at.tzinfo is None:
            self.resolved_at = self.resolved_at.replace(tzinfo=timezone.utc)

    def resolve(self, notes: str) -> None:
        self.status = IncidentStatus.RESOLVED
        self.resolution_notes = notes
        now = datetime.now(timezone.utc)
        self.resolved_at = now
        self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "component": self.component.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "trigger_event_id": self.trigger_event_id,
            "resolution_notes": self.resolution_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


class KillSwitchState(str, Enum):
    """Centralized kill switch state."""
    ARMED = "ARMED"
    TRIGGERED = "TRIGGERED"
    CLEARING = "CLEARING"


@dataclass
class KillSwitchSnapshot:
    """State record for global kill switch."""
    state: KillSwitchState = KillSwitchState.ARMED
    is_triggered: bool = False
    triggered_at: Optional[datetime] = None
    triggered_by: Optional[str] = None
    reason: Optional[str] = None
    cleared_at: Optional[datetime] = None
    cleared_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "is_triggered": self.is_triggered,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "triggered_by": self.triggered_by,
            "reason": self.reason,
            "cleared_at": self.cleared_at.isoformat() if self.cleared_at else None,
            "cleared_by": self.cleared_by,
            "metadata": self.metadata,
        }


@dataclass
class SafetyGateResult:
    """Evaluation result for an individual pre-flight safety gate."""
    gate_name: str
    passed: bool
    reason: str = "Passed"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass
class SafetyEvaluation:
    """Comprehensive evaluation of all 12 pre-flight safety gates."""
    is_permitted: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    gates: List[SafetyGateResult] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_permitted": self.is_permitted,
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "symbol": self.symbol,
            "rejection_reasons": self.rejection_reasons,
            "gates": [g.to_dict() for g in self.gates],
        }


@dataclass
class HeartbeatRecord:
    """Heartbeat record for tracked subsystem processes."""
    component: ComponentType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    interval_seconds: float = 60.0
    timeout_seconds: float = 120.0
    status: HealthStatus = HealthStatus.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_stale(self, as_of: Optional[datetime] = None) -> bool:
        now = as_of or datetime.now(timezone.utc)
        elapsed = (now - self.timestamp).total_seconds()
        return elapsed > self.timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component.value,
            "timestamp": self.timestamp.isoformat(),
            "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "metadata": self.metadata,
        }


@dataclass
class OperationalMetrics:
    """Operational monitoring metrics."""
    cycles_total: int = 0
    cycles_successful: int = 0
    cycles_failed: int = 0
    avg_cycle_duration_ms: float = 0.0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    partial_fills: int = 0
    broker_errors_total: int = 0
    broker_avg_latency_ms: float = 0.0
    rate_limit_events: int = 0
    reconciliation_mismatches: int = 0
    active_incidents_count: int = 0
    critical_incidents_count: int = 0
    kill_switch_activations: int = 0
    safety_gate_blocks: int = 0
    stale_data_events: int = 0
    model_inference_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycles_total": self.cycles_total,
            "cycles_successful": self.cycles_successful,
            "cycles_failed": self.cycles_failed,
            "avg_cycle_duration_ms": self.avg_cycle_duration_ms,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "partial_fills": self.partial_fills,
            "broker_errors_total": self.broker_errors_total,
            "broker_avg_latency_ms": self.broker_avg_latency_ms,
            "rate_limit_events": self.rate_limit_events,
            "reconciliation_mismatches": self.reconciliation_mismatches,
            "active_incidents_count": self.active_incidents_count,
            "critical_incidents_count": self.critical_incidents_count,
            "kill_switch_activations": self.kill_switch_activations,
            "safety_gate_blocks": self.safety_gate_blocks,
            "stale_data_events": self.stale_data_events,
            "model_inference_failures": self.model_inference_failures,
        }
