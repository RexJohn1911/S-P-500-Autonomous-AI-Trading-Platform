"""
Monitoring & Safety Subsystem (Phase 18).
Provides production-oriented observability, 12-point pre-flight safety gates,
incident lifecycle management, centralized global kill switch, alert deduplication,
heartbeats, operational metrics, and immutable audit logging.
"""

from backend.app.monitoring.alerts import AlertManager
from backend.app.monitoring.audit import SafetyAuditLogger, redact_sensitive_data
from backend.app.monitoring.heartbeat import HeartbeatTracker
from backend.app.monitoring.incident import IncidentManager
from backend.app.monitoring.kill_switch import GlobalKillSwitch
from backend.app.monitoring.metrics import MetricsCollector
from backend.app.monitoring.safety import SafetyGateEvaluator
from backend.app.monitoring.schemas import (
    ComponentHealthSnapshot,
    ComponentType,
    HealthStatus,
    HeartbeatRecord,
    IncidentStatus,
    KillSwitchSnapshot,
    KillSwitchState,
    OperationalMetrics,
    SafetyEvaluation,
    SafetyEvent,
    SafetyEventType,
    SafetyGateResult,
    SafetyIncident,
    SafetySeverity,
    SystemHealthSnapshot,
)
from backend.app.monitoring.service import MonitoringService
from backend.app.monitoring.storage import MonitoringStorage
from backend.app.monitoring.thresholds import MonitoringConfig
from backend.app.monitoring.watchdog import SystemWatchdog

__all__ = [
    "AlertManager",
    "ComponentHealthSnapshot",
    "ComponentType",
    "GlobalKillSwitch",
    "HealthStatus",
    "HeartbeatRecord",
    "HeartbeatTracker",
    "IncidentManager",
    "IncidentStatus",
    "KillSwitchSnapshot",
    "KillSwitchState",
    "MetricsCollector",
    "MonitoringConfig",
    "MonitoringService",
    "MonitoringStorage",
    "OperationalMetrics",
    "SafetyAuditLogger",
    "SafetyEvaluation",
    "SafetyEvent",
    "SafetyEventType",
    "SafetyGateEvaluator",
    "SafetyGateResult",
    "SafetyIncident",
    "SafetySeverity",
    "SystemHealthSnapshot",
    "SystemWatchdog",
    "redact_sensitive_data",
]
