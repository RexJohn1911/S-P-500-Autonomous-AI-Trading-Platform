"""
Safety Audit Logger (Phase 18).
Records append-only safety events, health transitions, pauses, resumes,
and kill switch activations with zero credential exposure.
"""

from datetime import datetime, timezone
import logging
import uuid
from typing import Any, Dict, List, Optional

from backend.app.monitoring.schemas import (
    ComponentType,
    SafetyEvent,
    SafetyEventType,
    SafetySeverity,
)
from backend.app.monitoring.storage import MonitoringStorage

logger = logging.getLogger(__name__)


def redact_sensitive_data(data: Any) -> Any:
    """Recursively redact API keys, secrets, tokens, and authorization headers."""
    if isinstance(data, dict):
        redacted = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(secret_term in k_lower for secret_term in ("key", "secret", "token", "auth", "password")):
                redacted[k] = "********"
            else:
                redacted[k] = redact_sensitive_data(v)
        return redacted
    elif isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    return data


class SafetyAuditLogger:
    """
    Append-only safety audit logger.
    """

    def __init__(self, storage: Optional[MonitoringStorage] = None):
        self.storage = storage or MonitoringStorage()

    def log_safety_event(
        self,
        component: ComponentType,
        event_type: SafetyEventType,
        severity: SafetySeverity,
        message: str,
        reason: Optional[str] = None,
        cycle_id: Optional[str] = None,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        previous_state: Optional[str] = None,
        new_state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SafetyEvent:
        """Create and persist a structured safety event."""
        safe_meta = redact_sensitive_data(metadata or {})
        event = SafetyEvent(
            event_id=f"evt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            component=component,
            event_type=event_type,
            message=message,
            reason=reason,
            cycle_id=cycle_id,
            order_id=order_id,
            symbol=symbol,
            previous_state=previous_state,
            new_state=new_state,
            metadata=safe_meta,
        )
        self.storage.append_audit_event(event)
        return event

    def get_recent_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent audit events from storage."""
        return self.storage.load_recent_audit_events(limit=limit)
