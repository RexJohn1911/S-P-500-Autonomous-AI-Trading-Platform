"""
Incident Lifecycle Manager (Phase 18).
Tracks operational incidents from creation through resolution.
Ensures critical incidents require explicit operational resolution.
"""

from datetime import datetime, timezone
import logging
from threading import RLock
from typing import Dict, List, Optional
import uuid

from backend.app.monitoring.schemas import (
    ComponentType,
    IncidentStatus,
    SafetyIncident,
    SafetySeverity,
)
from backend.app.monitoring.storage import MonitoringStorage

logger = logging.getLogger(__name__)


class IncidentManager:
    """
    Manages operational incidents with explicit lifecycle tracking.
    """

    def __init__(self, storage: Optional[MonitoringStorage] = None):
        self._lock = RLock()
        self.storage = storage or MonitoringStorage()
        self._incidents: Dict[str, SafetyIncident] = {}

    def create_incident(
        self,
        title: str,
        description: str,
        severity: SafetySeverity,
        component: ComponentType,
        trigger_event_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> SafetyIncident:
        """Create and track a new operational incident."""
        with self._lock:
            inc_id = f"inc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            incident = SafetyIncident(
                incident_id=inc_id,
                title=title,
                description=description,
                severity=severity,
                component=component,
                status=IncidentStatus.OPEN,
                trigger_event_id=trigger_event_id,
                metadata=metadata or {},
            )
            self._incidents[inc_id] = incident
            self.storage.append_incident(incident)
            logger.warning(
                "INCIDENT CREATED [%s] %s (severity=%s, component=%s)",
                inc_id,
                title,
                severity.value,
                component.value,
            )
            return incident

    def acknowledge_incident(self, incident_id: str) -> Optional[SafetyIncident]:
        """Mark an open incident as acknowledged."""
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident and incident.status == IncidentStatus.OPEN:
                incident.status = IncidentStatus.ACKNOWLEDGED
                incident.updated_at = datetime.now(timezone.utc)
                self.storage.append_incident(incident)
                return incident
            return None

    def resolve_incident(self, incident_id: str, notes: str) -> Optional[SafetyIncident]:
        """Explicitly resolve an active incident."""
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident:
                incident.resolve(notes)
                self.storage.append_incident(incident)
                logger.info("INCIDENT RESOLVED [%s]: %s", incident_id, notes)
                return incident
            return None

    def get_active_incidents(self) -> List[SafetyIncident]:
        """Return all unresolved incidents."""
        with self._lock:
            return [inc for inc in self._incidents.values() if inc.status.is_active()]

    def get_critical_incidents(self) -> List[SafetyIncident]:
        """Return all unresolved critical incidents."""
        with self._lock:
            return [
                inc for inc in self._incidents.values()
                if inc.status.is_active() and inc.severity == SafetySeverity.CRITICAL
            ]

    def has_unresolved_critical_incident(self) -> bool:
        """Return True if any critical incident remains active."""
        with self._lock:
            return len(self.get_critical_incidents()) > 0
