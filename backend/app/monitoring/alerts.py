"""
Safety Alert Deduplication and Event Dispatcher (Phase 18).
Prevents alert storms by deduplicating identical events within a configurable sliding window.
"""

from datetime import datetime, timezone
import logging
from threading import RLock
from typing import Dict, List, Optional

from backend.app.monitoring.schemas import SafetyEvent, SafetySeverity
from backend.app.monitoring.storage import MonitoringStorage

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages safety alerts, applying deterministic deduplication to prevent alert storms.
    """

    def __init__(
        self,
        dedup_window_seconds: float = 60.0,
        storage: Optional[MonitoringStorage] = None,
    ):
        self._lock = RLock()
        self.dedup_window_seconds = dedup_window_seconds
        self.storage = storage or MonitoringStorage()
        self._last_event_time_by_fingerprint: Dict[str, datetime] = {}
        self._recent_events: List[SafetyEvent] = []
        self._max_recent_events = 500

    def emit_event(self, event: SafetyEvent) -> bool:
        """
        Emit a safety event. Returns True if event was accepted and recorded,
        or False if suppressed as a duplicate within the dedup window.
        """
        with self._lock:
            fp = event.fingerprint
            now = event.timestamp

            # Check deduplication window
            last_time = self._last_event_time_by_fingerprint.get(fp)
            if last_time:
                elapsed = (now - last_time).total_seconds()
                if elapsed < self.dedup_window_seconds and event.severity != SafetySeverity.CRITICAL:
                    # Deduplicated (critical events are never suppressed)
                    return False

            self._last_event_time_by_fingerprint[fp] = now
            self._recent_events.append(event)
            if len(self._recent_events) > self._max_recent_events:
                self._recent_events.pop(0)

            # Persist to audit log
            self.storage.append_audit_event(event)

            # Log appropriately
            if event.severity == SafetySeverity.CRITICAL:
                logger.critical("SAFETY CRITICAL [%s] %s: %s", event.component.value, event.event_type.value, event.message)
            elif event.severity == SafetySeverity.HIGH:
                logger.error("SAFETY HIGH [%s] %s: %s", event.component.value, event.event_type.value, event.message)
            elif event.severity == SafetySeverity.WARNING:
                logger.warning("SAFETY WARNING [%s] %s: %s", event.component.value, event.event_type.value, event.message)
            else:
                logger.info("SAFETY INFO [%s] %s: %s", event.component.value, event.event_type.value, event.message)

            return True

    def get_recent_events(self, limit: int = 100) -> List[SafetyEvent]:
        """Return most recent safety events in chronological order."""
        with self._lock:
            return list(self._recent_events[-limit:])
