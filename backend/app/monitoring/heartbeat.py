"""
Heartbeat Tracker (Phase 18).
Monitors subsystem liveness, detecting stale heartbeats across autonomous loop,
market data, broker, and worker processes.
"""

from datetime import datetime, timezone
import logging
from threading import RLock
from typing import Dict, List, Optional

from backend.app.monitoring.schemas import (
    ComponentType,
    HealthStatus,
    HeartbeatRecord,
)

logger = logging.getLogger(__name__)


class HeartbeatTracker:
    """
    Tracks and evaluates periodic heartbeats from core subsystems.
    """

    def __init__(self, default_timeout_seconds: float = 120.0):
        self._lock = RLock()
        self.default_timeout_seconds = default_timeout_seconds
        self._heartbeats: Dict[ComponentType, HeartbeatRecord] = {}

    def record_heartbeat(
        self,
        component: ComponentType,
        interval_seconds: float = 60.0,
        timeout_seconds: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> HeartbeatRecord:
        """Record a successful heartbeat from a component."""
        with self._lock:
            timeout = timeout_seconds or self.default_timeout_seconds
            now = timestamp or datetime.now(timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            record = HeartbeatRecord(
                component=component,
                timestamp=now,
                interval_seconds=interval_seconds,
                timeout_seconds=timeout,
                status=HealthStatus.HEALTHY,
                metadata=metadata or {},
            )
            self._heartbeats[component] = record
            return record

    def check_heartbeats(self, as_of: Optional[datetime] = None) -> Dict[ComponentType, HeartbeatRecord]:
        """Check all registered heartbeats and update status based on staleness."""
        with self._lock:
            now = as_of or datetime.now(timezone.utc)
            updated: Dict[ComponentType, HeartbeatRecord] = {}
            for comp, record in self._heartbeats.items():
                elapsed = (now - record.timestamp).total_seconds()
                if elapsed > record.timeout_seconds:
                    record.status = HealthStatus.CRITICAL
                elif elapsed > (record.timeout_seconds * 0.75):
                    record.status = HealthStatus.WARNING
                else:
                    record.status = HealthStatus.HEALTHY
                updated[comp] = record
            return updated

    def get_stale_heartbeats(self, as_of: Optional[datetime] = None) -> List[HeartbeatRecord]:
        """Return list of heartbeats currently exceeding their timeout threshold."""
        with self._lock:
            checked = self.check_heartbeats(as_of=as_of)
            return [rec for rec in checked.values() if rec.is_stale(as_of=as_of)]

    def get_heartbeat_status(self, component: ComponentType, as_of: Optional[datetime] = None) -> HealthStatus:
        """Return current health status of a specific component's heartbeat."""
        with self._lock:
            checked = self.check_heartbeats(as_of=as_of)
            record = checked.get(component)
            if record is None:
                return HealthStatus.UNKNOWN
            return record.status
