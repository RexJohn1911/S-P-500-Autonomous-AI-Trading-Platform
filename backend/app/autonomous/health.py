"""
Autonomous Health & Heartbeat Monitor (Phase 15).
Tracks process liveness, trading health vs process health, stale data detection,
and error accounting.
"""

from datetime import datetime, timezone
import logging
from typing import Optional

from backend.app.autonomous.schemas import (
    AutonomousHealthSnapshot,
    HealthStatus,
    LoopState,
)

logger = logging.getLogger(__name__)


class AutonomousHealthMonitor:
    """
    Monitors process and trading health.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time: datetime = datetime.now(timezone.utc)
        self.last_cycle_start: Optional[datetime] = None
        self.last_cycle_completed: Optional[datetime] = None
        self.last_market_timestamp: Optional[datetime] = None
        self.current_cycle_id: Optional[str] = None
        self.total_cycles: int = 0
        self.successful_cycles: int = 0
        self.failed_cycles: int = 0
        self.last_error: Optional[str] = None
        self._current_health_status: HealthStatus = HealthStatus.HEALTHY

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    def record_cycle_start(self, cycle_id: str, market_timestamp: datetime) -> None:
        self.current_cycle_id = cycle_id
        self.last_cycle_start = datetime.now(timezone.utc)
        self.last_market_timestamp = market_timestamp
        self.total_cycles += 1

    def record_cycle_success(self, cycle_id: str) -> None:
        self.last_cycle_completed = datetime.now(timezone.utc)
        self.successful_cycles += 1
        self.current_cycle_id = None
        self._current_health_status = HealthStatus.HEALTHY

    def record_cycle_failure(self, cycle_id: str, error: str, health_status: HealthStatus = HealthStatus.DEGRADED) -> None:
        self.failed_cycles += 1
        self.last_error = error
        self.current_cycle_id = None
        self._current_health_status = health_status

    def get_snapshot(self, loop_state: LoopState) -> AutonomousHealthSnapshot:
        """Generate current point-in-time health snapshot."""
        # Process is alive if we are executing this method
        now = datetime.now(timezone.utc)
        
        status = self._current_health_status
        if loop_state == LoopState.PAUSED:
            status = HealthStatus.PAUSED
        elif loop_state == LoopState.FAILED:
            status = HealthStatus.FAILED

        return AutonomousHealthSnapshot(
            session_id=self.session_id,
            timestamp=now,
            process_alive=True,
            loop_state=loop_state,
            health_status=status,
            uptime_seconds=self.uptime_seconds,
            current_cycle_id=self.current_cycle_id,
            last_cycle_start=self.last_cycle_start,
            last_cycle_completed=self.last_cycle_completed,
            last_market_timestamp=self.last_market_timestamp,
            total_cycles=self.total_cycles,
            successful_cycles=self.successful_cycles,
            failed_cycles=self.failed_cycles,
            last_error=self.last_error,
        )
