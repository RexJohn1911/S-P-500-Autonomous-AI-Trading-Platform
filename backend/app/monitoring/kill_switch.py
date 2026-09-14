"""
Global Kill Switch (Phase 18).
Centralized, thread-safe, fail-closed kill switch with persistent state recovery
and verified safe clearing.
"""

from datetime import datetime, timezone
import logging
from threading import RLock
from typing import Any, Callable, Dict, Optional

from backend.app.monitoring.schemas import (
    KillSwitchSnapshot,
    KillSwitchState,
)
from backend.app.monitoring.storage import MonitoringStorage

logger = logging.getLogger(__name__)


class GlobalKillSwitch:
    """
    Centralized, fail-closed kill switch.
    When triggered, blocks all new order submissions immediately.
    """

    def __init__(self, storage: Optional[MonitoringStorage] = None):
        self._lock = RLock()
        self.storage = storage or MonitoringStorage()
        self._state = KillSwitchState.ARMED
        self._is_triggered = False
        self._triggered_at: Optional[datetime] = None
        self._triggered_by: Optional[str] = None
        self._reason: Optional[str] = None
        self._cleared_at: Optional[datetime] = None
        self._cleared_by: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

        # Restore persisted state across restarts
        persisted = self.storage.load_kill_switch_state()
        if persisted and persisted.is_triggered:
            self._state = KillSwitchState.TRIGGERED
            self._is_triggered = True
            self._triggered_at = persisted.triggered_at
            self._triggered_by = persisted.triggered_by
            self._reason = persisted.reason
            self._metadata = persisted.metadata
            logger.critical(
                "GlobalKillSwitch: Restored TRIGGERED state from storage (reason: %s, triggered_by: %s). Trading blocked.",
                self._reason,
                self._triggered_by,
            )

    @property
    def state(self) -> KillSwitchState:
        with self._lock:
            return self._state

    def is_triggered(self) -> bool:
        """Return True if kill switch is currently active (trading blocked)."""
        with self._lock:
            return self._is_triggered or self._state == KillSwitchState.TRIGGERED

    def is_armed(self) -> bool:
        """Return True if kill switch is armed and ready."""
        with self._lock:
            return self._state == KillSwitchState.ARMED and not self._is_triggered

    def trigger(
        self,
        reason: str,
        triggered_by: str = "SYSTEM",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KillSwitchSnapshot:
        """
        Trigger the kill switch to immediately halt order submission.
        """
        with self._lock:
            self._state = KillSwitchState.TRIGGERED
            self._is_triggered = True
            self._triggered_at = datetime.now(timezone.utc)
            self._triggered_by = triggered_by
            self._reason = reason
            self._metadata = metadata or {}
            self._cleared_at = None
            self._cleared_by = None

            snapshot = self.get_snapshot()
            self.storage.save_kill_switch_state(snapshot)
            logger.critical(
                "GLOBAL KILL SWITCH TRIGGERED by %s: %s (metadata=%s)",
                triggered_by,
                reason,
                metadata,
            )
            return snapshot

    def reset(
        self,
        cleared_by: str,
        verification_fn: Optional[Callable[[], tuple[bool, str]]] = None,
    ) -> tuple[bool, str]:
        """
        Attempt to reset the kill switch back to ARMED state.
        If a verification callback is provided, it must return (True, "") to succeed.
        """
        with self._lock:
            if not self.is_triggered():
                return True, "Kill switch already armed"

            self._state = KillSwitchState.CLEARING

            if verification_fn is not None:
                try:
                    can_reset, reason = verification_fn()
                    if not can_reset:
                        self._state = KillSwitchState.TRIGGERED
                        logger.warning("Kill switch reset rejected by verification check: %s", reason)
                        return False, f"Reset rejected by verification check: {reason}"
                except Exception as e:
                    self._state = KillSwitchState.TRIGGERED
                    logger.error("Kill switch reset verification raised error: %s", e)
                    return False, f"Verification failed with exception: {e}"

            self._state = KillSwitchState.ARMED
            self._is_triggered = False
            self._cleared_at = datetime.now(timezone.utc)
            self._cleared_by = cleared_by

            snapshot = self.get_snapshot()
            self.storage.save_kill_switch_state(snapshot)
            logger.info("GLOBAL KILL SWITCH RESET to ARMED by %s", cleared_by)
            return True, "Kill switch successfully armed"

    def get_snapshot(self) -> KillSwitchSnapshot:
        """Return current snapshot of kill switch state."""
        with self._lock:
            return KillSwitchSnapshot(
                state=self._state,
                is_triggered=self._is_triggered,
                triggered_at=self._triggered_at,
                triggered_by=self._triggered_by,
                reason=self._reason,
                cleared_at=self._cleared_at,
                cleared_by=self._cleared_by,
                metadata=dict(self._metadata),
            )
