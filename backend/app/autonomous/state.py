"""
Autonomous Loop State Machine (Phase 15).
Enforces valid lifecycle state transitions, safe transition guards, and failure containment.
"""

from datetime import datetime, timezone
import logging
from typing import Optional, Set

from backend.app.autonomous.schemas import LoopState

logger = logging.getLogger(__name__)


class AutonomousStateMachine:
    """
    Manages autonomous controller lifecycle transitions.
    """

    # Explicit allowed state machine transitions
    VALID_TRANSITIONS: dict[LoopState, Set[LoopState]] = {
        LoopState.CREATED: {LoopState.INITIALIZING, LoopState.FAILED},
        LoopState.INITIALIZING: {LoopState.READY, LoopState.FAILED},
        LoopState.READY: {LoopState.RUNNING, LoopState.STOPPING, LoopState.STOPPED, LoopState.FAILED},
        LoopState.RUNNING: {LoopState.PAUSED, LoopState.STOPPING, LoopState.FAILED},
        LoopState.PAUSED: {LoopState.RUNNING, LoopState.STOPPING, LoopState.STOPPED, LoopState.FAILED},
        LoopState.STOPPING: {LoopState.STOPPED, LoopState.FAILED},
        LoopState.STOPPED: {LoopState.INITIALIZING, LoopState.READY},  # Allows controlled restart
        LoopState.FAILED: {LoopState.RECOVERING, LoopState.STOPPED},
        LoopState.RECOVERING: {LoopState.READY, LoopState.RUNNING, LoopState.FAILED},
    }

    def __init__(self, initial_state: LoopState = LoopState.CREATED):
        self._current_state: LoopState = initial_state
        self._last_transition_time: datetime = datetime.now(timezone.utc)

    @property
    def current_state(self) -> LoopState:
        return self._current_state

    def can_transition_to(self, target_state: LoopState) -> bool:
        """Check if transition to target_state is permitted."""
        valid_next = self.VALID_TRANSITIONS.get(self._current_state, set())
        return target_state in valid_next

    def transition_to(self, target_state: LoopState, reason: Optional[str] = None) -> LoopState:
        """
        Transition to target_state or raise ValueError if invalid.
        """
        if self._current_state == target_state:
            return self._current_state

        if not self.can_transition_to(target_state):
            err_msg = f"Invalid autonomous lifecycle transition: {self._current_state.value} -> {target_state.value}"
            if reason:
                err_msg += f" (Reason: {reason})"
            logger.error(err_msg)
            raise ValueError(err_msg)

        old_state = self._current_state
        self._current_state = target_state
        self._last_transition_time = datetime.now(timezone.utc)
        logger.info("Autonomous state transitioned: %s -> %s%s", old_state.value, target_state.value, f" ({reason})" if reason else "")
        return self._current_state
