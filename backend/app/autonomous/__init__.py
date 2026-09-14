"""
Autonomous Trading Loop Subsystem (Phase 15).
Coordinates lifecycle state machines, schedulers, cycle pipeline execution,
checkpointing, crash recovery, health monitoring, and paper trading orchestration.
"""

from backend.app.autonomous.checkpoint import CheckpointManager
from backend.app.autonomous.controller import AutonomousTradingController
from backend.app.autonomous.cycle import TradingCycleExecutor
from backend.app.autonomous.health import AutonomousHealthMonitor
from backend.app.autonomous.recovery import CrashRecoveryManager
from backend.app.autonomous.scheduler import MarketDataScheduler
from backend.app.autonomous.schemas import (
    AutonomousConfig,
    AutonomousEvent,
    AutonomousEventType,
    AutonomousHealthSnapshot,
    AutonomousLoopResult,
    AutonomousSession,
    CycleCheckpoint,
    CycleMetadata,
    CycleStage,
    CycleStatus,
    FailureClass,
    HealthStatus,
    LoopState,
    MissingSymbolPolicy,
)
from backend.app.autonomous.state import AutonomousStateMachine
from backend.app.autonomous.storage import AutonomousStorage

__all__ = [
    "AutonomousConfig",
    "AutonomousEvent",
    "AutonomousEventType",
    "AutonomousHealthMonitor",
    "AutonomousHealthSnapshot",
    "AutonomousLoopResult",
    "AutonomousSession",
    "AutonomousStateMachine",
    "AutonomousStorage",
    "AutonomousTradingController",
    "CheckpointManager",
    "CrashRecoveryManager",
    "CycleCheckpoint",
    "CycleMetadata",
    "CycleStage",
    "CycleStatus",
    "FailureClass",
    "HealthStatus",
    "LoopState",
    "MarketDataScheduler",
    "MissingSymbolPolicy",
    "TradingCycleExecutor",
]
