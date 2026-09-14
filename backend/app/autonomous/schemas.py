"""
Autonomous Trading Loop Domain Schemas (Phase 15).
Strongly typed data contracts for autonomous trading orchestration, lifecycle state,
cycle execution stages, checkpoints, failure classification, health monitoring, and audit events.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Union


class LoopState(str, Enum):
    """Lifecycle states of the autonomous trading loop controller."""
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    RECOVERING = "RECOVERING"


class CycleStatus(str, Enum):
    """Status of an individual autonomous trading cycle."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RECOVERED = "RECOVERED"


class CycleStage(str, Enum):
    """Sequential stages within a canonical trading cycle pipeline."""
    DATA_ACQUISITION = "DATA_ACQUISITION"
    DATA_VALIDATION = "DATA_VALIDATION"
    FEATURE_GENERATION = "FEATURE_GENERATION"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    REGIME_DETECTION = "REGIME_DETECTION"
    SIGNAL_GENERATION = "SIGNAL_GENERATION"
    PORTFOLIO_CONSTRUCTION = "PORTFOLIO_CONSTRUCTION"
    RISK_ENGINE = "RISK_ENGINE"
    ORDER_GENERATION = "ORDER_GENERATION"
    PAPER_EXECUTION = "PAPER_EXECUTION"
    ACCOUNT_VALUATION = "ACCOUNT_VALUATION"
    RECONCILIATION = "RECONCILIATION"
    CHECKPOINTING = "CHECKPOINTING"
    COMPLETED = "COMPLETED"


class FailureClass(str, Enum):
    """Classification of errors encountered during autonomous execution."""
    TRANSIENT = "TRANSIENT"            # Bounded retry permitted
    VALIDATION = "VALIDATION"          # Skip/fail cycle; fail-closed
    RISK = "RISK"                      # Target blocked by risk engine
    EXECUTION = "EXECUTION"            # Order rejection / matching issue
    ACCOUNTING = "ACCOUNTING"          # Accounting corruption; halt
    STORAGE = "STORAGE"                # Persistence failure
    MODEL = "MODEL"                    # Model missing or failed inference; halt
    CONFIGURATION = "CONFIGURATION"    # Invalid configuration; halt
    SYSTEM = "SYSTEM"                  # Process/runtime failure


class HealthStatus(str, Enum):
    """Operational health status of the autonomous trading system."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    STALE_DATA = "STALE_DATA"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RISK_BLOCKED = "RISK_BLOCKED"
    RECONCILIATION_FAILED = "RECONCILIATION_FAILED"
    FAILED = "FAILED"


class MissingSymbolPolicy(str, Enum):
    """Policy when some symbols in configured universe are missing."""
    FAIL_CLOSED = "FAIL_CLOSED"
    ALLOW_PARTIAL_UNIVERSE = "ALLOW_PARTIAL_UNIVERSE"


class AutonomousEventType(str, Enum):
    """Event classification for autonomous loop audit logging."""
    LOOP_STARTED = "LOOP_STARTED"
    LOOP_PAUSED = "LOOP_PAUSED"
    LOOP_RESUMED = "LOOP_RESUMED"
    LOOP_STOP_REQUESTED = "LOOP_STOP_REQUESTED"
    LOOP_STOPPED = "LOOP_STOPPED"
    LOOP_FAILED = "LOOP_FAILED"
    CYCLE_STARTED = "CYCLE_STARTED"
    CYCLE_STAGE_STARTED = "CYCLE_STAGE_STARTED"
    CYCLE_STAGE_COMPLETED = "CYCLE_STAGE_COMPLETED"
    CYCLE_COMPLETED = "CYCLE_COMPLETED"
    CYCLE_FAILED = "CYCLE_FAILED"
    DATA_STALE = "DATA_STALE"
    DATA_MISSING = "DATA_MISSING"
    MODEL_READY = "MODEL_READY"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    TARGET_GENERATED = "TARGET_GENERATED"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_BLOCKED = "RISK_BLOCKED"
    ORDERS_GENERATED = "ORDERS_GENERATED"
    ORDERS_EXECUTED = "ORDERS_EXECUTED"
    RECONCILIATION_COMPLETED = "RECONCILIATION_COMPLETED"
    CHECKPOINT_CREATED = "CHECKPOINT_CREATED"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    HEARTBEAT = "HEARTBEAT"


@dataclass
class CycleMetadata:
    """
    Metadata and summary metrics for an autonomous trading cycle.
    """
    cycle_id: str
    session_id: str
    market_timestamp: datetime
    start_time: datetime
    end_time: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1Day"
    status: CycleStatus = CycleStatus.CREATED
    stage: CycleStage = CycleStage.DATA_ACQUISITION
    versions: Dict[str, str] = field(default_factory=dict)
    orders_generated_count: int = 0
    orders_executed_count: int = 0
    orders_rejected_count: int = 0
    error_message: Optional[str] = None
    failure_class: Optional[FailureClass] = None
    duration_seconds: float = 0.0
    stages_executed: List[Dict[str, Any]] = field(default_factory=list)
    model_telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "session_id": self.session_id,
            "market_timestamp": self.market_timestamp.isoformat(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "status": self.status.value,
            "stage": self.stage.value,
            "versions": self.versions,
            "orders_generated_count": self.orders_generated_count,
            "orders_executed_count": self.orders_executed_count,
            "orders_rejected_count": self.orders_rejected_count,
            "error_message": self.error_message,
            "failure_class": self.failure_class.value if self.failure_class else None,
            "duration_seconds": round(float(self.duration_seconds), 4),
            "stages_executed": self.stages_executed,
            "model_telemetry": self.model_telemetry,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleMetadata":
        return cls(
            cycle_id=data["cycle_id"],
            session_id=data["session_id"],
            market_timestamp=datetime.fromisoformat(data["market_timestamp"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            symbols=data.get("symbols", []),
            timeframe=data.get("timeframe", "1Day"),
            status=CycleStatus(data.get("status", "CREATED")),
            stage=CycleStage(data.get("stage", "DATA_ACQUISITION")),
            versions=data.get("versions", {}),
            orders_generated_count=int(data.get("orders_generated_count", 0)),
            orders_executed_count=int(data.get("orders_executed_count", 0)),
            orders_rejected_count=int(data.get("orders_rejected_count", 0)),
            error_message=data.get("error_message"),
            failure_class=FailureClass(data["failure_class"]) if data.get("failure_class") else None,
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            stages_executed=data.get("stages_executed", []),
            model_telemetry=data.get("model_telemetry", {}),
        )


@dataclass
class CycleCheckpoint:
    """
    Durable checkpoint created at critical cycle stages for idempotent crash recovery.
    """
    checkpoint_id: str
    session_id: str
    cycle_id: str
    market_timestamp: datetime
    last_successful_stage: CycleStage
    paper_session_id: str
    configuration_hash: str
    state_hash: str
    timestamp: datetime
    status: CycleStatus = CycleStatus.RUNNING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "cycle_id": self.cycle_id,
            "market_timestamp": self.market_timestamp.isoformat(),
            "last_successful_stage": self.last_successful_stage.value,
            "paper_session_id": self.paper_session_id,
            "configuration_hash": self.configuration_hash,
            "state_hash": self.state_hash,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleCheckpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            cycle_id=data["cycle_id"],
            market_timestamp=datetime.fromisoformat(data["market_timestamp"]),
            last_successful_stage=CycleStage(data["last_successful_stage"]),
            paper_session_id=data["paper_session_id"],
            configuration_hash=data["configuration_hash"],
            state_hash=data["state_hash"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=CycleStatus(data.get("status", "RUNNING")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AutonomousHealthSnapshot:
    """
    Point-in-time health and heartbeat metrics.
    """
    session_id: str
    timestamp: datetime
    process_alive: bool
    loop_state: LoopState
    health_status: HealthStatus
    uptime_seconds: float
    current_cycle_id: Optional[str] = None
    last_cycle_start: Optional[datetime] = None
    last_cycle_completed: Optional[datetime] = None
    last_market_timestamp: Optional[datetime] = None
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "process_alive": bool(self.process_alive),
            "loop_state": self.loop_state.value,
            "health_status": self.health_status.value,
            "uptime_seconds": round(float(self.uptime_seconds), 2),
            "current_cycle_id": self.current_cycle_id,
            "last_cycle_start": self.last_cycle_start.isoformat() if self.last_cycle_start else None,
            "last_cycle_completed": self.last_cycle_completed.isoformat() if self.last_cycle_completed else None,
            "last_market_timestamp": self.last_market_timestamp.isoformat() if self.last_market_timestamp else None,
            "total_cycles": int(self.total_cycles),
            "successful_cycles": int(self.successful_cycles),
            "failed_cycles": int(self.failed_cycles),
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomousHealthSnapshot":
        return cls(
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            process_alive=bool(data.get("process_alive", True)),
            loop_state=LoopState(data.get("loop_state", "READY")),
            health_status=HealthStatus(data.get("health_status", "HEALTHY")),
            uptime_seconds=float(data.get("uptime_seconds", 0.0)),
            current_cycle_id=data.get("current_cycle_id"),
            last_cycle_start=datetime.fromisoformat(data["last_cycle_start"]) if data.get("last_cycle_start") else None,
            last_cycle_completed=datetime.fromisoformat(data["last_cycle_completed"]) if data.get("last_cycle_completed") else None,
            last_market_timestamp=datetime.fromisoformat(data["last_market_timestamp"]) if data.get("last_market_timestamp") else None,
            total_cycles=int(data.get("total_cycles", 0)),
            successful_cycles=int(data.get("successful_cycles", 0)),
            failed_cycles=int(data.get("failed_cycles", 0)),
            last_error=data.get("last_error"),
        )


@dataclass
class AutonomousEvent:
    """
    Immutable audit log event for autonomous trading loop actions.
    """
    event_id: str
    session_id: str
    cycle_id: Optional[str]
    timestamp: datetime
    event_type: AutonomousEventType
    stage: Optional[CycleStage] = None
    status: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "stage": self.stage.value if self.stage else None,
            "status": self.status,
            "description": self.description,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomousEvent":
        return cls(
            event_id=data["event_id"],
            session_id=data["session_id"],
            cycle_id=data.get("cycle_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AutonomousEventType(data["event_type"]),
            stage=CycleStage(data["stage"]) if data.get("stage") else None,
            status=data.get("status"),
            description=data.get("description", ""),
            details=data.get("details", {}),
        )


@dataclass
class AutonomousConfig:
    """
    Configuration parameters for an autonomous trading controller session.
    """
    session_id: Optional[str] = None
    universe: List[str] = field(default_factory=lambda: ["AAPL"])
    timeframe: str = "1Day"
    schedule_interval: str = "daily"  # "daily", "hourly", "manual"
    max_stale_tolerance_seconds: float = 86400.0 * 5  # 5 days max staleness for daily bars
    missing_symbol_policy: MissingSymbolPolicy = MissingSymbolPolicy.FAIL_CLOSED
    max_retry_attempts: int = 3
    retry_backoff_base_seconds: float = 1.0
    auto_checkpoint_enabled: bool = True
    paper_initial_capital: float = 100_000.0
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0005
    bid_ask_spread_rate: float = 0.0002
    daily_borrow_rate: float = 0.0
    allow_short: bool = False
    version: str = "autonomous-v1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "universe": self.universe,
            "timeframe": self.timeframe,
            "schedule_interval": self.schedule_interval,
            "max_stale_tolerance_seconds": float(self.max_stale_tolerance_seconds),
            "missing_symbol_policy": self.missing_symbol_policy.value,
            "max_retry_attempts": int(self.max_retry_attempts),
            "retry_backoff_base_seconds": float(self.retry_backoff_base_seconds),
            "auto_checkpoint_enabled": bool(self.auto_checkpoint_enabled),
            "paper_initial_capital": float(self.paper_initial_capital),
            "commission_rate": float(self.commission_rate),
            "slippage_rate": float(self.slippage_rate),
            "bid_ask_spread_rate": float(self.bid_ask_spread_rate),
            "daily_borrow_rate": float(self.daily_borrow_rate),
            "allow_short": bool(self.allow_short),
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomousConfig":
        return cls(
            session_id=data.get("session_id"),
            universe=data.get("universe", ["AAPL"]),
            timeframe=data.get("timeframe", "1Day"),
            schedule_interval=data.get("schedule_interval", "daily"),
            max_stale_tolerance_seconds=float(data.get("max_stale_tolerance_seconds", 86400.0 * 5)),
            missing_symbol_policy=MissingSymbolPolicy(data.get("missing_symbol_policy", "FAIL_CLOSED")),
            max_retry_attempts=int(data.get("max_retry_attempts", 3)),
            retry_backoff_base_seconds=float(data.get("retry_backoff_base_seconds", 1.0)),
            auto_checkpoint_enabled=bool(data.get("auto_checkpoint_enabled", True)),
            paper_initial_capital=float(data.get("paper_initial_capital", 100000.0)),
            commission_rate=float(data.get("commission_rate", 0.0005)),
            slippage_rate=float(data.get("slippage_rate", 0.0005)),
            bid_ask_spread_rate=float(data.get("bid_ask_spread_rate", 0.0002)),
            daily_borrow_rate=float(data.get("daily_borrow_rate", 0.0)),
            allow_short=bool(data.get("allow_short", False)),
            version=data.get("version", "autonomous-v1.0"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AutonomousSession:
    """
    Session record for an autonomous trading controller instance.
    """
    session_id: str
    config: AutonomousConfig
    state: LoopState = LoopState.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    configuration_hash: str = ""
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    last_market_timestamp: Optional[datetime] = None
    paper_session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "configuration_hash": self.configuration_hash,
            "total_cycles": int(self.total_cycles),
            "successful_cycles": int(self.successful_cycles),
            "failed_cycles": int(self.failed_cycles),
            "last_market_timestamp": self.last_market_timestamp.isoformat() if self.last_market_timestamp else None,
            "paper_session_id": self.paper_session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomousSession":
        return cls(
            session_id=data["session_id"],
            config=AutonomousConfig.from_dict(data["config"]),
            state=LoopState(data.get("state", "CREATED")),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            stopped_at=datetime.fromisoformat(data["stopped_at"]) if data.get("stopped_at") else None,
            configuration_hash=data.get("configuration_hash", ""),
            total_cycles=int(data.get("total_cycles", 0)),
            successful_cycles=int(data.get("successful_cycles", 0)),
            failed_cycles=int(data.get("failed_cycles", 0)),
            last_market_timestamp=datetime.fromisoformat(data["last_market_timestamp"]) if data.get("last_market_timestamp") else None,
            paper_session_id=data.get("paper_session_id"),
        )


@dataclass
class AutonomousLoopResult:
    """
    Complete summary of autonomous loop execution.
    """
    session: AutonomousSession
    cycles: List[CycleMetadata]
    checkpoints: List[CycleCheckpoint]
    health_snapshots: List[AutonomousHealthSnapshot]
    audit_events: List[AutonomousEvent]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session.to_dict(),
            "cycles": [c.to_dict() for c in self.cycles],
            "checkpoints": [ck.to_dict() for ck in self.checkpoints],
            "health_snapshots": [h.to_dict() for h in self.health_snapshots],
            "audit_events": [a.to_dict() for a in self.audit_events],
            "provenance_hash": self.provenance_hash,
        }
