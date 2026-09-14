"""
Autonomous Trading Controller (Phase 15).
Master coordinator managing the lifecycle, scheduling, cycle execution,
checkpointing, crash recovery, and health monitoring for autonomous paper trading.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

from backend.app.autonomous.checkpoint import CheckpointManager
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
    CycleMetadata,
    CycleStatus,
    HealthStatus,
    LoopState,
)
from backend.app.autonomous.state import AutonomousStateMachine
from backend.app.autonomous.storage import AutonomousStorage
from backend.app.backtest.schemas import DividendEvent, SplitEvent
from backend.app.broker.paper import PaperBrokerAdapter
from backend.app.data.models import BarData
from backend.app.paper_trading.schemas import PaperTradingConfig
from backend.app.paper_trading.service import PaperTradingService
from backend.app.portfolio.service import PortfolioConstructionService
from backend.app.risk.schemas import RiskEngineConfig
from backend.app.strategy.schemas import SignalCandidate

logger = logging.getLogger(__name__)


class AutonomousTradingController:
    """
    Primary controller orchestrating autonomous paper trading loops.
    """

    def __init__(
        self,
        config: Optional[AutonomousConfig] = None,
        storage: Optional[AutonomousStorage] = None,
        paper_service: Optional[PaperTradingService] = None,
        portfolio_service: Optional[PortfolioConstructionService] = None,
        risk_config: Optional[RiskEngineConfig] = None,
        signal_generator: Optional[Callable[[datetime, Dict[str, BarData]], List[SignalCandidate]]] = None,
        session_id: Optional[str] = None,
    ):
        self.config = config or AutonomousConfig()
        self.storage = storage or AutonomousStorage()
        self.session_id = session_id or f"auto_ses_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.config.session_id = self.session_id

        # Calculate configuration hash
        config_dict = self.config.to_dict()
        self.config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()

        # Paper trading subsystem
        paper_cfg = PaperTradingConfig(
            initial_capital=self.config.paper_initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
            bid_ask_spread_rate=self.config.bid_ask_spread_rate,
            daily_borrow_rate=self.config.daily_borrow_rate,
            allow_short=self.config.allow_short,
        )
        self.paper_service = paper_service or PaperTradingService(
            config=paper_cfg,
            session_id=f"paper_{self.session_id}",
        )
        self.broker = PaperBrokerAdapter(
            simulated_broker=self.paper_service.broker,
            session_id=self.session_id,
        )

        # Core Autonomous sub-components
        self.state_machine = AutonomousStateMachine(initial_state=LoopState.CREATED)
        self.scheduler = MarketDataScheduler(config=self.config)
        self.checkpoint_manager = CheckpointManager(
            session_id=self.session_id,
            paper_session_id=self.paper_service.session_id,
            config_hash=self.config_hash,
        )
        self.health_monitor = AutonomousHealthMonitor(session_id=self.session_id)
        self.cycle_executor = TradingCycleExecutor(
            session_id=self.session_id,
            config=self.config,
            paper_service=self.paper_service,
            checkpoint_manager=self.checkpoint_manager,
            portfolio_service=portfolio_service,
            risk_config=risk_config,
            signal_generator=signal_generator,
        )

        # Session tracking
        self.session = AutonomousSession(
            session_id=self.session_id,
            config=self.config,
            state=LoopState.CREATED,
            configuration_hash=self.config_hash,
            paper_session_id=self.paper_service.session_id,
        )
        self.cycles_history: List[CycleMetadata] = []
        self.health_history: List[AutonomousHealthSnapshot] = []
        self.audit_events: List[AutonomousEvent] = []
        self._event_counter: int = 0
        self._cycle_counter: int = 0

        self._emit_event(
            event_type=AutonomousEventType.LOOP_STARTED,
            description=f"Autonomous controller created for session {self.session_id}",
            details={"config_hash": self.config_hash},
        )

    def _emit_event(
        self,
        event_type: AutonomousEventType,
        description: str,
        cycle_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> AutonomousEvent:
        """Record and append an audit event."""
        self._event_counter += 1
        ts = timestamp or datetime.now(timezone.utc)
        evt = AutonomousEvent(
            event_id=f"evt_{self.session_id}_{self._event_counter}",
            session_id=self.session_id,
            cycle_id=cycle_id,
            timestamp=ts,
            event_type=event_type,
            description=description,
            details=details or {},
        )
        self.audit_events.append(evt)
        return evt

    @property
    def current_state(self) -> LoopState:
        return self.state_machine.current_state

    def initialize(self) -> None:
        """Initialize controller components and verify crash recovery."""
        self.state_machine.transition_to(LoopState.INITIALIZING, "Starting controller initialization")
        
        # Check recovery state
        latest_chk = self.checkpoint_manager.get_latest_checkpoint()
        can_recover, action_desc, last_stage = CrashRecoveryManager.analyze_recovery_state(
            latest_checkpoint=latest_chk,
            paper_service=self.paper_service,
        )
        logger.info("Recovery analysis: %s", action_desc)

        self.state_machine.transition_to(LoopState.READY, "Controller initialized and ready")
        self.session.state = LoopState.READY

    def start(self) -> None:
        """Start autonomous loop execution."""
        if self.state_machine.current_state in (LoopState.CREATED, LoopState.STOPPED):
            self.initialize()

        self.state_machine.transition_to(LoopState.RUNNING, "Starting autonomous execution loop")
        self.session.state = LoopState.RUNNING
        self.session.started_at = datetime.now(timezone.utc)
        self.paper_service.start_session(symbols=self.config.universe)
        self._emit_event(AutonomousEventType.LOOP_STARTED, "Autonomous loop started")

    def pause(self) -> None:
        """Gracefully pause autonomous loop execution."""
        self.state_machine.transition_to(LoopState.PAUSED, "User/system requested pause")
        self.session.state = LoopState.PAUSED
        self.paper_service.pause_session()
        self._emit_event(AutonomousEventType.LOOP_PAUSED, "Autonomous loop paused")

    def resume(self) -> None:
        """Resume paused autonomous loop."""
        self.state_machine.transition_to(LoopState.RUNNING, "Resuming autonomous loop")
        self.session.state = LoopState.RUNNING
        self.paper_service.resume_session()
        self._emit_event(AutonomousEventType.LOOP_RESUMED, "Autonomous loop resumed")

    def stop(self) -> None:
        """Gracefully stop autonomous loop."""
        self.state_machine.transition_to(LoopState.STOPPING, "Graceful stop initiated")
        self.session.state = LoopState.STOPPING
        self.paper_service.stop_session()
        self.state_machine.transition_to(LoopState.STOPPED, "Graceful stop completed")
        self.session.state = LoopState.STOPPED
        self.session.stopped_at = datetime.now(timezone.utc)
        self._emit_event(AutonomousEventType.LOOP_STOPPED, "Autonomous loop stopped")

    def emergency_kill_switch(self, reason: str = "Emergency stop requested") -> None:
        """Immediate manual kill switch halting trading and releasing locks."""
        logger.critical("EMERGENCY KILL SWITCH ACTIVATED: %s", reason)
        self._emit_event(AutonomousEventType.LOOP_STOP_REQUESTED, f"Kill switch: {reason}")
        if self.scheduler.is_cycle_active():
            self.scheduler.release_cycle_lock(self.scheduler._active_cycle_id)  # type: ignore
        try:
            self.paper_service.stop_session()
        except Exception as e:
            logger.warning("Paper session stop error during kill switch: %s", str(e))
        
        self.state_machine._current_state = LoopState.STOPPED
        self.session.state = LoopState.STOPPED
        self.session.stopped_at = datetime.now(timezone.utc)
        self._emit_event(AutonomousEventType.LOOP_STOPPED, f"Autonomous loop halted via kill switch: {reason}")

    def execute_market_event(
        self,
        market_timestamp: datetime,
        current_bars: Dict[str, BarData],
        corporate_actions: Optional[List[Union[DividendEvent, SplitEvent]]] = None,
    ) -> CycleMetadata:
        """
        Execute an autonomous trading cycle triggered by a market-data event.
        Guarantees idempotency and concurrency isolation.
        """
        if self.state_machine.current_state != LoopState.RUNNING:
            if self.state_machine.current_state in (LoopState.CREATED, LoopState.READY):
                self.start()
            else:
                raise ValueError(f"Cannot execute market event in loop state: {self.state_machine.current_state.value}")

        # 1. Idempotency Check: Prevent duplicate processing of the same timestamp
        if self.scheduler.is_cycle_already_completed(self.session_id, market_timestamp):
            logger.info("Skipping market timestamp %s: already processed in session %s", market_timestamp.isoformat(), self.session_id)
            skip_meta = CycleMetadata(
                cycle_id=f"cyc_skip_{int(market_timestamp.timestamp())}",
                session_id=self.session_id,
                market_timestamp=market_timestamp,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                symbols=list(current_bars.keys()),
                status=CycleStatus.SKIPPED,
                error_message="Duplicate market timestamp event skipped",
            )
            return skip_meta

        # 2. Concurrency Lock: Prevent overlapping cycle execution
        self._cycle_counter += 1
        cycle_id = f"cyc_{self.session_id}_{self._cycle_counter}"
        if not self.scheduler.acquire_cycle_lock(cycle_id):
            logger.warning("Overlapping cycle attempt blocked for cycle %s", cycle_id)
            blocked_meta = CycleMetadata(
                cycle_id=cycle_id,
                session_id=self.session_id,
                market_timestamp=market_timestamp,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                status=CycleStatus.SKIPPED,
                error_message="Cycle blocked: another cycle is currently running",
            )
            return blocked_meta

        try:
            # 3. Market Data & Staleness Validation
            is_valid, val_err, missing_syms = self.scheduler.validate_market_data(current_bars)
            if not is_valid:
                logger.warning("Market data validation failed for cycle %s: %s", cycle_id, val_err)
                self._emit_event(AutonomousEventType.DATA_STALE, f"Data validation failure: {val_err}", cycle_id=cycle_id)
                failed_meta = CycleMetadata(
                    cycle_id=cycle_id,
                    session_id=self.session_id,
                    market_timestamp=market_timestamp,
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    status=CycleStatus.FAILED,
                    error_message=val_err,
                )
                self.cycles_history.append(failed_meta)
                self.session.total_cycles += 1
                self.session.failed_cycles += 1
                self.health_monitor.record_cycle_failure(cycle_id, val_err or "Validation failed")
                return failed_meta

            # 4. Record Cycle Start in Health Monitor
            self.health_monitor.record_cycle_start(cycle_id, market_timestamp)

            # 5. Execute Cycle Pipeline
            cycle_meta = self.cycle_executor.execute_cycle(
                cycle_id=cycle_id,
                market_timestamp=market_timestamp,
                current_bars=current_bars,
                corporate_actions=corporate_actions,
                event_callback=lambda evt: self.audit_events.append(evt),
            )

            # 6. Update Session Counters and Mark Completed
            self.cycles_history.append(cycle_meta)
            self.session.total_cycles += 1
            self.session.last_market_timestamp = market_timestamp

            if cycle_meta.status == CycleStatus.COMPLETED:
                self.session.successful_cycles += 1
                self.scheduler.mark_cycle_completed(self.session_id, market_timestamp)
                self.health_monitor.record_cycle_success(cycle_id)
            else:
                self.session.failed_cycles += 1
                self.health_monitor.record_cycle_failure(cycle_id, cycle_meta.error_message or "Execution failed")

            # Capture health snapshot
            health_snap = self.health_monitor.get_snapshot(self.current_state)
            self.health_history.append(health_snap)

            return cycle_meta

        finally:
            self.scheduler.release_cycle_lock(cycle_id)

    def run_replay(
        self,
        market_data: Dict[str, List[BarData]],
        corporate_actions: Optional[Dict[datetime, List[Union[DividendEvent, SplitEvent]]]] = None,
        save_artifacts: bool = False,
    ) -> AutonomousLoopResult:
        """
        Run continuous deterministic replay over a chronological series of market data bars.
        """
        # Build chronological timeline
        ts_set = set()
        for sym, bars in market_data.items():
            for b in bars:
                ts_set.add(b.timestamp)

        sorted_timestamps = sorted(ts_set)
        if not sorted_timestamps:
            raise ValueError("No market data bars available for replay")

        bar_lookup: Dict[datetime, Dict[str, BarData]] = {}
        for sym, bars in market_data.items():
            for b in bars:
                if b.timestamp not in bar_lookup:
                    bar_lookup[b.timestamp] = {}
                bar_lookup[b.timestamp][sym] = b

        self.start()

        for ts in sorted_timestamps:
            current_bars = bar_lookup.get(ts, {})
            if not current_bars:
                continue

            ca_list = corporate_actions.get(ts) if corporate_actions else None
            self.execute_market_event(
                market_timestamp=ts,
                current_bars=current_bars,
                corporate_actions=ca_list,
            )

        self.stop()
        result = self.get_result()

        if save_artifacts:
            self.storage.save_result(result)

        return result

    def get_result(self) -> AutonomousLoopResult:
        """Generate final immutable autonomous loop result with provenance hash."""
        summary = {
            "session_id": self.session_id,
            "config_hash": self.config_hash,
            "total_cycles": self.session.total_cycles,
            "successful_cycles": self.session.successful_cycles,
            "failed_cycles": self.session.failed_cycles,
            "paper_session_id": self.paper_service.session_id,
        }
        prov_hash = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()

        return AutonomousLoopResult(
            session=self.session,
            cycles=list(self.cycles_history),
            checkpoints=list(self.checkpoint_manager.checkpoints),
            health_snapshots=list(self.health_history),
            audit_events=list(self.audit_events),
            provenance_hash=prov_hash,
        )
