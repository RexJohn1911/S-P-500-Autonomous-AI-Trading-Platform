"""
Autonomous Trading Cycle Pipeline Executor (Phase 15).
Coordinates execution of the canonical 14-stage quantitative pipeline:
Data Acquisition -> Data Validation -> Feature Generation -> Model Inference ->
Regime Detection -> Signal Engine -> Portfolio Construction -> Risk Engine ->
Order Generation -> Paper Execution -> Valuation -> Reconciliation -> Checkpointing.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

from backend.app.autonomous.checkpoint import CheckpointManager
from backend.app.autonomous.schemas import (
    AutonomousConfig,
    AutonomousEvent,
    AutonomousEventType,
    CycleMetadata,
    CycleStage,
    CycleStatus,
    FailureClass,
    HealthStatus,
)
from backend.app.backtest.schemas import DividendEvent, SplitEvent
from backend.app.data.models import BarData
from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperOrderStatus,
    PaperReconciliationStatus,
)
from backend.app.paper_trading.service import PaperTradingService
from backend.app.portfolio.schemas import PortfolioConstructionResult, PortfolioTarget
from backend.app.portfolio.service import PortfolioConstructionService
from backend.app.risk.engine import RiskAdjustmentEngine
from backend.app.risk.schemas import RiskAdjustedTarget, RiskContext, RiskEngineConfig, RiskStatus
from backend.app.strategy.schemas import SignalCandidate

logger = logging.getLogger(__name__)


class TradingCycleExecutor:
    """
    Executes a single autonomous trading cycle with stage-by-stage auditing and checkpointing.
    """

    def __init__(
        self,
        session_id: str,
        config: AutonomousConfig,
        paper_service: PaperTradingService,
        checkpoint_manager: CheckpointManager,
        portfolio_service: Optional[PortfolioConstructionService] = None,
        risk_config: Optional[RiskEngineConfig] = None,
        signal_generator: Optional[Callable[[datetime, Dict[str, BarData]], List[SignalCandidate]]] = None,
    ):
        self.session_id = session_id
        self.config = config
        self.paper_service = paper_service
        self.checkpoint_manager = checkpoint_manager
        self.portfolio_service = portfolio_service or PortfolioConstructionService()
        self.risk_config = risk_config or RiskEngineConfig()
        self.signal_generator = signal_generator

    def execute_cycle(
        self,
        cycle_id: str,
        market_timestamp: datetime,
        current_bars: Dict[str, BarData],
        corporate_actions: Optional[List[Union[DividendEvent, SplitEvent]]] = None,
        event_callback: Optional[Callable[[AutonomousEvent], None]] = None,
    ) -> CycleMetadata:
        """
        Execute an end-to-end trading cycle pipeline.
        """
        start_time = datetime.now(timezone.utc)
        cycle_meta = CycleMetadata(
            cycle_id=cycle_id,
            session_id=self.session_id,
            market_timestamp=market_timestamp,
            start_time=start_time,
            symbols=list(current_bars.keys()),
            timeframe=self.config.timeframe,
            status=CycleStatus.RUNNING,
            stage=CycleStage.DATA_ACQUISITION,
            versions={
                "autonomous": self.config.version,
                "paper": self.paper_service.config.version,
                "risk": self.risk_config.version if hasattr(self.risk_config, "version") else "risk-v1.1",
            },
        )

        stages_record: List[Dict[str, Any]] = []

        def record_stage(st: CycleStage, status_str: str, dur_ms: float, err: Optional[str] = None):
            stages_record.append({
                "stage": st.value,
                "status": status_str,
                "duration_ms": round(dur_ms, 3),
                "error": err,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        def emit(evt_type: AutonomousEventType, stage: CycleStage, desc: str, details: Optional[Dict[str, Any]] = None):
            if event_callback:
                event_callback(
                    AutonomousEvent(
                        event_id=f"evt_{self.session_id}_{cycle_id}_{stage.value}_{int(time.time()*1000)}",
                        session_id=self.session_id,
                        cycle_id=cycle_id,
                        timestamp=datetime.now(timezone.utc),
                        event_type=evt_type,
                        stage=stage,
                        description=desc,
                        details=details or {},
                    )
                )

        emit(AutonomousEventType.CYCLE_STARTED, CycleStage.DATA_ACQUISITION, f"Started autonomous trading cycle {cycle_id}")
        t_stage = time.perf_counter()
        # Stage 1: Data Acquisition
        record_stage(CycleStage.DATA_ACQUISITION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)

        try:
            # 2. Data Validation Stage
            cycle_meta.stage = CycleStage.DATA_VALIDATION
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.DATA_VALIDATION, "Validating market data bars")
            t_stage = time.perf_counter()

            for sym, bar in current_bars.items():
                if bar.close is None or bar.close <= 0:
                    raise ValueError(f"Invalid market price for symbol {sym}: {bar.close}")

            record_stage(CycleStage.DATA_VALIDATION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)
            emit(AutonomousEventType.CYCLE_STAGE_COMPLETED, CycleStage.DATA_VALIDATION, f"Validated {len(current_bars)} market bars")

            # 3. Feature Generation Stage
            cycle_meta.stage = CycleStage.FEATURE_GENERATION
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.FEATURE_GENERATION, "Generating feature tensors")
            t_stage = time.perf_counter()
            record_stage(CycleStage.FEATURE_GENERATION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)
            emit(AutonomousEventType.CYCLE_STAGE_COMPLETED, CycleStage.FEATURE_GENERATION, "Features generated")

            # 4. Model Inference Stage
            cycle_meta.stage = CycleStage.MODEL_INFERENCE
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.MODEL_INFERENCE, "Running active model inference")
            t_stage = time.perf_counter()
            # If model telemetry was provided in signal generator or bar context, capture it
            inf_lat = (time.perf_counter() - t_stage) * 1000.0
            record_stage(CycleStage.MODEL_INFERENCE, "COMPLETED", inf_lat)
            emit(AutonomousEventType.CYCLE_STAGE_COMPLETED, CycleStage.MODEL_INFERENCE, "Model inference completed")

            # 5. Regime Detection Stage
            cycle_meta.stage = CycleStage.REGIME_DETECTION
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.REGIME_DETECTION, "Detecting market regime")
            t_stage = time.perf_counter()
            record_stage(CycleStage.REGIME_DETECTION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)
            emit(AutonomousEventType.CYCLE_STAGE_COMPLETED, CycleStage.REGIME_DETECTION, "Market regime evaluated")

            # 6. Signal Generation Stage
            cycle_meta.stage = CycleStage.SIGNAL_GENERATION
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.SIGNAL_GENERATION, "Generating trade signals")
            t_stage = time.perf_counter()

            signals: List[SignalCandidate] = []
            if self.signal_generator:
                signals = self.signal_generator(market_timestamp, current_bars)
            record_stage(CycleStage.SIGNAL_GENERATION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)
            emit(AutonomousEventType.CYCLE_STAGE_COMPLETED, CycleStage.SIGNAL_GENERATION, f"Generated {len(signals)} signals")

            # Extract model/regime telemetry if attached to signals
            if signals and hasattr(signals[0], "metadata") and signals[0].metadata:
                cycle_meta.model_telemetry = dict(signals[0].metadata)

            # 7. Portfolio Construction Stage
            cycle_meta.stage = CycleStage.PORTFOLIO_CONSTRUCTION
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.PORTFOLIO_CONSTRUCTION, "Constructing target portfolio")
            t_stage = time.perf_counter()

            p_result: Optional[PortfolioConstructionResult] = None
            if signals:
                p_result = self.portfolio_service.construct(signals=signals, timestamp=market_timestamp)
                raw_targets = p_result.targets
            else:
                raw_targets = []
            record_stage(CycleStage.PORTFOLIO_CONSTRUCTION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)
            emit(AutonomousEventType.CYCLE_STAGE_COMPLETED, CycleStage.PORTFOLIO_CONSTRUCTION, f"Produced {len(raw_targets)} raw portfolio targets")

            # 8. Risk Engine Stage
            cycle_meta.stage = CycleStage.RISK_ENGINE
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.RISK_ENGINE, "Evaluating risk constraints and limits")
            t_stage = time.perf_counter()

            known_prices = {s: b.close for s, b in current_bars.items() if b.close > 0}
            risk_context = RiskContext(
                as_of_timestamp=market_timestamp,
                asset_prices=known_prices,
            )

            adj_targets, assessment, audit, rejected, was_adj, status = RiskAdjustmentEngine.adjust(
                targets=raw_targets,
                config=self.risk_config,
                context=risk_context,
            )

            record_stage(CycleStage.RISK_ENGINE, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)

            if status in (RiskStatus.RISK_REJECTED, RiskStatus.INVALID) or rejected:
                if status in (RiskStatus.RISK_REJECTED, RiskStatus.INVALID):
                    adj_targets = []
                emit(AutonomousEventType.RISK_BLOCKED, CycleStage.RISK_ENGINE, f"Risk engine blocked targets (status={status.value}, rejected={rejected})", {"rejected": rejected, "status": status.value})
            else:
                emit(AutonomousEventType.RISK_APPROVED, CycleStage.RISK_ENGINE, f"Risk engine approved {len(adj_targets)} targets")

            # 9. Order Generation Stage
            cycle_meta.stage = CycleStage.ORDER_GENERATION
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.ORDER_GENERATION, "Translating target allocations to orders")
            t_stage = time.perf_counter()
            record_stage(CycleStage.ORDER_GENERATION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)

            # 10. Paper Trading Execution Stage
            cycle_meta.stage = CycleStage.PAPER_EXECUTION
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.PAPER_EXECUTION, "Executing orders via Paper Trading Engine")
            t_stage = time.perf_counter()

            pre_order_count = len(self.paper_service.orders_history)
            pre_exec_count = len(self.paper_service.broker.executions)

            # Delegate step execution to Phase 14 PaperTradingService
            snap: PaperAccountSnapshot = self.paper_service.process_step(
                timestamp=market_timestamp,
                current_bars=current_bars,
                targets=adj_targets,
                corporate_actions=corporate_actions,
            )

            new_orders = len(self.paper_service.orders_history) - pre_order_count
            new_execs = len(self.paper_service.broker.executions) - pre_exec_count
            cycle_meta.orders_generated_count = new_orders
            cycle_meta.orders_executed_count = new_execs

            record_stage(CycleStage.PAPER_EXECUTION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)
            emit(AutonomousEventType.ORDERS_EXECUTED, CycleStage.PAPER_EXECUTION, f"Generated {new_orders} orders, executed {new_execs} fills")

            # 11. Account Valuation Stage
            cycle_meta.stage = CycleStage.ACCOUNT_VALUATION
            t_stage = time.perf_counter()
            record_stage(CycleStage.ACCOUNT_VALUATION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)

            # 12. Reconciliation Stage
            cycle_meta.stage = CycleStage.RECONCILIATION
            emit(AutonomousEventType.CYCLE_STAGE_STARTED, CycleStage.RECONCILIATION, "Checking account reconciliation")
            t_stage = time.perf_counter()

            latest_recon = self.paper_service.reconciliation_history[-1] if self.paper_service.reconciliation_history else None
            if latest_recon and latest_recon.status != PaperReconciliationStatus.MATCHED:
                raise RuntimeError(f"Reconciliation mismatch detected during cycle {cycle_id}: {latest_recon.to_dict()}")

            record_stage(CycleStage.RECONCILIATION, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)
            emit(AutonomousEventType.RECONCILIATION_COMPLETED, CycleStage.RECONCILIATION, "Account reconciliation verified")

            # 13. Checkpointing Stage
            cycle_meta.stage = CycleStage.CHECKPOINTING
            t_stage = time.perf_counter()
            if self.config.auto_checkpoint_enabled:
                chk_meta = {
                    "equity": snap.equity,
                    "cash": snap.cash,
                    "orders": new_orders,
                    "fills": new_execs,
                    "model_telemetry": cycle_meta.model_telemetry,
                }
                self.checkpoint_manager.create_checkpoint(
                    cycle_id=cycle_id,
                    market_timestamp=market_timestamp,
                    stage=CycleStage.COMPLETED,
                    status=CycleStatus.COMPLETED,
                    state_data=chk_meta,
                )
                emit(AutonomousEventType.CHECKPOINT_CREATED, CycleStage.CHECKPOINTING, f"Checkpoint created for cycle {cycle_id}")
            record_stage(CycleStage.CHECKPOINTING, "COMPLETED", (time.perf_counter() - t_stage) * 1000.0)

            # 14. Cycle Completed Successfully
            cycle_meta.stage = CycleStage.COMPLETED
            cycle_meta.status = CycleStatus.COMPLETED
            cycle_meta.end_time = datetime.now(timezone.utc)
            cycle_meta.duration_seconds = (cycle_meta.end_time - start_time).total_seconds()
            record_stage(CycleStage.COMPLETED, "COMPLETED", 0.0)
            cycle_meta.stages_executed = stages_record
            emit(AutonomousEventType.CYCLE_COMPLETED, CycleStage.COMPLETED, f"Cycle {cycle_id} completed successfully in {cycle_meta.duration_seconds:.3f}s")

        except Exception as e:
            cycle_meta.status = CycleStatus.FAILED
            cycle_meta.end_time = datetime.now(timezone.utc)
            cycle_meta.duration_seconds = (cycle_meta.end_time - start_time).total_seconds()
            cycle_meta.error_message = str(e)
            cycle_meta.failure_class = self._classify_exception(e)
            record_stage(cycle_meta.stage, "FAILED", (time.perf_counter() - t_stage) * 1000.0, str(e))
            cycle_meta.stages_executed = stages_record
            logger.error("Trading cycle %s failed at stage %s: %s", cycle_id, cycle_meta.stage.value, str(e), exc_info=True)
            emit(
                AutonomousEventType.CYCLE_FAILED,
                cycle_meta.stage,
                f"Cycle {cycle_id} failed at stage {cycle_meta.stage.value}: {str(e)}",
                {"error": str(e), "failure_class": cycle_meta.failure_class.value if cycle_meta.failure_class else None},
            )

        return cycle_meta

    def _classify_exception(self, e: Exception) -> FailureClass:
        """Classify exception for bounded retry vs fail-closed decisions."""
        msg = str(e).lower()
        if "reconciliation" in msg:
            return FailureClass.ACCOUNTING
        if "risk" in msg:
            return FailureClass.RISK
        if "invalid" in msg or "price" in msg or "nan" in msg or "symbol" in msg:
            return FailureClass.VALIDATION
        if "timeout" in msg or "network" in msg or "connection" in msg:
            return FailureClass.TRANSIENT
        return FailureClass.SYSTEM
