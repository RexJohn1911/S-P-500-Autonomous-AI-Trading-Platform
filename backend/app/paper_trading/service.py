"""
Paper Trading Service Orchestrator (Phase 14).
Coordinates the complete simulated trading lifecycle:
- Session management (create, start, pause, resume, stop, complete)
- Upstream Strategy / Signal / Portfolio / Risk engine integration
- Order generation & risk-guarded dispatch
- Simulated broker execution & double-entry accounting
- Point-in-time valuation, mark-to-market ledger, and exposure tracking
- Automated ledger reconciliation & immutable audit logging
- Artifact persistence under models/paper_trading/{session_id}/
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

from backend.app.backtest.schemas import DividendEvent, SplitEvent
from backend.app.data.models import BarData
from backend.app.paper_trading.broker import SimulatedPaperBroker
from backend.app.paper_trading.orders import PaperOrderManager
from backend.app.paper_trading.reconciliation import PaperReconciliationEngine
from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperAuditEvent,
    PaperAuditEventType,
    PaperExecution,
    PaperOrder,
    PaperOrderStatus,
    PaperReconciliationReport,
    PaperReconciliationStatus,
    PaperSessionStatus,
    PaperTradingConfig,
    PaperTradingResult,
    PaperTradingSession,
)
from backend.app.paper_trading.storage import PaperTradingStorage
from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.portfolio.service import PortfolioConstructionService
from backend.app.risk.engine import RiskAdjustmentEngine
from backend.app.risk.schemas import RiskAdjustedTarget, RiskContext, RiskEngineConfig
from backend.app.strategy.schemas import SignalCandidate

logger = logging.getLogger(__name__)


class PaperTradingService:
    """
    Primary orchestrator for paper trading simulations and autonomous simulated sessions.
    """

    def __init__(
        self,
        config: Optional[PaperTradingConfig] = None,
        storage: Optional[PaperTradingStorage] = None,
        session_id: Optional[str] = None,
    ):
        self.config = config or PaperTradingConfig()
        self.storage = storage or PaperTradingStorage()
        self.session_id = session_id or f"paper_ses_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        self.session = PaperTradingSession(
            session_id=self.session_id,
            config=self.config,
            status=PaperSessionStatus.CREATED,
        )
        self.broker = SimulatedPaperBroker(
            session_id=self.session_id,
            config=self.config,
        )
        self.account_snapshots: List[PaperAccountSnapshot] = []
        self.orders_history: List[PaperOrder] = []
        self.reconciliation_history: List[PaperReconciliationReport] = []
        self.audit_events: List[PaperAuditEvent] = []
        self._audit_counter: int = 0

        self._record_audit_event(
            event_type=PaperAuditEventType.SESSION_CREATED,
            description=f"Paper trading session created with initial capital ${self.config.initial_capital:.2f}",
            details={"initial_capital": self.config.initial_capital, "config": self.config.to_dict()},
        )

    def _record_audit_event(
        self,
        event_type: PaperAuditEventType,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> PaperAuditEvent:
        """Record an immutable audit event."""
        self._audit_counter += 1
        ts = timestamp or datetime.now(timezone.utc)
        event = PaperAuditEvent(
            event_id=f"evt_{self.session_id}_{self._audit_counter}",
            session_id=self.session_id,
            timestamp=ts,
            event_type=event_type,
            description=description,
            details=details or {},
        )
        self.audit_events.append(event)
        return event

    def start_session(self, symbols: Optional[List[str]] = None) -> PaperTradingSession:
        """Start paper trading session."""
        if self.session.status in {PaperSessionStatus.RUNNING, PaperSessionStatus.COMPLETED, PaperSessionStatus.FAILED}:
            raise ValueError(f"Cannot start session in status: {self.session.status.value}")

        self.session.status = PaperSessionStatus.RUNNING
        self.session.started_at = datetime.now(timezone.utc)
        if symbols:
            self.session.symbols = list(symbols)

        self._record_audit_event(
            event_type=PaperAuditEventType.SESSION_STARTED,
            description="Paper trading session started",
            details={"symbols": self.session.symbols},
        )
        return self.session

    def pause_session(self) -> PaperTradingSession:
        """Pause session while preserving account state."""
        if self.session.status != PaperSessionStatus.RUNNING:
            raise ValueError(f"Cannot pause session in status: {self.session.status.value}")

        self.session.status = PaperSessionStatus.PAUSED
        self._record_audit_event(
            event_type=PaperAuditEventType.SESSION_PAUSED,
            description="Paper trading session paused",
        )
        return self.session

    def resume_session(self) -> PaperTradingSession:
        """Resume paused session."""
        if self.session.status != PaperSessionStatus.PAUSED:
            raise ValueError(f"Cannot resume session in status: {self.session.status.value}")

        self.session.status = PaperSessionStatus.RUNNING
        self._record_audit_event(
            event_type=PaperAuditEventType.SESSION_RESUMED,
            description="Paper trading session resumed",
        )
        return self.session

    def stop_session(self, status: PaperSessionStatus = PaperSessionStatus.STOPPED) -> PaperTradingSession:
        """Stop session."""
        self.session.status = status
        self.session.stopped_at = datetime.now(timezone.utc)
        
        # Final valuation
        current_prices = {s: p.market_price for s, p in self.broker.account.positions.items() if p.market_price > 0}
        final_snap = self.broker.get_account_snapshot(timestamp=self.session.stopped_at, current_prices=current_prices)
        self.session.final_equity = final_snap.equity
        self.session.total_pnl = final_snap.total_pnl
        self.session.total_trades = self.broker.account.total_trades_count
        self.session.total_costs = self.broker.account.total_transaction_cost

        event_type = PaperAuditEventType.SESSION_STOPPED if status != PaperSessionStatus.FAILED else PaperAuditEventType.SESSION_FAILED
        self._record_audit_event(
            event_type=event_type,
            description=f"Paper trading session ended with status {status.value}",
            details={"final_equity": self.session.final_equity, "total_pnl": self.session.total_pnl},
        )
        return self.session

    def process_step(
        self,
        timestamp: datetime,
        current_bars: Dict[str, BarData],
        targets: Optional[Union[List[RiskAdjustedTarget], List[PortfolioTarget], Dict[str, float]]] = None,
        corporate_actions: Optional[List[Union[DividendEvent, SplitEvent]]] = None,
        reference_liquidity: Optional[Dict[str, float]] = None,
    ) -> PaperAccountSnapshot:
        """
        Execute a single chronological point-in-time paper trading step:
        1. Process point-in-time corporate actions (dividends, splits).
        2. Apply short borrow financing fees if configured.
        3. Extract current prices ($P_t$).
        4. Generate and dispatch rebalancing orders from target portfolio.
        5. Simulate execution fills against $P_t$.
        6. Mark-to-market valuation & accounting snapshot.
        7. Run reconciliation check.
        """
        if self.session.status not in (PaperSessionStatus.RUNNING, PaperSessionStatus.CREATED, PaperSessionStatus.INITIALIZED):
            raise ValueError(f"Cannot process step in session status: {self.session.status.value}")

        if self.session.status != PaperSessionStatus.RUNNING:
            self.start_session(symbols=list(current_bars.keys()))

        self._record_audit_event(
            event_type=PaperAuditEventType.MARKET_DATA_RECEIVED,
            description=f"Received market data for {len(current_bars)} symbols at {timestamp.isoformat()}",
            timestamp=timestamp,
        )

        # 1. Corporate Actions
        if corporate_actions:
            for ca in corporate_actions:
                if isinstance(ca, DividendEvent):
                    delta = self.broker.account.apply_dividend(ca)
                    self._record_audit_event(
                        event_type=PaperAuditEventType.DIVIDEND_APPLIED,
                        description=f"Dividend {ca.amount_per_share}/sh applied to {ca.symbol}: cash delta ${delta:.2f}",
                        details=ca.to_dict(),
                        timestamp=timestamp,
                    )
                elif isinstance(ca, SplitEvent):
                    self.broker.account.apply_split(ca)
                    self._record_audit_event(
                        event_type=PaperAuditEventType.SPLIT_APPLIED,
                        description=f"Stock split ratio {ca.ratio} applied to {ca.symbol}",
                        details=ca.to_dict(),
                        timestamp=timestamp,
                    )

        # 2. Short Borrow Financing
        if self.config.daily_borrow_rate > 0:
            borrow_fee = self.broker.account.apply_borrow_cost(timestamp, self.config.daily_borrow_rate)
            if borrow_fee > 0:
                self._record_audit_event(
                    event_type=PaperAuditEventType.COST_APPLIED,
                    description=f"Short borrow financing fee ${borrow_fee:.2f} deducted",
                    details={"borrow_fee": borrow_fee},
                    timestamp=timestamp,
                )

        # 3. Extract prices
        current_prices: Dict[str, float] = {}
        for sym, bar in current_bars.items():
            if bar.close > 0:
                current_prices[sym] = float(bar.close)

        # Current valuation prior to order generation
        pre_snap = self.broker.get_account_snapshot(timestamp=timestamp, current_prices=current_prices)

        # 4. Generate orders from targets if provided
        if targets:
            self._record_audit_event(
                event_type=PaperAuditEventType.TARGET_GENERATED,
                description="Evaluating target portfolio for rebalancing order generation",
                timestamp=timestamp,
            )

            orders = self.broker.order_manager.generate_orders_from_targets(
                targets=targets,
                current_positions=self.broker.account.positions,
                current_prices=current_prices,
                portfolio_equity=pre_snap.equity,
                timestamp=timestamp,
            )

            for order in orders:
                self.orders_history.append(order)
                if order.status == PaperOrderStatus.REJECTED:
                    self._record_audit_event(
                        event_type=PaperAuditEventType.ORDER_REJECTED,
                        description=f"Order {order.order_id} ({order.side.value} {order.quantity} {order.symbol}) rejected: {order.rejection_reason}",
                        details=order.to_dict(),
                        timestamp=timestamp,
                    )
                    continue

                self._record_audit_event(
                    event_type=PaperAuditEventType.ORDER_SUBMITTED,
                    description=f"Order {order.order_id} ({order.side.value} {order.quantity} {order.symbol}) submitted",
                    details=order.to_dict(),
                    timestamp=timestamp,
                )

                # 5. Execute accepted market order
                liq = reference_liquidity.get(order.symbol) if reference_liquidity else None
                execution = self.broker.execute_market_order(
                    order=order,
                    market_price=current_prices[order.symbol],
                    timestamp=timestamp,
                    reference_liquidity=liq,
                )

                self._record_audit_event(
                    event_type=PaperAuditEventType.ORDER_FILLED,
                    description=f"Order {order.order_id} filled: {execution.quantity} @ ${execution.executed_price:.4f} (realized P&L: ${execution.realized_pnl:.2f})",
                    details=execution.to_dict(),
                    timestamp=timestamp,
                )

        # 6. Post-execution valuation snapshot
        snapshot = self.broker.get_account_snapshot(timestamp=timestamp, current_prices=current_prices)
        self.account_snapshots.append(snapshot)

        self._record_audit_event(
            event_type=PaperAuditEventType.ACCOUNT_VALUED,
            description=f"Account valued: equity=${snapshot.equity:.2f}, cash=${snapshot.cash:.2f}, positions={snapshot.positions_count}",
            details={"equity": snapshot.equity, "cash": snapshot.cash, "unrealized_pnl": snapshot.unrealized_pnl, "realized_pnl": snapshot.realized_pnl},
            timestamp=timestamp,
        )

        # 7. Reconciliation
        recon = PaperReconciliationEngine.reconcile(
            session_id=self.session_id,
            broker=self.broker,
            current_prices=current_prices,
            timestamp=timestamp,
        )
        self.reconciliation_history.append(recon)

        if recon.status == PaperReconciliationStatus.MATCHED:
            self._record_audit_event(
                event_type=PaperAuditEventType.RECONCILIATION_PASSED,
                description=f"Reconciliation passed at {timestamp.isoformat()}",
                timestamp=timestamp,
            )
        else:
            self._record_audit_event(
                event_type=PaperAuditEventType.RECONCILIATION_FAILED,
                description=f"Reconciliation failed at {timestamp.isoformat()}",
                details=recon.to_dict(),
                timestamp=timestamp,
            )

        return snapshot

    def run_integrated_pipeline_session(
        self,
        market_data: Dict[str, List[BarData]],
        signal_generator: Callable[[datetime, Dict[str, BarData]], List[SignalCandidate]],
        portfolio_service: Optional[PortfolioConstructionService] = None,
        risk_config: Optional[RiskEngineConfig] = None,
        corporate_actions: Optional[Dict[datetime, List[Union[DividendEvent, SplitEvent]]]] = None,
        save_artifacts: bool = False,
    ) -> PaperTradingResult:
        """
        Execute continuous paper trading event loop over a sequence of chronological market bars:
        At bar t:
          1. Observe information available <= t
          2. Generate signals at t
          3. Construct portfolio at t
          4. Evaluate & adjust with Risk Engine at t
          5. Process step at t (corporate actions, execution against t, accounting, reconciliation)
        """
        p_service = portfolio_service or PortfolioConstructionService()
        r_config = risk_config or RiskEngineConfig()

        # Build chronological timeline
        timestamp_set = set()
        for sym, bars in market_data.items():
            for b in bars:
                timestamp_set.add(b.timestamp)

        sorted_timestamps = sorted(timestamp_set)
        if not sorted_timestamps:
            raise ValueError("No valid bars provided for paper trading session")

        bar_lookup: Dict[datetime, Dict[str, BarData]] = {}
        for sym, bars in market_data.items():
            for b in bars:
                if b.timestamp not in bar_lookup:
                    bar_lookup[b.timestamp] = {}
                bar_lookup[b.timestamp][sym] = b

        self.start_session(symbols=sorted(list(market_data.keys())))

        for ts in sorted_timestamps:
            current_bars = bar_lookup.get(ts, {})
            if not current_bars:
                continue

            # 1. Signals generated at t
            signals = signal_generator(ts, current_bars)

            # 2. Portfolio construction at t
            p_result = p_service.construct(signals=signals, timestamp=ts)

            # 3. Risk Engine adjustment at t
            known_prices = {s: b.close for s, b in current_bars.items() if b.close > 0}
            risk_context = RiskContext(
                as_of_timestamp=ts,
                asset_prices=known_prices,
            )

            adj_targets, assessment, audit, rejected, was_adj, status = RiskAdjustmentEngine.adjust(
                targets=p_result.targets,
                config=r_config,
                context=risk_context,
            )

            # 4. Process step
            ca_list = corporate_actions.get(ts) if corporate_actions else None
            self.process_step(
                timestamp=ts,
                current_bars=current_bars,
                targets=adj_targets,
                corporate_actions=ca_list,
            )

        self.stop_session(status=PaperSessionStatus.COMPLETED)
        result = self.get_result()

        if save_artifacts:
            self.storage.save_result(result)

        return result

    def get_result(self) -> PaperTradingResult:
        """Construct immutable PaperTradingResult with provenance hash."""
        # Calculate deterministic provenance hash
        summary_payload = {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "initial_capital": self.config.initial_capital,
            "final_equity": self.session.final_equity,
            "total_trades": self.session.total_trades,
            "total_costs": self.session.total_costs,
            "executions_count": len(self.broker.executions),
        }
        prov_hash = hashlib.sha256(json.dumps(summary_payload, sort_keys=True).encode()).hexdigest()

        return PaperTradingResult(
            session=self.session,
            account_snapshots=list(self.account_snapshots),
            orders=list(self.orders_history),
            executions=list(self.broker.executions),
            reconciliation_reports=list(self.reconciliation_history),
            audit_events=list(self.audit_events),
            provenance_hash=prov_hash,
        )
