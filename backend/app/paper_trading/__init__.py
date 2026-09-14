"""
Paper Trading Subsystem (Phase 14).
Provides research-grade simulated paper trading execution, double-entry account ledger,
order lifecycle state machine, corporate action processing, transaction cost modeling,
reconciliation engine, and audit trail persistence.
"""

from backend.app.paper_trading.account import PaperAccount
from backend.app.paper_trading.broker import BasePaperBroker, SimulatedPaperBroker
from backend.app.paper_trading.costs import PaperCostModel
from backend.app.paper_trading.execution import PaperExecutionEngine
from backend.app.paper_trading.orders import PaperOrderManager
from backend.app.paper_trading.reconciliation import PaperReconciliationEngine
from backend.app.paper_trading.risk_guard import PaperExecutionRiskGuard
from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperAuditEvent,
    PaperAuditEventType,
    PaperExecution,
    PaperFifoLot,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
    PaperPosition,
    PaperPositionSide,
    PaperReconciliationReport,
    PaperReconciliationStatus,
    PaperSessionStatus,
    PaperTimeInForce,
    PaperTradingConfig,
    PaperTradingResult,
    PaperTradingSession,
)
from backend.app.paper_trading.service import PaperTradingService
from backend.app.paper_trading.storage import PaperTradingStorage

__all__ = [
    "BasePaperBroker",
    "PaperAccount",
    "PaperAccountSnapshot",
    "PaperAuditEvent",
    "PaperAuditEventType",
    "PaperCostModel",
    "PaperExecution",
    "PaperExecutionEngine",
    "PaperExecutionRiskGuard",
    "PaperFifoLot",
    "PaperOrder",
    "PaperOrderManager",
    "PaperOrderSide",
    "PaperOrderStatus",
    "PaperOrderType",
    "PaperPosition",
    "PaperPositionSide",
    "PaperReconciliationEngine",
    "PaperReconciliationReport",
    "PaperReconciliationStatus",
    "PaperSessionStatus",
    "PaperTimeInForce",
    "PaperTradingConfig",
    "PaperTradingResult",
    "PaperTradingService",
    "PaperTradingSession",
    "PaperTradingStorage",
    "SimulatedPaperBroker",
]
