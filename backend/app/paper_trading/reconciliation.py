"""
Paper Trading Reconciliation Engine (Phase 14).
Compares expected internal state vs broker ledger state, verifying:
- Cash balance consistency
- Position quantities and side
- Total equity accounting identity
- Open vs filled orders
Produces structured PaperReconciliationReport.
"""

from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, List, Optional
import uuid

from backend.app.paper_trading.broker import SimulatedPaperBroker
from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperPosition,
    PaperReconciliationReport,
    PaperReconciliationStatus,
)

logger = logging.getLogger(__name__)


class PaperReconciliationEngine:
    """
    Validates ledger consistency and detects mismatches between account state and broker records.
    """

    @classmethod
    def reconcile(
        cls,
        session_id: str,
        broker: SimulatedPaperBroker,
        expected_cash: Optional[float] = None,
        expected_positions: Optional[Dict[str, float]] = None,
        current_prices: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
        tolerance: float = 1e-4,
    ) -> PaperReconciliationReport:
        """
        Perform complete reconciliation check.
        """
        ts = timestamp or datetime.now(timezone.utc)
        prices = current_prices or {}
        snapshot = broker.get_account_snapshot(timestamp=ts, current_prices=prices)

        actual_cash = snapshot.cash
        actual_equity = snapshot.equity
        actual_positions = broker.get_positions()

        positions_mismatches: List[Dict[str, Any]] = []
        orders_mismatches: List[Dict[str, Any]] = []
        is_matched = True

        # 1. Cash Reconciliation
        cash_exp = expected_cash if expected_cash is not None else actual_cash
        if abs(cash_exp - actual_cash) > tolerance:
            is_matched = False
            logger.error(
                "Cash reconciliation mismatch for session %s: expected=%.4f, actual=%.4f, diff=%.4f",
                session_id, cash_exp, actual_cash, abs(cash_exp - actual_cash)
            )

        # 2. Position Quantities Reconciliation
        if expected_positions is not None:
            all_symbols = set(expected_positions.keys()).union(set(actual_positions.keys()))
            for sym in sorted(all_symbols):
                exp_qty = expected_positions.get(sym, 0.0)
                act_pos = actual_positions.get(sym)
                act_qty = act_pos.quantity if act_pos else 0.0

                if abs(exp_qty - act_qty) > tolerance:
                    is_matched = False
                    mismatch = {
                        "symbol": sym,
                        "expected_quantity": exp_qty,
                        "actual_quantity": act_qty,
                        "delta": exp_qty - act_qty,
                    }
                    positions_mismatches.append(mismatch)
                    logger.error("Position mismatch for %s: expected=%.4f, actual=%.4f", sym, exp_qty, act_qty)

        # 3. Accounting Identity Check: equity == cash + sum(market_value)
        total_market_val = sum(p.market_value for p in actual_positions.values())
        calculated_equity = actual_cash + total_market_val
        if abs(calculated_equity - actual_equity) > 0.05:  # $0.05 rounding tolerance
            is_matched = False
            logger.error(
                "Accounting identity failure for session %s: cash(%.4f) + mv(%.4f) = %.4f != equity(%.4f)",
                session_id, actual_cash, total_market_val, calculated_equity, actual_equity
            )

        # 4. Open Orders Check
        open_orders = broker.get_open_orders()
        for o in open_orders:
            if o.is_terminal():
                orders_mismatches.append({
                    "order_id": o.order_id,
                    "issue": f"Order in terminal status {o.status.value} marked open",
                })
                is_matched = False

        status = PaperReconciliationStatus.MATCHED if is_matched else PaperReconciliationStatus.MISMATCH
        recon_id = f"recon_{session_id}_{int(ts.timestamp())}"

        return PaperReconciliationReport(
            reconciliation_id=recon_id,
            session_id=session_id,
            timestamp=ts,
            status=status,
            cash_expected=round(float(cash_exp), 4),
            cash_actual=round(float(actual_cash), 4),
            equity_expected=round(float(calculated_equity), 4),
            equity_actual=round(float(actual_equity), 4),
            positions_mismatches=positions_mismatches,
            orders_mismatches=orders_mismatches,
            details={
                "positions_count": len(actual_positions),
                "open_orders_count": len(open_orders),
                "total_executions": len(broker.executions),
            },
        )
