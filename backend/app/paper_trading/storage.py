"""
Paper Trading Persistence Manager (Phase 14).
Saves and restores paper trading sessions, account snapshots, orders,
executions, reconciliation reports, and audit logs under models/paper_trading/{session_id}/.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.app.paper_trading.schemas import (
    PaperAccountSnapshot,
    PaperAuditEvent,
    PaperExecution,
    PaperOrder,
    PaperReconciliationReport,
    PaperTradingResult,
    PaperTradingSession,
)

logger = logging.getLogger(__name__)


class PaperTradingStorage:
    """
    Manages filesystem persistence and recovery of paper trading sessions and artifacts.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.base_dir = root_dir / "models" / "paper_trading"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def save_session(self, session: PaperTradingSession) -> Path:
        """Persist session metadata."""
        session_dir = self._get_session_dir(session.session_id)
        file_path = session_dir / "session.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2)
        return file_path

    def load_session(self, session_id: str) -> Optional[PaperTradingSession]:
        """Load session metadata from disk."""
        session_dir = self._get_session_dir(session_id)
        file_path = session_dir / "session.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PaperTradingSession.from_dict(data)

    def save_result(self, result: PaperTradingResult) -> Path:
        """
        Save complete session result (JSON summary, CSV/Parquet snapshots, orders, executions).
        """
        session_dir = self._get_session_dir(result.session.session_id)

        # 1. Main result JSON
        result_file = session_dir / "paper_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        # 2. Account snapshots Parquet / CSV
        if result.account_snapshots:
            snap_df = pd.DataFrame([s.to_dict() for s in result.account_snapshots])
            # Drop nested positions dict for tabular storage
            if "positions" in snap_df.columns:
                snap_df = snap_df.drop(columns=["positions"])
            snap_df.to_parquet(session_dir / "snapshots.parquet", index=False)
            snap_df.to_csv(session_dir / "snapshots.csv", index=False)

        # 3. Orders Parquet / CSV
        if result.orders:
            ord_df = pd.DataFrame([o.to_dict() for o in result.orders])
            if "metadata" in ord_df.columns:
                ord_df = ord_df.drop(columns=["metadata"])
            ord_df.to_parquet(session_dir / "orders.parquet", index=False)
            ord_df.to_csv(session_dir / "orders.csv", index=False)

        # 4. Executions Parquet / CSV
        if result.executions:
            exec_df = pd.DataFrame([e.to_dict() for e in result.executions])
            exec_df.to_parquet(session_dir / "executions.parquet", index=False)
            exec_df.to_csv(session_dir / "executions.csv", index=False)

        # 5. Audit Events JSON
        if result.audit_events:
            audit_file = session_dir / "audit_events.json"
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump([a.to_dict() for a in result.audit_events], f, indent=2)

        logger.info("Saved paper trading artifacts for session %s to %s", result.session.session_id, session_dir)
        return session_dir

    def load_result(self, session_id: str) -> Optional[PaperTradingResult]:
        """Load complete PaperTradingResult from disk."""
        session_dir = self._get_session_dir(session_id)
        result_file = session_dir / "paper_result.json"
        if not result_file.exists():
            return None

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = PaperTradingSession.from_dict(data["session"])
        snapshots = [PaperAccountSnapshot.from_dict(s) for s in data.get("account_snapshots", [])]
        orders = [PaperOrder.from_dict(o) for o in data.get("orders", [])]
        executions = [PaperExecution.from_dict(e) for e in data.get("executions", [])]
        reconciliations = [PaperReconciliationReport.from_dict(r) for r in data.get("reconciliation_reports", [])]
        audit_events = [PaperAuditEvent.from_dict(a) for a in data.get("audit_events", [])]

        return PaperTradingResult(
            session=session,
            account_snapshots=snapshots,
            orders=orders,
            executions=executions,
            reconciliation_reports=reconciliations,
            audit_events=audit_events,
            provenance_hash=data.get("provenance_hash", ""),
        )
