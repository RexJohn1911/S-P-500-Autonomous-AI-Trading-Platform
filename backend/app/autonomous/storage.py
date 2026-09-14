"""
Autonomous Trading Persistence Manager (Phase 15).
Saves and restores autonomous loop sessions, cycle executions, checkpoints,
health snapshots, and audit event logs under models/autonomous/{session_id}/.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.app.autonomous.schemas import (
    AutonomousEvent,
    AutonomousHealthSnapshot,
    AutonomousLoopResult,
    AutonomousSession,
    CycleCheckpoint,
    CycleMetadata,
)

logger = logging.getLogger(__name__)


class AutonomousStorage:
    """
    Manages filesystem persistence and recovery of autonomous loop sessions.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.base_dir = root_dir / "models" / "autonomous"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def save_session(self, session: AutonomousSession) -> Path:
        """Persist session metadata."""
        session_dir = self._get_session_dir(session.session_id)
        file_path = session_dir / "session.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2)
        return file_path

    def load_session(self, session_id: str) -> Optional[AutonomousSession]:
        """Load session metadata from disk."""
        session_dir = self._get_session_dir(session_id)
        file_path = session_dir / "session.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AutonomousSession.from_dict(data)

    def save_result(self, result: AutonomousLoopResult) -> Path:
        """
        Save complete autonomous loop execution result.
        """
        session_dir = self._get_session_dir(result.session.session_id)

        # 1. Main result summary JSON
        result_file = session_dir / "results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        # 2. Cycles JSONL / Parquet
        if result.cycles:
            cycle_file = session_dir / "cycles.jsonl"
            with open(cycle_file, "w", encoding="utf-8") as f:
                for c in result.cycles:
                    f.write(json.dumps(c.to_dict()) + "\n")
            cycle_dicts = []
            for c in result.cycles:
                cd = c.to_dict()
                if "versions" in cd and isinstance(cd["versions"], (dict, list)):
                    cd["versions"] = json.dumps(cd["versions"])
                if "symbols" in cd and isinstance(cd["symbols"], (dict, list)):
                    cd["symbols"] = json.dumps(cd["symbols"])
                if "stages_executed" in cd and isinstance(cd["stages_executed"], (dict, list)):
                    cd["stages_executed"] = json.dumps(cd["stages_executed"])
                if "model_telemetry" in cd and isinstance(cd["model_telemetry"], (dict, list)):
                    cd["model_telemetry"] = json.dumps(cd["model_telemetry"])
                cycle_dicts.append(cd)
            df = pd.DataFrame(cycle_dicts)
            df.to_parquet(session_dir / "cycles.parquet", index=False)

        # 3. Checkpoints JSON
        if result.checkpoints:
            chk_file = session_dir / "checkpoints.json"
            with open(chk_file, "w", encoding="utf-8") as f:
                json.dump([ck.to_dict() for ck in result.checkpoints], f, indent=2)

        # 4. Health Snapshots JSON
        if result.health_snapshots:
            health_file = session_dir / "health.json"
            with open(health_file, "w", encoding="utf-8") as f:
                json.dump([h.to_dict() for h in result.health_snapshots], f, indent=2)

        # 5. Audit Events JSONL
        if result.audit_events:
            events_file = session_dir / "events.jsonl"
            with open(events_file, "w", encoding="utf-8") as f:
                for e in result.audit_events:
                    f.write(json.dumps(e.to_dict()) + "\n")

        logger.info("Saved autonomous loop artifacts for session %s to %s", result.session.session_id, session_dir)
        return session_dir

    def load_result(self, session_id: str) -> Optional[AutonomousLoopResult]:
        """Load complete AutonomousLoopResult from disk."""
        session_dir = self._get_session_dir(session_id)
        result_file = session_dir / "results.json"
        if not result_file.exists():
            return None

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = AutonomousSession.from_dict(data["session"])
        cycles = [CycleMetadata.from_dict(c) for c in data.get("cycles", [])]
        checkpoints = [CycleCheckpoint.from_dict(ck) for ck in data.get("checkpoints", [])]
        health_snaps = [AutonomousHealthSnapshot.from_dict(h) for h in data.get("health_snapshots", [])]
        events = [AutonomousEvent.from_dict(e) for e in data.get("audit_events", [])]

        return AutonomousLoopResult(
            session=session,
            cycles=cycles,
            checkpoints=checkpoints,
            health_snapshots=health_snaps,
            audit_events=events,
            provenance_hash=data.get("provenance_hash", ""),
        )
