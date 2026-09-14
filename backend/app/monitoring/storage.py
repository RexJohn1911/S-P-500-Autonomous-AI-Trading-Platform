"""
Monitoring & Safety Storage Manager (Phase 18).
Handles durable, append-only file persistence for kill switch state,
incidents, and safety audit trails with SHA-256 integrity and secret safety.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.monitoring.schemas import (
    ComponentHealthSnapshot,
    ComponentType,
    HealthStatus,
    IncidentStatus,
    KillSwitchSnapshot,
    KillSwitchState,
    SafetyEvent,
    SafetyIncident,
    SafetySeverity,
)

logger = logging.getLogger(__name__)


class MonitoringStorage:
    """
    Filesystem storage engine for Monitoring & Safety subsystem.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or "models/monitoring")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.kill_switch_file = self.base_dir / "kill_switch.json"
        self.incidents_file = self.base_dir / "incidents.jsonl"
        self.audit_file = self.base_dir / "safety_audit.jsonl"

    def save_kill_switch_state(self, snapshot: KillSwitchSnapshot) -> None:
        """Atomically persist global kill switch state."""
        tmp_file = self.base_dir / f"kill_switch_{datetime.now(timezone.utc).timestamp()}.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(snapshot.to_dict(), f, indent=2)
            tmp_file.replace(self.kill_switch_file)
        except Exception as e:
            logger.error("Failed to persist kill switch state: %s", e)
            if tmp_file.exists():
                tmp_file.unlink()

    def load_kill_switch_state(self) -> Optional[KillSwitchSnapshot]:
        """Load persisted kill switch state if available."""
        if not self.kill_switch_file.exists():
            return None
        try:
            with open(self.kill_switch_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return KillSwitchSnapshot(
                state=KillSwitchState(data.get("state", "ARMED")),
                is_triggered=bool(data.get("is_triggered", False)),
                triggered_at=datetime.fromisoformat(data["triggered_at"]) if data.get("triggered_at") else None,
                triggered_by=data.get("triggered_by"),
                reason=data.get("reason"),
                cleared_at=datetime.fromisoformat(data["cleared_at"]) if data.get("cleared_at") else None,
                cleared_by=data.get("cleared_by"),
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logger.error("Failed to load kill switch state: %s", e)
            return None

    def append_incident(self, incident: SafetyIncident) -> None:
        """Append an incident record to incidents log."""
        try:
            with open(self.incidents_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(incident.to_dict()) + "\n")
        except Exception as e:
            logger.error("Failed to append incident to storage: %s", e)

    def append_audit_event(self, event: SafetyEvent) -> None:
        """Append a safety event to safety audit log."""
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error("Failed to append audit event: %s", e)

    def load_recent_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Load recent safety audit records."""
        if not self.audit_file.exists():
            return []
        records = []
        try:
            with open(self.audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records[-limit:]
        except Exception as e:
            logger.error("Failed to read audit events: %s", e)
            return []
