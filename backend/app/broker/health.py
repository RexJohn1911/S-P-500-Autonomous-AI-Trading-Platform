"""
Broker Health & Status Verification (Phase 16).
Defines health reporting structures for broker connections.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.app.broker.schemas import BrokerStatus


@dataclass
class BrokerHealthSnapshot:
    """Normalized broker health and liveness record."""
    status: BrokerStatus
    provider: str
    execution_mode: str
    connected: bool
    latency_ms: Optional[float] = None
    message: str = "OK"
    timestamp: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        elif self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "provider": self.provider,
            "execution_mode": self.execution_mode,
            "connected": self.connected,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "details": self.details,
        }
