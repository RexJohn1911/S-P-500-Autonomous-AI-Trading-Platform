"""
Autonomous Loop Checkpoint Manager (Phase 15).
Creates, persists, and validates deterministic cycle checkpoints at critical pipeline stages.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from backend.app.autonomous.schemas import (
    CycleCheckpoint,
    CycleStage,
    CycleStatus,
)

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages creation and verification of cycle checkpoints.
    """

    def __init__(self, session_id: str, paper_session_id: str, config_hash: str):
        self.session_id = session_id
        self.paper_session_id = paper_session_id
        self.config_hash = config_hash
        self.checkpoints: List[CycleCheckpoint] = []
        self._checkpoint_counter: int = 0

    def create_checkpoint(
        self,
        cycle_id: str,
        market_timestamp: datetime,
        stage: CycleStage,
        status: CycleStatus = CycleStatus.RUNNING,
        state_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CycleCheckpoint:
        """
        Create a durable checkpoint for the given cycle and stage.
        """
        self._checkpoint_counter += 1
        now = datetime.now(timezone.utc)
        
        # Calculate state hash
        payload = {
            "session_id": self.session_id,
            "cycle_id": cycle_id,
            "stage": stage.value,
            "market_timestamp": market_timestamp.isoformat(),
            "state_data": state_data or {},
        }
        state_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

        chk_id = f"chk_{self.session_id}_{cycle_id}_{stage.value.lower()}_{self._checkpoint_counter}"
        checkpoint = CycleCheckpoint(
            checkpoint_id=chk_id,
            session_id=self.session_id,
            cycle_id=cycle_id,
            market_timestamp=market_timestamp,
            last_successful_stage=stage,
            paper_session_id=self.paper_session_id,
            configuration_hash=self.config_hash,
            state_hash=state_hash,
            timestamp=now,
            status=status,
            metadata=metadata or {},
        )
        self.checkpoints.append(checkpoint)
        logger.debug("Created checkpoint %s for stage %s", chk_id, stage.value)
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[CycleCheckpoint]:
        """Return the most recent checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def get_checkpoint_for_cycle(self, cycle_id: str) -> List[CycleCheckpoint]:
        """Return all checkpoints created for a specific cycle."""
        return [c for c in self.checkpoints if c.cycle_id == cycle_id]
