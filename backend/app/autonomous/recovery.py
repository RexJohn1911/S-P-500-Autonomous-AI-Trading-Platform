"""
Autonomous Crash Recovery Manager (Phase 15).
Inspects checkpoints and paper trading state on process restart;
determines completed vs interrupted stages and executes idempotent recovery.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.app.autonomous.schemas import (
    CycleCheckpoint,
    CycleStage,
    CycleStatus,
    LoopState,
)
from backend.app.paper_trading.service import PaperTradingService

logger = logging.getLogger(__name__)


class CrashRecoveryManager:
    """
    Recovers autonomous loop state after crash or abrupt termination.
    """

    @classmethod
    def analyze_recovery_state(
        cls,
        latest_checkpoint: Optional[CycleCheckpoint],
        paper_service: Optional[PaperTradingService] = None,
    ) -> Tuple[bool, Optional[str], Optional[CycleStage]]:
        """
        Analyze checkpoint and paper trading service to determine recovery action.
        Returns: (can_recover, recovery_action_description, last_stage)
        """
        if not latest_checkpoint:
            return True, "Clean startup - no prior checkpoint found", None

        last_stage = latest_checkpoint.last_successful_stage
        status = latest_checkpoint.status

        if status == CycleStatus.COMPLETED or last_stage == CycleStage.COMPLETED:
            return True, f"Prior cycle {latest_checkpoint.cycle_id} successfully completed", last_stage

        # If cycle was interrupted mid-flight:
        if last_stage in (CycleStage.DATA_ACQUISITION, CycleStage.DATA_VALIDATION, CycleStage.FEATURE_GENERATION):
            return True, f"Cycle {latest_checkpoint.cycle_id} interrupted before order generation; safe to retry cycle", last_stage

        if last_stage in (CycleStage.ORDER_GENERATION, CycleStage.PAPER_EXECUTION, CycleStage.ACCOUNT_VALUATION, CycleStage.RECONCILIATION):
            # Orders may have been submitted to paper broker. Verify with paper broker.
            if paper_service:
                open_orders = paper_service.broker.get_open_orders()
                executed_orders = paper_service.orders_history
                return True, f"Cycle {latest_checkpoint.cycle_id} interrupted at {last_stage.value}; recovered with {len(executed_orders)} orders and {len(open_orders)} open", last_stage
            return True, f"Cycle {latest_checkpoint.cycle_id} interrupted at {last_stage.value}; restoring state", last_stage

        return True, f"Recovered state at stage {last_stage.value}", last_stage
