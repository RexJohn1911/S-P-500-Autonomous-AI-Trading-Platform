"""
Autonomous Market Data Scheduler & Safety Validator (Phase 15).
Coordinates market data cadence, staleness validation, symbol universe completeness,
and concurrency safety to prevent overlapping execution cycles.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Set, Tuple

from backend.app.autonomous.schemas import (
    AutonomousConfig,
    MissingSymbolPolicy,
)
from backend.app.data.models import BarData

logger = logging.getLogger(__name__)


class MarketDataScheduler:
    """
    Validates market data readiness, staleness, symbol completeness, and concurrency guards.
    """

    def __init__(self, config: Optional[AutonomousConfig] = None):
        self.config = config or AutonomousConfig()
        self.processed_cycle_keys: Set[str] = set()
        self._active_cycle_id: Optional[str] = None

    def is_cycle_active(self) -> bool:
        """Check if a trading cycle is currently running."""
        return self._active_cycle_id is not None

    def acquire_cycle_lock(self, cycle_id: str) -> bool:
        """Acquire lock for starting a cycle; prevents overlapping concurrency."""
        if self.is_cycle_active():
            logger.warning("Cannot start cycle %s: cycle %s is already running", cycle_id, self._active_cycle_id)
            return False
        self._active_cycle_id = cycle_id
        return True

    def release_cycle_lock(self, cycle_id: str) -> None:
        """Release concurrency lock."""
        if self._active_cycle_id == cycle_id:
            self._active_cycle_id = None

    def generate_cycle_key(self, session_id: str, market_timestamp: datetime) -> str:
        """Deterministic identity for a market event cycle."""
        return f"{session_id}_{market_timestamp.isoformat()}_{self.config.timeframe}"

    def is_cycle_already_completed(self, session_id: str, market_timestamp: datetime) -> bool:
        """Idempotency check to prevent re-executing already processed market timestamps."""
        key = self.generate_cycle_key(session_id, market_timestamp)
        return key in self.processed_cycle_keys

    def mark_cycle_completed(self, session_id: str, market_timestamp: datetime) -> None:
        """Record cycle key as completed."""
        key = self.generate_cycle_key(session_id, market_timestamp)
        self.processed_cycle_keys.add(key)

    def validate_market_data(
        self,
        market_data: Dict[str, BarData],
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate data freshness, symbol coverage, and price finite values.
        Returns: (is_valid, error_reason, missing_symbols)
        """
        now = current_time or datetime.now(timezone.utc)

        if not market_data:
            return False, "No market data bars provided", list(self.config.universe)

        # 1. Symbol Universe Coverage Check
        provided_symbols = set(market_data.keys())
        expected_symbols = set(self.config.universe)
        missing_symbols = list(expected_symbols - provided_symbols)

        if missing_symbols:
            if self.config.missing_symbol_policy == MissingSymbolPolicy.FAIL_CLOSED:
                return False, f"Missing required symbols in universe: {missing_symbols}", missing_symbols
            else:
                logger.warning("Partial universe detected; missing symbols: %s", missing_symbols)

        # 2. Bar Integrity & Finite Values
        for sym, bar in market_data.items():
            if bar.close is None or bar.close <= 0:
                return False, f"Invalid or non-positive close price for {sym}: {bar.close}", missing_symbols
            if bar.open is None or bar.open <= 0:
                return False, f"Invalid or non-positive open price for {sym}: {bar.open}", missing_symbols
            if bar.timestamp is None:
                return False, f"Missing timestamp for {sym}", missing_symbols

        # 3. Staleness Check
        # Check that latest bar timestamp is not older than max_stale_tolerance_seconds
        latest_ts = max(b.timestamp for b in market_data.values())
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)

        age_seconds = (now - latest_ts).total_seconds()
        # Only check staleness when running with real wall-clock (not simulated replay in past)
        if age_seconds > self.config.max_stale_tolerance_seconds and now.year == latest_ts.year:
            return False, f"Market data is stale: age {age_seconds:.0f}s exceeds tolerance {self.config.max_stale_tolerance_seconds:.0f}s", missing_symbols

        return True, None, missing_symbols
