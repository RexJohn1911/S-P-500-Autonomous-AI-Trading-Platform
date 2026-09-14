"""
Portfolio Construction Service Orchestrator (Phase 11).
Coordinates signal ingestion, ranking, proportional allocation, constraint validation, and portfolio target result generation.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.portfolio.allocator import BasePortfolioAllocator, SignalProportionalAllocator
from backend.app.portfolio.constraints import PortfolioConstraintEnforcer
from backend.app.portfolio.schemas import (
    PortfolioConstructionConfig,
    PortfolioConstructionRequest,
    PortfolioConstructionResult,
    PortfolioTarget,
)
from backend.app.portfolio.storage import PortfolioStorage
from backend.app.strategy.schemas import SignalCandidate

logger = logging.getLogger(__name__)


class PortfolioConstructionService:
    """
    Primary service for synthesizing Phase 10 SignalCandidate streams into portfolio-level target allocations.
    
    IMPORTANT ARCHITECTURAL PROPERTIES:
    - Purely mathematical, deterministic allocation engine.
    - Strictly causal at timestamp t with zero future look-ahead.
    - Generates target weights only. Does NOT execute trades, manage stop-loss risk, or simulate broker orders.
    """

    def __init__(
        self,
        config: Optional[PortfolioConstructionConfig] = None,
        allocator: Optional[BasePortfolioAllocator] = None,
        constraint_enforcer: Optional[PortfolioConstraintEnforcer] = None,
        storage: Optional[PortfolioStorage] = None,
    ):
        self.config = config or PortfolioConstructionConfig()
        self.allocator = allocator or SignalProportionalAllocator()
        self.constraint_enforcer = constraint_enforcer or PortfolioConstraintEnforcer()
        self.storage = storage or PortfolioStorage()

    def construct(
        self,
        signals: List[SignalCandidate],
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PortfolioConstructionResult:
        """
        Construct portfolio-level target allocations from a list of SignalCandidate objects at timestamp t.

        Args:
            signals: List of SignalCandidate instances available at timestamp t.
            timestamp: Point-in-time timestamp t.
            metadata: Optional contextual metadata.

        Returns:
            PortfolioConstructionResult with targets, summary exposures, and rejected reasons.
        """
        eval_time = timestamp or (signals[0].timestamp if signals else datetime.now(timezone.utc))

        # 1. Run allocator to produce target weights
        targets, rejected = self.allocator.allocate(
            signals=signals,
            config=self.config,
            timestamp=eval_time,
        )

        # 2. Validate constraints
        is_valid, violations = self.constraint_enforcer.validate_targets(
            targets=targets,
            config=self.config,
        )
        if not is_valid:
            logger.warning(f"Portfolio construction constraint warnings: {violations}")

        # 3. Calculate portfolio exposure metrics
        total_long = float(sum(t.target_weight for t in targets if t.target_weight > 0.0))
        total_short = float(sum(abs(t.target_weight) for t in targets if t.target_weight < 0.0))
        gross_exposure = total_long + total_short
        net_exposure = total_long - total_short
        active_count = len([t for t in targets if abs(t.target_weight) > 1e-6])
        cash_weight = max(0.0, 1.0 - gross_exposure)

        meta = dict(metadata or {})
        if violations:
            meta["constraint_violations"] = violations

        return PortfolioConstructionResult(
            timestamp=eval_time,
            targets=targets,
            total_long_weight=round(total_long, 6),
            total_short_weight=round(total_short, 6),
            gross_exposure=round(gross_exposure, 6),
            net_exposure=round(net_exposure, 6),
            active_position_count=active_count,
            allocated_cash_weight=round(cash_weight, 6),
            rejected_signals=rejected,
            construction_version=self.config.version,
            metadata=meta,
        )

    def construct_request(
        self,
        request: PortfolioConstructionRequest,
    ) -> PortfolioConstructionResult:
        """Execute portfolio construction from a structured request container."""
        cfg = request.config or self.config
        allocator = self.allocator

        # Temporarily use request-specific config if provided
        prev_cfg = self.config
        try:
            self.config = cfg
            return self.construct(
                signals=request.signals,
                timestamp=request.timestamp,
                metadata=request.metadata,
            )
        finally:
            self.config = prev_cfg
