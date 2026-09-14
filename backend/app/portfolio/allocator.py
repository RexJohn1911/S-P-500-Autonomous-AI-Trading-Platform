"""
Portfolio Allocation Module.
Implements deterministic signal-to-weight mapping algorithms for constructing portfolio targets from Phase 10 signals.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.app.portfolio.schemas import PortfolioConstructionConfig, PortfolioMode, PortfolioTarget
from backend.app.strategy.schemas import SignalCandidate, SignalDirection


class BasePortfolioAllocator(ABC):
    """Abstract base class for portfolio allocation strategies."""

    @abstractmethod
    def allocate(
        self,
        signals: List[SignalCandidate],
        config: PortfolioConstructionConfig,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[List[PortfolioTarget], Dict[str, str]]:
        """
        Transform a list of SignalCandidate objects into PortfolioTarget instances.

        Returns:
            (targets: List[PortfolioTarget], rejected_signals: Dict[str, str])
        """
        pass


class SignalProportionalAllocator(BasePortfolioAllocator):
    """
    Standard signal-proportional allocator.
    Maps signal strength and conviction into target portfolio weights.
    
    ALLOCATION METHODOLOGY (ENGINEERING BASELINE):
    - Raw score: score_i = |signal_strength_i| * confidence_i
    - FLAT signals receive 0 weight and are filtered out.
    - SHORT signals are rejected in LONG_ONLY mode and mapped to negative weights in LONG_SHORT mode.
    - Candidates are ranked deterministically by (raw_score DESC, confidence DESC, symbol ASC).
    - Top K <= max_positions candidates are selected.
    - Pruned if initial proportional weight is below min_position_weight.
    - Capped iteratively at max_position_weight with surplus redistribution to uncapped positions.
    - Surplus capital that cannot be allocated without violating constraints remains unallocated cash.
    """

    def allocate(
        self,
        signals: List[SignalCandidate],
        config: PortfolioConstructionConfig,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[List[PortfolioTarget], Dict[str, str]]:
        rejected_signals: Dict[str, str] = {}
        eligible_candidates: List[Tuple[SignalCandidate, float]] = []

        if not signals:
            return [], rejected_signals

        # Determine effective timestamp from signals if not explicitly provided
        eval_time = timestamp or signals[0].timestamp

        # 1. Deduplicate by symbol (keep the latest or highest strength signal)
        unique_signals: Dict[str, SignalCandidate] = {}
        for s in signals:
            if s.symbol not in unique_signals:
                unique_signals[s.symbol] = s
            else:
                # Deterministic tie-break: keep signal with higher strength, then confidence
                existing = unique_signals[s.symbol]
                if (abs(s.signal_strength), s.confidence) > (abs(existing.signal_strength), existing.confidence):
                    unique_signals[s.symbol] = s

        # 2. Filter and compute raw allocation score
        for symbol, sig in unique_signals.items():
            # Validate numerical sanity
            if (
                np.isnan(sig.signal_strength)
                or np.isinf(sig.signal_strength)
                or np.isnan(sig.confidence)
                or np.isinf(sig.confidence)
            ):
                rejected_signals[symbol] = "INVALID_SIGNAL"
                continue

            # FLAT signals receive zero allocation
            if sig.signal == SignalDirection.FLAT:
                rejected_signals[symbol] = "FLAT_SIGNAL"
                continue

            # Long-only mode rejecting SHORT signals
            if config.mode == PortfolioMode.LONG_ONLY and sig.signal == SignalDirection.SHORT:
                rejected_signals[symbol] = "SHORT_IN_LONG_ONLY"
                continue

            raw_score = abs(sig.signal_strength) * sig.confidence
            if raw_score <= 1e-6:
                rejected_signals[symbol] = "ZERO_ALLOCATION_SCORE"
                continue

            eligible_candidates.append((sig, float(raw_score)))

        if not eligible_candidates:
            return [], rejected_signals

        # 3. Deterministic candidate ranking:
        #    1. raw_score descending
        #    2. confidence descending
        #    3. symbol ascending (tie-breaker)
        eligible_candidates.sort(
            key=lambda item: (-item[1], -item[0].confidence, item[0].symbol)
        )

        # 4. Enforce max_positions constraint
        selected_candidates: List[Tuple[SignalCandidate, float]] = []
        for rank, (sig, raw_score) in enumerate(eligible_candidates):
            if rank < config.max_positions:
                selected_candidates.append((sig, raw_score))
            else:
                rejected_signals[sig.symbol] = "MAX_POSITIONS_EXCEEDED"

        if not selected_candidates:
            return [], rejected_signals

        # 5. Filter out positions whose initial unconstrained proportional weight < min_position_weight
        total_initial_score = sum(score for _, score in selected_candidates)
        surviving_candidates: List[Tuple[SignalCandidate, float]] = []
        for sig, score in selected_candidates:
            initial_prop = config.max_gross_exposure * (score / total_initial_score)
            if initial_prop < config.min_position_weight - 1e-7:
                rejected_signals[sig.symbol] = "BELOW_MIN_WEIGHT"
            else:
                surviving_candidates.append((sig, score))

        if not surviving_candidates:
            return [], rejected_signals

        # 6. Iterative Weight Capping and Surplus Redistribution Routine
        def _compute_capped_weights(
            candidates: List[Tuple[SignalCandidate, float]],
            target_gross_exposure: float,
            max_weight: float,
        ) -> Dict[str, float]:
            """Allocate weights proportionally up to target_gross_exposure with max_weight cap."""
            weights: Dict[str, float] = {}
            active = list(candidates)
            remaining_exposure = target_gross_exposure

            while active:
                total_active_score = sum(score for _, score in active)
                if total_active_score <= 0.0:
                    break

                newly_capped = []
                for sig, score in active:
                    prop_weight = remaining_exposure * (score / total_active_score)
                    if prop_weight >= max_weight - 1e-7:
                        weights[sig.symbol] = max_weight
                        newly_capped.append(sig.symbol)
                    else:
                        weights[sig.symbol] = prop_weight

                if newly_capped and len(newly_capped) < len(active):
                    capped_exposure = len(newly_capped) * max_weight
                    remaining_exposure = max(0.0, remaining_exposure - capped_exposure)
                    active = [(sig, score) for sig, score in active if sig.symbol not in newly_capped]
                else:
                    break

            return weights

        weights_map = _compute_capped_weights(
            surviving_candidates,
            config.max_gross_exposure,
            config.max_position_weight,
        )

        # 7. Final minimum weight verification on capped weights
        final_candidates: List[Tuple[SignalCandidate, float]] = []
        for sig, score in surviving_candidates:
            w = weights_map.get(sig.symbol, 0.0)
            if w < config.min_position_weight - 1e-7:
                rejected_signals[sig.symbol] = "BELOW_MIN_WEIGHT"
                if sig.symbol in weights_map:
                    del weights_map[sig.symbol]
            else:
                final_candidates.append((sig, score))

        # 8. Construct final PortfolioTarget objects with signed weights
        targets: List[PortfolioTarget] = []
        for sig, _ in final_candidates:
            w_magnitude = weights_map.get(sig.symbol, 0.0)
            if w_magnitude <= 0.0:
                continue

            # Signed weight: positive for LONG, negative for SHORT
            signed_weight = w_magnitude if sig.signal == SignalDirection.LONG else -w_magnitude

            targets.append(
                PortfolioTarget(
                    symbol=sig.symbol,
                    target_weight=signed_weight,
                    signal_direction=sig.signal,
                    signal_strength=sig.signal_strength,
                    confidence=sig.confidence,
                    expected_return=sig.expected_return,
                    signal_version=sig.signal_version,
                    timestamp=eval_time,
                    metadata={
                        "raw_score": round(float(abs(sig.signal_strength) * sig.confidence), 6),
                        "signal_horizon": sig.signal_horizon,
                    },
                )
            )

        # Sort targets deterministically by symbol
        targets.sort(key=lambda t: t.symbol)

        return targets, rejected_signals
