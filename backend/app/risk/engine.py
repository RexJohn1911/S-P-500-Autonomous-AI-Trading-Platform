"""
Risk Adjustment Engine (Phase 12 Hardening v1.1).
Applies deterministic, conservative adjustments to portfolio targets to enforce compliance
with configured risk constraints and maintains a structured audit trail.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.risk.limits import RiskLimitEvaluator
from backend.app.risk.metrics import RiskMetricsCalculator
from backend.app.risk.schemas import (
    PortfolioCapitalContext,
    RiskAdjustedTarget,
    RiskAdjustmentAction,
    RiskAdjustmentRecord,
    RiskAssessment,
    RiskContext,
    RiskEngineConfig,
    RiskLimitSeverity,
    RiskStatus,
    RiskViolation,
    RiskViolationCode,
)


class RiskAdjustmentEngine:
    """
    Executes deterministic constraint enforcement and risk adjustment.
    
    IMPORTANT ARCHITECTURAL PROPERTY:
    The Risk Engine is NOT a second strategy optimizer. It applies conservative,
    deterministic clipping and scaling to enforce hard boundaries without inventing
    or optimizing allocations.
    """

    @classmethod
    def adjust(
        cls,
        targets: List[PortfolioTarget],
        config: RiskEngineConfig,
        context: Optional[RiskContext] = None,
    ) -> Tuple[
        List[RiskAdjustedTarget],
        RiskAssessment,
        List[RiskAdjustmentRecord],
        Dict[str, str],
        bool,
        RiskStatus,
    ]:
        """
        Evaluate targets, apply deterministic adjustments if needed/enabled,
        and return (adjusted_targets, final_assessment, audit_trail, rejected_positions, was_adjusted, status).
        """
        # Fail-closed check: validate capital context if provided
        if context and context.capital:
            cap = context.capital
            if np.isnan(cap.equity) or np.isinf(cap.equity) or cap.equity <= 0:
                empty_metrics = RiskMetricsCalculator.calculate(targets, context)
                empty_assessment = RiskLimitEvaluator.evaluate(targets, empty_metrics, config, context)
                rejected_pos = {t.symbol: "INVALID_CAPITAL_FAIL_CLOSED" for t in targets}
                audit_records = [
                    RiskAdjustmentRecord(
                        symbol=t.symbol,
                        original_weight=t.target_weight,
                        final_weight=0.0,
                        adjustment_amount=-t.target_weight,
                        action=RiskAdjustmentAction.REJECTED,
                        adjustment_reason="Portfolio equity invalid (<=0 or non-finite)",
                        violation_code=RiskViolationCode.INVALID_CAPITAL,
                    )
                    for t in targets
                ]
                adjusted_targets = [
                    RiskAdjustedTarget(
                        symbol=t.symbol,
                        original_weight=t.target_weight,
                        adjusted_weight=0.0,
                        signal_direction=t.signal_direction,
                        signal_strength=t.signal_strength,
                        confidence=t.confidence,
                        action=RiskAdjustmentAction.REJECTED,
                        was_adjusted=True,
                        adjustment_reason="Portfolio equity invalid (<=0 or non-finite)",
                        target_notional=0.0,
                        shares=0.0 if (context and context.asset_prices and t.symbol in context.asset_prices) else None,
                        expected_return=t.expected_return,
                        timestamp=t.timestamp,
                        metadata=t.metadata,
                    )
                    for t in targets
                ]
                return adjusted_targets, empty_assessment, audit_records, rejected_pos, True, RiskStatus.INVALID

        if not targets:
            empty_metrics = RiskMetricsCalculator.calculate([], context)
            empty_assessment = RiskLimitEvaluator.evaluate([], empty_metrics, config, context)
            return [], empty_assessment, [], {}, False, RiskStatus.VALID

        # Phase 1: Initial compliance evaluation
        initial_metrics = RiskMetricsCalculator.calculate(targets, context)
        initial_assessment = RiskLimitEvaluator.evaluate(targets, initial_metrics, config, context)

        equity = context.capital.equity if (context and context.capital) else None
        prices = context.asset_prices if context else None

        if initial_assessment.is_compliant or not config.adjustment_enabled:
            # Portfolio is already compliant, or adjustment is explicitly disabled
            status = RiskStatus.VALID if initial_assessment.is_compliant else RiskStatus.RISK_REJECTED
            adjusted_targets = []
            audit_trail = []

            # Deterministic sorting by symbol for consistent audit trail
            sorted_compliant_targets = sorted(targets, key=lambda x: x.symbol)

            for t in sorted_compliant_targets:
                notional = (t.target_weight * equity) if equity is not None else None
                px = prices.get(t.symbol) if prices else None
                shares = (notional / px) if (notional is not None and px is not None and px > 0) else None

                adjusted_targets.append(
                    RiskAdjustedTarget(
                        symbol=t.symbol,
                        original_weight=t.target_weight,
                        adjusted_weight=t.target_weight,
                        signal_direction=t.signal_direction,
                        signal_strength=t.signal_strength,
                        confidence=t.confidence,
                        action=RiskAdjustmentAction.UNCHANGED,
                        was_adjusted=False,
                        adjustment_reason=None,
                        target_notional=round(notional, 2) if notional is not None else None,
                        shares=round(shares, 4) if shares is not None else None,
                        expected_return=t.expected_return,
                        timestamp=t.timestamp,
                        metadata=t.metadata,
                    )
                )
                audit_trail.append(
                    RiskAdjustmentRecord(
                        symbol=t.symbol,
                        original_weight=t.target_weight,
                        final_weight=t.target_weight,
                        adjustment_amount=0.0,
                        action=RiskAdjustmentAction.UNCHANGED,
                        adjustment_reason=None,
                        violation_code=None,
                        original_notional=round(notional, 2) if notional is not None else None,
                        final_notional=round(notional, 2) if notional is not None else None,
                    )
                )

            return adjusted_targets, initial_assessment, audit_trail, {}, False, status

        # Phase 2: Deterministic risk adjustments
        was_adjusted = False
        rejected_positions: Dict[str, str] = {}

        # 2a. Deduplicate symbols and sanitize non-finite weights
        sanitized_targets: List[PortfolioTarget] = []
        seen_symbols = set()

        for t in targets:
            if t.symbol in seen_symbols:
                rejected_positions[t.symbol] = "DUPLICATE_SYMBOL_REJECTED"
                was_adjusted = True
                continue
            seen_symbols.add(t.symbol)

            weight = float(t.target_weight)
            if np.isnan(weight) or np.isinf(weight):
                rejected_positions[t.symbol] = "NON_FINITE_WEIGHT_REJECTED"
                was_adjusted = True
                weight = 0.0

            sanitized_targets.append(
                PortfolioTarget(
                    symbol=t.symbol,
                    target_weight=weight,
                    signal_direction=t.signal_direction,
                    signal_strength=t.signal_strength,
                    confidence=t.confidence,
                    expected_return=t.expected_return,
                    signal_version=t.signal_version,
                    timestamp=t.timestamp,
                    metadata=t.metadata,
                )
            )

        # 2b. Enforce max_positions limit
        active_candidates = [t for t in sanitized_targets if abs(t.target_weight) > 1e-7]
        inactive_candidates = [t for t in sanitized_targets if abs(t.target_weight) <= 1e-7]

        if len(active_candidates) > config.max_positions:
            # Deterministic ranking: (|weight| DESC, confidence DESC, symbol ASC)
            sorted_active = sorted(
                active_candidates,
                key=lambda x: (-abs(x.target_weight), -float(x.confidence), x.symbol),
            )
            kept_active = sorted_active[: config.max_positions]
            pruned_active = sorted_active[config.max_positions :]

            for p in pruned_active:
                rejected_positions[p.symbol] = "MAX_POSITIONS_EXCEEDED"
                was_adjusted = True
                inactive_candidates.append(
                    PortfolioTarget(
                        symbol=p.symbol,
                        target_weight=0.0,
                        signal_direction=p.signal_direction,
                        signal_strength=p.signal_strength,
                        confidence=p.confidence,
                        expected_return=p.expected_return,
                        signal_version=p.signal_version,
                        timestamp=p.timestamp,
                        metadata=p.metadata,
                    )
                )
            active_candidates = kept_active

        # 2c. Enforce individual max_position_weight concentration limit
        working_weights: Dict[str, float] = {}
        target_map: Dict[str, PortfolioTarget] = {t.symbol: t for t in active_candidates}

        for t in active_candidates:
            w = float(t.target_weight)
            if abs(w) > config.max_position_weight:
                capped_w = float(np.sign(w) * config.max_position_weight)
                working_weights[t.symbol] = capped_w
                was_adjusted = True
            else:
                working_weights[t.symbol] = w

        # 2d. Enforce max_long_exposure limit
        long_sum = sum(w for w in working_weights.values() if w > 0)
        if long_sum > config.max_long_exposure + 1e-7:
            long_scale = config.max_long_exposure / long_sum
            for sym, w in list(working_weights.items()):
                if w > 0:
                    working_weights[sym] = round(w * long_scale, 6)
            was_adjusted = True

        # 2e. Enforce max_short_exposure limit
        short_sum = sum(abs(w) for w in working_weights.values() if w < 0)
        if short_sum > config.max_short_exposure + 1e-7:
            short_scale = config.max_short_exposure / short_sum
            for sym, w in list(working_weights.items()):
                if w < 0:
                    working_weights[sym] = round(w * short_scale, 6)
            was_adjusted = True

        # 2f. Enforce max_gross_exposure limit
        gross_sum = sum(abs(w) for w in working_weights.values())
        if gross_sum > config.max_gross_exposure + 1e-7:
            gross_scale = config.max_gross_exposure / gross_sum
            for sym, w in list(working_weights.items()):
                working_weights[sym] = round(w * gross_scale, 6)
            was_adjusted = True

        # Assemble intermediate adjusted targets
        intermediate_targets: List[PortfolioTarget] = []
        original_map: Dict[str, float] = {t.symbol: t.target_weight for t in targets}

        for sym, t in target_map.items():
            adj_w = working_weights[sym]
            intermediate_targets.append(
                PortfolioTarget(
                    symbol=sym,
                    target_weight=adj_w,
                    signal_direction=t.signal_direction,
                    signal_strength=t.signal_strength,
                    confidence=t.confidence,
                    expected_return=t.expected_return,
                    signal_version=t.signal_version,
                    timestamp=t.timestamp,
                    metadata=t.metadata,
                )
            )

        # Include inactive/pruned targets
        for t in inactive_candidates:
            intermediate_targets.append(t)

        # Phase 3: Final compliance evaluation on adjusted targets
        final_metrics = RiskMetricsCalculator.calculate(intermediate_targets, context)
        final_assessment = RiskLimitEvaluator.evaluate(
            intermediate_targets, final_metrics, config, context
        )

        # Construct strongly typed RiskAdjustedTarget objects and RiskAdjustmentRecords
        adjusted_target_results: List[RiskAdjustedTarget] = []
        audit_trail: List[RiskAdjustmentRecord] = []

        # Sort symbols deterministically for audit trail
        sorted_targets = sorted(intermediate_targets, key=lambda x: x.symbol)

        for t in sorted_targets:
            orig_w = original_map.get(t.symbol, t.target_weight)
            diff = t.target_weight - orig_w
            is_adj = abs(diff) > 1e-6

            action = RiskAdjustmentAction.UNCHANGED
            reason = None
            v_code = None

            if t.symbol in rejected_positions:
                action = RiskAdjustmentAction.REJECTED if "INVALID" in rejected_positions[t.symbol] else RiskAdjustmentAction.REMOVED
                reason = rejected_positions[t.symbol]
                v_code = RiskViolationCode.MAX_POSITIONS if reason == "MAX_POSITIONS_EXCEEDED" else RiskViolationCode.DUPLICATE_SYMBOL
            elif is_adj:
                if abs(orig_w) > config.max_position_weight:
                    action = RiskAdjustmentAction.REDUCED
                    reason = f"Capped at max position weight {config.max_position_weight:.1%}"
                    v_code = RiskViolationCode.MAX_POSITION_WEIGHT
                elif gross_sum > config.max_gross_exposure:
                    action = RiskAdjustmentAction.SCALED
                    reason = f"Scaled proportionally for gross exposure limit {config.max_gross_exposure:.1%}"
                    v_code = RiskViolationCode.MAX_GROSS_EXPOSURE
                elif long_sum > config.max_long_exposure:
                    action = RiskAdjustmentAction.SCALED
                    reason = f"Scaled proportionally for long exposure limit {config.max_long_exposure:.1%}"
                    v_code = RiskViolationCode.MAX_LONG_EXPOSURE
                elif short_sum > config.max_short_exposure:
                    action = RiskAdjustmentAction.SCALED
                    reason = f"Scaled proportionally for short exposure limit {config.max_short_exposure:.1%}"
                    v_code = RiskViolationCode.MAX_SHORT_EXPOSURE
                else:
                    action = RiskAdjustmentAction.REDUCED
                    reason = "Adjusted for risk limits"
                    v_code = RiskViolationCode.MAX_POSITION_WEIGHT

            orig_notional = (orig_w * equity) if equity is not None else None
            final_notional = (t.target_weight * equity) if equity is not None else None
            px = prices.get(t.symbol) if prices else None
            shares = (final_notional / px) if (final_notional is not None and px is not None and px > 0) else None

            adjusted_target_results.append(
                RiskAdjustedTarget(
                    symbol=t.symbol,
                    original_weight=orig_w,
                    adjusted_weight=t.target_weight,
                    signal_direction=t.signal_direction,
                    signal_strength=t.signal_strength,
                    confidence=t.confidence,
                    action=action,
                    was_adjusted=is_adj,
                    adjustment_reason=reason,
                    target_notional=round(final_notional, 2) if final_notional is not None else None,
                    shares=round(shares, 4) if shares is not None else None,
                    expected_return=t.expected_return,
                    timestamp=t.timestamp,
                    metadata=t.metadata,
                )
            )

            audit_trail.append(
                RiskAdjustmentRecord(
                    symbol=t.symbol,
                    original_weight=orig_w,
                    final_weight=t.target_weight,
                    adjustment_amount=diff,
                    action=action,
                    adjustment_reason=reason,
                    violation_code=v_code,
                    original_notional=round(orig_notional, 2) if orig_notional is not None else None,
                    final_notional=round(final_notional, 2) if final_notional is not None else None,
                )
            )

        # Update violation adjustment status
        if was_adjusted:
            for v in final_assessment.violations:
                v.was_adjusted = True
                v.adjustment_details = "Target weights adjusted by deterministic risk engine"

        status = RiskStatus.RISK_ADJUSTED if was_adjusted else RiskStatus.VALID
        return adjusted_target_results, final_assessment, audit_trail, rejected_positions, was_adjusted, status
