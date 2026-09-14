"""
Risk Limits and Constraints Evaluation Subsystem (Phase 12 Hardening v1.1).
Evaluates portfolio targets against hard and soft risk constraints with fail-closed safety.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.risk.schemas import (
    RiskAssessment,
    RiskCheck,
    RiskContext,
    RiskEngineConfig,
    RiskLimitSeverity,
    RiskMetrics,
    RiskViolation,
    RiskViolationCode,
)


class RiskLimitEvaluator:
    """
    Evaluates portfolio targets and calculated risk metrics against configured limits.
    Separates hard constraints (must not be violated) from soft constraints (warnings).
    Enforces fail-closed safety on corrupted or invalid inputs.
    """

    @classmethod
    def evaluate(
        cls,
        targets: List[PortfolioTarget],
        metrics: RiskMetrics,
        config: RiskEngineConfig,
        context: Optional[RiskContext] = None,
    ) -> RiskAssessment:
        """
        Run all configured risk checks and collect violations.
        """
        checks: List[RiskCheck] = []
        violations: List[RiskViolation] = []

        # 1. Non-finite weight check
        non_finite_symbols = [
            t.symbol for t in targets
            if np.isnan(t.target_weight) or np.isinf(t.target_weight)
        ]
        if non_finite_symbols:
            checks.append(RiskCheck(
                name="non_finite_weight",
                passed=False,
                severity=RiskLimitSeverity.HARD,
                message=f"Non-finite weights found in symbols: {non_finite_symbols}",
            ))
            violations.append(RiskViolation(
                code=RiskViolationCode.NON_FINITE_WEIGHT,
                severity=RiskLimitSeverity.HARD,
                threshold=1.0,
                observed_value=float("nan"),
                message="Targets contain non-finite (NaN or Inf) weights",
                affected_symbols=non_finite_symbols,
            ))
        else:
            checks.append(RiskCheck(
                name="non_finite_weight",
                passed=True,
                severity=RiskLimitSeverity.HARD,
                message="All target weights are finite numbers",
            ))

        # 2. Duplicate symbol check
        seen_symbols = set()
        duplicate_symbols = set()
        for t in targets:
            if t.symbol in seen_symbols:
                duplicate_symbols.add(t.symbol)
            seen_symbols.add(t.symbol)

        if duplicate_symbols:
            checks.append(RiskCheck(
                name="duplicate_symbols",
                passed=False,
                severity=RiskLimitSeverity.HARD,
                message=f"Duplicate target entries for symbols: {sorted(duplicate_symbols)}",
            ))
            violations.append(RiskViolation(
                code=RiskViolationCode.DUPLICATE_SYMBOL,
                severity=RiskLimitSeverity.HARD,
                threshold=1.0,
                observed_value=float(len(duplicate_symbols)),
                message="Multiple target allocations provided for identical symbols",
                affected_symbols=sorted(duplicate_symbols),
            ))
        else:
            checks.append(RiskCheck(
                name="duplicate_symbols",
                passed=True,
                severity=RiskLimitSeverity.HARD,
                message="No duplicate symbols present in portfolio targets",
            ))

        # 3. Capital validity check (Fail-Closed)
        if context and context.capital:
            cap = context.capital
            if np.isnan(cap.equity) or np.isinf(cap.equity) or cap.equity <= 0:
                checks.append(RiskCheck(
                    name="capital_validity",
                    passed=False,
                    severity=RiskLimitSeverity.HARD,
                    message=f"Invalid portfolio equity: {cap.equity}",
                ))
                violations.append(RiskViolation(
                    code=RiskViolationCode.INVALID_CAPITAL,
                    severity=RiskLimitSeverity.HARD,
                    threshold=0.0,
                    observed_value=cap.equity,
                    message=f"Portfolio capital equity must be finite and > 0, got {cap.equity}",
                ))
            else:
                checks.append(RiskCheck(
                    name="capital_validity",
                    passed=True,
                    severity=RiskLimitSeverity.HARD,
                    message=f"Portfolio capital equity valid: {cap.base_currency} {cap.equity:,.2f}",
                ))

        # 4. Position concentration check (Max Position Weight)
        overweight_symbols = [
            t.symbol for t in targets
            if abs(t.target_weight) > config.max_position_weight + 1e-7
        ]
        max_pos_passed = len(overweight_symbols) == 0
        checks.append(RiskCheck(
            name="max_position_weight",
            passed=max_pos_passed,
            severity=RiskLimitSeverity.HARD,
            threshold=config.max_position_weight,
            observed_value=metrics.max_observed_position_weight,
            message="Passed" if max_pos_passed else f"Overweight symbols: {overweight_symbols}",
        ))
        if not max_pos_passed:
            violations.append(RiskViolation(
                code=RiskViolationCode.MAX_POSITION_WEIGHT,
                severity=RiskLimitSeverity.HARD,
                threshold=config.max_position_weight,
                observed_value=metrics.max_observed_position_weight,
                message=f"Position weight exceeds maximum limit of {config.max_position_weight:.1%}",
                affected_symbols=overweight_symbols,
            ))

        # 5. Gross exposure check
        gross_passed = metrics.gross_exposure <= config.max_gross_exposure + 1e-7
        checks.append(RiskCheck(
            name="max_gross_exposure",
            passed=gross_passed,
            severity=RiskLimitSeverity.HARD,
            threshold=config.max_gross_exposure,
            observed_value=metrics.gross_exposure,
            message="Passed" if gross_passed else f"Gross exposure {metrics.gross_exposure:.1%} exceeds limit",
        ))
        if not gross_passed:
            violations.append(RiskViolation(
                code=RiskViolationCode.MAX_GROSS_EXPOSURE,
                severity=RiskLimitSeverity.HARD,
                threshold=config.max_gross_exposure,
                observed_value=metrics.gross_exposure,
                message=f"Gross exposure {metrics.gross_exposure:.1%} exceeds maximum limit {config.max_gross_exposure:.1%}",
            ))

        # 6. Net exposure check
        net_passed = abs(metrics.net_exposure) <= config.max_net_exposure + 1e-7
        checks.append(RiskCheck(
            name="max_net_exposure",
            passed=net_passed,
            severity=RiskLimitSeverity.HARD,
            threshold=config.max_net_exposure,
            observed_value=abs(metrics.net_exposure),
            message="Passed" if net_passed else f"Net exposure {metrics.net_exposure:.1%} exceeds limit",
        ))
        if not net_passed:
            violations.append(RiskViolation(
                code=RiskViolationCode.MAX_NET_EXPOSURE,
                severity=RiskLimitSeverity.HARD,
                threshold=config.max_net_exposure,
                observed_value=abs(metrics.net_exposure),
                message=f"Net exposure {metrics.net_exposure:.1%} exceeds maximum limit {config.max_net_exposure:.1%}",
            ))

        # 7. Long exposure check
        long_passed = metrics.total_long_weight <= config.max_long_exposure + 1e-7
        checks.append(RiskCheck(
            name="max_long_exposure",
            passed=long_passed,
            severity=RiskLimitSeverity.HARD,
            threshold=config.max_long_exposure,
            observed_value=metrics.total_long_weight,
            message="Passed" if long_passed else f"Long exposure {metrics.total_long_weight:.1%} exceeds limit",
        ))
        if not long_passed:
            violations.append(RiskViolation(
                code=RiskViolationCode.MAX_LONG_EXPOSURE,
                severity=RiskLimitSeverity.HARD,
                threshold=config.max_long_exposure,
                observed_value=metrics.total_long_weight,
                message=f"Long exposure {metrics.total_long_weight:.1%} exceeds maximum limit {config.max_long_exposure:.1%}",
            ))

        # 8. Short exposure check
        short_passed = metrics.total_short_weight <= config.max_short_exposure + 1e-7
        checks.append(RiskCheck(
            name="max_short_exposure",
            passed=short_passed,
            severity=RiskLimitSeverity.HARD,
            threshold=config.max_short_exposure,
            observed_value=metrics.total_short_weight,
            message="Passed" if short_passed else f"Short exposure {metrics.total_short_weight:.1%} exceeds limit",
        ))
        if not short_passed:
            violations.append(RiskViolation(
                code=RiskViolationCode.MAX_SHORT_EXPOSURE,
                severity=RiskLimitSeverity.HARD,
                threshold=config.max_short_exposure,
                observed_value=metrics.total_short_weight,
                message=f"Short exposure {metrics.total_short_weight:.1%} exceeds maximum limit {config.max_short_exposure:.1%}",
            ))

        # 9. Maximum position count check
        positions_passed = metrics.active_position_count <= config.max_positions
        checks.append(RiskCheck(
            name="max_positions",
            passed=positions_passed,
            severity=RiskLimitSeverity.HARD,
            threshold=float(config.max_positions),
            observed_value=float(metrics.active_position_count),
            message="Passed" if positions_passed else f"Active positions ({metrics.active_position_count}) exceeds limit ({config.max_positions})",
        ))
        if not positions_passed:
            violations.append(RiskViolation(
                code=RiskViolationCode.MAX_POSITIONS,
                severity=RiskLimitSeverity.HARD,
                threshold=float(config.max_positions),
                observed_value=float(metrics.active_position_count),
                message=f"Active position count {metrics.active_position_count} exceeds maximum allowable {config.max_positions}",
            ))

        # 10. Leverage check (Leverage = Gross Notional / Portfolio Equity)
        leverage_passed = metrics.leverage <= config.max_leverage + 1e-7
        checks.append(RiskCheck(
            name="max_leverage",
            passed=leverage_passed,
            severity=RiskLimitSeverity.HARD,
            threshold=config.max_leverage,
            observed_value=metrics.leverage,
            message="Passed" if leverage_passed else f"Leverage {metrics.leverage:.2f}x exceeds limit {config.max_leverage:.2f}x",
        ))
        if not leverage_passed:
            violations.append(RiskViolation(
                code=RiskViolationCode.MAX_LEVERAGE,
                severity=RiskLimitSeverity.HARD,
                threshold=config.max_leverage,
                observed_value=metrics.leverage,
                message=f"Leverage {metrics.leverage:.2f}x exceeds maximum allowable limit {config.max_leverage:.2f}x",
            ))

        # 11. Volatility limit check (Soft Limit by default)
        if config.volatility_limit is not None:
            if metrics.portfolio_volatility is not None:
                vol_passed = metrics.portfolio_volatility <= config.volatility_limit + 1e-7
                checks.append(RiskCheck(
                    name="volatility_limit",
                    passed=vol_passed,
                    severity=RiskLimitSeverity.SOFT,
                    threshold=config.volatility_limit,
                    observed_value=metrics.portfolio_volatility,
                    message="Passed" if vol_passed else f"Portfolio volatility {metrics.portfolio_volatility:.1%} exceeds soft limit {config.volatility_limit:.1%}",
                ))
                if not vol_passed:
                    violations.append(RiskViolation(
                        code=RiskViolationCode.VOLATILITY_LIMIT,
                        severity=RiskLimitSeverity.SOFT,
                        threshold=config.volatility_limit,
                        observed_value=metrics.portfolio_volatility,
                        message=f"Estimated portfolio volatility {metrics.portfolio_volatility:.1%} exceeds soft cap {config.volatility_limit:.1%}",
                    ))
            else:
                checks.append(RiskCheck(
                    name="volatility_limit",
                    passed=True,
                    severity=RiskLimitSeverity.SOFT,
                    threshold=config.volatility_limit,
                    observed_value=None,
                    message="Volatility limit check skipped: insufficient point-in-time return data",
                ))

        # 12. Correlation limit check (Soft Limit by default)
        if config.correlation_limit is not None:
            if metrics.average_correlation is not None:
                corr_passed = metrics.average_correlation <= config.correlation_limit + 1e-7
                checks.append(RiskCheck(
                    name="correlation_limit",
                    passed=corr_passed,
                    severity=RiskLimitSeverity.SOFT,
                    threshold=config.correlation_limit,
                    observed_value=metrics.average_correlation,
                    message="Passed" if corr_passed else f"Average correlation {metrics.average_correlation:.2f} exceeds limit {config.correlation_limit:.2f}",
                ))
                if not corr_passed:
                    violations.append(RiskViolation(
                        code=RiskViolationCode.CORRELATION_LIMIT,
                        severity=RiskLimitSeverity.SOFT,
                        threshold=config.correlation_limit,
                        observed_value=metrics.average_correlation,
                        message=f"Average pairwise correlation {metrics.average_correlation:.2f} exceeds soft threshold {config.correlation_limit:.2f}",
                    ))
            else:
                checks.append(RiskCheck(
                    name="correlation_limit",
                    passed=True,
                    severity=RiskLimitSeverity.SOFT,
                    threshold=config.correlation_limit,
                    observed_value=None,
                    message="Correlation limit check skipped: insufficient point-in-time return data",
                ))

        # 13. Turnover limit check (if configured and available)
        if config.turnover_limit is not None:
            if metrics.turnover is not None:
                turn_passed = metrics.turnover <= config.turnover_limit + 1e-7
                checks.append(RiskCheck(
                    name="turnover_limit",
                    passed=turn_passed,
                    severity=RiskLimitSeverity.SOFT,
                    threshold=config.turnover_limit,
                    observed_value=metrics.turnover,
                    message="Passed" if turn_passed else f"Turnover {metrics.turnover:.1%} exceeds limit {config.turnover_limit:.1%}",
                ))
                if not turn_passed:
                    violations.append(RiskViolation(
                        code=RiskViolationCode.TURNOVER_LIMIT,
                        severity=RiskLimitSeverity.SOFT,
                        threshold=config.turnover_limit,
                        observed_value=metrics.turnover,
                        message=f"Portfolio turnover {metrics.turnover:.1%} exceeds soft threshold {config.turnover_limit:.1%}",
                    ))

        # Compliance is True only if zero HARD violations exist
        hard_violations = [v for v in violations if v.severity == RiskLimitSeverity.HARD]
        is_compliant = len(hard_violations) == 0

        return RiskAssessment(
            is_compliant=is_compliant,
            checks=checks,
            violations=violations,
            metrics=metrics,
        )
