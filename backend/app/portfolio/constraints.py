"""
Portfolio Constraints Validation Module.
Enforces deterministic portfolio construction constraints and invariants without implementing risk management logic.
"""

from typing import Dict, List, Tuple
import numpy as np

from backend.app.portfolio.schemas import PortfolioConstructionConfig, PortfolioMode, PortfolioTarget


class PortfolioConstraintEnforcer:
    """
    Validates that a constructed portfolio target allocation satisfies all configured constraints.
    """

    @staticmethod
    def validate_targets(
        targets: List[PortfolioTarget],
        config: PortfolioConstructionConfig,
        tolerance: float = 1e-5,
    ) -> Tuple[bool, List[str]]:
        """
        Check that target allocations satisfy configuration limits.

        Returns:
            (is_valid: bool, violations: List[str])
        """
        violations: List[str] = []

        if not targets:
            return True, []

        active_targets = [t for t in targets if abs(t.target_weight) > tolerance]
        active_count = len(active_targets)

        # 1. Max positions check
        if active_count > config.max_positions:
            violations.append(
                f"Active position count {active_count} exceeds max_positions {config.max_positions}"
            )

        total_long = sum(t.target_weight for t in active_targets if t.target_weight > 0.0)
        total_short = sum(abs(t.target_weight) for t in active_targets if t.target_weight < 0.0)
        gross_exposure = total_long + total_short
        net_exposure = total_long - total_short

        # 2. Portfolio mode check
        if config.mode == PortfolioMode.LONG_ONLY:
            for t in active_targets:
                if t.target_weight < -tolerance:
                    violations.append(
                        f"Negative target weight {t.target_weight} found for symbol {t.symbol} in LONG_ONLY mode"
                    )

        # 3. Individual position weight limits
        for t in active_targets:
            w_abs = abs(t.target_weight)
            if np.isnan(t.target_weight) or np.isinf(t.target_weight):
                violations.append(f"Non-finite target weight {t.target_weight} for symbol {t.symbol}")
            elif w_abs > config.max_position_weight + tolerance:
                violations.append(
                    f"Position weight {t.target_weight:.6f} for symbol {t.symbol} exceeds max_position_weight {config.max_position_weight}"
                )
            elif w_abs < config.min_position_weight - tolerance:
                violations.append(
                    f"Active position weight {t.target_weight:.6f} for symbol {t.symbol} is below min_position_weight {config.min_position_weight}"
                )

        # 4. Gross exposure limit
        if gross_exposure > config.max_gross_exposure + tolerance:
            violations.append(
                f"Gross exposure {gross_exposure:.6f} exceeds max_gross_exposure {config.max_gross_exposure}"
            )

        # 5. Net exposure limit
        if abs(net_exposure) > config.max_net_exposure + tolerance:
            violations.append(
                f"Absolute net exposure {abs(net_exposure):.6f} exceeds max_net_exposure {config.max_net_exposure}"
            )

        return len(violations) == 0, violations
