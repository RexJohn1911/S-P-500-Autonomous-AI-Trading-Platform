"""
Corporate Action Discontinuity Validator
Detects unadjusted price discontinuities potentially caused by stock splits, reverse splits, or large distributions.
"""

from typing import List, Optional
from backend.app.data.models import BarData, TimeFrame, AssetClass
from backend.app.data.validation.models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator

# Common Stock Split Ratios (Fraction drop / multiplier)
# 2:1 split -> price drops ~50% (ratio ~0.50)
# 3:1 split -> price drops ~66.7% (ratio ~0.333)
# 4:1 split -> price drops ~75% (ratio ~0.25)
# 3:2 split -> price drops ~33.3% (ratio ~0.667)
COMMON_SPLIT_RATIOS = [
    (0.50, "Potential 2:1 Stock Split"),
    (0.3333, "Potential 3:1 Stock Split"),
    (0.25, "Potential 4:1 Stock Split"),
    (0.20, "Potential 5:1 Stock Split"),
    (0.10, "Potential 10:1 Stock Split"),
    (0.6667, "Potential 3:2 Stock Split"),
    (2.0, "Potential 1:2 Reverse Stock Split"),
    (5.0, "Potential 1:5 Reverse Stock Split"),
    (10.0, "Potential 1:10 Reverse Stock Split"),
]


class CorporateActionAnomalyValidator(BaseValidator):
    def __init__(self, split_ratio_tolerance: float = 0.08):
        self.split_ratio_tolerance = split_ratio_tolerance

    @property
    def name(self) -> str:
        return "corporate_action_anomaly_validator"

    @property
    def description(self) -> str:
        return "Detects unadjusted price discontinuities matching common stock split/reverse split patterns."

    def validate(
        self,
        bars: List[BarData],
        expected_symbol: Optional[str] = None,
        expected_timeframe: Optional[TimeFrame] = None,
    ) -> ValidationResult:
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.PASS,
            records_inspected=len(bars),
            description=self.description,
        )

        for i in range(1, len(bars)):
            prev_bar = bars[i - 1]
            curr_bar = bars[i]

            if not isinstance(prev_bar, BarData) or not isinstance(curr_bar, BarData):
                continue
            if prev_bar.close <= 0 or curr_bar.close <= 0:
                continue

            ratio = curr_bar.open / prev_bar.close

            for expected_ratio, label in COMMON_SPLIT_RATIOS:
                if abs(ratio - expected_ratio) <= self.split_ratio_tolerance:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Suspicious price discontinuity ({prev_bar.close} -> {curr_bar.open}, "
                                f"ratio={ratio:.3f}) matching {label} at index {i}"
                            ),
                            symbol=curr_bar.symbol,
                            timeframe=curr_bar.timeframe,
                            timestamp=curr_bar.timestamp,
                            record_index=i,
                            field_name="open",
                            value_observed=f"Ratio {ratio:.3f}",
                            expected="Adjusted continuous pricing",
                        )
                    )
                    break

        return result
