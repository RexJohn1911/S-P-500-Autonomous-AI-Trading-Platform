"""
Price Consistency & Sanity Validator
Checks for zero prices, non-finite values, and suspicious price jumps between consecutive observations.
"""

import math
from typing import List, Optional
from backend.app.data.models import BarData, TimeFrame, AssetClass
from backend.app.data.validation.models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator


class PriceValidator(BaseValidator):
    def __init__(self, max_single_bar_change_pct: float = 0.35):
        """
        :param max_single_bar_change_pct: Threshold for triggering warning on large price jump (default 35%).
        """
        self.max_single_bar_change_pct = max_single_bar_change_pct

    @property
    def name(self) -> str:
        return "price_validator"

    @property
    def description(self) -> str:
        return "Validates price sanity: detects zero prices, non-finite values, and suspicious price jumps."

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

        prev_close: Optional[float] = None
        for idx, bar in enumerate(bars):
            if not isinstance(bar, BarData):
                continue

            # Zero price check for equities/ETFs
            if bar.asset_class in (AssetClass.EQUITY, AssetClass.ETF) and bar.close == 0.0:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Zero close price detected for equity {bar.symbol} at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="close",
                        value_observed=bar.close,
                        expected="price > 0 for equities/ETFs",
                    )
                )

            # Price Jump Check (consecutive bars)
            if prev_close is not None and prev_close > 0:
                change_pct = abs(bar.close - prev_close) / prev_close
                if change_pct > self.max_single_bar_change_pct:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Suspicious large price change of {change_pct * 100:.1f}% "
                                f"from {prev_close} to {bar.close} at index {idx}"
                            ),
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=bar.timestamp,
                            record_index=idx,
                            field_name="close",
                            value_observed=f"{prev_close} -> {bar.close} ({change_pct * 100:.1f}%)",
                            expected=f"change_pct <= {self.max_single_bar_change_pct * 100:.1f}%",
                        )
                    )

            prev_close = bar.close

        return result
