"""
OHLC Mathematical Consistency Validator
Verifies mathematical relationships between open, high, low, close, and volume.
"""

from typing import List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator


class OHLCValidator(BaseValidator):
    @property
    def name(self) -> str:
        return "ohlc_validator"

    @property
    def description(self) -> str:
        return "Validates mathematical consistency of OHLC bars (high >= low, high >= open/close, low <= open/close, non-negative)."

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

        for idx, bar in enumerate(bars):
            if not isinstance(bar, BarData):
                continue

            o, h, l, c, v = bar.open, bar.high, bar.low, bar.close, bar.volume

            # Check 1: High >= Low
            if h < l:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"High ({h}) is strictly less than Low ({l}) at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="high",
                        value_observed=f"H={h}, L={l}",
                        expected="high >= low",
                    )
                )

            # Check 2: High >= max(Open, Close)
            max_body = max(o, c)
            if h < max_body:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"High ({h}) is less than body maximum ({max_body}) at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="high",
                        value_observed=f"H={h}, O={o}, C={c}",
                        expected="high >= max(open, close)",
                    )
                )

            # Check 3: Low <= min(Open, Close)
            min_body = min(o, c)
            if l > min_body:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Low ({l}) is greater than body minimum ({min_body}) at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="low",
                        value_observed=f"L={l}, O={o}, C={c}",
                        expected="low <= min(open, close)",
                    )
                )

            # Check 4: Non-negative prices
            for name, val in [("open", o), ("high", h), ("low", l), ("close", c)]:
                if val < 0:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Negative price in '{name}' ({val}) at index {idx}",
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=bar.timestamp,
                            record_index=idx,
                            field_name=name,
                            value_observed=val,
                            expected="price >= 0",
                        )
                    )

            # Check 5: Non-negative volume
            if v < 0:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Negative volume ({v}) at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="volume",
                        value_observed=v,
                        expected="volume >= 0",
                    )
                )

            # Check 6: VWAP consistency (if present)
            if bar.vwap is not None:
                # VWAP should reasonably fall between Low and High (or with slight tolerance for rounding)
                if bar.vwap < l * 0.999 or bar.vwap > h * 1.001:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.WARNING,
                            message=f"VWAP ({bar.vwap}) outside [Low={l}, High={h}] range at index {idx}",
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=bar.timestamp,
                            record_index=idx,
                            field_name="vwap",
                            value_observed=bar.vwap,
                        )
                    )

        return result
