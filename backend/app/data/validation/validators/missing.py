"""
Missing Value Validator
Detects missing, null, None, and empty observations in essential OHLCV fields.
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


class MissingValueValidator(BaseValidator):
    @property
    def name(self) -> str:
        return "missing_value_validator"

    @property
    def description(self) -> str:
        return "Detects missing or null values in open, high, low, close, volume, timestamp, and symbol."

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

            if bar.symbol is None or bar.symbol == "":
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Missing symbol at index {idx}",
                        record_index=idx,
                        field_name="symbol",
                    )
                )

            if bar.timestamp is None:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Missing timestamp at index {idx}",
                        record_index=idx,
                        field_name="timestamp",
                    )
                )

            for field_name in ["open", "high", "low", "close", "volume"]:
                val = getattr(bar, field_name, None)
                if val is None:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.ERROR,
                            message=f"Missing essential field '{field_name}' at index {idx}",
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=bar.timestamp,
                            record_index=idx,
                            field_name=field_name,
                        )
                    )

        return result
