"""
Timestamp & Ordering Validator
Validates UTC timezone compliance, reasonable historical bounds, and strictly chronological ordering.
"""

from datetime import datetime, timezone
from typing import List, Optional
from backend.app.data.models import BarData, TimeFrame, UTC
from backend.app.data.validation.models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator


class TimestampValidator(BaseValidator):
    def __init__(self, min_year: int = 1970, max_future_buffer_hours: float = 24.0):
        self.min_year = min_year
        self.max_future_buffer_hours = max_future_buffer_hours

    @property
    def name(self) -> str:
        return "timestamp_validator"

    @property
    def description(self) -> str:
        return "Verifies UTC timezone normalization, reasonable date bounds, and chronological sorting."

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

        prev_ts: Optional[datetime] = None
        now_utc = datetime.now(timezone.utc)

        for idx, bar in enumerate(bars):
            if not isinstance(bar, BarData):
                continue

            ts = bar.timestamp

            # 1. Timezone Check (Must be UTC)
            if ts.tzinfo is None:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Naive (timezone-unaware) timestamp found at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=ts,
                        record_index=idx,
                        field_name="timestamp",
                        value_observed=str(ts),
                        expected="Timezone-aware datetime in UTC",
                    )
                )
            elif ts.utcoffset() != UTC.utcoffset(ts):
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Non-UTC timezone offset ({ts.tzinfo}) found at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=ts,
                        record_index=idx,
                        field_name="timestamp",
                        value_observed=str(ts.tzinfo),
                        expected="UTC timezone offset (+00:00)",
                    )
                )

            # 2. Date Bounds Check
            if ts.year < self.min_year:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Timestamp year {ts.year} precedes minimum allowable year {self.min_year} at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=ts,
                        record_index=idx,
                        field_name="timestamp",
                        value_observed=ts.isoformat(),
                        expected=f"year >= {self.min_year}",
                    )
                )

            # 3. Future date check
            if (ts - now_utc).total_seconds() > (self.max_future_buffer_hours * 3600):
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.WARNING,
                        message=f"Timestamp {ts.isoformat()} is in the distant future relative to current time at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=ts,
                        record_index=idx,
                        field_name="timestamp",
                        value_observed=ts.isoformat(),
                    )
                )

            # 4. Chronological Ordering Check
            if prev_ts is not None:
                if ts < prev_ts:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.ERROR,
                            message=(
                                f"Out-of-order timestamp at index {idx}: {ts.isoformat()} "
                                f"is earlier than preceding timestamp {prev_ts.isoformat()}"
                            ),
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=ts,
                            record_index=idx,
                            field_name="timestamp",
                            value_observed=f"{ts.isoformat()} < {prev_ts.isoformat()}",
                            expected="timestamp[i] >= timestamp[i-1]",
                        )
                    )

            prev_ts = ts

        return result
