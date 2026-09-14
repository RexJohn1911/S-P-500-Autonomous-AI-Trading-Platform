"""
Timeframe Gap & Continuity Validator
Detects unexpected gaps in financial time-series observations while accounting for weekends and market closures.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from backend.app.data.models import BarData, TimeFrame, NY_TZ
from backend.app.data.validation.models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator


class GapValidator(BaseValidator):
    def __init__(self, max_allowed_daily_gap_days: int = 4):
        """
        :param max_allowed_daily_gap_days: Maximum calendar days allowed between daily bars (to allow for 3-day holiday weekends).
        """
        self.max_allowed_daily_gap_days = max_allowed_daily_gap_days

    @property
    def name(self) -> str:
        return "gap_validator"

    @property
    def description(self) -> str:
        return "Detects missing expected periods/bars in time series, accounting for weekends and standard closures."

    def _expected_delta(self, timeframe: TimeFrame) -> timedelta:
        mapping = {
            TimeFrame.MINUTE_1: timedelta(minutes=1),
            TimeFrame.MINUTE_5: timedelta(minutes=5),
            TimeFrame.MINUTE_15: timedelta(minutes=15),
            TimeFrame.HOUR_1: timedelta(hours=1),
            TimeFrame.DAY_1: timedelta(days=1),
        }
        return mapping.get(timeframe, timedelta(days=1))

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

        if len(bars) < 2:
            return result

        timeframe = expected_timeframe or (bars[0].timeframe if isinstance(bars[0].timeframe, TimeFrame) else TimeFrame.DAY_1)
        expected_step = self._expected_delta(timeframe)

        for i in range(1, len(bars)):
            prev_bar = bars[i - 1]
            curr_bar = bars[i]

            diff = curr_bar.timestamp - prev_bar.timestamp
            if diff <= timedelta(0):
                continue  # Out of order or duplicate handled by other validators

            # Daily Bar Gap Handling
            if timeframe == TimeFrame.DAY_1:
                # Normal step is 1 day, Friday to Monday is 3 days. A 4-day weekend (e.g. Easter/Thanksgiving) is 4 days.
                if diff > timedelta(days=self.max_allowed_daily_gap_days):
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Unexpected gap of {diff.days} days in daily series between "
                                f"{prev_bar.timestamp.date()} and {curr_bar.timestamp.date()} at index {i}"
                            ),
                            symbol=curr_bar.symbol,
                            timeframe=timeframe,
                            timestamp=curr_bar.timestamp,
                            record_index=i,
                            value_observed=f"{diff.days} days",
                            expected=f"<= {self.max_allowed_daily_gap_days} days",
                        )
                    )

            # Intraday Bar Gap Handling (1Min, 5Min, 15Min, 1Hour)
            else:
                prev_ny = prev_bar.timestamp.astimezone(NY_TZ)
                curr_ny = curr_bar.timestamp.astimezone(NY_TZ)

                # If on the same calendar day
                if prev_ny.date() == curr_ny.date():
                    # Gap within same trading session
                    if diff > expected_step * 3:
                        result.add_issue(
                            ValidationIssue(
                                validator_name=self.name,
                                severity=ValidationSeverity.WARNING,
                                message=(
                                    f"Intraday gap of {diff} ({diff / expected_step:.0f} missing {timeframe.value} bars) "
                                    f"between {prev_ny.strftime('%H:%M')} and {curr_ny.strftime('%H:%M')} at index {i}"
                                ),
                                symbol=curr_bar.symbol,
                                timeframe=timeframe,
                                timestamp=curr_bar.timestamp,
                                record_index=i,
                                value_observed=str(diff),
                                expected=f"Step size ~ {expected_step}",
                            )
                        )
                else:
                    # Overnight gap: check calendar day jump
                    days_diff = (curr_ny.date() - prev_ny.date()).days
                    if days_diff > self.max_allowed_daily_gap_days:
                        result.add_issue(
                            ValidationIssue(
                                validator_name=self.name,
                                severity=ValidationSeverity.WARNING,
                                message=(
                                    f"Multi-day gap of {days_diff} days in intraday series between "
                                    f"{prev_ny.date()} and {curr_ny.date()} at index {i}"
                                ),
                                symbol=curr_bar.symbol,
                                timeframe=timeframe,
                                timestamp=curr_bar.timestamp,
                                record_index=i,
                                value_observed=f"{days_diff} days",
                            )
                        )

        return result
