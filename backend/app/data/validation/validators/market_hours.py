"""
Market Trading Session Hours Validator
Validates whether timestamps fall within expected US equity market trading sessions.
"""

from datetime import time
from typing import List, Optional
from backend.app.data.models import BarData, TimeFrame, NY_TZ
from backend.app.data.validation.models import (
    MarketSessionMode,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator

# Standard US Equity Market Session Limits (America/New_York)
REGULAR_MARKET_OPEN = time(9, 30)
REGULAR_MARKET_CLOSE = time(16, 0)

EXTENDED_MARKET_OPEN = time(4, 0)
EXTENDED_MARKET_CLOSE = time(20, 0)


class MarketHoursValidator(BaseValidator):
    def __init__(self, session_mode: MarketSessionMode = MarketSessionMode.REGULAR_HOURS):
        self.session_mode = session_mode

    @property
    def name(self) -> str:
        return "market_hours_validator"

    @property
    def description(self) -> str:
        return f"Verifies that timestamps align with US equity trading sessions (mode: {self.session_mode.value})."

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

        if self.session_mode == MarketSessionMode.ALL_HOURS:
            return result

        for idx, bar in enumerate(bars):
            if not isinstance(bar, BarData):
                continue

            # Daily bars represent aggregate sessions, so we primarily check intraday bars
            if bar.timeframe == TimeFrame.DAY_1:
                continue

            ny_time = bar.timestamp.astimezone(NY_TZ)
            bar_time = ny_time.time()
            weekday = ny_time.weekday()  # 0=Monday, 6=Sunday

            # Weekend Check
            if weekday >= 5:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.WARNING,
                        message=f"Intraday bar observed on weekend ({ny_time.strftime('%A')}) at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="timestamp",
                        value_observed=ny_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                        expected="Monday through Friday",
                    )
                )
                continue

            # Regular Hours Session Check
            if self.session_mode == MarketSessionMode.REGULAR_HOURS:
                if bar_time < REGULAR_MARKET_OPEN or bar_time > REGULAR_MARKET_CLOSE:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Timestamp {ny_time.strftime('%H:%M:%S')} is outside regular US market hours "
                                f"(09:30-16:00 ET) at index {idx}"
                            ),
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=bar.timestamp,
                            record_index=idx,
                            field_name="timestamp",
                            value_observed=ny_time.strftime("%H:%M:%S ET"),
                            expected="09:30:00 - 16:00:00 ET",
                        )
                    )

            # Extended Hours Session Check
            elif self.session_mode == MarketSessionMode.EXTENDED_HOURS:
                if bar_time < EXTENDED_MARKET_OPEN or bar_time > EXTENDED_MARKET_CLOSE:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Timestamp {ny_time.strftime('%H:%M:%S')} is outside extended US market hours "
                                f"(04:00-20:00 ET) at index {idx}"
                            ),
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=bar.timestamp,
                            record_index=idx,
                            field_name="timestamp",
                            value_observed=ny_time.strftime("%H:%M:%S ET"),
                            expected="04:00:00 - 20:00:00 ET",
                        )
                    )

        return result
