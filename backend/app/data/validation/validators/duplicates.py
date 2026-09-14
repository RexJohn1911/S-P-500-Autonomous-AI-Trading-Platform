"""
Duplicate Timestamp & Record Validator
Detects duplicate observations based on (symbol, timeframe, timestamp) composite key.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator


class DuplicateValidator(BaseValidator):
    @property
    def name(self) -> str:
        return "duplicate_validator"

    @property
    def description(self) -> str:
        return "Identifies duplicate bars sharing identical symbol, timeframe, and timestamp."

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

        seen_keys: Dict[tuple, List[int]] = defaultdict(list)
        for idx, bar in enumerate(bars):
            if not isinstance(bar, BarData):
                continue
            key = (bar.symbol, bar.timeframe.value if isinstance(bar.timeframe, TimeFrame) else str(bar.timeframe), bar.timestamp)
            seen_keys[key].append(idx)

        duplicates_found = 0
        for (sym, tf, ts), indices in seen_keys.items():
            if len(indices) > 1:
                duplicates_found += len(indices) - 1
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Duplicate timestamp detected for {sym} [{tf}] at {ts.isoformat()} across {len(indices)} records (indices: {indices})",
                        symbol=sym,
                        timestamp=ts,
                        record_index=indices[0],
                        value_observed=len(indices),
                        expected="Unique timestamp per symbol/timeframe",
                    )
                )

        return result
