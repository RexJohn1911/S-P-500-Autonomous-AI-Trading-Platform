"""
Dataset Consistency & Homogeneity Validator
Ensures that all records in a batch adhere to the specified symbol, timeframe, and asset class.
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


class ConsistencyValidator(BaseValidator):
    @property
    def name(self) -> str:
        return "consistency_validator"

    @property
    def description(self) -> str:
        return "Verifies dataset homogeneity (consistent symbol, timeframe, and metadata across all records)."

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

        if not bars:
            return result

        target_symbol = expected_symbol or bars[0].symbol
        target_tf = expected_timeframe or bars[0].timeframe
        target_ac = bars[0].asset_class

        for idx, bar in enumerate(bars):
            if not isinstance(bar, BarData):
                continue

            # Symbol Mismatch
            if bar.symbol != target_symbol:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Inconsistent symbol '{bar.symbol}' found (expected '{target_symbol}') at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="symbol",
                        value_observed=bar.symbol,
                        expected=target_symbol,
                    )
                )

            # Timeframe Mismatch
            if bar.timeframe != target_tf:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Inconsistent timeframe '{bar.timeframe}' found (expected '{target_tf}') at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="timeframe",
                        value_observed=str(bar.timeframe),
                        expected=str(target_tf),
                    )
                )

            # AssetClass Mismatch
            if bar.asset_class != target_ac:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.WARNING,
                        message=f"Inconsistent asset class '{bar.asset_class}' found (expected '{target_ac}') at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="asset_class",
                        value_observed=str(bar.asset_class),
                        expected=str(target_ac),
                    )
                )

        return result
