"""
Schema and Type Validator
Verifies record structure, field presence, and datatype conformity against BarData specification.
"""

from datetime import datetime
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


class SchemaValidator(BaseValidator):
    @property
    def name(self) -> str:
        return "schema_validator"

    @property
    def description(self) -> str:
        return "Validates data types, required fields, and structural integrity of BarData records."

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
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Record at index {idx} is not an instance of BarData (type: {type(bar).__name__})",
                        record_index=idx,
                    )
                )
                continue

            # Symbol check
            if not isinstance(bar.symbol, str) or not bar.symbol.strip():
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid or empty symbol at index {idx}: {bar.symbol}",
                        record_index=idx,
                        field_name="symbol",
                        value_observed=bar.symbol,
                    )
                )

            # Timestamp check
            if not isinstance(bar.timestamp, datetime):
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid timestamp type at index {idx}: {type(bar.timestamp).__name__}",
                        record_index=idx,
                        field_name="timestamp",
                    )
                )

            # Numeric fields check
            for field_name, val in [
                ("open", bar.open),
                ("high", bar.high),
                ("low", bar.low),
                ("close", bar.close),
                ("volume", bar.volume),
            ]:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.ERROR,
                            message=f"Non-finite or non-numeric {field_name} at index {idx}: {val}",
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=bar.timestamp,
                            record_index=idx,
                            field_name=field_name,
                            value_observed=val,
                        )
                    )

            # Timeframe and AssetClass enum checks
            if not isinstance(bar.timeframe, TimeFrame):
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid TimeFrame enum type at index {idx}: {bar.timeframe}",
                        record_index=idx,
                        field_name="timeframe",
                    )
                )

            if not isinstance(bar.asset_class, AssetClass):
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid AssetClass enum type at index {idx}: {bar.asset_class}",
                        record_index=idx,
                        field_name="asset_class",
                    )
                )

        return result
