"""
Volume Consistency & Anomaly Validator
Validates volume metrics: detects negative volumes, zero volume context, and extreme volume spikes.
"""

import math
from typing import List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator


class VolumeValidator(BaseValidator):
    def __init__(self, spike_multiplier_threshold: float = 25.0, allow_zero_volume: bool = True):
        self.spike_multiplier_threshold = spike_multiplier_threshold
        self.allow_zero_volume = allow_zero_volume

    @property
    def name(self) -> str:
        return "volume_validator"

    @property
    def description(self) -> str:
        return "Checks for negative volume, unexpected zero volume, and extreme volume anomalies."

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

        recent_volumes: List[float] = []
        for idx, bar in enumerate(bars):
            if not isinstance(bar, BarData):
                continue

            v = bar.volume

            # Negative Volume
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

            # Zero volume
            elif v == 0.0 and not self.allow_zero_volume:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.WARNING,
                        message=f"Zero trading volume observed at index {idx}",
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=idx,
                        field_name="volume",
                        value_observed=v,
                    )
                )

            # Volume spike detection (relative to rolling mean of last 20 bars)
            if len(recent_volumes) >= 10:
                avg_vol = sum(recent_volumes) / len(recent_volumes)
                if avg_vol > 0 and v > avg_vol * self.spike_multiplier_threshold:
                    result.add_issue(
                        ValidationIssue(
                            validator_name=self.name,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Extreme volume spike ({v:,.0f} vs rolling avg {avg_vol:,.0f}, "
                                f"{v / avg_vol:.1f}x) at index {idx}"
                            ),
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            timestamp=bar.timestamp,
                            record_index=idx,
                            field_name="volume",
                            value_observed=f"{v} ({v / avg_vol:.1f}x avg)",
                            expected=f"volume <= {self.spike_multiplier_threshold}x rolling avg",
                        )
                    )

            recent_volumes.append(v)
            if len(recent_volumes) > 20:
                recent_volumes.pop(0)

        return result
