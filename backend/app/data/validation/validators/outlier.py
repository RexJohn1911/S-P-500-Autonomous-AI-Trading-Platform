"""
Conservative Statistical Outlier & Anomaly Validator
Utilizes robust statistical metrics (Interquartile Range IQR and Median Absolute Deviation MAD)
to detect price and return anomalies without discarding legitimate volatile market events.
"""

import math
from typing import List, Optional
import numpy as np
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator


class OutlierValidator(BaseValidator):
    def __init__(
        self,
        mad_zscore_warning_threshold: float = 6.0,
        mad_zscore_error_threshold: float = 20.0,
        min_sample_size: int = 15,
    ):
        """
        :param mad_zscore_warning_threshold: Robust Z-score (using MAD) to flag as statistically unusual WARNING.
        :param mad_zscore_error_threshold: Robust Z-score threshold to flag as implausible data ERROR.
        :param min_sample_size: Minimum observations required to perform statistical distribution checks.
        """
        self.mad_zscore_warning_threshold = mad_zscore_warning_threshold
        self.mad_zscore_error_threshold = mad_zscore_error_threshold
        self.min_sample_size = min_sample_size

    @property
    def name(self) -> str:
        return "outlier_validator"

    @property
    def description(self) -> str:
        return "Detects statistical price anomalies using robust median absolute deviation (MAD) scoring."

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

        if len(bars) < self.min_sample_size:
            return result

        closes = [b.close for b in bars if isinstance(b, BarData) and b.close > 0]
        if len(closes) < self.min_sample_size:
            return result

        # Compute log returns
        returns = []
        for i in range(1, len(closes)):
            ret = math.log(closes[i] / closes[i - 1])
            returns.append(ret)

        returns_arr = np.array(returns)
        median_ret = float(np.median(returns_arr))
        
        # To avoid outlier masking on small-to-medium samples, estimate dispersion on trimmed distribution
        sorted_ret = np.sort(returns_arr)
        trim_k = max(1, int(len(sorted_ret) * 0.05))
        core_ret = sorted_ret[trim_k:-trim_k] if len(sorted_ret) > 2 * trim_k else sorted_ret

        core_mad = float(np.median(np.abs(core_ret - np.median(core_ret))))
        core_std = float(np.std(core_ret))

        if core_mad > 1e-4:
            effective_scale = 1.4826 * core_mad
        elif core_std > 1e-4:
            effective_scale = core_std
        else:
            # Baseline minimum volatility scale (50 bps)
            effective_scale = 0.005

        for i, ret in enumerate(returns):
            bar_idx = i + 1  # Corresponds to bars[i+1]
            bar = bars[bar_idx]
            robust_z = abs(ret - median_ret) / effective_scale

            if robust_z >= self.mad_zscore_error_threshold:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Implausible extreme return anomaly (Robust Z-score={robust_z:.1f} >= {self.mad_zscore_error_threshold}, "
                            f"Return={math.exp(ret) - 1:.1%}) at index {bar_idx}"
                        ),
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=bar_idx,
                        field_name="close",
                        value_observed=f"{bar.close} (Z={robust_z:.1f})",
                        expected=f"Robust Z-score < {self.mad_zscore_error_threshold}",
                    )
                )
            elif robust_z >= self.mad_zscore_warning_threshold:
                result.add_issue(
                    ValidationIssue(
                        validator_name=self.name,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Statistically unusual price movement (Robust Z-score={robust_z:.1f}, "
                            f"Return={math.exp(ret) - 1:.1%}) at index {bar_idx}"
                        ),
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        timestamp=bar.timestamp,
                        record_index=bar_idx,
                        field_name="close",
                        value_observed=f"{bar.close} (Z={robust_z:.1f})",
                        expected=f"Robust Z-score < {self.mad_zscore_warning_threshold}",
                    )
                )

        return result
