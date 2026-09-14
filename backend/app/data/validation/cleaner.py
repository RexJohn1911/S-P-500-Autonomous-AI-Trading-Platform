"""
Data Cleaning & Safe Repair Engine
Applies strict, deterministic, auditable data cleaning rules without fabricating market data.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging
from backend.app.data.models import BarData, TimeFrame, ensure_utc
from backend.app.data.validation.models import (
    DeduplicationPolicy,
    ValidationIssue,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Executes safe, deterministic normalization, deduplication, and sorting on BarData sequences.
    Strictly prohibits inventing prices, fabricating volume, or interpolating missing bars.
    """

    def __init__(
        self,
        dedup_policy: DeduplicationPolicy = DeduplicationPolicy.KEEP_LAST,
        auto_sort: bool = True,
        auto_normalize_utc: bool = True,
    ):
        self.dedup_policy = dedup_policy
        self.auto_sort = auto_sort
        self.auto_normalize_utc = auto_normalize_utc

    def clean(
        self,
        bars: List[BarData],
        expected_symbol: Optional[str] = None,
        expected_timeframe: Optional[TimeFrame] = None,
    ) -> Tuple[List[BarData], List[ValidationIssue]]:
        """
        Cleans the input sequence of BarData and returns the cleaned list plus repair audit log.
        """
        if not bars:
            return [], []

        repairs: List[ValidationIssue] = []
        working_bars: List[BarData] = []

        # Step 1: Normalize Timestamps & Filter Invalid Type Objects
        for idx, bar in enumerate(bars):
            if not isinstance(bar, BarData):
                continue

            # Ensure UTC Normalization
            if self.auto_normalize_utc and (bar.timestamp.tzinfo is None or bar.timestamp.tzinfo != timezone.utc):
                normalized_dt = ensure_utc(bar.timestamp)
                # Reconstruct bar with UTC timestamp
                reconstructed_bar = BarData(
                    symbol=bar.symbol,
                    timestamp=normalized_dt,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap=bar.vwap,
                    trade_count=bar.trade_count,
                    timeframe=bar.timeframe,
                    asset_class=bar.asset_class,
                )
                working_bars.append(reconstructed_bar)
                repairs.append(
                    ValidationIssue(
                        validator_name="data_cleaner",
                        severity=ValidationSeverity.INFO,
                        message=f"Normalized timestamp from {bar.timestamp} to UTC at index {idx}",
                        symbol=bar.symbol,
                        timestamp=normalized_dt,
                        record_index=idx,
                        was_repaired=True,
                        repair_action="UTC_TIMESTAMP_NORMALIZATION",
                    )
                )
            else:
                working_bars.append(bar)

        # Step 2: Deterministic Deduplication
        if self.dedup_policy != DeduplicationPolicy.REJECT_ON_DUPLICATE:
            groups: Dict[tuple, List[Tuple[int, BarData]]] = defaultdict(list)
            for idx, bar in enumerate(working_bars):
                key = (bar.symbol, bar.timeframe.value if isinstance(bar.timeframe, TimeFrame) else str(bar.timeframe), bar.timestamp)
                groups[key].append((idx, bar))

            deduped_bars: List[BarData] = []
            for key, bar_entries in groups.items():
                if len(bar_entries) == 1:
                    deduped_bars.append(bar_entries[0][1])
                else:
                    sym, tf, ts = key
                    # Resolve duplicate based on policy
                    if self.dedup_policy == DeduplicationPolicy.KEEP_FIRST:
                        chosen = bar_entries[0][1]
                    elif self.dedup_policy == DeduplicationPolicy.KEEP_LAST:
                        chosen = bar_entries[-1][1]
                    elif self.dedup_policy == DeduplicationPolicy.KEEP_HIGHER_VOLUME:
                        chosen = max(bar_entries, key=lambda x: x[1].volume)[1]
                    else:
                        chosen = bar_entries[-1][1]

                    deduped_bars.append(chosen)
                    repairs.append(
                        ValidationIssue(
                            validator_name="data_cleaner",
                            severity=ValidationSeverity.INFO,
                            message=f"Deduplicated {len(bar_entries)} conflicting bars for {sym} [{tf}] at {ts.isoformat()} using {self.dedup_policy.value}",
                            symbol=sym,
                            timestamp=ts,
                            value_observed=f"{len(bar_entries)} duplicates",
                            was_repaired=True,
                            repair_action=f"DEDUPLICATION_{self.dedup_policy.value}",
                        )
                    )
            working_bars = deduped_bars

        # Step 3: Deterministic Chronological Sorting
        if self.auto_sort:
            is_unsorted = any(
                working_bars[i].timestamp < working_bars[i - 1].timestamp
                for i in range(1, len(working_bars))
            )
            if is_unsorted:
                working_bars.sort(key=lambda b: b.timestamp)
                repairs.append(
                    ValidationIssue(
                        validator_name="data_cleaner",
                        severity=ValidationSeverity.INFO,
                        message=f"Sorted {len(working_bars)} bars into chronological order",
                        symbol=expected_symbol or (working_bars[0].symbol if working_bars else None),
                        was_repaired=True,
                        repair_action="CHRONOLOGICAL_SORT",
                    )
                )

        return working_bars, repairs
