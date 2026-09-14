"""
Data Validation & Cleaning Service Orchestrator
Coordinates multi-stage data quality validation, safe cleaning, and processed storage.
"""

from typing import List, Optional, Tuple
import logging
from backend.app.config.settings import get_settings
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.models import (
    DataQualityReport,
    DeduplicationPolicy,
    MarketSessionMode,
    ValidationResult,
    ValidationStatus,
)
from backend.app.data.validation.validators.base import BaseValidator
from backend.app.data.validation.validators.schema import SchemaValidator
from backend.app.data.validation.validators.missing import MissingValueValidator
from backend.app.data.validation.validators.duplicates import DuplicateValidator
from backend.app.data.validation.validators.ohlc import OHLCValidator
from backend.app.data.validation.validators.price import PriceValidator
from backend.app.data.validation.validators.volume import VolumeValidator
from backend.app.data.validation.validators.timestamp import TimestampValidator
from backend.app.data.validation.validators.gap import GapValidator
from backend.app.data.validation.validators.market_hours import MarketHoursValidator
from backend.app.data.validation.validators.consistency import ConsistencyValidator
from backend.app.data.validation.validators.outlier import OutlierValidator
from backend.app.data.validation.validators.corporate_actions import CorporateActionAnomalyValidator
from backend.app.data.validation.cleaner import DataCleaner
from backend.app.data.validation.storage import ProcessedDataStorage

logger = logging.getLogger(__name__)


class DataValidationService:
    """
    High-level Data Validation & Cleaning Service.
    Executes modular validation suites, applies deterministic cleaning, and persists trusted data.
    """

    def __init__(
        self,
        validators: Optional[List[BaseValidator]] = None,
        cleaner: Optional[DataCleaner] = None,
        storage: Optional[ProcessedDataStorage] = None,
        max_price_jump_warning_pct: float = 0.35,
        session_mode: MarketSessionMode = MarketSessionMode.REGULAR_HOURS,
        dedup_policy: DeduplicationPolicy = DeduplicationPolicy.KEEP_LAST,
    ):
        settings = get_settings()

        self.storage = storage or ProcessedDataStorage()
        self.cleaner = cleaner or DataCleaner(dedup_policy=dedup_policy)

        if validators is not None:
            self.validators = validators
        else:
            # Default institutional validation suite
            self.validators = [
                SchemaValidator(),
                MissingValueValidator(),
                DuplicateValidator(),
                OHLCValidator(),
                PriceValidator(max_single_bar_change_pct=max_price_jump_warning_pct),
                VolumeValidator(spike_multiplier_threshold=25.0),
                TimestampValidator(),
                GapValidator(max_allowed_daily_gap_days=4),
                MarketHoursValidator(session_mode=session_mode),
                ConsistencyValidator(),
                OutlierValidator(mad_zscore_warning_threshold=6.0, mad_zscore_error_threshold=20.0),
                CorporateActionAnomalyValidator(),
            ]

        logger.info(f"DataValidationService initialized with {len(self.validators)} validators.")

    def validate(
        self,
        bars: List[BarData],
        expected_symbol: Optional[str] = None,
        expected_timeframe: Optional[TimeFrame] = None,
    ) -> DataQualityReport:
        """
        Execute all configured validators over the input bars sequence without altering data.
        """
        symbol = expected_symbol or (bars[0].symbol if bars else "UNKNOWN")
        timeframe = expected_timeframe or (bars[0].timeframe if bars else TimeFrame.DAY_1)

        logger.info(f"Starting data validation for {symbol} [{timeframe.value}] ({len(bars)} bars)...")

        results: List[ValidationResult] = []
        overall_status = ValidationStatus.PASS

        for validator in self.validators:
            res = validator.validate(
                bars,
                expected_symbol=expected_symbol,
                expected_timeframe=expected_timeframe,
            )
            results.append(res)
            if res.status == ValidationStatus.FAIL:
                overall_status = ValidationStatus.FAIL
            elif res.status == ValidationStatus.WARNING and overall_status != ValidationStatus.FAIL:
                overall_status = ValidationStatus.WARNING

        valid_count = len(bars) if overall_status != ValidationStatus.FAIL else 0

        report = DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_records=len(bars),
            valid_records_count=valid_count,
            overall_status=overall_status,
            results=results,
        )

        logger.info(
            f"Validation completed for {symbol} [{timeframe.value}]: Status={overall_status.value} "
            f"(Errors: {report.total_errors}, Warnings: {report.total_warnings})"
        )
        return report

    def clean(
        self,
        bars: List[BarData],
        expected_symbol: Optional[str] = None,
        expected_timeframe: Optional[TimeFrame] = None,
    ) -> Tuple[List[BarData], DataQualityReport]:
        """
        Apply safe automatic cleaning repairs (UTC normalization, deduplication, sorting),
        then run the full validation suite on the cleaned data.
        """
        cleaned_bars, repairs = self.cleaner.clean(
            bars,
            expected_symbol=expected_symbol,
            expected_timeframe=expected_timeframe,
        )

        report = self.validate(
            cleaned_bars,
            expected_symbol=expected_symbol,
            expected_timeframe=expected_timeframe,
        )

        # Inject cleaner repair issues into the report
        if repairs:
            cleaner_res = ValidationResult(
                validator_name="data_cleaner",
                status=ValidationStatus.PASS,
                records_inspected=len(bars),
                repairs_performed=len(repairs),
                issues=repairs,
                description="Safe automatic cleaning repairs applied (sorting, deduplication, normalization).",
            )
            report.results.append(cleaner_res)

        return cleaned_bars, report

    def validate_and_clean(
        self,
        bars: List[BarData],
        expected_symbol: Optional[str] = None,
        expected_timeframe: Optional[TimeFrame] = None,
        persist_processed: bool = True,
        storage_format: str = "parquet",
    ) -> Tuple[List[BarData], DataQualityReport]:
        """
        High-level pipeline: clean -> validate -> persist to data/processed/ (if valid).
        """
        cleaned_bars, report = self.clean(
            bars,
            expected_symbol=expected_symbol,
            expected_timeframe=expected_timeframe,
        )

        if persist_processed and cleaned_bars and report.is_valid:
            try:
                self.storage.save_bars(cleaned_bars, format=storage_format)
                logger.info(
                    f"Persisted {len(cleaned_bars)} validated bars to processed storage in {storage_format} format."
                )
            except Exception as e:
                logger.error(f"Failed to persist cleaned bars to processed storage: {e}")

        return cleaned_bars, report
