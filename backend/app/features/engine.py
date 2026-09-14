"""
Feature Engine Orchestrator
Coordinates quantitative feature calculation, zero-leakage enforcement, metadata tracking, and persistence.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional
import logging
from backend.app.config.settings import get_settings
from backend.app.data.models import BarData, TimeFrame
from backend.app.features.generators.base import BaseFeatureGenerator
from backend.app.features.generators.returns import ReturnFeatureGenerator
from backend.app.features.generators.volatility import VolatilityFeatureGenerator
from backend.app.features.generators.trend import TrendFeatureGenerator
from backend.app.features.generators.volume import VolumeFeatureGenerator
from backend.app.features.generators.market_relative import MarketRelativeFeatureGenerator
from backend.app.features.generators.beta import BetaFeatureGenerator
from backend.app.features.generators.sector import SectorFeatureGenerator
from backend.app.features.generators.regime import RegimeFeaturePlaceholder
from backend.app.features.models import FeatureDataset, FeatureMetadata, FeatureRecord
from backend.app.features.storage import FeatureStorage

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Quantitative Feature Engineering Engine.
    Executes modular feature generators with strict causal (zero future leakage) guarantees.
    """

    def __init__(
        self,
        generators: Optional[List[BaseFeatureGenerator]] = None,
        storage: Optional[FeatureStorage] = None,
    ):
        settings = get_settings()
        self.storage = storage or FeatureStorage()

        if generators is not None:
            self.generators = generators
        else:
            # Default institutional feature generator suite
            return_windows = getattr(settings, "FEATURE_RETURN_WINDOWS", [1, 5, 20])
            vol_window = getattr(settings, "FEATURE_VOLATILITY_WINDOW", 20)
            atr_window = getattr(settings, "FEATURE_ATR_WINDOW", 14)
            trend_windows = getattr(settings, "FEATURE_TREND_WINDOWS", [20, 50, 200])
            volume_window = getattr(settings, "FEATURE_VOLUME_ZSCORE_WINDOW", 20)
            beta_window = getattr(settings, "FEATURE_BETA_WINDOW", 60)
            rel_window = getattr(settings, "FEATURE_RELATIVE_WINDOW", 20)

            self.generators = [
                ReturnFeatureGenerator(windows=return_windows),
                VolatilityFeatureGenerator(vol_window=vol_window, atr_window=atr_window),
                TrendFeatureGenerator(windows=trend_windows),
                VolumeFeatureGenerator(zscore_window=volume_window),
                MarketRelativeFeatureGenerator(window=rel_window),
                BetaFeatureGenerator(window=beta_window),
                SectorFeatureGenerator(window=rel_window),
                RegimeFeaturePlaceholder(),
            ]

        logger.info(f"FeatureEngine initialized with {len(self.generators)} feature generators.")

    def generate_features(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
    ) -> FeatureDataset:
        """
        Generate feature dataset from validated BarData sequence.
        Guarantees that feature calculation at bar i strictly uses data at or before index i.
        """
        if not bars:
            return FeatureDataset(symbol="UNKNOWN", timeframe=TimeFrame.DAY_1, records=[], metadata={})

        symbol = bars[0].symbol
        timeframe = bars[0].timeframe
        n = len(bars)

        logger.info(f"Computing features for {symbol} [{timeframe.value}] ({n} bars)...")

        # 1. Collect all feature columns and metadata
        all_features: Dict[str, List[Optional[float]]] = {}
        metadata_map: Dict[str, FeatureMetadata] = {}

        for generator in self.generators:
            # Generate features
            feature_cols = generator.generate(bars, context=context)
            all_features.update(feature_cols)

            # Collect metadata
            gen_meta = generator.get_metadata(timeframe)
            for m in gen_meta:
                metadata_map[m.feature_name] = m

        # 2. Assemble time-indexed FeatureRecords with non-finite protection
        records: List[FeatureRecord] = []
        for i in range(n):
            rec_features: Dict[str, Optional[float]] = {}
            for col_name, col_values in all_features.items():
                val = col_values[i]
                # Sanitize non-finite values (NaN / Inf) to None
                if val is None or math.isnan(val) or math.isinf(val):
                    rec_features[col_name] = None
                else:
                    rec_features[col_name] = float(val)

            records.append(
                FeatureRecord(
                    symbol=symbol,
                    timestamp=bars[i].timestamp,
                    timeframe=timeframe,
                    features=rec_features,
                )
            )

        dataset = FeatureDataset(
            symbol=symbol,
            timeframe=timeframe,
            records=records,
            metadata=metadata_map,
        )

        logger.info(
            f"Feature generation completed for {symbol} [{timeframe.value}]: "
            f"{len(records)} records, {len(dataset.feature_names)} features."
        )
        return dataset

    def generate_and_save(
        self,
        bars: List[BarData],
        context: Optional[Dict[str, Any]] = None,
        format: str = "parquet",
        save_metadata: bool = True,
    ) -> FeatureDataset:
        """High-level pipeline: generate features and persist to data/features/."""
        dataset = self.generate_features(bars, context=context)
        if dataset and len(dataset) > 0:
            self.storage.save_dataset(dataset, format=format, save_metadata_json=save_metadata)
        return dataset
