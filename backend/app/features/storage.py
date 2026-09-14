"""
Feature Dataset Storage Manager
Persists and retrieves computed feature datasets and feature metadata under data/features/{symbol}/{timeframe}/.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import pandas as pd
from backend.app.data.models import TimeFrame
from backend.app.features.models import FeatureDataset, FeatureMetadata

logger = logging.getLogger(__name__)


class FeatureStorage:
    """
    Manages quantitative feature file persistence under data/features/{symbol}/{timeframe}/.
    Preserves immutability of raw and processed market data.
    """

    def __init__(
        self,
        base_storage_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
    ):
        target_base = base_storage_dir or base_dir
        if target_base is None:
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.base_dir = root_dir / "data" / "features"
        else:
            self.base_dir = Path(target_base)

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_symbol_dir(self, symbol: str, timeframe: TimeFrame) -> Path:
        clean_symbol = symbol.replace("^", "INDEX_").replace("/", "_")
        target_dir = self.base_dir / clean_symbol / timeframe.value
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def get_metadata_path(self, symbol: str, timeframe: TimeFrame) -> Path:
        """Return the path to feature_metadata.json for given symbol and timeframe."""
        return self._get_symbol_dir(symbol, timeframe) / "feature_metadata.json"

    def feature_exists(
        self,
        symbol: str,
        timeframe: TimeFrame,
        format: str = "parquet",
    ) -> bool:
        """Check whether features file exists on disk."""
        target_dir = self._get_symbol_dir(symbol, timeframe)
        file_name = "features.parquet" if format == "parquet" else "features.csv"
        return (target_dir / file_name).exists()

    def save_dataset(
        self,
        dataset: FeatureDataset,
        format: str = "parquet",
        save_metadata_json: bool = True,
    ) -> Path:
        """Save a FeatureDataset to disk in Parquet or CSV format along with metadata JSON."""
        target_dir = self._get_symbol_dir(dataset.symbol, dataset.timeframe)
        df = dataset.to_dataframe()

        if format == "parquet":
            file_path = target_dir / "features.parquet"
            df.to_parquet(file_path, index=False)
        elif format == "csv":
            file_path = target_dir / "features.csv"
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported feature storage format: {format}")

        logger.info(f"Saved {len(dataset)} feature records for {dataset.symbol} to {file_path}")

        if save_metadata_json and dataset.metadata:
            meta_path = target_dir / "feature_metadata.json"
            meta_dict = {k: v.to_dict() for k, v in dataset.metadata.items()}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)
            logger.info(f"Saved feature metadata to {meta_path}")

        return file_path

    def load_dataframe(
        self,
        symbol: str,
        timeframe: TimeFrame,
        format: str = "parquet",
    ) -> Optional[pd.DataFrame]:
        """Load feature dataset directly into a pandas DataFrame."""
        target_dir = self._get_symbol_dir(symbol, timeframe)
        file_name = "features.parquet" if format == "parquet" else "features.csv"
        file_path = target_dir / file_name
        if not file_path.exists():
            return None
        if format == "parquet":
            return pd.read_parquet(file_path)
        elif format == "csv":
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_metadata(
        self,
        symbol: str,
        timeframe: TimeFrame,
    ) -> Dict[str, Any]:
        """Load feature metadata dictionary from disk."""
        meta_path = self.get_metadata_path(symbol, timeframe)
        if not meta_path.exists():
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dataset(
        self,
        symbol: str,
        timeframe: TimeFrame,
        format: str = "parquet",
    ) -> Optional[FeatureDataset]:
        """Load a persisted FeatureDataset and its associated metadata."""
        target_dir = self._get_symbol_dir(symbol, timeframe)

        metadata: Dict[str, FeatureMetadata] = {}
        meta_path = target_dir / "feature_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
            metadata = {k: FeatureMetadata.from_dict(v) for k, v in raw_meta.items()}

        if format == "parquet":
            file_path = target_dir / "features.parquet"
            if not file_path.exists():
                return None
            df = pd.read_parquet(file_path)
            return FeatureDataset.from_dataframe(df, symbol=symbol, timeframe=timeframe, metadata=metadata)
        elif format == "csv":
            file_path = target_dir / "features.csv"
            if not file_path.exists():
                return None
            df = pd.read_csv(file_path)
            return FeatureDataset.from_dataframe(df, symbol=symbol, timeframe=timeframe, metadata=metadata)
        else:
            raise ValueError(f"Unsupported feature storage format: {format}")
