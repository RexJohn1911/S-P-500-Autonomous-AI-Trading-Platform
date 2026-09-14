"""
Raw Market Data Storage Manager
Persists and retrieves raw historical market data payloads to disk in structured partitions.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Optional
import logging
import pandas as pd
from backend.app.data.models import BarData, TimeFrame

logger = logging.getLogger(__name__)


class RawDataStorage:
    """
    Manages raw market data file persistence in data/raw/{symbol}/{timeframe}/ format.
    Supports CSV, Parquet, and JSON representations with time-series indexing.
    """

    def __init__(self, base_storage_dir: Optional[Path] = None):
        if base_storage_dir is None:
            # Default to root data/raw
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.base_dir = root_dir / "data" / "raw"
        else:
            self.base_dir = Path(base_storage_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_symbol_dir(self, symbol: str, timeframe: TimeFrame) -> Path:
        clean_symbol = symbol.replace("^", "INDEX_").replace("/", "_")
        target_dir = self.base_dir / clean_symbol / timeframe.value
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def save_bars(self, bars: List[BarData], format: str = "json") -> Path:
        """Save a list of BarData objects to raw data storage."""
        if not bars:
            raise ValueError("Cannot save empty bars list")

        symbol = bars[0].symbol
        timeframe = bars[0].timeframe
        target_dir = self._get_symbol_dir(symbol, timeframe)

        if format == "json":
            file_path = target_dir / "bars.json"
            data = [b.to_dict() for b in bars]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(bars)} raw bars to {file_path}")
            return file_path

        elif format == "csv":
            file_path = target_dir / "bars.csv"
            df = pd.DataFrame([b.to_dict() for b in bars])
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(bars)} raw bars to {file_path}")
            return file_path

        elif format == "parquet":
            file_path = target_dir / "bars.parquet"
            df = pd.DataFrame([b.to_dict() for b in bars])
            df.to_parquet(file_path, index=False)
            logger.info(f"Saved {len(bars)} raw bars to {file_path}")
            return file_path

        else:
            raise ValueError(f"Unsupported storage format: {format}")

    def load_bars(self, symbol: str, timeframe: TimeFrame, format: str = "json") -> List[BarData]:
        """Load stored raw bars for a given symbol and timeframe."""
        target_dir = self._get_symbol_dir(symbol, timeframe)
        
        if format == "json":
            file_path = target_dir / "bars.json"
            if not file_path.exists():
                return []
            with open(file_path, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            return [BarData.from_dict(item) for item in raw_list]

        elif format == "csv":
            file_path = target_dir / "bars.csv"
            if not file_path.exists():
                return []
            df = pd.read_csv(file_path)
            return [BarData.from_dict(row.to_dict()) for _, row in df.iterrows()]

        elif format == "parquet":
            file_path = target_dir / "bars.parquet"
            if not file_path.exists():
                return []
            df = pd.read_parquet(file_path)
            return [BarData.from_dict(row.to_dict()) for _, row in df.iterrows()]

        else:
            raise ValueError(f"Unsupported storage format: {format}")
