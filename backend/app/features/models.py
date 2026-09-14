"""
Feature Engineering Domain Models & Schemas
Provides structured, reproducible, and metadata-rich entities for quantitative feature datasets.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from backend.app.data.models import TimeFrame, ensure_utc


@dataclass(frozen=True)
class FeatureMetadata:
    """Detailed metadata describing an individual computed feature."""
    feature_name: str
    description: str
    formula: str
    window: Optional[int]
    source_columns: List[str]
    timeframe: TimeFrame
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "description": self.description,
            "formula": self.formula,
            "window": self.window,
            "source_columns": self.source_columns,
            "timeframe": self.timeframe.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMetadata":
        created = data.get("created_at")
        if isinstance(created, str):
            created_dt = datetime.fromisoformat(created)
        else:
            created_dt = datetime.now(timezone.utc)
        return cls(
            feature_name=data["feature_name"],
            description=data["description"],
            formula=data["formula"],
            window=data.get("window"),
            source_columns=data.get("source_columns", ["close"]),
            timeframe=TimeFrame(data.get("timeframe", TimeFrame.DAY_1.value)),
            version=data.get("version", "1.0.0"),
            created_at=ensure_utc(created_dt),
        )


@dataclass(frozen=True)
class FeatureRecord:
    """Individual time-indexed feature observation for an asset."""
    symbol: str
    timestamp: datetime
    timeframe: TimeFrame
    features: Dict[str, Optional[float]]

    def __post_init__(self):
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe.value,
            **self.features,
        }


@dataclass
class FeatureDataset:
    """Complete multi-feature time-series dataset with metadata."""
    symbol: str
    timeframe: TimeFrame
    records: List[FeatureRecord]
    metadata: Dict[str, FeatureMetadata] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def feature_names(self) -> List[str]:
        """List of feature column names in this dataset."""
        if not self.records:
            return list(self.metadata.keys())
        return list(self.records[0].features.keys())

    def __len__(self) -> int:
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert feature records into a Pandas DataFrame."""
        if not self.records:
            return pd.DataFrame()
        
        rows = [r.to_dict() for r in self.records]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        timeframe: TimeFrame,
        metadata: Optional[Dict[str, FeatureMetadata]] = None,
    ) -> "FeatureDataset":
        """Construct FeatureDataset from a Pandas DataFrame."""
        if df.empty:
            return cls(symbol=symbol, timeframe=timeframe, records=[], metadata=metadata or {})

        meta = metadata or {}
        non_feature_cols = {"symbol", "timestamp", "timeframe"}
        feature_cols = [c for c in df.columns if c not in non_feature_cols]

        records: List[FeatureRecord] = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts_dt = datetime.fromisoformat(ts)
            elif isinstance(ts, pd.Timestamp):
                ts_dt = ts.to_pydatetime()
            else:
                ts_dt = ts
            
            features_dict = {}
            for col in feature_cols:
                val = row[col]
                if pd.isna(val) or val is None:
                    features_dict[col] = None
                else:
                    features_dict[col] = float(val)

            records.append(
                FeatureRecord(
                    symbol=symbol,
                    timestamp=ensure_utc(ts_dt),
                    timeframe=timeframe,
                    features=features_dict,
                )
            )

        return cls(symbol=symbol, timeframe=timeframe, records=records, metadata=meta)
