"""
Machine Learning Dataset Builder and Time-Series Splitter
Constructs predictive target labels, extracts feature matrices, enforces zero target leakage,
and performs chronological train/validation/test partitioning.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.data.models import BarData, TimeFrame, ensure_utc
from backend.app.features.models import FeatureDataset
from backend.app.models.schemas import DatasetSplit, TargetConfig

logger = logging.getLogger(__name__)


class MLDatasetBuilder:
    """
    Constructs ML-ready tabular feature matrices (X) and forward-horizon target labels (y)
    from Phase 06 FeatureDatasets and underlying BarData.
    Strictly isolates future target calculation from feature columns.
    """

    def __init__(
        self,
        target_config: Optional[TargetConfig] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        self.target_config = target_config or TargetConfig()
        self.feature_columns = feature_columns

    def build_dataset(
        self,
        features: FeatureDataset,
        bars: List[BarData],
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime], List[str], np.ndarray]:
        """
        Build aligned (X, y, timestamps, feature_names, future_returns) for a single asset.
        
        Rules:
        1. Feature matrix X at index t uses only features[t].
        2. Target y at index t uses (close[t + horizon] - close[t]) / close[t] > threshold.
        3. Future targets are NEVER appended or mixed into X.
        4. Rows with incomplete warm-up features or missing forward horizon are dropped cleanly.
        """
        if len(features) == 0 or len(bars) == 0:
            raise ValueError("Feature dataset and bars sequence must not be empty.")

        horizon = self.target_config.forward_horizon
        threshold = self.target_config.return_threshold

        # 1. Map bars by timestamp to ensure perfect time alignment
        bar_map: Dict[datetime, BarData] = {b.timestamp: b for b in bars}
        
        # Sort feature records chronologically
        records = sorted(features.records, key=lambda r: r.timestamp)
        n_records = len(records)

        # 2. Determine feature column names (excluding non-predictive identifiers and completely empty features)
        if self.feature_columns is not None:
            active_cols = [c for c in self.feature_columns if c in records[0].features]
        else:
            # Inspect columns and keep only those that have at least some non-null values (>30% of records)
            all_cols = sorted(list(records[0].features.keys()))
            active_cols = []
            for col in all_cols:
                valid_count = sum(1 for r in records if r.features.get(col) is not None and not np.isnan(r.features.get(col)) and not np.isinf(r.features.get(col)))
                if valid_count >= max(5, int(n_records * 0.2)):
                    active_cols.append(col)

        logger.info(f"Building ML dataset for {features.symbol} with {len(active_cols)} features and horizon={horizon}d...")

        valid_X_rows: List[List[float]] = []
        valid_y_rows: List[int] = []
        valid_timestamps: List[datetime] = []
        valid_future_returns: List[float] = []

        # Chronological iteration
        for t_idx in range(n_records):
            rec = records[t_idx]
            ts = rec.timestamp

            # Forward horizon check: must have bar at t + horizon
            if t_idx + horizon >= n_records:
                continue

            current_bar = bar_map.get(ts)
            future_ts = records[t_idx + horizon].timestamp
            future_bar = bar_map.get(future_ts)

            if current_bar is None or future_bar is None:
                continue

            # Calculate future return y
            curr_close = current_bar.close
            fut_close = future_bar.close

            if curr_close <= 0:
                continue

            fut_ret = (fut_close - curr_close) / curr_close
            target_label = 1 if fut_ret > threshold else 0

            # Extract feature vector X[t]
            feat_vals: List[float] = []
            has_missing = False

            for col in active_cols:
                v = rec.features.get(col)
                if v is None or np.isnan(v) or np.isinf(v):
                    has_missing = True
                    break
                feat_vals.append(float(v))

            # Drop row if warm-up period feature is missing
            if has_missing:
                continue

            valid_X_rows.append(feat_vals)
            valid_y_rows.append(target_label)
            valid_timestamps.append(ts)
            valid_future_returns.append(fut_ret)

        if not valid_X_rows:
            raise ValueError(
                f"No complete samples available for {features.symbol} after warm-up and forward horizon alignment."
            )

        X = np.array(valid_X_rows, dtype=np.float64)
        y = np.array(valid_y_rows, dtype=np.int64)
        future_returns = np.array(valid_future_returns, dtype=np.float64)

        logger.info(
            f"Dataset constructed for {features.symbol}: {len(X)} samples, "
            f"positive target ratio = {float(np.mean(y)):.2%}"
        )
        return X, y, valid_timestamps, active_cols, future_returns

    def build_multi_symbol_dataset(
        self,
        datasets: Dict[str, Tuple[FeatureDataset, List[BarData]]],
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime], List[str], np.ndarray]:
        """
        Build combined multi-symbol dataset without cross-sectional leakage.
        Each symbol's targets and features are computed independently before combining in chronological order.
        """
        all_samples: List[Tuple[datetime, List[float], int, float]] = []
        shared_cols: Optional[List[str]] = None

        for sym, (feat_ds, bars) in datasets.items():
            X_sym, y_sym, ts_sym, cols_sym, ret_sym = self.build_dataset(feat_ds, bars)
            if shared_cols is None:
                shared_cols = cols_sym
            elif shared_cols != cols_sym:
                raise ValueError(f"Feature column mismatch across symbols for {sym}")

            for i in range(len(ts_sym)):
                all_samples.append((ts_sym[i], list(X_sym[i]), int(y_sym[i]), float(ret_sym[i])))

        # Sort combined samples chronologically by timestamp
        all_samples.sort(key=lambda s: s[0])

        X = np.array([s[1] for s in all_samples], dtype=np.float64)
        y = np.array([s[2] for s in all_samples], dtype=np.int64)
        timestamps = [s[0] for s in all_samples]
        future_returns = np.array([s[3] for s in all_samples], dtype=np.float64)

        return X, y, timestamps, shared_cols or [], future_returns


class TimeSeriesSplitter:
    """
    Chronological time-aware train/validation/test splitter.
    Guarantees that train < validation < test in time without random shuffle leakage.
    """

    def __init__(
        self,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ):
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Split ratios must sum to 1.0 (got {total:.3f})")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_indices(self, n: int) -> Tuple[int, int]:
        """Compute chronological boundary indices (train_end, val_end) for n observations."""
        if n < 3:
            raise ValueError(f"Insufficient samples for 3-way split (n={n})")
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        # Clamp to ensure non-empty partitions if n >= 3
        train_end = max(1, train_end)
        val_end = max(train_end + 1, min(val_end, n - 1))
        return train_end, val_end

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: List[datetime],
        feature_names: List[str],
        target_config: TargetConfig,
        symbol: str = "MULTI",
        timeframe: TimeFrame = TimeFrame.DAY_1,
        future_returns: Optional[np.ndarray] = None,
    ) -> DatasetSplit:
        """
        Partition dataset into strictly chronological Train, Validation, and Test splits.
        """
        n = len(X)
        if n < 10:
            raise ValueError(f"Dataset too small for 3-way chronological split (n={n})")

        train_end, val_end = self.split_indices(n)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        ts_train = timestamps[:train_end]
        ts_val = timestamps[train_end:val_end]
        ts_test = timestamps[val_end:]

        ret_train = future_returns[:train_end] if future_returns is not None else None
        ret_val = future_returns[train_end:val_end] if future_returns is not None else None
        ret_test = future_returns[val_end:] if future_returns is not None else None

        # Verify temporal monotonicity
        max_train_ts = max(ts_train)
        min_val_ts = min(ts_val)
        max_val_ts = max(ts_val)
        min_test_ts = min(ts_test)

        if max_train_ts > min_val_ts or max_val_ts > min_test_ts:
            raise ValueError("Temporal ordering violated: train/val/test timestamps overlap.")

        logger.info(
            f"Temporal split completed: Train={len(y_train)} ({ts_train[0].date()} to {ts_train[-1].date()}), "
            f"Val={len(y_val)} ({ts_val[0].date()} to {ts_val[-1].date()}), "
            f"Test={len(y_test)} ({ts_test[0].date()} to {ts_test[-1].date()})"
        )

        return DatasetSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_timestamps=ts_train,
            val_timestamps=ts_val,
            test_timestamps=ts_test,
            feature_names=feature_names,
            target_config=target_config,
            symbol=symbol,
            timeframe=timeframe,
            future_returns_train=ret_train,
            future_returns_val=ret_val,
            future_returns_test=ret_test,
        )


class SequenceDatasetBuilder:
    """
    Constructs 3D chronological sequence tensors (samples, sequence_length, features)
    for recurrent (LSTM) and attention-based (Transformer) models.
    Guarantees that sequence[t] contains observations strictly <= t with target y[t].
    """

    def __init__(self, sequence_length: int = 15):
        if sequence_length < 2:
            raise ValueError(f"Sequence length must be at least 2 (got {sequence_length})")
        self.sequence_length = sequence_length

    def build_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: List[datetime],
        future_returns: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime], Optional[np.ndarray]]:
        """
        Transform 2D (N, D) array into 3D (N - L + 1, L, D) temporal sequence tensor.
        """
        n, d = X.shape
        L = self.sequence_length

        if n < L:
            raise ValueError(f"Insufficient samples (n={n}) to construct sequence of length L={L}.")

        n_sequences = n - L + 1
        X_seq = np.zeros((n_sequences, L, d), dtype=np.float64)
        y_seq = np.zeros(n_sequences, dtype=np.int64)
        ts_seq: List[datetime] = []
        ret_seq: Optional[np.ndarray] = np.zeros(n_sequences, dtype=np.float64) if future_returns is not None else None

        for i in range(n_sequences):
            # Window of features from i to i + L (length L)
            X_seq[i] = X[i : i + L]
            # Target is the label at the end of the sequence (timestamp of i + L - 1)
            y_seq[i] = y[i + L - 1]
            ts_seq.append(timestamps[i + L - 1])
            if future_returns is not None and ret_seq is not None:
                ret_seq[i] = future_returns[i + L - 1]

        return X_seq, y_seq, ts_seq, ret_seq

    def build_multi_symbol_sequences(
        self,
        symbol_splits: Dict[str, Tuple[np.ndarray, np.ndarray, List[datetime], Optional[np.ndarray]]],
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime], Optional[np.ndarray]]:
        """
        Build sequences across multiple symbols ensuring that NO sequence spans across different symbols.
        """
        all_X_seq = []
        all_y_seq = []
        all_ts_seq = []
        all_ret_seq = []

        for sym, (X_sym, y_sym, ts_sym, ret_sym) in symbol_splits.items():
            if len(X_sym) >= self.sequence_length:
                X_s, y_s, ts_s, ret_s = self.build_sequences(X_sym, y_sym, ts_sym, ret_sym)
                all_X_seq.append(X_s)
                all_y_seq.append(y_s)
                all_ts_seq.extend(ts_s)
                if ret_s is not None:
                    all_ret_seq.append(ret_s)

        if not all_X_seq:
            raise ValueError("No valid sequences could be formed across provided symbols.")

        combined_X = np.concatenate(all_X_seq, axis=0)
        combined_y = np.concatenate(all_y_seq, axis=0)
        combined_ret = np.concatenate(all_ret_seq, axis=0) if all_ret_seq else None

        # Sort chronologically by timestamp
        sort_indices = np.argsort(all_ts_seq)
        combined_X = combined_X[sort_indices]
        combined_y = combined_y[sort_indices]
        sorted_ts = [all_ts_seq[i] for i in sort_indices]
        if combined_ret is not None:
            combined_ret = combined_ret[sort_indices]

        return combined_X, combined_y, sorted_ts, combined_ret
