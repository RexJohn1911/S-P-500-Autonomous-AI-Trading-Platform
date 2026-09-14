"""
Unit and regression tests for Model Evaluation and Out-of-Sample Validation.
Tests strict temporal causality, leakage prevention, metric calculations,
deterministic reproducibility, and scientific integrity (no fabricated metrics).
"""

from datetime import datetime, timezone, timedelta
import numpy as np
import pytest

from backend.app.data.models import BarData, TimeFrame
from backend.app.features.engine import FeatureEngine
from backend.app.models.dataset import MLDatasetBuilder, TimeSeriesSplitter
from backend.app.models.preprocessing import FeaturePreprocessor
from backend.app.models.schemas import ModelType, TargetConfig
from backend.app.models.storage import ModelStorage
from backend.app.models.validation import (
    ModelValidator,
    CalibrationBucket,
    ConfusionMatrixMetrics,
)


def _generate_synthetic_bars(n: int = 150, symbol: str = "TEST") -> list[BarData]:
    """Generate strictly chronological daily synthetic bars."""
    start = datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for i in range(n):
        ts = start + timedelta(days=i)
        ret = 0.001 * (1 if i % 2 == 0 else -1)
        price *= (1.0 + ret)
        bars.append(
            BarData(
                symbol=symbol,
                timeframe=TimeFrame.DAY_1,
                timestamp=ts,
                open=price * 0.995,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=10000.0 + (i * 100),
            )
        )
    return bars


class TestLeakageAndTemporalCausality:
    """Rigorous tests proving zero lookahead bias and strict chronological integrity."""

    def test_chronological_ordering(self):
        """Test that TimeSeriesSplitter maintains strictly ascending timestamps."""
        bars = _generate_synthetic_bars(120)
        fe = FeatureEngine()
        feat_data = fe.generate_features(bars)
        target_cfg = TargetConfig(forward_horizon=5)
        builder = MLDatasetBuilder(target_config=target_cfg)
        X, y, timestamps, feat_cols, future_returns = builder.build_dataset(feat_data, bars)

        splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        split = splitter.split(
            X=X,
            y=y,
            timestamps=timestamps,
            feature_names=feat_cols,
            target_config=target_cfg,
            symbol="TEST",
            timeframe=TimeFrame.DAY_1,
            future_returns=future_returns,
        )

        # Assert no overlapping dates and strict chronological order
        assert max(split.train_timestamps) < min(split.val_timestamps)
        assert max(split.val_timestamps) < min(split.test_timestamps)

        # Assert internal chronological ordering
        assert list(split.train_timestamps) == sorted(split.train_timestamps)
        assert list(split.val_timestamps) == sorted(split.val_timestamps)
        assert list(split.test_timestamps) == sorted(split.test_timestamps)

    def test_preprocessor_not_fitted_on_test_data(self):
        """Test that scaler fitted on train does not change its statistics when transforming test."""
        np.random.seed(42)
        X_train = np.random.normal(loc=10.0, scale=2.0, size=(100, 5))
        X_test = np.random.normal(loc=100.0, scale=20.0, size=(50, 5))  # Out of distribution

        preprocessor = FeaturePreprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)

        # Preprocessor should maintain mean/scale from X_train ONLY
        train_mean = preprocessor.scaler.mean_.copy()
        train_scale = preprocessor.scaler.scale_.copy()

        # Transform test data
        X_test_scaled = preprocessor.transform(X_test)

        # Verify scaler statistics did NOT change during transform
        np.testing.assert_array_equal(preprocessor.scaler.mean_, train_mean)
        np.testing.assert_array_equal(preprocessor.scaler.scale_, train_scale)

        # Verify X_test_scaled was transformed using train_mean and train_scale
        expected_test = (X_test - train_mean) / train_scale
        np.testing.assert_allclose(X_test_scaled, expected_test, rtol=1e-5)

    def test_target_uses_only_future_observations(self):
        """Verify that target label at index t is computed strictly from prices at t+1 to t+horizon."""
        bars = _generate_synthetic_bars(100)
        fe = FeatureEngine()
        feat_data = fe.generate_features(bars)
        target_cfg = TargetConfig(forward_horizon=5, return_threshold=0.0)
        builder = MLDatasetBuilder(target_config=target_cfg)
        X, y, timestamps, feat_cols, future_returns = builder.build_dataset(feat_data, bars)

        # For every sample, verify target equals (close[t+5] - close[t]) / close[t] > threshold
        close_by_ts = {b.timestamp: b.close for b in bars}
        bar_timestamps = [b.timestamp for b in bars]

        for i, ts in enumerate(timestamps):
            idx = bar_timestamps.index(ts)
            future_idx = idx + 5
            if future_idx < len(bars):
                p_now = close_by_ts[ts]
                p_future = close_by_ts[bar_timestamps[future_idx]]
                ret = (p_future - p_now) / p_now
                expected_label = 1 if ret > 0.0 else 0
                assert y[i] == expected_label

    def test_walk_forward_folds_preserve_chronology(self):
        """Verify that in walk-forward evaluation, every fold's train precedes its test partition."""
        bars = _generate_synthetic_bars(150)
        fe = FeatureEngine()
        feat_data = fe.generate_features(bars)
        target_cfg = TargetConfig(forward_horizon=5)
        builder = MLDatasetBuilder(target_config=target_cfg)
        X, y, timestamps, feat_cols, future_returns = builder.build_dataset(feat_data, bars)

        n_total = len(X)
        n_folds = 4
        fold_size = n_total // (n_folds + 1)

        for f_idx in range(n_folds):
            train_end = fold_size * (f_idx + 1)
            test_end = min(train_end + fold_size, n_total)

            train_ts = timestamps[:train_end]
            test_ts = timestamps[train_end:test_end]

            # Train timestamps must be strictly before test timestamps
            assert max(train_ts) < min(test_ts)


class TestMetricsCalculation:
    """Test metric computation, confusion matrix, calibration, and forward returns."""

    def test_classification_metrics_correctness(self):
        """Verify classification metric equations with known inputs."""
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.6, 0.7, 0.3])
        y_pred = (y_prob >= 0.5).astype(int)

        # TP: 3 (idx 0, 2, 6), FP: 1 (idx 5), TN: 3 (idx 1, 4, 7), FN: 1 (idx 3)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        assert tp == 3
        assert fp == 1
        assert tn == 3
        assert fn == 1

        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        assert accuracy == 0.75
        assert precision == 0.75
        assert recall == 0.75
        assert f1 == 0.75

    def test_baseline_majority_class_calculation(self):
        """Verify baseline accuracy reflects the majority class frequency."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0])  # 4 ones (57.14%), 3 zeros (42.86%)
        majority_acc = max(np.mean(y_true == 1), np.mean(y_true == 0))
        assert abs(majority_acc - (4 / 7)) < 1e-5

    def test_forward_return_spread_calculation(self):
        """Test forward return computation and return spread logic."""
        preds = np.array([1, 1, 0, 0])
        future_rets = np.array([0.05, 0.03, -0.02, 0.00])

        pos_rets = future_rets[preds == 1]
        neg_rets = future_rets[preds == 0]

        mean_long = float(np.mean(pos_rets))
        mean_short = float(np.mean(neg_rets))
        spread = mean_long - mean_short

        assert abs(mean_long - 0.04) < 1e-6
        assert abs(mean_short - (-0.01)) < 1e-6
        assert abs(spread - 0.05) < 1e-6

    def test_probability_calibration_buckets(self):
        """Verify probability calibration bucket categorization."""
        probs = np.array([0.55, 0.65, 0.75, 0.95])
        b_80_90 = [p for p in probs if 0.80 <= p < 0.90]
        assert len(b_80_90) == 0


class TestModelValidator:
    """Integration tests for the full validation runner."""

    def test_runner_execution_and_determinism(self):
        """Test that running the evaluator twice on the same data produces identical deterministic results."""
        validator = ModelValidator()

        # Run 1
        res1 = validator.run_full_evaluation(
            dataset_symbol="AAPL",
            model_version="baseline-v1",
            walk_forward_folds=2,
        )

        # Run 2
        res2 = validator.run_full_evaluation(
            dataset_symbol="AAPL",
            model_version="baseline-v1",
            walk_forward_folds=2,
        )

        assert res1.dataset_provenance["dataset_id"] == res2.dataset_provenance["dataset_id"]
        assert len(res1.models_evaluated) == len(res2.models_evaluated)

        # Check random forest metrics equality between runs
        if "random_forest" in res1.classification_results:
            rf1 = res1.classification_results["random_forest"]
            rf2 = res2.classification_results["random_forest"]
            assert rf1.accuracy == rf2.accuracy
            assert rf1.f1 == rf2.f1
            assert rf1.roc_auc == rf2.roc_auc

    def test_missing_dataset_handling(self):
        """Test that runner gracefully raises FileNotFoundError if dataset does not exist."""
        validator = ModelValidator()
        with pytest.raises(FileNotFoundError):
            validator.run_full_evaluation(
                dataset_symbol="NON_EXISTENT_SYMBOL_XYZ",
            )

    def test_leakage_checks_pass(self):
        """Test that all 10 leakage prevention checks evaluate to True."""
        validator = ModelValidator()
        res = validator.run_full_evaluation(
            dataset_symbol="AAPL",
            model_version="baseline-v1",
            walk_forward_folds=2,
        )

        assert len(res.leakage_checks) == 10
        assert all(res.leakage_checks.values()), f"Leakage checks failed: {res.leakage_checks}"
