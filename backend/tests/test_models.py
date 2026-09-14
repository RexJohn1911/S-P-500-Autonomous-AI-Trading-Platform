"""
Comprehensive Baseline Machine Learning Test Suite (PHASE 07)
Tests target construction, target leakage prevention, temporal time-series splitting,
train-only preprocessing, 4 baseline models (Logistic Regression, Random Forest, XGBoost, LightGBM),
reproducibility, artifact persistence, trading diagnostics, multi-symbol handling,
and end-to-end integration across Phase 04 -> 05 -> 06 -> 07.
"""

import dataclasses
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import numpy as np
import pytest

from backend.app.data import MockMarketDataProvider
from backend.app.data.models import BarData, TimeFrame, AssetClass
from backend.app.data.service import MarketDataService
from backend.app.data.storage import RawDataStorage
from backend.app.data.validation.service import DataValidationService
from backend.app.data.validation.storage import ProcessedDataStorage
from backend.app.features.engine import FeatureEngine
from backend.app.features.storage import FeatureStorage
from backend.app.models import (
    BaseMLModel,
    DatasetSplit,
    EvaluationMetrics,
    FeaturePreprocessor,
    LightGBMModel,
    LogisticRegressionModel,
    MLDatasetBuilder,
    ModelEvaluator,
    ModelMetadata,
    ModelStorage,
    ModelTrainingService,
    ModelType,
    RandomForestModel,
    TargetConfig,
    TimeSeriesSplitter,
    XGBoostModel,
)


def generate_synthetic_bars(
    symbol: str = "AAPL",
    count: int = 150,
    start_price: float = 100.0,
    daily_trend: float = 0.001,
    timeframe: TimeFrame = TimeFrame.DAY_1,
    start_dt: datetime = None,
) -> list[BarData]:
    """Helper to generate deterministic, realistic BarData sequences."""
    if start_dt is None:
        start_dt = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)

    bars = []
    for i in range(count):
        ts = start_dt + timedelta(days=i)
        variation = math.sin(i * 0.3) * 3.0 + (i * daily_trend * start_price)
        close = round(start_price + variation, 4)
        high = round(close + 1.5, 4)
        low = round(close - 1.5, 4)
        open_price = round(close - 0.2, 4)
        volume = 1000000.0 + (i % 10) * 50000.0

        bar = BarData(
            symbol=symbol,
            timestamp=ts,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timeframe=timeframe,
            asset_class=AssetClass.EQUITY,
        )
        bars.append(bar)
    return bars


# =====================================================================
# 1. Target Construction Tests
# =====================================================================

def test_target_construction_and_horizon():
    """Verify target calculation: (close[t+5] - close[t]) / close[t] > threshold."""
    bars = generate_synthetic_bars(count=60)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    target_cfg = TargetConfig(forward_horizon=5, return_threshold=0.01)
    builder = MLDatasetBuilder(target_config=target_cfg)
    X, y, timestamps, feat_names, fut_rets = builder.build_dataset(features, bars)

    assert len(X) == len(y) == len(timestamps) == len(fut_rets)
    assert len(X) > 0

    # Verify target label matches mathematical formula
    # For any sample i, fut_ret should equal (close[t+5] - close[t]) / close[t]
    bar_map = {b.timestamp: b for b in bars}
    for i in range(min(10, len(timestamps))):
        ts = timestamps[i]
        curr_bar = bar_map[ts]
        # Find index in bars
        idx = bars.index(curr_bar)
        fut_bar = bars[idx + 5]

        expected_ret = (fut_bar.close - curr_bar.close) / curr_bar.close
        assert pytest.approx(fut_rets[i], rel=1e-5) == expected_ret

        expected_label = 1 if expected_ret > 0.01 else 0
        assert y[i] == expected_label


def test_target_excluded_from_features():
    """Verify that target and future returns NEVER enter the feature matrix X."""
    bars = generate_synthetic_bars(count=60)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    builder = MLDatasetBuilder()
    X, y, timestamps, feat_names, fut_rets = builder.build_dataset(features, bars)

    forbidden_names = ["target", "future_return", "label", "future_close", "close_t+5"]
    for fname in feat_names:
        for f in forbidden_names:
            assert f not in fname.lower(), f"Forbidden target column found in feature matrix: {fname}"


# =====================================================================
# 2. Critical Target & Feature Leakage Prevention Test
# =====================================================================

def test_target_leakage_isolation():
    """
    Verify that altering future prices (after t) modifies only future targets y[t],
    while keeping the feature vector X[t] strictly unchanged.
    """
    base_bars = generate_synthetic_bars(count=70)
    engine = FeatureEngine()
    feat_1 = engine.generate_features(base_bars)

    builder = MLDatasetBuilder(target_config=TargetConfig(forward_horizon=5))
    X1, y1, ts1, cols1, rets1 = builder.build_dataset(feat_1, base_bars)

    # Modify future prices from bar 45 onwards
    mod_bars = list(base_bars[:45])
    for i in range(45, len(base_bars)):
        b = base_bars[i]
        new_close = b.close * 2.5
        mod_bars.append(
            dataclasses.replace(
                b,
                close=new_close,
                high=new_close + 2.0,
                low=max(0.1, new_close - 2.0),
            )
        )

    feat_2 = engine.generate_features(mod_bars)
    X2, y2, ts2, cols2, rets2 = builder.build_dataset(feat_2, mod_bars)

    # Features at t < 45 must be IDENTICAL between X1 and X2
    for i in range(len(ts1)):
        if ts1[i] < base_bars[45].timestamp:
            np.testing.assert_allclose(
                X1[i],
                X2[i],
                rtol=1e-7,
                atol=1e-7,
                err_msg=f"Feature leakage detected in X at index {i}",
            )


# =====================================================================
# 3. Time-Aware Dataset Splitter Tests
# =====================================================================

def test_chronological_time_series_split():
    """Verify strict chronological ordering: max(train) < min(val) and max(val) < min(test)."""
    bars = generate_synthetic_bars(count=120)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    builder = MLDatasetBuilder()
    X, y, timestamps, feat_names, fut_rets = builder.build_dataset(features, bars)

    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    split = splitter.split(
        X=X,
        y=y,
        timestamps=timestamps,
        feature_names=feat_names,
        target_config=builder.target_config,
        future_returns=fut_rets,
    )

    assert split.train_size > 0
    assert split.val_size > 0
    assert split.test_size > 0
    assert split.total_size == len(X)

    max_train_ts = max(split.train_timestamps)
    min_val_ts = min(split.val_timestamps)
    max_val_ts = max(split.val_timestamps)
    min_test_ts = min(split.test_timestamps)

    assert max_train_ts < min_val_ts, "Train and Validation timestamps overlap or are not chronological!"
    assert max_val_ts < min_test_ts, "Validation and Test timestamps overlap or are not chronological!"


def test_split_invalid_ratios():
    """Verify splitter rejects ratios not summing to 1.0."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        TimeSeriesSplitter(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


# =====================================================================
# 4. Train-Only Preprocessing Tests
# =====================================================================

def test_train_only_preprocessor_leakage_prevention():
    """Verify preprocessing parameters (mean, scale) are fitted strictly on train data."""
    X_train = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    X_test = np.array([[1000.0, 5000.0]])

    prep = FeaturePreprocessor()
    X_train_scaled = prep.fit_transform(X_train)

    # Scaler mean must match X_train mean exactly
    assert pytest.approx(prep.scaler.mean_[0]) == 20.0
    assert pytest.approx(prep.scaler.mean_[1]) == 200.0

    # Transforming X_test must NOT alter fitted scaler mean
    X_test_scaled = prep.transform(X_test)
    assert pytest.approx(prep.scaler.mean_[0]) == 20.0
    assert pytest.approx(prep.scaler.mean_[1]) == 200.0

    # Test values scaled using train statistics
    assert X_test_scaled[0, 0] > 10.0


# =====================================================================
# 5. Baseline Models Training & Evaluation Tests
# =====================================================================

@pytest.fixture
def sample_dataset_split():
    bars = generate_synthetic_bars(count=120)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    builder = MLDatasetBuilder(target_config=TargetConfig(forward_horizon=5))
    X, y, timestamps, feat_names, fut_rets = builder.build_dataset(features, bars)

    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    return splitter.split(
        X=X,
        y=y,
        timestamps=timestamps,
        feature_names=feat_names,
        target_config=builder.target_config,
        future_returns=fut_rets,
    )


def test_logistic_regression_model(sample_dataset_split):
    """Test Logistic Regression baseline model."""
    split = sample_dataset_split
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(split.X_train)
    X_val = prep.transform(split.X_val)
    X_test = prep.transform(split.X_test)

    model = LogisticRegressionModel(feature_names=split.feature_names)
    model.fit(X_train, split.y_train)

    assert model.is_fitted is True
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    assert len(y_pred) == len(split.y_test)
    assert y_prob.shape == (len(split.y_test), 2)

    metrics = model.evaluate(X_test, split.y_test, split.future_returns_test)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.confusion_matrix is not None

    importances = model.get_feature_importance()
    assert len(importances) == len(split.feature_names)


def test_random_forest_model(sample_dataset_split):
    """Test Random Forest baseline model."""
    split = sample_dataset_split
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(split.X_train)
    X_test = prep.transform(split.X_test)

    model = RandomForestModel(n_estimators=30, max_depth=4, feature_names=split.feature_names)
    model.fit(X_train, split.y_train)

    assert model.is_fitted is True
    metrics = model.evaluate(X_test, split.y_test, split.future_returns_test)
    assert 0.0 <= metrics.accuracy <= 1.0

    importances = model.get_feature_importance()
    assert len(importances) == len(split.feature_names)
    assert all(v >= 0.0 for v in importances.values())


def test_xgboost_model(sample_dataset_split):
    """Test XGBoost baseline model."""
    split = sample_dataset_split
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(split.X_train)
    X_val = prep.transform(split.X_val)
    X_test = prep.transform(split.X_test)

    model = XGBoostModel(n_estimators=30, max_depth=3, feature_names=split.feature_names)
    model.fit(X_train, split.y_train, X_val, split.y_val)

    assert model.is_fitted is True
    metrics = model.evaluate(X_test, split.y_test, split.future_returns_test)
    assert 0.0 <= metrics.accuracy <= 1.0

    importances = model.get_feature_importance()
    assert len(importances) == len(split.feature_names)


def test_lightgbm_model(sample_dataset_split):
    """Test LightGBM baseline model."""
    split = sample_dataset_split
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(split.X_train)
    X_val = prep.transform(split.X_val)
    X_test = prep.transform(split.X_test)

    model = LightGBMModel(n_estimators=30, max_depth=3, feature_names=split.feature_names)
    model.fit(X_train, split.y_train, X_val, split.y_val)

    assert model.is_fitted is True
    metrics = model.evaluate(X_test, split.y_test, split.future_returns_test)
    assert 0.0 <= metrics.accuracy <= 1.0

    importances = model.get_feature_importance()
    assert len(importances) == len(split.feature_names)


# =====================================================================
# 6. Reproducibility & Determinism Tests
# =====================================================================

def test_model_training_reproducibility(sample_dataset_split):
    """Verify that training twice with the same seed produces identical predictions and metrics."""
    split = sample_dataset_split
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(split.X_train)
    X_test = prep.transform(split.X_test)

    m1 = RandomForestModel(n_estimators=20, random_state=42)
    m1.fit(X_train, split.y_train)
    p1 = m1.predict_proba(X_test)

    m2 = RandomForestModel(n_estimators=20, random_state=42)
    m2.fit(X_train, split.y_train)
    p2 = m2.predict_proba(X_test)

    np.testing.assert_allclose(p1, p2, atol=1e-7)


# =====================================================================
# 7. Model Storage, Serialization & Round-Trip Tests
# =====================================================================

def test_model_storage_roundtrip(tmp_path, sample_dataset_split):
    """Verify model artifact saving, loading, and prediction equivalence."""
    split = sample_dataset_split
    storage = ModelStorage(base_storage_dir=tmp_path / "models")

    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(split.X_train)
    X_test = prep.transform(split.X_test)

    model = LogisticRegressionModel(model_version="test-v1", feature_names=split.feature_names)
    model.fit(X_train, split.y_train)

    metrics = model.evaluate(X_test, split.y_test)
    metadata = ModelMetadata(
        model_name="logistic_regression",
        model_type=ModelType.LOGISTIC_REGRESSION,
        model_version="test-v1",
        trained_at=datetime.now(timezone.utc),
        feature_names=split.feature_names,
        target_config=split.target_config,
        hyperparameters=model.hyperparameters,
        test_metrics=metrics,
    )

    # Save
    storage.save_model(model, metadata=metadata, preprocessor=prep)
    assert storage.model_exists(ModelType.LOGISTIC_REGRESSION, "test-v1")

    # Load
    loaded_model, loaded_prep, loaded_meta = storage.load_model(ModelType.LOGISTIC_REGRESSION, "test-v1")
    assert loaded_model.is_fitted is True
    assert loaded_prep.is_fitted is True
    assert loaded_meta.model_name == "logistic_regression"

    # Prediction equivalence
    orig_preds = model.predict(X_test)
    loaded_preds = loaded_model.predict(loaded_prep.transform(split.X_test))
    np.testing.assert_array_equal(orig_preds, loaded_preds)


# =====================================================================
# 8. Multi-Symbol Dataset Building Tests
# =====================================================================

def test_multi_symbol_dataset_builder():
    """Verify combining multiple symbols with symbol isolation and chronological alignment."""
    engine = FeatureEngine()
    bars_aapl = generate_synthetic_bars(symbol="AAPL", count=50, start_price=150.0)
    bars_msft = generate_synthetic_bars(symbol="MSFT", count=50, start_price=300.0)

    feat_aapl = engine.generate_features(bars_aapl)
    feat_msft = engine.generate_features(bars_msft)

    builder = MLDatasetBuilder(target_config=TargetConfig(forward_horizon=3))
    datasets = {
        "AAPL": (feat_aapl, bars_aapl),
        "MSFT": (feat_msft, bars_msft),
    }

    X, y, timestamps, cols, rets = builder.build_multi_symbol_dataset(datasets)
    assert len(X) == len(y) == len(timestamps) == len(rets)
    assert len(X) > 0

    # Ensure combined timestamps are monotonically sorted
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


# =====================================================================
# 9. End-to-End Integration Test (Phase 04 -> Phase 05 -> Phase 06 -> Phase 07)
# =====================================================================

@pytest.mark.asyncio
async def test_full_pipeline_ingest_validate_features_train_models(tmp_path):
    """
    End-to-End Pipeline Integration Test:
    Phase 04: Ingest Market Data via Mock Provider
    Phase 05: Clean & Validate Data
    Phase 06: Compute Quantitative Feature Dataset
    Phase 07: Split, Preprocess, Train all 4 Baseline ML Models, Evaluate, & Persist
    """
    raw_storage = RawDataStorage(base_storage_dir=tmp_path / "raw")
    proc_storage = ProcessedDataStorage(base_storage_dir=tmp_path / "processed")
    feat_storage = FeatureStorage(base_storage_dir=tmp_path / "features")
    model_storage = ModelStorage(base_storage_dir=tmp_path / "models")

    # Step 1: Phase 04 Ingestion
    provider = MockMarketDataProvider()
    market_service = MarketDataService(provider=provider, storage=raw_storage)
    raw_bars = await market_service.get_historical_bars(
        symbol="AAPL",
        timeframe=TimeFrame.DAY_1,
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 8, 1, tzinfo=timezone.utc),
    )
    assert len(raw_bars) > 100

    # Step 2: Phase 05 Cleaning & Validation
    validation_service = DataValidationService(storage=proc_storage)
    cleaned_bars, report = validation_service.validate_and_clean(
        bars=raw_bars,
        expected_symbol="AAPL",
        expected_timeframe=TimeFrame.DAY_1,
        persist_processed=True,
    )
    assert report.is_valid is True
    assert len(cleaned_bars) > 100

    # Step 3: Phase 06 Feature Engineering
    feature_engine = FeatureEngine(storage=feat_storage)
    feature_dataset = feature_engine.generate_and_save(
        bars=cleaned_bars,
        format="parquet",
        save_metadata=True,
    )
    assert len(feature_dataset) == len(cleaned_bars)

    # Step 4: Phase 07 Baseline Model Training Service
    ml_service = ModelTrainingService(storage=model_storage)
    split = ml_service.prepare_dataset_split(feature_dataset, cleaned_bars)

    assert split.train_size > 0
    assert split.val_size > 0
    assert split.test_size > 0

    results = ml_service.train_and_evaluate_all(split, save_artifacts=True)

    assert len(results) == 4
    assert "logistic_regression" in results
    assert "random_forest" in results
    assert "xgboost" in results
    assert "lightgbm" in results

    # Verify all 4 models were persisted
    assert model_storage.model_exists(ModelType.LOGISTIC_REGRESSION)
    assert model_storage.model_exists(ModelType.RANDOM_FOREST)
    assert model_storage.model_exists(ModelType.XGBOOST)
    assert model_storage.model_exists(ModelType.LIGHTGBM)

    # Verify model comparison DataFrame
    comparison_df = ml_service.compare_models(results)
    assert len(comparison_df) == 4
    assert "Accuracy" in comparison_df.columns
    assert "ROC-AUC" in comparison_df.columns
    assert "Spread" in comparison_df.columns
