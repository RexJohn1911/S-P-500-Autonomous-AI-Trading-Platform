"""
Comprehensive Advanced AI Models Test Suite (PHASE 08)
Tests temporal sequence construction, multi-symbol sequence isolation, sequence leakage prevention,
MLP, LSTM, and Transformer neural models, parameter count reporting, training history tracking,
reproducibility, artifact serialization/deserialization roundtrip, and end-to-end integration.
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
    LSTMModel,
    MLDatasetBuilder,
    MLPModel,
    ModelEvaluator,
    ModelMetadata,
    ModelStorage,
    ModelTrainingService,
    ModelType,
    RandomForestModel,
    SequenceDatasetBuilder,
    TargetConfig,
    TimeSeriesSplitter,
    TransformerModel,
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
# 1. Temporal Sequence Dataset Builder Tests
# =====================================================================

def test_sequence_construction_dimensions():
    """Verify sequence builder transforms 2D (N, D) array into 3D (N-L+1, L, D) tensor."""
    X = np.arange(100).reshape(50, 2).astype(np.float64)
    y = np.arange(50).astype(np.int64)
    timestamps = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(50)]
    fut_rets = np.linspace(-0.05, 0.05, 50)

    seq_builder = SequenceDatasetBuilder(sequence_length=10)
    X_seq, y_seq, ts_seq, rets_seq = seq_builder.build_sequences(X, y, timestamps, fut_rets)

    assert X_seq.shape == (41, 10, 2)
    assert len(y_seq) == 41
    assert len(ts_seq) == 41
    assert len(rets_seq) == 41

    # First sequence: observations from index 0 to 9, target at index 9
    np.testing.assert_array_equal(X_seq[0], X[0:10])
    assert y_seq[0] == y[9]
    assert ts_seq[0] == timestamps[9]
    assert rets_seq[0] == fut_rets[9]


def test_sequence_leakage_isolation():
    """Verify modifying observations after t does not alter sequence ending at t."""
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)
    ts = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(50)]

    seq_builder = SequenceDatasetBuilder(sequence_length=5)
    X_seq1, y_seq1, _, _ = seq_builder.build_sequences(X, y, ts)

    # Modify future observations from index 30 onwards
    X_mod = np.copy(X)
    X_mod[30:] *= 10.0

    X_seq2, y_seq2, _, _ = seq_builder.build_sequences(X_mod, y, ts)

    # All sequences ending before index 30 (sequence indices 0..25) must be identical
    for i in range(26):
        np.testing.assert_array_equal(
            X_seq1[i],
            X_seq2[i],
            err_msg=f"Sequence leakage detected at sequence index {i}",
        )


def test_multi_symbol_sequence_isolation():
    """Verify that multi-symbol sequence generation NEVER crosses symbol boundaries."""
    seq_builder = SequenceDatasetBuilder(sequence_length=5)

    X_aapl = np.full((10, 2), 1.0)
    y_aapl = np.zeros(10, dtype=int)
    ts_aapl = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(10)]

    X_msft = np.full((10, 2), 2.0)
    y_msft = np.ones(10, dtype=int)
    ts_msft = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(10)]

    datasets = {
        "AAPL": (X_aapl, y_aapl, ts_aapl, None),
        "MSFT": (X_msft, y_msft, ts_msft, None),
    }

    X_comb, y_comb, ts_comb, _ = seq_builder.build_multi_symbol_sequences(datasets)
    assert len(X_comb) == 12  # 6 from AAPL (10 - 5 + 1) + 6 from MSFT (10 - 5 + 1)

    # Verify every sequence contains PURE AAPL (all 1.0) or PURE MSFT (all 2.0)
    for i in range(len(X_comb)):
        seq = X_comb[i]
        is_pure_aapl = np.all(seq == 1.0)
        is_pure_msft = np.all(seq == 2.0)
        assert is_pure_aapl or is_pure_msft, f"Cross-symbol sequence contamination detected at index {i}!"


# =====================================================================
# 2. MLP Neural Network Model Tests
# =====================================================================

def test_mlp_model_training_and_evaluation():
    """Verify MLP feed-forward neural model construction, training, prediction, and parameter reporting."""
    X_train = np.random.randn(80, 8)
    y_train = np.random.randint(0, 2, 80)
    X_test = np.random.randn(20, 8)
    y_test = np.random.randint(0, 2, 20)

    model = MLPModel(
        hidden_units=[32, 16],
        epochs=5,
        batch_size=16,
        dropout_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train, X_test, y_test)

    assert model.is_fitted is True
    assert model.total_params > 0
    assert model.trainable_params > 0
    assert "loss" in model.training_history
    assert len(model.training_history["loss"]) > 0

    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    assert probs.shape == (20, 2)
    assert len(preds) == 20
    assert np.all((probs[:, 0] + probs[:, 1]) >= 0.99)

    metrics = model.evaluate(X_test, y_test)
    assert 0.0 <= metrics.accuracy <= 1.0


# =====================================================================
# 3. LSTM Recurrent Neural Network Model Tests
# =====================================================================

def test_lstm_model_training_and_evaluation():
    """Verify LSTM sequential model construction, 3D tensor training, and prediction."""
    # 3D Sequence: (samples, seq_len=10, features=6)
    X_train = np.random.randn(60, 10, 6)
    y_train = np.random.randint(0, 2, 60)
    X_test = np.random.randn(20, 10, 6)
    y_test = np.random.randint(0, 2, 20)

    model = LSTMModel(
        sequence_length=10,
        lstm_units=16,
        dense_units=8,
        epochs=5,
        batch_size=16,
        random_state=42,
    )
    model.fit(X_train, y_train, X_test, y_test)

    assert model.is_fitted is True
    assert model.total_params > 0
    assert "loss" in model.training_history

    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    assert probs.shape == (20, 2)
    assert len(preds) == 20

    metrics = model.evaluate(X_test, y_test)
    assert 0.0 <= metrics.accuracy <= 1.0


# =====================================================================
# 4. Transformer Attention Model Tests
# =====================================================================

def test_transformer_model_training_and_evaluation():
    """Verify Transformer self-attention model construction, 3D tensor training, and forward pass."""
    X_train = np.random.randn(60, 10, 6)
    y_train = np.random.randint(0, 2, 60)
    X_test = np.random.randn(20, 10, 6)
    y_test = np.random.randint(0, 2, 20)

    model = TransformerModel(
        sequence_length=10,
        num_heads=2,
        key_dim=8,
        ff_dim=16,
        d_model=16,
        epochs=5,
        batch_size=16,
        random_state=42,
    )
    model.fit(X_train, y_train, X_test, y_test)

    assert model.is_fitted is True
    assert model.total_params > 0
    assert "loss" in model.training_history

    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    assert probs.shape == (20, 2)
    assert len(preds) == 20

    metrics = model.evaluate(X_test, y_test)
    assert 0.0 <= metrics.accuracy <= 1.0


# =====================================================================
# 5. Neural Artifact Serialization & Round-Trip Tests
# =====================================================================

def test_advanced_models_storage_roundtrip(tmp_path):
    """Verify MLP, LSTM, and Transformer serialization, deserialization, and prediction parity."""
    storage = ModelStorage(base_storage_dir=tmp_path / "models")

    # 1. MLP Roundtrip
    X_mlp = np.random.randn(40, 5)
    y_mlp = np.random.randint(0, 2, 40)
    mlp = MLPModel(hidden_units=[16, 8], epochs=3, random_state=42, model_version="test-v1")
    mlp.fit(X_mlp, y_mlp)

    mlp_preds_orig = mlp.predict_proba(X_mlp)
    meta_mlp = ModelMetadata(
        model_name="mlp",
        model_type=ModelType.MLP,
        model_version="test-v1",
        trained_at=datetime.now(timezone.utc),
        feature_names=["f1", "f2", "f3", "f4", "f5"],
        target_config=TargetConfig(),
        hyperparameters=mlp.hyperparameters,
        trainable_parameters=mlp.trainable_params,
        total_parameters=mlp.total_params,
    )
    storage.save_model(mlp, metadata=meta_mlp)
    assert storage.model_exists(ModelType.MLP, "test-v1")

    loaded_mlp, _, loaded_meta = storage.load_model(ModelType.MLP, "test-v1")
    mlp_preds_loaded = loaded_mlp.predict_proba(X_mlp)
    np.testing.assert_allclose(mlp_preds_orig, mlp_preds_loaded, atol=1e-5)
    assert loaded_meta.total_parameters == mlp.total_params

    # 2. LSTM Roundtrip
    X_seq = np.random.randn(30, 8, 4)
    y_seq = np.random.randint(0, 2, 30)
    lstm = LSTMModel(sequence_length=8, lstm_units=12, epochs=3, random_state=42, model_version="test-v1")
    lstm.fit(X_seq, y_seq)

    lstm_preds_orig = lstm.predict_proba(X_seq)
    meta_lstm = ModelMetadata(
        model_name="lstm",
        model_type=ModelType.LSTM,
        model_version="test-v1",
        trained_at=datetime.now(timezone.utc),
        feature_names=["f1", "f2", "f3", "f4"],
        target_config=TargetConfig(),
        hyperparameters=lstm.hyperparameters,
    )
    storage.save_model(lstm, metadata=meta_lstm)
    assert storage.model_exists(ModelType.LSTM, "test-v1")

    loaded_lstm, _, _ = storage.load_model(ModelType.LSTM, "test-v1")
    lstm_preds_loaded = loaded_lstm.predict_proba(X_seq)
    np.testing.assert_allclose(lstm_preds_orig, lstm_preds_loaded, atol=1e-5)

    # 3. Transformer Roundtrip
    transformer = TransformerModel(sequence_length=8, num_heads=2, key_dim=8, d_model=12, epochs=3, random_state=42, model_version="test-v1")
    transformer.fit(X_seq, y_seq)

    tf_preds_orig = transformer.predict_proba(X_seq)
    meta_tf = ModelMetadata(
        model_name="transformer",
        model_type=ModelType.TRANSFORMER,
        model_version="test-v1",
        trained_at=datetime.now(timezone.utc),
        feature_names=["f1", "f2", "f3", "f4"],
        target_config=TargetConfig(),
        hyperparameters=transformer.hyperparameters,
    )
    storage.save_model(transformer, metadata=meta_tf)
    assert storage.model_exists(ModelType.TRANSFORMER, "test-v1")

    loaded_tf, _, _ = storage.load_model(ModelType.TRANSFORMER, "test-v1")
    tf_preds_loaded = loaded_tf.predict_proba(X_seq)
    np.testing.assert_allclose(tf_preds_orig, tf_preds_loaded, atol=1e-5)


# =====================================================================
# 6. End-to-End Pipeline Integration Test (Phase 04 -> Phase 08)
# =====================================================================

@pytest.mark.asyncio
async def test_full_pipeline_all_models_including_advanced(tmp_path):
    """
    End-to-End Pipeline Verification across all 7 model architectures:
    Phase 04: Ingest Market Data
    Phase 05: Clean & Validate Data
    Phase 06: Compute Features
    Phase 07: Split, Preprocess, Train Baselines (Logistic, RF, XGB, LGBM)
    Phase 08: Train Advanced Models (MLP, LSTM, Transformer), Evaluate, & Compare
    """
    raw_storage = RawDataStorage(base_storage_dir=tmp_path / "raw")
    proc_storage = ProcessedDataStorage(base_storage_dir=tmp_path / "processed")
    feat_storage = FeatureStorage(base_storage_dir=tmp_path / "features")
    model_storage = ModelStorage(base_storage_dir=tmp_path / "models")

    # Step 1: Ingestion
    provider = MockMarketDataProvider()
    market_service = MarketDataService(provider=provider, storage=raw_storage)
    raw_bars = await market_service.get_historical_bars(
        symbol="AAPL",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 8, 1, tzinfo=timezone.utc),
        timeframe=TimeFrame.DAY_1,
    )
    assert len(raw_bars) > 100

    # Step 2: Validation & Cleaning
    val_service = DataValidationService(storage=proc_storage)
    cleaned_bars, report = val_service.validate_and_clean(raw_bars, persist_processed=True)
    assert report.is_valid is True

    # Step 3: Feature Engineering
    feature_engine = FeatureEngine(storage=feat_storage)
    feature_dataset = feature_engine.generate_and_save(cleaned_bars, save_metadata=True)

    # Step 4: ML Training Service with all 7 models
    ml_service = ModelTrainingService(storage=model_storage)
    split = ml_service.prepare_dataset_split(feature_dataset, cleaned_bars)

    # Create compact models for fast test execution
    compact_models = [
        LogisticRegressionModel(feature_names=split.feature_names),
        RandomForestModel(n_estimators=10, max_depth=3, feature_names=split.feature_names),
        XGBoostModel(n_estimators=10, max_depth=2, feature_names=split.feature_names),
        LightGBMModel(n_estimators=10, max_depth=2, feature_names=split.feature_names),
        MLPModel(hidden_units=[16, 8], epochs=3, feature_names=split.feature_names),
        LSTMModel(sequence_length=8, lstm_units=12, epochs=3, feature_names=split.feature_names),
        TransformerModel(sequence_length=8, num_heads=2, key_dim=8, d_model=12, epochs=3, feature_names=split.feature_names),
    ]

    results = ml_service.train_and_evaluate_all(split, models=compact_models, save_artifacts=True)

    assert len(results) == 7
    assert "logistic_regression" in results
    assert "random_forest" in results
    assert "xgboost" in results
    assert "lightgbm" in results
    assert "mlp" in results
    assert "lstm" in results
    assert "transformer" in results

    # Verify model comparison DataFrame across all 7 models
    comparison_df = ml_service.compare_models(results)
    assert len(comparison_df) == 7
    assert "Model" in comparison_df.columns
    assert "Accuracy" in comparison_df.columns
    assert "ROC-AUC" in comparison_df.columns

    # Verify advanced model diagnostics table
    advanced_df = ml_service.compare_advanced_models(results)
    assert len(advanced_df) == 7
    assert "Parameters" in advanced_df.columns
    assert "Seq Len" in advanced_df.columns
