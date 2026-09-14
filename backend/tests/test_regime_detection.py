"""
Comprehensive Market Regime Detection Test Suite (Phase 09).
Verifies zero temporal leakage, train-only preprocessing, rule-based baseline, K-Means, GMM, HMM,
semantic mapping stability, transition matrices, regime duration, reproducibility, and end-to-end integration.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.service import DataValidationService
from backend.app.features.engine import FeatureEngine
from backend.app.features.models import FeatureDataset, FeatureRecord
from backend.app.models.dataset import TimeSeriesSplitter
from backend.app.regime.analysis import RegimeAnalytics
from backend.app.regime.gmm_detector import GMMRegimeDetector
from backend.app.regime.hmm_detector import HMM_AVAILABLE, HMMRegimeDetector
from backend.app.regime.kmeans_detector import KMeansRegimeDetector
from backend.app.regime.preprocessor import RegimeFeaturePreprocessor
from backend.app.regime.rule_based import RuleBasedRegimeDetector
from backend.app.regime.schemas import DetectorType, MarketRegimeType
from backend.app.regime.service import MarketRegimeService
from backend.app.regime.storage import RegimeStorage


def generate_synthetic_regime_bars(
    count: int = 150,
    symbol: str = "SPY",
    seed: int = 42,
) -> list[BarData]:
    """Generate realistic synthetic BarData with market regime shifts."""
    np.random.seed(seed)
    base_time = datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars = []
    price = 100.0

    # 3 distinct periods: Bull low-vol -> Bear high-vol -> Sideways neutral
    for i in range(count):
        if i < count // 3:
            # Bull trending low vol
            ret = np.random.normal(0.0015, 0.008)
        elif i < 2 * (count // 3):
            # Bear trending high vol
            ret = np.random.normal(-0.0020, 0.025)
        else:
            # Sideways normal vol
            ret = np.random.normal(0.0001, 0.012)

        price *= (1.0 + ret)
        open_p = float(round(price * (1.0 + np.random.normal(0, 0.002)), 4))
        close_p = float(round(price, 4))
        spread = float(abs(np.random.normal(0, 0.005)) * price + 0.2)
        high = float(round(max(open_p, close_p) + spread, 4))
        low = float(round(min(open_p, close_p) - spread, 4))
        vol = float(int(np.random.lognormal(14, 0.5)))

        bars.append(
            BarData(
                symbol=symbol,
                timestamp=base_time + timedelta(days=i),
                open=open_p,
                high=high,
                low=low,
                close=close_p,
                volume=vol,
                timeframe=TimeFrame.DAY_1,
            )
        )
    return bars


def test_regime_feature_preprocessor_train_only():
    """Verify preprocessor fits statistics strictly on training partition without future leakage."""
    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ])
    X_test = np.array([
        [100.0, 1000.0],  # Outlier in test set
    ])

    preprocessor = RegimeFeaturePreprocessor(feature_names=["f1", "f2"], scale_features=True)
    X_train_scaled = preprocessor.fit_transform(X_train)

    # Means should be exactly [2.0, 20.0] derived from X_train
    assert np.allclose(preprocessor.scaler.mean_, [2.0, 20.0])

    # Transform test set using train means
    X_test_scaled = preprocessor.transform(X_test)
    assert X_test_scaled.shape == (1, 2)
    # Test sample scaled value must reflect train mean, not test mean
    assert X_test_scaled[0, 0] > 10.0


def test_temporal_ordering_and_zero_leakage():
    """Verify regime feature extraction and temporal splits strictly preserve chronological causality."""
    bars = generate_synthetic_regime_bars(count=90)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    service = MarketRegimeService(
        splitter=TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    )
    split = service.prepare_regime_split(features, bars)

    # 1. Check strict chronological split invariant
    assert max(split.train_timestamps) < min(split.val_timestamps)
    assert max(split.val_timestamps) < min(split.test_timestamps)

    # 2. Check no data shuffling
    assert len(split.X_train_raw) + len(split.X_val_raw) + len(split.X_test_raw) == len(bars)

    # 3. Modifying future bar at t+1 cannot alter past features at t
    bars_mutated = list(bars)
    bars_mutated[-1] = BarData(
        symbol="SPY",
        timestamp=bars[-1].timestamp,
        open=9999.0,
        high=9999.0,
        low=9999.0,
        close=9999.0,
        volume=999999,
    )
    features_mutated = engine.generate_features(bars_mutated)
    # Check that first 80 feature records are bit-for-bit identical
    for i in range(80):
        assert features.records[i].features == features_mutated.records[i].features


def test_rule_based_regime_detector():
    """Verify deterministic RuleBasedRegimeDetector behavior and threshold calibration."""
    bars = generate_synthetic_regime_bars(count=60)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    service = MarketRegimeService()
    split = service.prepare_regime_split(features, bars)

    detector = RuleBasedRegimeDetector(feature_names=split.feature_names)
    preprocessor = RegimeFeaturePreprocessor(feature_names=split.feature_names)
    X_tr = preprocessor.fit_transform(split.X_train_raw)

    detector.fit(X_tr, returns=split.train_returns, raw_features_df=split.X_train_raw)
    assert detector.is_fitted is True
    assert detector.calibrated_vol_threshold is not None

    preds = detector.predict(X_tr)
    probs = detector.predict_proba(X_tr)

    assert len(preds) == len(X_tr)
    assert probs.shape == (len(X_tr), detector.n_regimes)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3})


def test_kmeans_regime_detector_lifecycle(tmp_path):
    """Verify KMeans fitting, semantic mapping from train data, probability calibration, and serialization."""
    bars = generate_synthetic_regime_bars(count=90)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    service = MarketRegimeService()
    split = service.prepare_regime_split(features, bars)

    preprocessor = RegimeFeaturePreprocessor(feature_names=split.feature_names)
    X_tr = preprocessor.fit_transform(split.X_train_raw)
    X_te = preprocessor.transform(split.X_test_raw)

    detector = KMeansRegimeDetector(n_clusters=3, random_state=42, feature_names=split.feature_names)
    detector.fit(X_tr, returns=split.train_returns, raw_features_df=split.X_train_raw)

    # 1. Semantic mapping check
    assert len(detector.semantic_mapping) == 3
    for cid, name in detector.semantic_mapping.items():
        assert isinstance(name, str)
        assert len(name) > 0

    # 2. Prediction and probabilities check
    preds_orig = detector.predict(X_te)
    probs_orig = detector.predict_proba(X_te)
    assert len(preds_orig) == len(X_te)
    assert probs_orig.shape == (len(X_te), 3)
    assert np.allclose(np.sum(probs_orig, axis=1), 1.0)

    # 3. Save & Load roundtrip
    model_dir = tmp_path / "kmeans_test"
    detector.save(model_dir)
    loaded_detector = KMeansRegimeDetector.load(model_dir)

    preds_loaded = loaded_detector.predict(X_te)
    probs_loaded = loaded_detector.predict_proba(X_te)

    np.testing.assert_array_equal(preds_orig, preds_loaded)
    np.testing.assert_allclose(probs_orig, probs_loaded, atol=1e-5)
    assert loaded_detector.semantic_mapping == detector.semantic_mapping


def test_gmm_regime_detector_lifecycle(tmp_path):
    """Verify Gaussian Mixture Model posterior probabilities, BIC/AIC diagnostics, and persistence."""
    bars = generate_synthetic_regime_bars(count=90)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    service = MarketRegimeService()
    split = service.prepare_regime_split(features, bars)

    preprocessor = RegimeFeaturePreprocessor(feature_names=split.feature_names)
    X_tr = preprocessor.fit_transform(split.X_train_raw)
    X_te = preprocessor.transform(split.X_test_raw)

    detector = GMMRegimeDetector(n_components=3, random_state=42, feature_names=split.feature_names)
    detector.fit(X_tr, returns=split.train_returns, raw_features_df=split.X_train_raw)

    # 1. Diagnostics check
    bic = detector.get_bic(X_tr)
    aic = detector.get_aic(X_tr)
    ll = detector.get_log_likelihood(X_tr)
    assert not np.isnan(bic)
    assert not np.isnan(aic)
    assert not np.isnan(ll)

    # 2. Probability check
    probs = detector.predict_proba(X_te)
    assert probs.shape == (len(X_te), 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # 3. Persistence roundtrip
    model_dir = tmp_path / "gmm_test"
    detector.save(model_dir)
    loaded_detector = GMMRegimeDetector.load(model_dir)

    preds_orig = detector.predict(X_te)
    preds_loaded = loaded_detector.predict(X_te)
    np.testing.assert_array_equal(preds_orig, preds_loaded)


@pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn not installed in environment")
def test_hmm_regime_detector_lifecycle(tmp_path):
    """Verify Hidden Markov Model sequence modeling, transition matrix extraction, and persistence."""
    bars = generate_synthetic_regime_bars(count=90)
    engine = FeatureEngine()
    features = engine.generate_features(bars)

    service = MarketRegimeService()
    split = service.prepare_regime_split(features, bars)

    preprocessor = RegimeFeaturePreprocessor(feature_names=split.feature_names)
    X_tr = preprocessor.fit_transform(split.X_train_raw)
    X_te = preprocessor.transform(split.X_test_raw)

    detector = HMMRegimeDetector(n_components=3, random_state=42, feature_names=split.feature_names)
    detector.fit(X_tr, returns=split.train_returns, raw_features_df=split.X_train_raw)

    # 1. Transition matrix check
    transmat = detector.get_transition_matrix()
    assert transmat is not None
    assert transmat.shape == (3, 3)
    assert np.allclose(np.sum(transmat, axis=1), 1.0)

    # 2. Viterbi decode & posterior check
    preds_orig = detector.predict(X_te)
    probs_orig = detector.predict_proba(X_te)
    assert len(preds_orig) == len(X_te)
    assert probs_orig.shape == (len(X_te), 3)
    assert np.allclose(np.sum(probs_orig, axis=1), 1.0)

    # 3. Roundtrip
    model_dir = tmp_path / "hmm_test"
    detector.save(model_dir)
    loaded_detector = HMMRegimeDetector.load(model_dir)

    preds_loaded = loaded_detector.predict(X_te)
    np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_semantic_mapping_stability():
    """Verify semantic mapping is strictly determined by training data and cannot be altered by test observations."""
    X_tr = np.random.randn(50, 4)
    rets_tr = np.array([0.02] * 25 + [-0.02] * 25)

    detector = KMeansRegimeDetector(n_clusters=2, random_state=42)
    detector.fit(X_tr, returns=rets_tr)
    mapping_initial = dict(detector.semantic_mapping)

    # Provide extreme test samples
    X_te = np.random.randn(30, 4) * 100.0
    _ = detector.predict(X_te)
    _ = detector.predict_proba(X_te)

    # Semantic mapping must remain exactly the training mapping
    assert detector.semantic_mapping == mapping_initial


def test_transition_matrix_and_regime_durations():
    """Verify empirical Markov transition matrix row normalization and consecutive run-length duration math."""
    # Sequence: 0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2
    seq = [0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2]
    mapping = {0: "BULL", 1: "BEAR", 2: "SIDEWAYS"}

    trans_mat = RegimeAnalytics.compute_transition_matrix(seq, mapping)
    assert trans_mat.total_transitions == len(seq) - 1
    assert len(trans_mat.matrix) == 3

    # Check each row sums to 1.0
    for row in trans_mat.matrix:
        assert np.isclose(sum(row), 1.0)

    # Check durations:
    # 0 has runs of length 3 and 2 -> mean = 2.5
    # 1 has run of length 2 -> mean = 2.0
    # 2 has run of length 4 -> mean = 4.0
    durations = RegimeAnalytics.calculate_regime_durations(seq)
    assert durations[0] == [3, 2]
    assert durations[1] == [2]
    assert durations[2] == [4]

    rets = np.random.normal(0.001, 0.01, len(seq))
    stats = RegimeAnalytics.compute_regime_statistics(seq, rets, semantic_mapping=mapping)
    assert "BULL" in stats
    assert stats["BULL"].mean_duration_bars == 2.5
    assert stats["SIDEWAYS"].max_duration_bars == 4


def test_reproducibility():
    """Verify deterministic training produces identical regime assignments with identical seed."""
    bars = generate_synthetic_regime_bars(count=60, seed=123)
    engine = FeatureEngine()
    features = engine.generate_features(bars)
    service = MarketRegimeService()
    split = service.prepare_regime_split(features, bars)

    preprocessor = RegimeFeaturePreprocessor(feature_names=split.feature_names)
    X_tr = preprocessor.fit_transform(split.X_train_raw)

    det1 = KMeansRegimeDetector(n_clusters=3, random_state=42)
    det1.fit(X_tr, returns=split.train_returns)

    det2 = KMeansRegimeDetector(n_clusters=3, random_state=42)
    det2.fit(X_tr, returns=split.train_returns)

    preds1 = det1.predict(X_tr)
    preds2 = det2.predict(X_tr)
    np.testing.assert_array_equal(preds1, preds2)
    assert det1.semantic_mapping == det2.semantic_mapping


def test_multi_symbol_isolation():
    """Verify symbol-specific features and bar sequences are processed independently without cross-contamination."""
    aapl_bars = generate_synthetic_regime_bars(count=60, symbol="AAPL", seed=10)
    msft_bars = generate_synthetic_regime_bars(count=60, symbol="MSFT", seed=20)

    engine = FeatureEngine()
    aapl_feats = engine.generate_features(aapl_bars)
    msft_feats = engine.generate_features(msft_bars)

    service = MarketRegimeService()
    aapl_split = service.prepare_regime_split(aapl_feats, aapl_bars)
    msft_split = service.prepare_regime_split(msft_feats, msft_bars)

    assert aapl_split.symbol == "AAPL"
    assert msft_split.symbol == "MSFT"
    assert len(aapl_split.X_train_raw) == len(msft_split.X_train_raw)
    # Different prices must result in different feature vectors
    assert not aapl_split.X_train_raw.equals(msft_split.X_train_raw)


def test_error_handling_and_insufficient_samples():
    """Verify explicit error handling on empty or invalid inputs."""
    detector = KMeansRegimeDetector(n_clusters=5)
    # Less samples than n_clusters must raise ValueError
    with pytest.raises(ValueError, match="Insufficient training samples"):
        detector.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # Calling detect on unfitted detector must raise ValueError
    with pytest.raises(ValueError, match="must be fitted"):
        detector.detect(np.array([[1.0, 2.0]]), [datetime.now(timezone.utc)])


def test_full_pipeline_ingestion_validation_features_regime_storage(tmp_path):
    """End-to-end integration test covering raw data validation -> features -> all 4 regime detectors -> storage."""
    raw_bars = generate_synthetic_regime_bars(count=120)

    # 1. Validation & Cleaning
    val_service = DataValidationService()
    cleaned_bars, val_rep = val_service.validate_and_clean(raw_bars)
    assert val_rep.is_valid is True

    # 2. Feature Generation
    feature_engine = FeatureEngine()
    feature_dataset = feature_engine.generate_features(cleaned_bars)
    assert len(feature_dataset) == len(cleaned_bars)

    # 3. Market Regime Service
    storage = RegimeStorage(base_storage_dir=tmp_path / "regimes")
    regime_service = MarketRegimeService(storage=storage)
    split = regime_service.prepare_regime_split(feature_dataset, cleaned_bars)

    # 4. Fit all detectors
    results = regime_service.fit_and_evaluate_all(split, save_artifacts=True)

    assert len(results) >= 3
    assert "rule_based" in results
    assert "kmeans" in results
    assert "gmm" in results
    if HMM_AVAILABLE:
        assert "hmm" in results

    # 5. Verify comparative reporting
    comp_df = regime_service.compare_detectors(results)
    assert len(comp_df) == len(results)
    assert "Detector" in comp_df.columns
    assert "Silhouette" in comp_df.columns
    assert "Avg Persistence" in comp_df.columns

    # 6. Verify storage artifact existence and loadability
    assert storage.detector_exists(DetectorType.KMEANS, "regime-v1")
    loaded_km, loaded_prep, loaded_meta = storage.load_detector(DetectorType.KMEANS, "regime-v1")
    assert loaded_km.is_fitted is True
    assert loaded_meta.n_regimes == loaded_km.n_regimes
    assert loaded_meta.transition_matrix is not None


# =====================================================================
# DEDICATED LEAKAGE AUDIT REGRESSION TEST SUITE (PHASE 09 AUDIT)
# =====================================================================

def test_leakage_audit_future_mutation_invariance():
    """Test A: Mutating observations strictly after timestamp t has zero effect on features at t."""
    bars = generate_synthetic_regime_bars(count=80, seed=101)
    engine = FeatureEngine()
    features_original = engine.generate_features(bars)

    # Pick evaluation cutoff index t = 50
    t_idx = 50
    feat_at_t_orig = dict(features_original.records[t_idx].features)

    # Mutate all future bars after t (indices 51..79) with drastic values
    bars_mutated = list(bars[:t_idx + 1])
    for j in range(t_idx + 1, len(bars)):
        bars_mutated.append(
            BarData(
                symbol="SPY",
                timestamp=bars[j].timestamp,
                open=99999.0,
                high=100000.0,
                low=99998.0,
                close=99999.0,
                volume=100000000.0,
                timeframe=TimeFrame.DAY_1,
            )
        )

    features_mutated = engine.generate_features(bars_mutated)
    feat_at_t_mutated = dict(features_mutated.records[t_idx].features)

    # Features at t must be bit-for-bit identical
    assert feat_at_t_orig == feat_at_t_mutated


def test_leakage_audit_preprocessing_train_isolation():
    """Test B: Test data distribution cannot alter fitted scaler means or imputer statistics."""
    X_train = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
    ])
    X_test_wild = np.array([
        [10000.0, 50000.0],
        [20000.0, 90000.0],
    ])

    preprocessor = RegimeFeaturePreprocessor(feature_names=["f1", "f2"], scale_features=True)
    preprocessor.fit(X_train)

    train_means_before = np.copy(preprocessor.scaler.mean_)
    train_vars_before = np.copy(preprocessor.scaler.var_)
    assert np.allclose(train_means_before, [20.0, 200.0])

    # Transform test set
    _ = preprocessor.transform(X_test_wild)

    # Assert fitted preprocessor statistics are completely untouched by test transformation
    assert np.array_equal(preprocessor.scaler.mean_, train_means_before)
    assert np.array_equal(preprocessor.scaler.var_, train_vars_before)


def test_leakage_audit_semantic_mapping_train_only():
    """Test C: Out-of-sample data passes cannot alter semantic mapping of any detector."""
    X_train = np.random.randn(60, 4)
    rets_train = np.array([0.02] * 30 + [-0.02] * 30)

    detectors = [
        KMeansRegimeDetector(n_clusters=2, random_state=42),
        GMMRegimeDetector(n_components=2, random_state=42),
    ]
    if HMM_AVAILABLE:
        detectors.append(HMMRegimeDetector(n_components=2, random_state=42))

    for det in detectors:
        det.fit(X_train, returns=rets_train)
        mapping_before = dict(det.semantic_mapping)

        # Predict on extreme out-of-sample test samples
        X_test_wild = np.random.randn(40, 4) * 50.0 + 500.0
        _ = det.predict(X_test_wild)
        _ = det.predict_proba(X_test_wild)

        assert det.semantic_mapping == mapping_before, f"Semantic mapping altered for {det.name}"


def test_leakage_audit_temporal_split_monotonicity():
    """Test D: Monotonic chronological ordering max(train) < min(val) < min(test)."""
    bars = generate_synthetic_regime_bars(count=100)
    engine = FeatureEngine()
    features = engine.generate_features(bars)
    service = MarketRegimeService(
        splitter=TimeSeriesSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    )
    split = service.prepare_regime_split(features, bars)

    assert max(split.train_timestamps) < min(split.val_timestamps)
    assert max(split.val_timestamps) < min(split.test_timestamps)


def test_leakage_audit_no_future_target_columns():
    """Test E: Regime feature list contains zero forward-looking target columns."""
    service = MarketRegimeService()
    forbidden_substrings = ["future", "forward", "target", "label", "t_plus", "lead"]

    for feat in service.default_features:
        for forbidden in forbidden_substrings:
            assert forbidden not in feat.lower(), f"Forbidden target keyword '{forbidden}' found in regime feature '{feat}'"


def test_leakage_audit_train_only_detector_fitting():
    """Test F: Detector internal parameters remain strictly unchanged when inferring on test sets."""
    X_train = np.random.randn(80, 4)
    rets_train = np.random.normal(0.001, 0.01, 80)
    X_test = np.random.randn(30, 4)

    # 1. KMeans Centroid Immutability
    km = KMeansRegimeDetector(n_clusters=3, random_state=42)
    km.fit(X_train, returns=rets_train)
    centers_before = np.copy(km.model.cluster_centers_)
    _ = km.predict(X_test)
    _ = km.predict_proba(X_test)
    np.testing.assert_array_equal(km.model.cluster_centers_, centers_before)

    # 2. GMM Means Immutability
    gmm = GMMRegimeDetector(n_components=3, random_state=42)
    gmm.fit(X_train, returns=rets_train)
    gmm_means_before = np.copy(gmm.model.means_)
    _ = gmm.predict(X_test)
    _ = gmm.predict_proba(X_test)
    np.testing.assert_array_equal(gmm.model.means_, gmm_means_before)

    # 3. HMM Transition Matrix Immutability
    if HMM_AVAILABLE:
        hmm = HMMRegimeDetector(n_components=3, random_state=42)
        hmm.fit(X_train, returns=rets_train)
        trans_before = np.copy(hmm.model.transmat_)
        _ = hmm.predict(X_test)
        _ = hmm.predict_proba(X_test)
        np.testing.assert_array_equal(hmm.model.transmat_, trans_before)


@pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn not installed in environment")
def test_leakage_audit_hmm_sequential_temporal_safety():
    """Test G: HMM parameters are estimated on train sequence only without refitting on test sequence."""
    X_train = np.random.randn(90, 3)
    rets_train = np.random.normal(0.001, 0.01, 90)
    X_test = np.random.randn(30, 3)

    hmm = HMMRegimeDetector(n_components=3, random_state=42)
    hmm.fit(X_train, returns=rets_train)

    train_transmat = np.copy(hmm.model.transmat_)
    train_startprob = np.copy(hmm.model.startprob_)
    train_means = np.copy(hmm.model.means_)

    # Inference on test sequence (Viterbi decoding and posterior state calculation)
    test_states = hmm.predict(X_test)
    test_probs = hmm.predict_proba(X_test)

    assert len(test_states) == len(X_test)
    assert test_probs.shape == (len(X_test), 3)

    # Assert HMM model parameters were NOT updated or refitted
    np.testing.assert_array_equal(hmm.model.transmat_, train_transmat)
    np.testing.assert_array_equal(hmm.model.startprob_, train_startprob)
    np.testing.assert_array_equal(hmm.model.means_, train_means)

