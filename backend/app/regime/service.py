"""
Market Regime Detection Orchestration Service.
Coordinates feature extraction, temporal chronological splitting, train-only preprocessing,
multi-detector fitting, semantic mapping, transition dynamics, and comparative reporting.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.config.settings import get_settings
from backend.app.data.models import BarData
from backend.app.features.models import FeatureDataset
from backend.app.models.dataset import TimeSeriesSplitter
from backend.app.regime.analysis import RegimeAnalytics
from backend.app.regime.base import BaseRegimeDetector
from backend.app.regime.gmm_detector import GMMRegimeDetector
from backend.app.regime.hmm_detector import HMM_AVAILABLE, HMMRegimeDetector
from backend.app.regime.kmeans_detector import KMeansRegimeDetector
from backend.app.regime.preprocessor import RegimeFeaturePreprocessor
from backend.app.regime.rule_based import RuleBasedRegimeDetector
from backend.app.regime.schemas import (
    ClusteringDiagnostics,
    DetectorType,
    MarketRegimeState,
    RegimeDetectorMetadata,
    RegimeStatistics,
    TransitionMatrix,
)
from backend.app.regime.storage import RegimeStorage

logger = logging.getLogger(__name__)


@dataclass
class RegimeDatasetSplit:
    """Chronologically partitioned feature matrices and bar data for regime detection."""
    symbol: str
    feature_names: List[str]
    X_train_raw: pd.DataFrame
    X_val_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    train_timestamps: List[datetime]
    val_timestamps: List[datetime]
    test_timestamps: List[datetime]
    train_returns: np.ndarray
    val_returns: np.ndarray
    test_returns: np.ndarray
    train_volatility: Optional[np.ndarray] = None
    val_volatility: Optional[np.ndarray] = None
    test_volatility: Optional[np.ndarray] = None
    train_vix: Optional[np.ndarray] = None
    val_vix: Optional[np.ndarray] = None
    test_vix: Optional[np.ndarray] = None


class MarketRegimeService:
    """
    High-level orchestration service for the Market Regime Detection subsystem.
    Executes leakage-safe training, semantic mapping, diagnostics, and comparative evaluation.
    """

    def __init__(
        self,
        storage: Optional[RegimeStorage] = None,
        splitter: Optional[TimeSeriesSplitter] = None,
    ):
        settings = get_settings()
        self.storage = storage or RegimeStorage()

        train_r = getattr(settings, "ML_TRAIN_RATIO", 0.70)
        val_r = getattr(settings, "ML_VALIDATION_RATIO", 0.15)
        test_r = getattr(settings, "ML_TEST_RATIO", 0.15)

        self.splitter = splitter or TimeSeriesSplitter(train_ratio=train_r, val_ratio=val_r, test_ratio=test_r)
        self.default_features = getattr(
            settings,
            "REGIME_FEATURES",
            [
                "return_20d",
                "rolling_volatility_20d",
                "price_vs_sma_50",
                "price_vs_sma_200",
                "vix_level",
                "vix_change_5d",
            ],
        )
        self.default_n_clusters = getattr(settings, "REGIME_N_CLUSTERS", 3)
        self.default_seed = getattr(settings, "REGIME_RANDOM_SEED", 42)
        self.default_version = getattr(settings, "REGIME_VERSION", "regime-v1")
        self.hmm_enabled = getattr(settings, "REGIME_HMM_ENABLED", True)

    def prepare_regime_split(
        self,
        features: FeatureDataset,
        bars: List[BarData],
        feature_names: Optional[List[str]] = None,
    ) -> RegimeDatasetSplit:
        """
        Extract regime features and partition chronologically into Train, Validation, and Test.
        """
        selected_feats = feature_names or self.default_features
        preprocessor = RegimeFeaturePreprocessor(feature_names=selected_feats)
        df, timestamps = preprocessor.extract_features_df(features)

        if len(df) == 0 or len(bars) == 0:
            raise ValueError("Cannot prepare regime split from empty feature dataset or bar sequence.")

        # Align close price returns for descriptive statistics
        bar_dict = {b.timestamp: b for b in bars}
        aligned_bars = [bar_dict.get(ts) for ts in timestamps]

        rets = []
        for i in range(len(aligned_bars)):
            if i == 0 or aligned_bars[i] is None or aligned_bars[i - 1] is None:
                rets.append(0.0)
            else:
                prev_c = aligned_bars[i - 1].close
                curr_c = aligned_bars[i].close
                rets.append((curr_c - prev_c) / prev_c if prev_c > 0 else 0.0)
        returns_arr = np.asarray(rets, dtype=float)

        # Partition indices chronologically
        n = len(df)
        train_end, val_end = self.splitter.split_indices(n)

        X_tr = df.iloc[:train_end].copy()
        X_va = df.iloc[train_end:val_end].copy()
        X_te = df.iloc[val_end:].copy()

        ts_tr = timestamps[:train_end]
        ts_va = timestamps[train_end:val_end]
        ts_te = timestamps[val_end:]

        ret_tr = returns_arr[:train_end]
        ret_va = returns_arr[train_end:val_end]
        ret_te = returns_arr[val_end:]

        vol_tr = X_tr["rolling_volatility_20d"].values if "rolling_volatility_20d" in X_tr.columns else None
        vol_va = X_va["rolling_volatility_20d"].values if "rolling_volatility_20d" in X_va.columns else None
        vol_te = X_te["rolling_volatility_20d"].values if "rolling_volatility_20d" in X_te.columns else None

        vix_tr = X_tr["vix_level"].values if "vix_level" in X_tr.columns else None
        vix_va = X_va["vix_level"].values if "vix_level" in X_va.columns else None
        vix_te = X_te["vix_level"].values if "vix_level" in X_te.columns else None

        return RegimeDatasetSplit(
            symbol=features.symbol,
            feature_names=selected_feats,
            X_train_raw=X_tr,
            X_val_raw=X_va,
            X_test_raw=X_te,
            train_timestamps=ts_tr,
            val_timestamps=ts_va,
            test_timestamps=ts_te,
            train_returns=ret_tr,
            val_returns=ret_va,
            test_returns=ret_te,
            train_volatility=vol_tr,
            val_volatility=vol_va,
            test_volatility=vol_te,
            train_vix=vix_tr,
            val_vix=vix_va,
            test_vix=vix_te,
        )

    def instantiate_all_detectors(
        self,
        feature_names: List[str],
        n_regimes: Optional[int] = None,
        version: Optional[str] = None,
    ) -> List[BaseRegimeDetector]:
        """Instantiate the complete suite of regime detectors."""
        k = n_regimes or self.default_n_clusters
        ver = version or self.default_version
        seed = self.default_seed

        detectors: List[BaseRegimeDetector] = [
            RuleBasedRegimeDetector(
                feature_names=feature_names,
                detector_version=ver,
            ),
            KMeansRegimeDetector(
                n_clusters=k,
                random_state=seed,
                detector_version=ver,
                feature_names=feature_names,
            ),
            GMMRegimeDetector(
                n_components=k,
                covariance_type="full",
                random_state=seed,
                detector_version=ver,
                feature_names=feature_names,
            ),
        ]

        if self.hmm_enabled and HMM_AVAILABLE:
            detectors.append(
                HMMRegimeDetector(
                    n_components=k,
                    covariance_type="diag",
                    random_state=seed,
                    detector_version=ver,
                    feature_names=feature_names,
                )
            )

        return detectors

    def fit_and_evaluate_all(
        self,
        split: RegimeDatasetSplit,
        detectors: Optional[List[BaseRegimeDetector]] = None,
        save_artifacts: bool = True,
    ) -> Dict[str, Tuple[BaseRegimeDetector, RegimeDetectorMetadata, TransitionMatrix, Dict[str, RegimeStatistics]]]:
        """
        Execute end-to-end regime discovery and validation pipeline:
        1. Fit preprocessor strictly on X_train.
        2. Transform Train, Val, Test.
        3. Fit detectors on Train set and learn data-driven semantic mappings.
        4. Calculate clustering diagnostics on Train, Val, Test.
        5. Calculate empirical transition matrices and duration stats on Test set.
        6. Persist artifacts and metadata.
        """
        logger.info(f"Starting regime fitting pipeline for {split.symbol} ({len(split.X_train_raw)} train samples)...")

        # 1. Preprocessing (Strictly Train-Only)
        preprocessor = RegimeFeaturePreprocessor(feature_names=split.feature_names)
        X_tr_scaled = preprocessor.fit_transform(split.X_train_raw)
        X_va_scaled = preprocessor.transform(split.X_val_raw)
        X_te_scaled = preprocessor.transform(split.X_test_raw)

        if detectors is None:
            detectors = self.instantiate_all_detectors(split.feature_names)

        results = {}

        for detector in detectors:
            logger.info(f"Fitting regime detector: {detector.name} [{detector.detector_version}]...")

            # 2. Fit on training partition
            detector.fit(
                X=X_tr_scaled,
                returns=split.train_returns,
                timestamps=split.train_timestamps,
                raw_features_df=split.X_train_raw,
            )

            # 3. Predict states across partitions
            train_labels = detector.predict(X_tr_scaled)
            val_labels = detector.predict(X_va_scaled) if len(X_va_scaled) > 0 else np.empty(0, dtype=int)
            test_labels = detector.predict(X_te_scaled) if len(X_te_scaled) > 0 else np.empty(0, dtype=int)

            # 4. Compute clustering diagnostics
            bic_tr = detector.get_bic(X_tr_scaled) if hasattr(detector, "get_bic") else None
            aic_tr = detector.get_aic(X_tr_scaled) if hasattr(detector, "get_aic") else None
            ll_tr = detector.get_log_likelihood(X_tr_scaled) if hasattr(detector, "get_log_likelihood") else None

            bic_va = detector.get_bic(X_va_scaled) if hasattr(detector, "get_bic") and len(X_va_scaled) > 0 else None
            aic_va = detector.get_aic(X_va_scaled) if hasattr(detector, "get_aic") and len(X_va_scaled) > 0 else None
            ll_va = detector.get_log_likelihood(X_va_scaled) if hasattr(detector, "get_log_likelihood") and len(X_va_scaled) > 0 else None

            bic_te = detector.get_bic(X_te_scaled) if hasattr(detector, "get_bic") and len(X_te_scaled) > 0 else None
            aic_te = detector.get_aic(X_te_scaled) if hasattr(detector, "get_aic") and len(X_te_scaled) > 0 else None
            ll_te = detector.get_log_likelihood(X_te_scaled) if hasattr(detector, "get_log_likelihood") and len(X_te_scaled) > 0 else None

            train_diag = RegimeAnalytics.compute_clustering_diagnostics(
                X_tr_scaled, train_labels, bic=bic_tr, aic=aic_tr, log_likelihood=ll_tr
            )
            val_diag = RegimeAnalytics.compute_clustering_diagnostics(
                X_va_scaled, val_labels, bic=bic_va, aic=aic_va, log_likelihood=ll_va
            ) if len(val_labels) > 0 else None
            test_diag = RegimeAnalytics.compute_clustering_diagnostics(
                X_te_scaled, test_labels, bic=bic_te, aic=aic_te, log_likelihood=ll_te
            ) if len(test_labels) > 0 else None

            # 5. Compute transition matrix and regime statistics (on full chronological sequence or test set)
            trans_matrix = RegimeAnalytics.compute_transition_matrix(
                regime_sequence=test_labels if len(test_labels) > 0 else train_labels,
                semantic_mapping=detector.semantic_mapping,
            )
            regime_stats = RegimeAnalytics.compute_regime_statistics(
                regime_sequence=test_labels if len(test_labels) > 0 else train_labels,
                returns=split.test_returns if len(test_labels) > 0 else split.train_returns,
                volatilities=split.test_volatility if len(test_labels) > 0 else split.train_volatility,
                vix_series=split.test_vix if len(test_labels) > 0 else split.train_vix,
                semantic_mapping=detector.semantic_mapping,
            )

            # 6. Assemble metadata
            meta = RegimeDetectorMetadata(
                detector_type=detector.detector_type.value,
                detector_version=detector.detector_version,
                trained_at=datetime.now(timezone.utc),
                feature_names=split.feature_names,
                n_regimes=detector.n_regimes,
                semantic_mapping=detector.semantic_mapping,
                hyperparameters={
                    "n_regimes": detector.n_regimes,
                    "random_state": detector.random_state,
                },
                train_period_start=split.train_timestamps[0] if split.train_timestamps else None,
                train_period_end=split.train_timestamps[-1] if split.train_timestamps else None,
                val_period_start=split.val_timestamps[0] if split.val_timestamps else None,
                val_period_end=split.val_timestamps[-1] if split.val_timestamps else None,
                test_period_start=split.test_timestamps[0] if split.test_timestamps else None,
                test_period_end=split.test_timestamps[-1] if split.test_timestamps else None,
                train_diagnostics=train_diag,
                val_diagnostics=val_diag,
                test_diagnostics=test_diag,
                statistics_per_regime={k: v.to_dict() for k, v in regime_stats.items()},
                transition_matrix=trans_matrix.to_dict(),
            )

            # 7. Persist artifact if requested
            if save_artifacts:
                self.storage.save_detector(detector, metadata=meta, preprocessor=preprocessor)

            results[detector.name] = (detector, meta, trans_matrix, regime_stats)

        logger.info(f"Regime detection pipeline completed across {len(results)} detectors.")
        return results

    def compare_detectors(
        self,
        results: Dict[str, Tuple[BaseRegimeDetector, RegimeDetectorMetadata, TransitionMatrix, Dict[str, RegimeStatistics]]],
    ) -> pd.DataFrame:
        """
        Generate standardized comparison DataFrame across all fitted regime detectors.
        """
        rows = []
        for name, (det, meta, trans, stats) in results.items():
            test_d = meta.test_diagnostics or meta.train_diagnostics
            sil = f"{test_d.silhouette_score:.4f}" if test_d and test_d.silhouette_score is not None else "N/A"
            db = f"{test_d.davies_bouldin_index:.4f}" if test_d and test_d.davies_bouldin_index is not None else "N/A"
            ch = f"{test_d.calinski_harabasz_score:.2f}" if test_d and test_d.calinski_harabasz_score is not None else "N/A"

            # Average persistence across regimes
            persists = list(trans.persistence_probabilities.values())
            avg_pers = float(np.mean(persists)) if persists else 1.0

            # Average duration across regimes
            durations = [s.mean_duration_bars for s in stats.values()]
            avg_dur = float(np.mean(durations)) if durations else 0.0

            rows.append({
                "Detector": name,
                "Regimes": det.n_regimes,
                "Silhouette": sil,
                "Davies-Bouldin": db,
                "Calinski-Harabasz": ch,
                "Avg Persistence": f"{avg_pers:.2%}",
                "Avg Duration (bars)": f"{avg_dur:.1f}",
                "Total Transitions": trans.total_transitions,
            })

        return pd.DataFrame(rows)
