"""
Market Regime Detection Subsystem Package.
Provides multi-model market regime classification, unsupervised clustering, transition modeling, and persistence.
"""

from backend.app.regime.schemas import (
    MarketRegimeType,
    VolatilityState,
    TrendState,
    DetectorType,
    MarketRegimeState,
    RegimeStatistics,
    TransitionMatrix,
    ClusteringDiagnostics,
    RegimeDetectorMetadata,
)
from backend.app.regime.preprocessor import (
    RegimeFeaturePreprocessor,
)
from backend.app.regime.analysis import (
    RegimeAnalytics,
)
from backend.app.regime.base import (
    BaseRegimeDetector,
)
from backend.app.regime.rule_based import (
    RuleBasedRegimeDetector,
)
from backend.app.regime.kmeans_detector import (
    KMeansRegimeDetector,
)
from backend.app.regime.gmm_detector import (
    GMMRegimeDetector,
)
from backend.app.regime.hmm_detector import (
    HMMRegimeDetector,
    HMM_AVAILABLE,
)
from backend.app.regime.storage import (
    RegimeStorage,
)
from backend.app.regime.service import (
    MarketRegimeService,
    RegimeDatasetSplit,
)

__all__ = [
    "MarketRegimeType",
    "VolatilityState",
    "TrendState",
    "DetectorType",
    "MarketRegimeState",
    "RegimeStatistics",
    "TransitionMatrix",
    "ClusteringDiagnostics",
    "RegimeDetectorMetadata",
    "RegimeFeaturePreprocessor",
    "RegimeAnalytics",
    "BaseRegimeDetector",
    "RuleBasedRegimeDetector",
    "KMeansRegimeDetector",
    "GMMRegimeDetector",
    "HMMRegimeDetector",
    "HMM_AVAILABLE",
    "RegimeStorage",
    "MarketRegimeService",
    "RegimeDatasetSplit",
]
