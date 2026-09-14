"""
Market Regime Artifact Storage Manager.
Persists and retrieves trained regime detectors, preprocessors, metadata, transition matrices, and statistics.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from backend.app.regime.base import BaseRegimeDetector
from backend.app.regime.gmm_detector import GMMRegimeDetector
from backend.app.regime.hmm_detector import HMMRegimeDetector
from backend.app.regime.kmeans_detector import KMeansRegimeDetector
from backend.app.regime.preprocessor import RegimeFeaturePreprocessor
from backend.app.regime.rule_based import RuleBasedRegimeDetector
from backend.app.regime.schemas import DetectorType, RegimeDetectorMetadata

logger = logging.getLogger(__name__)

DETECTOR_CLASS_MAP = {
    DetectorType.RULE_BASED: RuleBasedRegimeDetector,
    DetectorType.KMEANS: KMeansRegimeDetector,
    DetectorType.GMM: GMMRegimeDetector,
    DetectorType.HMM: HMMRegimeDetector,
}


class RegimeStorage:
    """
    Manages persistence of market regime detector weights, preprocessors, and full audit metadata.
    Root path defaults to models/regimes/.
    """

    def __init__(self, base_storage_dir: Optional[Union[str, Path]] = None):
        if base_storage_dir is not None:
            self.base_dir = Path(base_storage_dir)
        else:
            self.base_dir = Path("models/regimes")

        self.metadata_dir = self.base_dir / "metadata"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage root directories if they do not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_detector_dir(self, detector_type: Union[DetectorType, str], version: str) -> Path:
        dtype_str = detector_type.value if isinstance(detector_type, DetectorType) else str(detector_type)
        target = self.base_dir / dtype_str / version
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_detector(
        self,
        detector: BaseRegimeDetector,
        metadata: RegimeDetectorMetadata,
        preprocessor: Optional[RegimeFeaturePreprocessor] = None,
    ) -> Path:
        """
        Save detector artifact, preprocessor, and full metadata JSON to disk.
        """
        detector_dir = self._get_detector_dir(detector.detector_type, detector.detector_version)

        # 1. Save detector weights and configuration
        detector.save(detector_dir)

        # 2. Save preprocessor if provided
        if preprocessor is not None:
            preprocessor.save(detector_dir / "preprocessor.joblib")

        # 3. Save semantic mapping JSON
        mapping_file = detector_dir / "semantic_mapping.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in detector.semantic_mapping.items()}, f, indent=2)

        # 4. Save metadata JSON in model directory
        meta_file = detector_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # 5. Save copy in root metadata directory
        meta_root = self.metadata_dir / f"{detector.name}_{detector.detector_version}.json"
        with open(meta_root, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Regime detector {detector.name} [{detector.detector_version}] saved successfully to {detector_dir}")
        return detector_dir

    def load_detector(
        self,
        detector_type: Union[DetectorType, str],
        version: str = "regime-v1",
    ) -> Tuple[BaseRegimeDetector, Optional[RegimeFeaturePreprocessor], Optional[RegimeDetectorMetadata]]:
        """
        Load trained regime detector, preprocessor, and metadata from disk.
        """
        dtype = DetectorType(detector_type) if isinstance(detector_type, str) else detector_type
        detector_dir = self._get_detector_dir(dtype, version)

        cls = DETECTOR_CLASS_MAP.get(dtype)
        if cls is None:
            raise ValueError(f"Unsupported regime detector type for loading: {dtype}")

        # 1. Load detector
        detector = cls.load(detector_dir)

        # 2. Load preprocessor if exists
        preprocessor = None
        prep_file = detector_dir / "preprocessor.joblib"
        if prep_file.exists():
            preprocessor = RegimeFeaturePreprocessor.load(prep_file)

        # 3. Load metadata if exists
        metadata = None
        meta_file = detector_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = RegimeDetectorMetadata.from_dict(json.load(f))

        return detector, preprocessor, metadata

    def detector_exists(
        self,
        detector_type: Union[DetectorType, str],
        version: str = "regime-v1",
    ) -> bool:
        """Check if regime detector artifact exists on disk."""
        dtype = DetectorType(detector_type) if isinstance(detector_type, str) else detector_type
        detector_dir = self._get_detector_dir(dtype, version)
        return (
            (detector_dir / "config.json").exists()
            or (detector_dir / "detector.joblib").exists()
        )
