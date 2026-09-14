"""
Machine Learning Model Storage Manager
Persists and retrieves trained model weights, preprocessing pipelines, and full reproducibility metadata.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from backend.app.models.base import BaseMLModel
from backend.app.models.logistic_regression import LogisticRegressionModel
from backend.app.models.random_forest import RandomForestModel
from backend.app.models.xgboost_model import XGBoostModel
from backend.app.models.lightgbm_model import LightGBMModel
from backend.app.models.mlp_model import MLPModel
from backend.app.models.lstm_model import LSTMModel
from backend.app.models.transformer_model import TransformerModel
from backend.app.models.preprocessing import FeaturePreprocessor
from backend.app.models.schemas import ModelMetadata, ModelType

logger = logging.getLogger(__name__)

MODEL_CLASS_MAP = {
    ModelType.LOGISTIC_REGRESSION: LogisticRegressionModel,
    ModelType.RANDOM_FOREST: RandomForestModel,
    ModelType.XGBOOST: XGBoostModel,
    ModelType.LIGHTGBM: LightGBMModel,
    ModelType.MLP: MLPModel,
    ModelType.LSTM: LSTMModel,
    ModelType.TRANSFORMER: TransformerModel,
}


class ModelStorage:
    """
    Manages persistence of trained models and metadata in models/trained/ and models/metadata/.
    """

    def __init__(
        self,
        base_storage_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
    ):
        target_base = base_storage_dir or base_dir
        if target_base is None:
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.base_dir = root_dir / "models"
        else:
            self.base_dir = Path(target_base)

        self.trained_dir = self.base_dir / "trained"
        self.metadata_dir = self.base_dir / "metadata"
        self.trained_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self, model_type: Union[ModelType, str], model_version: str) -> Path:
        type_str = model_type.value if isinstance(model_type, ModelType) else str(model_type)
        target = self.trained_dir / type_str / model_version
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_model(
        self,
        model: BaseMLModel,
        metadata: ModelMetadata,
        preprocessor: Optional[FeaturePreprocessor] = None,
    ) -> Path:
        """
        Save model artifact, preprocessor pipeline, and metadata to disk.
        """
        model_dir = self._get_model_dir(model.model_type, model.model_version)
        
        # 1. Save model weights
        model.save(model_dir)

        # 2. Save preprocessor if provided
        if preprocessor is not None:
            preprocessor.save(model_dir / "preprocessor.joblib")

        # 3. Save metadata JSON in model directory
        meta_file = model_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # 4. Also copy/save metadata JSON in root models/metadata/
        meta_root_file = self.metadata_dir / f"{model.model_name}_{model.model_version}.json"
        with open(meta_root_file, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Model {model.model_name} [{model.model_version}] saved successfully to {model_dir}")
        return model_dir

    def load_model(
        self,
        model_type: Union[ModelType, str],
        model_version: str = "baseline-v1",
    ) -> Tuple[BaseMLModel, Optional[FeaturePreprocessor], Optional[ModelMetadata]]:
        """
        Load trained model, preprocessor, and metadata from disk.
        """
        mtype = ModelType(model_type) if isinstance(model_type, str) else model_type
        model_dir = self._get_model_dir(mtype, model_version)

        cls = MODEL_CLASS_MAP.get(mtype)
        if cls is None:
            raise ValueError(f"Unsupported model type for loading: {mtype}")

        # Load model
        model = cls.load(model_dir)

        # Load preprocessor if exists
        preprocessor = None
        prep_file = model_dir / "preprocessor.joblib"
        if prep_file.exists():
            preprocessor = FeaturePreprocessor.load(prep_file)

        # Load metadata if exists
        metadata = None
        meta_file = model_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = ModelMetadata.from_dict(json.load(f))

        return model, preprocessor, metadata

    def model_exists(
        self,
        model_type: Union[ModelType, str],
        model_version: str = "baseline-v1",
    ) -> bool:
        """Check if model artifact exists on disk."""
        mtype = ModelType(model_type) if isinstance(model_type, str) else model_type
        model_dir = self._get_model_dir(mtype, model_version)
        return (
            (model_dir / "model.joblib").exists()
            or (model_dir / "model.keras").exists()
            or (model_dir / "config.json").exists()
        )
