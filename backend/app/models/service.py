"""
Model Training and Orchestration Service
Coordinates dataset preparation, chronological splitting, train-only preprocessing,
model training across baseline and advanced neural architectures, validation, test evaluation, comparison, and storage.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.config.settings import get_settings
from backend.app.data.models import BarData, TimeFrame, ensure_utc
from backend.app.features.models import FeatureDataset
from backend.app.models.base import BaseMLModel
from backend.app.models.dataset import MLDatasetBuilder, SequenceDatasetBuilder, TimeSeriesSplitter
from backend.app.models.evaluator import ModelEvaluator
from backend.app.models.logistic_regression import LogisticRegressionModel
from backend.app.models.random_forest import RandomForestModel
from backend.app.models.xgboost_model import XGBoostModel
from backend.app.models.lightgbm_model import LightGBMModel
from backend.app.models.mlp_model import MLPModel
from backend.app.models.lstm_model import LSTMModel
from backend.app.models.transformer_model import TransformerModel
from backend.app.models.preprocessing import FeaturePreprocessor
from backend.app.models.schemas import DatasetSplit, EvaluationMetrics, ModelMetadata, ModelType, TargetConfig
from backend.app.models.storage import ModelStorage

logger = logging.getLogger(__name__)


class ModelTrainingService:
    """
    High-level Machine Learning Training and Evaluation Service.
    Supports both standard 2D tabular baselines and 3D temporal sequence neural models.
    """

    def __init__(
        self,
        storage: Optional[ModelStorage] = None,
        dataset_builder: Optional[MLDatasetBuilder] = None,
        splitter: Optional[TimeSeriesSplitter] = None,
    ):
        settings = get_settings()
        self.storage = storage or ModelStorage()

        # Target and splitting configurations
        target_horizon = getattr(settings, "ML_TARGET_FORWARD_HORIZON", 5)
        target_thresh = getattr(settings, "ML_TARGET_RETURN_THRESHOLD", 0.0)
        target_cfg = TargetConfig(forward_horizon=target_horizon, return_threshold=target_thresh)

        train_r = getattr(settings, "ML_TRAIN_RATIO", 0.60)
        val_r = getattr(settings, "ML_VALIDATION_RATIO", 0.20)
        test_r = getattr(settings, "ML_TEST_RATIO", 0.20)

        self.dataset_builder = dataset_builder or MLDatasetBuilder(target_config=target_cfg)
        self.splitter = splitter or TimeSeriesSplitter(train_ratio=train_r, val_ratio=val_r, test_ratio=test_r)
        self.random_seed = getattr(settings, "ML_RANDOM_SEED", 42)
        self.default_version = getattr(settings, "ML_MODEL_VERSION", "baseline-v1")
        self.default_seq_len = getattr(settings, "ML_SEQUENCE_LENGTH", 15)

    def prepare_dataset_split(
        self,
        features: FeatureDataset,
        bars: List[BarData],
        target_config: Optional[TargetConfig] = None,
    ) -> DatasetSplit:
        """
        Build feature/target matrix and partition chronologically into Train/Val/Test.
        """
        builder = self.dataset_builder
        if target_config is not None:
            builder = MLDatasetBuilder(target_config=target_config, feature_columns=builder.feature_columns)

        X, y, timestamps, feat_names, fut_rets = builder.build_dataset(features, bars)

        split = self.splitter.split(
            X=X,
            y=y,
            timestamps=timestamps,
            feature_names=feat_names,
            target_config=builder.target_config,
            symbol=features.symbol,
            timeframe=features.timeframe,
            future_returns=fut_rets,
        )
        return split

    def instantiate_baseline_models(
        self,
        feature_names: List[str],
        version: Optional[str] = None,
    ) -> List[BaseMLModel]:
        """Instantiate all 4 standardized baseline ML architectures."""
        settings = get_settings()
        ver = version or self.default_version
        seed = self.random_seed

        log_c = getattr(settings, "ML_LOGISTIC_C", 1.0)
        rf_n = getattr(settings, "ML_RF_N_ESTIMATORS", 100)
        rf_depth = getattr(settings, "ML_RF_MAX_DEPTH", 5)
        xgb_n = getattr(settings, "ML_XGB_N_ESTIMATORS", 100)
        xgb_lr = getattr(settings, "ML_XGB_LEARNING_RATE", 0.05)
        xgb_depth = getattr(settings, "ML_XGB_MAX_DEPTH", 3)
        lgbm_n = getattr(settings, "ML_LGBM_N_ESTIMATORS", 100)
        lgbm_lr = getattr(settings, "ML_LGBM_LEARNING_RATE", 0.05)
        lgbm_depth = getattr(settings, "ML_LGBM_MAX_DEPTH", 3)

        return [
            LogisticRegressionModel(
                C=log_c,
                random_state=seed,
                model_version=ver,
                feature_names=feature_names,
            ),
            RandomForestModel(
                n_estimators=rf_n,
                max_depth=rf_depth,
                random_state=seed,
                model_version=ver,
                feature_names=feature_names,
            ),
            XGBoostModel(
                n_estimators=xgb_n,
                learning_rate=xgb_lr,
                max_depth=xgb_depth,
                random_state=seed,
                model_version=ver,
                feature_names=feature_names,
            ),
            LightGBMModel(
                n_estimators=lgbm_n,
                learning_rate=lgbm_lr,
                max_depth=lgbm_depth,
                random_state=seed,
                model_version=ver,
                feature_names=feature_names,
            ),
        ]

    def instantiate_advanced_models(
        self,
        feature_names: List[str],
        sequence_length: Optional[int] = None,
        version: Optional[str] = None,
    ) -> List[BaseMLModel]:
        """Instantiate all 3 Advanced AI deep learning architectures (MLP, LSTM, Transformer)."""
        settings = get_settings()
        ver = version or self.default_version
        seed = getattr(settings, "ML_NEURAL_RANDOM_SEED", self.random_seed)
        seq_len = sequence_length or self.default_seq_len

        epochs = getattr(settings, "ML_NEURAL_EPOCHS", 30)
        batch_size = getattr(settings, "ML_NEURAL_BATCH_SIZE", 32)
        lr = getattr(settings, "ML_NEURAL_LEARNING_RATE", 0.001)
        dropout = getattr(settings, "ML_NEURAL_DROPOUT", 0.2)
        patience = getattr(settings, "ML_EARLY_STOPPING_PATIENCE", 5)

        mlp_units = getattr(settings, "ML_MLP_HIDDEN_UNITS", [64, 32])
        lstm_units = getattr(settings, "ML_LSTM_UNITS", 32)
        tf_heads = getattr(settings, "ML_TRANSFORMER_HEADS", 2)
        tf_key_dim = getattr(settings, "ML_TRANSFORMER_KEY_DIM", 16)
        tf_ff_dim = getattr(settings, "ML_TRANSFORMER_FF_DIM", 32)

        return [
            MLPModel(
                hidden_units=mlp_units,
                learning_rate=lr,
                dropout_rate=dropout,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                random_state=seed,
                model_version=ver,
                feature_names=feature_names,
            ),
            LSTMModel(
                sequence_length=seq_len,
                lstm_units=lstm_units,
                dropout_rate=dropout,
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                random_state=seed,
                model_version=ver,
                feature_names=feature_names,
            ),
            TransformerModel(
                sequence_length=seq_len,
                num_heads=tf_heads,
                key_dim=tf_key_dim,
                ff_dim=tf_ff_dim,
                dropout_rate=dropout,
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                random_state=seed,
                model_version=ver,
                feature_names=feature_names,
            ),
        ]

    def instantiate_all_models(
        self,
        feature_names: List[str],
        sequence_length: Optional[int] = None,
        version: Optional[str] = None,
    ) -> List[BaseMLModel]:
        """Instantiate all 7 models across baseline and advanced suites."""
        return self.instantiate_baseline_models(feature_names, version=version) + self.instantiate_advanced_models(
            feature_names, sequence_length=sequence_length, version=version
        )

    def train_and_evaluate_all(
        self,
        split: DatasetSplit,
        models: Optional[List[BaseMLModel]] = None,
        save_artifacts: bool = True,
    ) -> Dict[str, Tuple[BaseMLModel, EvaluationMetrics, ModelMetadata]]:
        """
        End-to-end training and evaluation:
        1. Fit FeaturePreprocessor strictly on X_train.
        2. Transform X_train, X_val, X_test.
        3. For tabular models: train on 2D arrays.
        4. For sequence models (LSTM, Transformer): build 3D sequence tensors without look-ahead.
        5. Evaluate validation and untouched test sets.
        6. Persist artifacts and metadata.
        """
        logger.info(f"Starting training run for {split.symbol} on {split.total_size} total samples...")

        # 1. Preprocessing (Fitted ONLY on training observations)
        preprocessor = FeaturePreprocessor()
        X_train_scaled = preprocessor.fit_transform(split.X_train)
        X_val_scaled = preprocessor.transform(split.X_val)
        X_test_scaled = preprocessor.transform(split.X_test)

        if models is None:
            models = self.instantiate_baseline_models(split.feature_names)

        results: Dict[str, Tuple[BaseMLModel, EvaluationMetrics, ModelMetadata]] = {}

        for model in models:
            logger.info(f"Training {model.model_name} [{model.model_version}]...")

            # Determine if model requires 3D sequential tensors
            is_sequence_model = model.model_type in (ModelType.LSTM, ModelType.TRANSFORMER)

            if is_sequence_model:
                seq_len = getattr(model, "sequence_length", self.default_seq_len)
                seq_builder = SequenceDatasetBuilder(sequence_length=seq_len)

                # Construct sequence partitions
                X_tr, y_tr, ts_tr, ret_tr = seq_builder.build_sequences(
                    X_train_scaled, split.y_train, split.train_timestamps, split.future_returns_train
                )
                X_va, y_va, ts_va, ret_va = seq_builder.build_sequences(
                    X_val_scaled, split.y_val, split.val_timestamps, split.future_returns_val
                )
                X_te, y_te, ts_te, ret_te = seq_builder.build_sequences(
                    X_test_scaled, split.y_test, split.test_timestamps, split.future_returns_test
                )
            else:
                X_tr, y_tr, ts_tr, ret_tr = X_train_scaled, split.y_train, split.train_timestamps, split.future_returns_train
                X_va, y_va, ts_va, ret_va = X_val_scaled, split.y_val, split.val_timestamps, split.future_returns_val
                X_te, y_te, ts_te, ret_te = X_test_scaled, split.y_test, split.test_timestamps, split.future_returns_test

            # 2. Fit model
            model.fit(
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_va,
                y_val=y_va,
            )

            # 3. Evaluate partitions
            train_m = model.evaluate(X_tr, y_tr, ret_tr)
            val_m = model.evaluate(X_va, y_va, ret_va)
            test_m = model.evaluate(X_te, y_te, ret_te)

            importances = model.get_feature_importance()

            trainable_p = getattr(model, "trainable_params", None)
            non_trainable_p = getattr(model, "non_trainable_params", None)
            total_p = getattr(model, "total_params", None)
            history_dict = getattr(model, "training_history", None)

            # 4. Construct metadata
            metadata = ModelMetadata(
                model_name=model.model_name,
                model_type=model.model_type,
                model_version=model.model_version,
                trained_at=datetime.now(timezone.utc),
                feature_names=split.feature_names,
                target_config=split.target_config,
                hyperparameters=model.hyperparameters,
                train_metrics=train_m,
                val_metrics=val_m,
                test_metrics=test_m,
                feature_importances=importances,
                trainable_parameters=trainable_p,
                non_trainable_parameters=non_trainable_p,
                total_parameters=total_p,
                training_history=history_dict,
                train_period_start=ts_tr[0] if ts_tr else split.train_timestamps[0],
                train_period_end=ts_tr[-1] if ts_tr else split.train_timestamps[-1],
                val_period_start=ts_va[0] if ts_va else split.val_timestamps[0],
                val_period_end=ts_va[-1] if ts_va else split.val_timestamps[-1],
                test_period_start=ts_te[0] if ts_te else split.test_timestamps[0],
                test_period_end=ts_te[-1] if ts_te else split.test_timestamps[-1],
                dataset_summary={
                    "symbol": split.symbol,
                    "timeframe": split.timeframe.value,
                    "train_samples": len(y_tr),
                    "val_samples": len(y_va),
                    "test_samples": len(y_te),
                    "train_positive_ratio": float(np.mean(y_tr)) if len(y_tr) > 0 else 0.0,
                    "test_positive_ratio": float(np.mean(y_te)) if len(y_te) > 0 else 0.0,
                },
            )

            # 5. Persist artifacts
            if save_artifacts:
                self.storage.save_model(model, metadata=metadata, preprocessor=preprocessor)

            results[model.model_name] = (model, test_m, metadata)

        logger.info(f"Training completed for all {len(models)} models.")
        return results

    def compare_models(
        self,
        results: Dict[str, Tuple[BaseMLModel, EvaluationMetrics, ModelMetadata]],
    ) -> pd.DataFrame:
        """Create standardized comparison summary DataFrame."""
        test_metrics_map = {name: res[1] for name, res in results.items()}
        return ModelEvaluator.create_comparison_dataframe(test_metrics_map)

    def compare_advanced_models(
        self,
        results: Dict[str, Tuple[BaseMLModel, EvaluationMetrics, ModelMetadata]],
    ) -> pd.DataFrame:
        """Create specialized comparison DataFrame reporting parameter counts and training characteristics."""
        rows = []
        for name, (model, test_m, meta) in results.items():
            diag = test_m.trading_diagnostics
            seq_l = meta.hyperparameters.get("sequence_length", "N/A")
            params = meta.total_parameters if meta.total_parameters is not None else "N/A"
            val_loss = round(meta.val_metrics.log_loss, 4) if meta.val_metrics and meta.val_metrics.log_loss else (
                round(min(meta.training_history.get("val_loss", [0.0])), 4) if meta.training_history and "val_loss" in meta.training_history else "N/A"
            )

            rows.append({
                "Model": name,
                "Type": meta.model_type.value,
                "Parameters": params,
                "Seq Len": seq_l,
                "Val Loss": val_loss,
                "Test Accuracy": f"{test_m.accuracy:.2%}",
                "Test ROC-AUC": f"{test_m.roc_auc:.4f}" if test_m.roc_auc is not None else "N/A",
                "Test F1": f"{test_m.f1:.4f}",
                "Return Spread": f"{diag.return_spread:.2%}" if diag and diag.return_spread is not None else "N/A",
            })
        return pd.DataFrame(rows)
