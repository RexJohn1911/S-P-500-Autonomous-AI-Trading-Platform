"""
Machine Learning Domain Schemas & Data Structures
Provides structured, type-safe data containers for ML targets, dataset splits, evaluation metrics, and model metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from backend.app.data.models import TimeFrame, ensure_utc


class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    MLP = "mlp"
    LSTM = "lstm"
    TRANSFORMER = "transformer"


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for predictive machine learning target construction."""
    forward_horizon: int = 5
    return_threshold: float = 0.0
    task_type: str = "binary_classification"
    target_name: str = "future_return_5d_pos"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forward_horizon": self.forward_horizon,
            "return_threshold": self.return_threshold,
            "task_type": self.task_type,
            "target_name": self.target_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetConfig":
        return cls(
            forward_horizon=data.get("forward_horizon", 5),
            return_threshold=data.get("return_threshold", 0.0),
            task_type=data.get("task_type", "binary_classification"),
            target_name=data.get("target_name", "future_return_5d_pos"),
        )


@dataclass
class TradingDiagnostics:
    """Trading-relevant evaluation diagnostics for predicted classes."""
    mean_return_predicted_positive: Optional[float] = None
    mean_return_predicted_negative: Optional[float] = None
    return_spread: Optional[float] = None
    positive_prediction_count: int = 0
    negative_prediction_count: int = 0
    positive_class_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_return_predicted_positive": self.mean_return_predicted_positive,
            "mean_return_predicted_negative": self.mean_return_predicted_negative,
            "return_spread": self.return_spread,
            "positive_prediction_count": self.positive_prediction_count,
            "negative_prediction_count": self.negative_prediction_count,
            "positive_class_ratio": self.positive_class_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingDiagnostics":
        return cls(
            mean_return_predicted_positive=data.get("mean_return_predicted_positive"),
            mean_return_predicted_negative=data.get("mean_return_predicted_negative"),
            return_spread=data.get("return_spread"),
            positive_prediction_count=data.get("positive_prediction_count", 0),
            negative_prediction_count=data.get("negative_prediction_count", 0),
            positive_class_ratio=data.get("positive_class_ratio", 0.0),
        )


@dataclass
class EvaluationMetrics:
    """Comprehensive performance metrics for model evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    trading_diagnostics: Optional[TradingDiagnostics] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "pr_auc": round(self.pr_auc, 4) if self.pr_auc is not None else None,
            "log_loss": round(self.log_loss, 4) if self.log_loss is not None else None,
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "trading_diagnostics": self.trading_diagnostics.to_dict() if self.trading_diagnostics else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationMetrics":
        diag = None
        if data.get("trading_diagnostics"):
            diag = TradingDiagnostics.from_dict(data["trading_diagnostics"])
        return cls(
            accuracy=data.get("accuracy", 0.0),
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1=data.get("f1", 0.0),
            roc_auc=data.get("roc_auc"),
            pr_auc=data.get("pr_auc"),
            log_loss=data.get("log_loss"),
            brier_score=data.get("brier_score"),
            confusion_matrix=data.get("confusion_matrix"),
            trading_diagnostics=diag,
        )


@dataclass
class DatasetSplit:
    """Time-series chronologically split dataset partition."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_timestamps: List[datetime]
    val_timestamps: List[datetime]
    test_timestamps: List[datetime]
    feature_names: List[str]
    target_config: TargetConfig
    symbol: str
    timeframe: TimeFrame
    future_returns_train: Optional[np.ndarray] = None
    future_returns_val: Optional[np.ndarray] = None
    future_returns_test: Optional[np.ndarray] = None

    @property
    def train_size(self) -> int:
        return len(self.y_train)

    @property
    def val_size(self) -> int:
        return len(self.y_val)

    @property
    def test_size(self) -> int:
        return len(self.y_test)

    @property
    def total_size(self) -> int:
        return self.train_size + self.val_size + self.test_size


@dataclass
class ModelMetadata:
    """Complete lineage and reproducibility metadata for a trained model artifact."""
    model_name: str
    model_type: ModelType
    model_version: str
    trained_at: datetime
    feature_names: List[str]
    target_config: TargetConfig
    hyperparameters: Dict[str, Any]
    train_metrics: Optional[EvaluationMetrics] = None
    val_metrics: Optional[EvaluationMetrics] = None
    test_metrics: Optional[EvaluationMetrics] = None
    feature_importances: Optional[Dict[str, float]] = None
    trainable_parameters: Optional[int] = None
    non_trainable_parameters: Optional[int] = None
    total_parameters: Optional[int] = None
    training_history: Optional[Dict[str, List[float]]] = None
    train_period_start: Optional[datetime] = None
    train_period_end: Optional[datetime] = None
    val_period_start: Optional[datetime] = None
    val_period_end: Optional[datetime] = None
    test_period_start: Optional[datetime] = None
    test_period_end: Optional[datetime] = None
    dataset_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "model_version": self.model_version,
            "trained_at": self.trained_at.isoformat(),
            "feature_names": self.feature_names,
            "target_config": self.target_config.to_dict(),
            "hyperparameters": self.hyperparameters,
            "train_metrics": self.train_metrics.to_dict() if self.train_metrics else None,
            "val_metrics": self.val_metrics.to_dict() if self.val_metrics else None,
            "test_metrics": self.test_metrics.to_dict() if self.test_metrics else None,
            "feature_importances": self.feature_importances,
            "trainable_parameters": self.trainable_parameters,
            "non_trainable_parameters": self.non_trainable_parameters,
            "total_parameters": self.total_parameters,
            "training_history": self.training_history,
            "train_period_start": self.train_period_start.isoformat() if self.train_period_start else None,
            "train_period_end": self.train_period_end.isoformat() if self.train_period_end else None,
            "val_period_start": self.val_period_start.isoformat() if self.val_period_start else None,
            "val_period_end": self.val_period_end.isoformat() if self.val_period_end else None,
            "test_period_start": self.test_period_start.isoformat() if self.test_period_start else None,
            "test_period_end": self.test_period_end.isoformat() if self.test_period_end else None,
            "dataset_summary": self.dataset_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        trained_at = datetime.fromisoformat(data["trained_at"]) if isinstance(data["trained_at"], str) else data["trained_at"]
        t_start = datetime.fromisoformat(data["train_period_start"]) if data.get("train_period_start") else None
        t_end = datetime.fromisoformat(data["train_period_end"]) if data.get("train_period_end") else None
        v_start = datetime.fromisoformat(data["val_period_start"]) if data.get("val_period_start") else None
        v_end = datetime.fromisoformat(data["val_period_end"]) if data.get("val_period_end") else None
        test_start = datetime.fromisoformat(data["test_period_start"]) if data.get("test_period_start") else None
        test_end = datetime.fromisoformat(data["test_period_end"]) if data.get("test_period_end") else None

        train_m = EvaluationMetrics.from_dict(data["train_metrics"]) if data.get("train_metrics") else None
        val_m = EvaluationMetrics.from_dict(data["val_metrics"]) if data.get("val_metrics") else None
        test_m = EvaluationMetrics.from_dict(data["test_metrics"]) if data.get("test_metrics") else None

        return cls(
            model_name=data["model_name"],
            model_type=ModelType(data["model_type"]),
            model_version=data["model_version"],
            trained_at=ensure_utc(trained_at),
            feature_names=data["feature_names"],
            target_config=TargetConfig.from_dict(data["target_config"]),
            hyperparameters=data["hyperparameters"],
            train_metrics=train_m,
            val_metrics=val_m,
            test_metrics=test_m,
            feature_importances=data.get("feature_importances"),
            trainable_parameters=data.get("trainable_parameters"),
            non_trainable_parameters=data.get("non_trainable_parameters"),
            total_parameters=data.get("total_parameters"),
            training_history=data.get("training_history"),
            train_period_start=ensure_utc(t_start) if t_start else None,
            train_period_end=ensure_utc(t_end) if t_end else None,
            val_period_start=ensure_utc(v_start) if v_start else None,
            val_period_end=ensure_utc(v_end) if v_end else None,
            test_period_start=ensure_utc(test_start) if test_start else None,
            test_period_end=ensure_utc(test_end) if test_end else None,
            dataset_summary=data.get("dataset_summary"),
        )
