"""
Model Evaluation and Performance Metrics Suite
Computes comprehensive statistical, probabilistic, and trading-relevant diagnostic metrics for classification models.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from backend.app.models.schemas import EvaluationMetrics, TradingDiagnostics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for classification models and trading diagnostics.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        future_returns: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """
        Compute full statistical, probabilistic, and trading metrics for model predictions.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Evaluation arrays cannot be empty.")

        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        # 1. Core Classification Metrics
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred).tolist()

        # 2. Probability & Calibration Metrics
        roc_auc: Optional[float] = None
        pr_auc: Optional[float] = None
        loss: Optional[float] = None
        brier: Optional[float] = None

        if y_prob is not None and len(y_prob) > 0:
            y_prob = np.asarray(y_prob, dtype=float)
            # Ensure probabilities are 1D for positive class
            if y_prob.ndim == 2:
                prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
            else:
                prob_pos = y_prob

            # Check if both classes exist in y_true
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                try:
                    roc_auc = float(roc_auc_score(y_true, prob_pos))
                    pr_auc = float(average_precision_score(y_true, prob_pos))
                    loss = float(log_loss(y_true, prob_pos, labels=[0, 1]))
                except Exception as e:
                    logger.warning(f"Error computing probabilistic metrics: {e}")
            brier = float(brier_score_loss(y_true, prob_pos))

        # 3. Trading-Relevant Diagnostics
        diagnostics = ModelEvaluator._compute_trading_diagnostics(y_pred, future_returns)

        return EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            log_loss=loss,
            brier_score=brier,
            confusion_matrix=cm,
            trading_diagnostics=diagnostics,
        )

    @staticmethod
    def _compute_trading_diagnostics(
        y_pred: np.ndarray,
        future_returns: Optional[np.ndarray] = None,
    ) -> TradingDiagnostics:
        """Calculate class distribution and mean forward returns for predicted cohorts."""
        n = len(y_pred)
        pos_mask = y_pred == 1
        neg_mask = y_pred == 0

        pos_count = int(np.sum(pos_mask))
        neg_count = int(np.sum(neg_mask))
        pos_ratio = float(pos_count / n) if n > 0 else 0.0

        mean_pos_ret: Optional[float] = None
        mean_neg_ret: Optional[float] = None
        spread: Optional[float] = None

        if future_returns is not None and len(future_returns) == n:
            ret_arr = np.asarray(future_returns, dtype=float)
            if pos_count > 0:
                mean_pos_ret = float(np.mean(ret_arr[pos_mask]))
            if neg_count > 0:
                mean_neg_ret = float(np.mean(ret_arr[neg_mask]))
            if mean_pos_ret is not None and mean_neg_ret is not None:
                spread = float(mean_pos_ret - mean_neg_ret)

        return TradingDiagnostics(
            mean_return_predicted_positive=mean_pos_ret,
            mean_return_predicted_negative=mean_neg_ret,
            return_spread=spread,
            positive_prediction_count=pos_count,
            negative_prediction_count=neg_count,
            positive_class_ratio=pos_ratio,
        )

    @staticmethod
    def create_comparison_dataframe(
        model_metrics_map: Dict[str, EvaluationMetrics],
    ) -> pd.DataFrame:
        """Build standardized comparison DataFrame across all models."""
        rows = []
        for name, m in model_metrics_map.items():
            diag = m.trading_diagnostics
            rows.append({
                "Model": name,
                "Accuracy": m.accuracy,
                "Precision": m.precision,
                "Recall": m.recall,
                "F1": m.f1,
                "ROC-AUC": m.roc_auc,
                "PR-AUC": m.pr_auc,
                "Log Loss": m.log_loss,
                "Brier Score": m.brier_score,
                "Pos Class %": f"{diag.positive_class_ratio:.1%}" if diag else None,
                "Mean Ret Pos": f"{diag.mean_return_predicted_positive:.2%}" if diag and diag.mean_return_predicted_positive is not None else None,
                "Mean Ret Neg": f"{diag.mean_return_predicted_negative:.2%}" if diag and diag.mean_return_predicted_negative is not None else None,
                "Spread": f"{diag.return_spread:.2%}" if diag and diag.return_spread is not None else None,
            })
        return pd.DataFrame(rows)
