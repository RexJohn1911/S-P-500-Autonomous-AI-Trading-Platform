"""
Model Prediction Adapters.
Translates heterogeneous raw ML model outputs (classification probabilities, regression scores) into unified StandardizedPrediction representations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.strategy.schemas import StandardizedPrediction


class BasePredictionAdapter(ABC):
    """Abstract interface for adapting model outputs to StandardizedPrediction."""

    @abstractmethod
    def adapt(
        self,
        symbol: str,
        timestamp: datetime,
        model_name: str,
        raw_output: Any,
        model_type: Optional[str] = None,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StandardizedPrediction:
        """Adapt a single raw prediction output at a specific timestamp."""
        pass


class BinaryClassificationAdapter(BasePredictionAdapter):
    """
    Adapter for binary classification models predicting upward vs downward price movement.
    Converts P(y=1) in [0, 1] into:
      - directional_score in [-1, +1] where s = 2 * P(y=1) - 1
      - confidence in [0, 1] where c = 2 * |P(y=1) - 0.5|
    """

    def adapt(
        self,
        symbol: str,
        timestamp: datetime,
        model_name: str,
        raw_output: Union[float, np.ndarray, List[float]],
        model_type: Optional[str] = None,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StandardizedPrediction:
        # Handle 2D probability array [P(0), P(1)], 1D probability, or single float
        if isinstance(raw_output, (list, tuple, np.ndarray)):
            arr = np.asarray(raw_output).flatten()
            if len(arr) == 2:
                prob_up = float(arr[1])
            elif len(arr) == 1:
                prob_up = float(arr[0])
            else:
                raise ValueError(f"Invalid probability array length: {len(arr)}")
        elif isinstance(raw_output, (float, int, np.floating, np.integer)):
            prob_up = float(raw_output)
        else:
            raise TypeError(f"Unsupported raw_output type: {type(raw_output)}")

        # Clip within [0.0, 1.0] to guard against numerical precision issues
        prob_up = max(0.0, min(1.0, prob_up))

        directional_score = 2.0 * prob_up - 1.0
        confidence = 2.0 * abs(prob_up - 0.5)

        return StandardizedPrediction(
            symbol=symbol,
            timestamp=timestamp,
            model_name=model_name,
            model_type=model_type,
            probability_up=prob_up,
            directional_score=directional_score,
            confidence=confidence,
            expected_return=None,
            weight=weight,
            metadata=metadata or {},
        )


class DirectPredictionAdapter(BasePredictionAdapter):
    """
    Adapter for models, signals, or experts that directly output directional scores or probabilities.
    """

    def adapt(
        self,
        symbol: str,
        timestamp: datetime,
        model_name: str,
        raw_output: Dict[str, Any],
        model_type: Optional[str] = None,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StandardizedPrediction:
        prob_up = float(raw_output.get("probability_up", 0.5))
        prob_up = max(0.0, min(1.0, prob_up))

        directional_score = float(raw_output.get("directional_score", 2.0 * prob_up - 1.0))
        directional_score = max(-1.0, min(1.0, directional_score))

        confidence = float(raw_output.get("confidence", 2.0 * abs(prob_up - 0.5)))
        confidence = max(0.0, min(1.0, confidence))

        expected_return = raw_output.get("expected_return")
        if expected_return is not None:
            expected_return = float(expected_return)

        meta = dict(metadata or {})
        if "metadata" in raw_output:
            meta.update(raw_output["metadata"])

        return StandardizedPrediction(
            symbol=symbol,
            timestamp=timestamp,
            model_name=model_name,
            model_type=model_type or raw_output.get("model_type"),
            probability_up=prob_up,
            directional_score=directional_score,
            confidence=confidence,
            expected_return=expected_return,
            weight=weight,
            metadata=meta,
        )
