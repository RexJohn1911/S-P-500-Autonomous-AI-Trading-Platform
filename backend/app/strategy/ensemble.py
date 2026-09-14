"""
Signal Ensemble Module.
Combines multiple standardized model predictions into a single weighted ensemble score, computing agreement metrics, confidence, and model contributions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from backend.app.strategy.schemas import StandardizedPrediction


class SignalEnsemble:
    """
    Weighted ensemble engine for synthesizing standardized predictions from multiple ML models.
    Supports dynamic weight re-normalization across available models, voting consensus, and disagreement metrics.
    
    NOTE ON MODEL WEIGHTS & CONFIDENCE:
    - Default model weights are configurable engineering defaults and not established as scientifically optimal.
    - Confidence represents model conviction / evidence strength in [0.0, 1.0], not an empirically calibrated probability of profit.
    - Expected return is aggregated strictly from models that legitimately provide return forecasts; it is never fabricated.
    """

    def __init__(
        self,
        configured_weights: Optional[Dict[str, float]] = None,
    ):
        self.configured_weights = configured_weights or {}

    def aggregate(
        self,
        predictions: List[StandardizedPrediction],
    ) -> Dict[str, Any]:
        """
        Synthesize a list of predictions for a single timestamp into ensemble metrics.

        Returns a dictionary containing:
            - ensemble_score: float in [-1.0, 1.0]
            - direction_probability: float in [0.0, 1.0]
            - confidence: float in [0.0, 1.0] (model conviction, not calibrated profit probability)
            - model_agreement: float in [0.0, 1.0]
            - bullish_votes: int
            - bearish_votes: int
            - neutral_votes: int
            - active_model_count: int
            - model_contributions: Dict[str, float]
            - active_weights: Dict[str, float]
            - expected_return: Optional[float] (None if no upstream models provide returns)
        """
        if not predictions:
            return {
                "ensemble_score": 0.0,
                "direction_probability": 0.5,
                "confidence": 0.0,
                "model_agreement": 0.0,
                "bullish_votes": 0,
                "bearish_votes": 0,
                "neutral_votes": 0,
                "active_model_count": 0,
                "model_contributions": {},
                "active_weights": {},
                "expected_return": None,
            }

        # Deduplicate predictions by model_name (keep first or highest weight)
        unique_preds: Dict[str, StandardizedPrediction] = {}
        for p in predictions:
            if p.model_name not in unique_preds:
                unique_preds[p.model_name] = p

        active_models = list(unique_preds.values())
        n_active = len(active_models)

        # Determine effective raw weights
        raw_weights: Dict[str, float] = {}
        for p in active_models:
            # Check configured_weights first, fallback to prediction weight, default to 1.0
            w = self.configured_weights.get(p.model_name, p.weight)
            if w < 0.0:
                raise ValueError(f"Negative weight {w} for model {p.model_name}")
            raw_weights[p.model_name] = float(w)

        total_weight = sum(raw_weights.values())
        if total_weight <= 0.0:
            # If all weights are 0, assign equal weights
            norm_weights = {p.model_name: 1.0 / n_active for p in active_models}
        else:
            norm_weights = {m: w / total_weight for m, w in raw_weights.items()}

        # Compute weighted ensemble directional score
        ensemble_score = 0.0
        model_contributions: Dict[str, float] = {}
        bullish_votes = 0
        bearish_votes = 0
        neutral_votes = 0
        weighted_returns: List[float] = []
        return_weights: List[float] = []

        for p in active_models:
            w = norm_weights[p.model_name]
            score_contrib = w * p.directional_score
            ensemble_score += score_contrib
            model_contributions[p.model_name] = score_contrib

            # Voting based on directional score
            if p.directional_score > 0.0:
                bullish_votes += 1
            elif p.directional_score < 0.0:
                bearish_votes += 1
            else:
                neutral_votes += 1

            if p.expected_return is not None:
                weighted_returns.append(p.expected_return * w)
                return_weights.append(w)

        # Bound ensemble score strictly within [-1.0, 1.0]
        ensemble_score = max(-1.0, min(1.0, float(ensemble_score)))

        # Directional probability in [0.0, 1.0]
        direction_probability = max(0.0, min(1.0, (ensemble_score + 1.0) / 2.0))

        # Model agreement ratio: proportion of active models aligning with dominant direction
        max_directional_votes = max(bullish_votes, bearish_votes)
        model_agreement = float(max_directional_votes / n_active) if n_active > 0 else 0.0

        # Conviction confidence: magnitude of ensemble score scaled by model agreement
        # When all models strongly agree, confidence equals |ensemble_score|.
        # When models conflict, confidence is proportionately dampened.
        raw_confidence = abs(ensemble_score)
        confidence = float(raw_confidence * (0.5 + 0.5 * model_agreement))
        confidence = max(0.0, min(1.0, confidence))

        # Expected return aggregation: strictly sourced from genuine upstream estimates
        expected_return: Optional[float] = None
        if weighted_returns and sum(return_weights) > 0:
            expected_return = float(sum(weighted_returns) / sum(return_weights))

        return {
            "ensemble_score": round(ensemble_score, 6),
            "direction_probability": round(direction_probability, 6),
            "confidence": round(confidence, 6),
            "model_agreement": round(model_agreement, 6),
            "bullish_votes": bullish_votes,
            "bearish_votes": bearish_votes,
            "neutral_votes": neutral_votes,
            "active_model_count": n_active,
            "model_contributions": model_contributions,
            "active_weights": norm_weights,
            "expected_return": expected_return,
        }
