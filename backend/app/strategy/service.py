"""
Signal Engine Service Module.
Coordinates model prediction ingestion, normalization, ensemble combination, regime gating, and signal generation.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.regime.schemas import MarketRegimeState, MarketRegimeType
from backend.app.strategy.adapters import BinaryClassificationAdapter, DirectPredictionAdapter
from backend.app.strategy.ensemble import SignalEnsemble
from backend.app.strategy.evaluator import SignalEvaluator
from backend.app.strategy.regime_gate import RegimeGate
from backend.app.strategy.schemas import (
    ReasonCode,
    SignalCandidate,
    SignalDirection,
    SignalEngineConfig,
    SignalEvaluationReport,
    StandardizedPrediction,
)
from backend.app.strategy.storage import SignalStorage

logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Modular Signal Engine for synthesizing machine learning model outputs and regime state into directional trade signal candidates.
    Strictly causal at timestamp t; produces auditable reason codes and confidence measures without position sizing or execution logic.
    """

    def __init__(
        self,
        config: Optional[SignalEngineConfig] = None,
        ensemble: Optional[SignalEnsemble] = None,
        regime_gate: Optional[RegimeGate] = None,
        storage: Optional[SignalStorage] = None,
    ):
        self.config = config or SignalEngineConfig()
        self.ensemble = ensemble or SignalEnsemble(configured_weights=self.config.model_weights)
        self.regime_gate = regime_gate or RegimeGate(config=self.config)
        self.storage = storage or SignalStorage()
        self.classification_adapter = BinaryClassificationAdapter()
        self.direct_adapter = DirectPredictionAdapter()

    def generate_signal(
        self,
        symbol: str,
        timestamp: datetime,
        predictions: List[StandardizedPrediction],
        regime: Optional[Union[str, MarketRegimeType, MarketRegimeState]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignalCandidate:
        """
        Synthesize a directional trade signal candidate at timestamp t.

        Args:
            symbol: Ticker symbol (e.g., 'SPY').
            timestamp: Point-in-time timestamp t.
            predictions: List of StandardizedPrediction objects available at t.
            regime: Optional MarketRegimeState, MarketRegimeType, or string name at t.
            metadata: Optional arbitrary context metadata.

        Returns:
            SignalCandidate with direction, confidence, strength, agreement, and reason codes.
        """
        # Resolve standardized regime string name
        regime_str: Optional[str] = None
        if isinstance(regime, MarketRegimeState):
            regime_str = regime.regime_name
        elif isinstance(regime, MarketRegimeType):
            regime_str = regime.value
        elif isinstance(regime, str):
            regime_str = regime

        # Handle zero predictions
        if not predictions:
            return SignalCandidate(
                timestamp=timestamp,
                symbol=symbol,
                signal=SignalDirection.FLAT,
                signal_strength=0.0,
                confidence=0.0,
                direction_probability=0.5,
                expected_return=None,
                regime=regime_str,
                model_contributions={},
                model_agreement=0.0,
                bullish_votes=0,
                bearish_votes=0,
                neutral_votes=0,
                active_model_count=0,
                reason_codes=[ReasonCode.NO_PREDICTIONS_AVAILABLE.value],
                signal_horizon=self.config.default_horizon,
                signal_version=self.config.version,
                metadata=metadata or {},
            )

        # 1. Ensemble aggregation
        ens_res = self.ensemble.aggregate(predictions)
        ensemble_score = float(ens_res["ensemble_score"])
        direction_prob = float(ens_res["direction_probability"])
        confidence = float(ens_res["confidence"])
        model_agreement = float(ens_res["model_agreement"])
        bullish_votes = int(ens_res["bullish_votes"])
        bearish_votes = int(ens_res["bearish_votes"])
        neutral_votes = int(ens_res["neutral_votes"])
        active_model_count = int(ens_res["active_model_count"])
        model_contributions = ens_res["model_contributions"]
        expected_return = ens_res.get("expected_return")

        reason_codes: List[str] = []

        # 2. Minimum active models check
        if active_model_count < self.config.min_active_models:
            proposed_signal = SignalDirection.FLAT
            reason_codes.append(ReasonCode.INSUFFICIENT_ACTIVE_MODELS.value)
        # 3. Model agreement check
        elif model_agreement < self.config.min_agreement:
            proposed_signal = SignalDirection.FLAT
            reason_codes.append(ReasonCode.MODEL_DISAGREEMENT.value)
        # 4. Threshold & confidence check
        elif ensemble_score >= self.config.long_threshold:
            if confidence >= self.config.min_confidence:
                proposed_signal = SignalDirection.LONG
                reason_codes.append(ReasonCode.BULLISH_ENSEMBLE_SCORE.value)
                if model_agreement >= 0.75:
                    reason_codes.append(ReasonCode.MODEL_CONSENSUS.value)
            else:
                proposed_signal = SignalDirection.FLAT
                reason_codes.append(ReasonCode.LOW_CONFIDENCE.value)
        elif ensemble_score <= self.config.short_threshold:
            if confidence >= self.config.min_confidence:
                proposed_signal = SignalDirection.SHORT
                reason_codes.append(ReasonCode.BEARISH_ENSEMBLE_SCORE.value)
                if model_agreement >= 0.75:
                    reason_codes.append(ReasonCode.MODEL_CONSENSUS.value)
            else:
                proposed_signal = SignalDirection.FLAT
                reason_codes.append(ReasonCode.LOW_CONFIDENCE.value)
        else:
            proposed_signal = SignalDirection.FLAT
            reason_codes.append(ReasonCode.NEUTRAL_SCORE.value)

        # 5. Regime-aware gating
        final_signal, gate_reasons = self.regime_gate.evaluate_gate(
            proposed_signal=proposed_signal,
            ensemble_score=ensemble_score,
            confidence=confidence,
            regime=regime,
        )
        reason_codes.extend(gate_reasons)

        # Signal strength: magnitude matches score if active, 0.0 if FLAT due to filter/abstention
        signal_strength = ensemble_score if final_signal != SignalDirection.FLAT else 0.0

        return SignalCandidate(
            timestamp=timestamp,
            symbol=symbol,
            signal=final_signal,
            signal_strength=signal_strength,
            confidence=confidence,
            direction_probability=direction_prob,
            expected_return=expected_return,
            regime=regime_str,
            model_contributions=model_contributions,
            model_agreement=model_agreement,
            bullish_votes=bullish_votes,
            bearish_votes=bearish_votes,
            neutral_votes=neutral_votes,
            active_model_count=active_model_count,
            reason_codes=reason_codes,
            signal_horizon=self.config.default_horizon,
            signal_version=self.config.version,
            metadata=metadata or {},
        )

    def generate_signals_sequence(
        self,
        symbol: str,
        timestamps: List[datetime],
        predictions_by_time: List[List[StandardizedPrediction]],
        regimes_by_time: Optional[List[Optional[Union[str, MarketRegimeType, MarketRegimeState]]]] = None,
    ) -> List[SignalCandidate]:
        """
        Generate a chronological sequence of signals across multiple time steps.
        """
        n_steps = len(timestamps)
        if len(predictions_by_time) != n_steps:
            raise ValueError(f"Mismatch between timestamps ({n_steps}) and predictions ({len(predictions_by_time)})")

        regimes = regimes_by_time if regimes_by_time is not None else [None] * n_steps
        if len(regimes) != n_steps:
            raise ValueError(f"Mismatch between timestamps ({n_steps}) and regimes ({len(regimes)})")

        signals: List[SignalCandidate] = []
        for t, preds, reg in zip(timestamps, predictions_by_time, regimes):
            sig = self.generate_signal(
                symbol=symbol,
                timestamp=t,
                predictions=preds,
                regime=reg,
            )
            signals.append(sig)

        return signals

    def evaluate(
        self,
        signals: List[SignalCandidate],
        future_returns: Optional[Union[np.ndarray, List[float]]] = None,
        realized_directions: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> SignalEvaluationReport:
        """
        Run statistical diagnostic evaluation on generated signals.
        """
        return SignalEvaluator.evaluate(
            signals=signals,
            future_returns=future_returns,
            realized_directions=realized_directions,
        )
