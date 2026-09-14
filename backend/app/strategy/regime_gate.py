"""
Regime-Aware Signal Gating Module.
Applies configurable regime-based gating and directional filters without altering position sizing or portfolio risk constraints.
Consumes standardized Phase 09 semantic regime representations (MarketRegimeState, MarketRegimeType, or string names)
and does not interpret raw detector cluster IDs directly.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from backend.app.regime.schemas import MarketRegimeState, MarketRegimeType
from backend.app.strategy.schemas import ReasonCode, SignalDirection, SignalEngineConfig


class RegimeGate:
    """
    Evaluates proposed directional signals against macroeconomic / market regime context.
    Provides transparent permission filtering and elevated evidence thresholds.
    """

    def __init__(self, config: Optional[SignalEngineConfig] = None):
        self.config = config or SignalEngineConfig()

    def evaluate_gate(
        self,
        proposed_signal: SignalDirection,
        ensemble_score: float,
        confidence: float,
        regime: Optional[Union[str, MarketRegimeType, MarketRegimeState, Any]] = None,
    ) -> Tuple[SignalDirection, List[str]]:
        """
        Evaluate proposed signal against regime state.

        Args:
            proposed_signal: Direction proposed by ensemble (LONG, SHORT, FLAT).
            ensemble_score: Continuous directional score in [-1.0, 1.0].
            confidence: Model conviction score in [0.0, 1.0].
            regime: Standardized Phase 09 regime (MarketRegimeState, MarketRegimeType, or semantic string name).

        Returns:
            - effective_signal: SignalDirection (LONG, SHORT, or FLAT)
            - reason_codes: List[str] explaining the gating decision
        """
        if proposed_signal == SignalDirection.FLAT:
            return SignalDirection.FLAT, []

        if not self.config.regime_gating_enabled:
            return proposed_signal, [ReasonCode.REGIME_GATING_DISABLED.value]

        # Extract semantic regime string name safely
        regime_name: Optional[str] = None
        if isinstance(regime, MarketRegimeState):
            regime_name = regime.regime_name
        elif isinstance(regime, MarketRegimeType):
            regime_name = regime.value
        elif isinstance(regime, str):
            regime_name = regime
        else:
            # Numeric cluster IDs or unrecognized types are not interpreted as semantic regimes
            return proposed_signal, [ReasonCode.REGIME_UNKNOWN_DEFAULT.value]

        if not regime_name or regime_name.upper() in ("UNKNOWN", "TRANSITIONAL", "NONE"):
            return proposed_signal, [ReasonCode.REGIME_UNKNOWN_DEFAULT.value]

        reg_upper = regime_name.upper()
        policy = self.config.counter_trend_policy.lower()

        # 1. Bullish Regimes: BULL_TRENDING, LOW_VOLATILITY_BULL
        if "BULL" in reg_upper:
            if proposed_signal == SignalDirection.LONG:
                return SignalDirection.LONG, [ReasonCode.REGIME_PERMITTED_BULL.value]
            elif proposed_signal == SignalDirection.SHORT:
                if policy == "filter":
                    return SignalDirection.FLAT, [ReasonCode.REGIME_BLOCKED_COUNTER_TREND.value]
                elif policy == "penalize":
                    # Elevated criteria: require more extreme negative score and higher confidence
                    elevated_short_threshold = self.config.short_threshold * 1.5  # e.g. -0.20 -> -0.30
                    elevated_confidence = self.config.min_confidence * self.config.sideways_confidence_multiplier
                    if ensemble_score <= elevated_short_threshold and confidence >= elevated_confidence:
                        return SignalDirection.SHORT, [ReasonCode.REGIME_PERMITTED_BULL.value]
                    else:
                        return SignalDirection.FLAT, [ReasonCode.REGIME_ELEVATED_THRESHOLD_REJECTED.value]
                else:  # "allow"
                    return SignalDirection.SHORT, [ReasonCode.REGIME_PERMITTED_BULL.value]

        # 2. Bearish Regimes: BEAR_TRENDING, HIGH_VOLATILITY_BEAR
        if "BEAR" in reg_upper:
            if proposed_signal == SignalDirection.SHORT:
                return SignalDirection.SHORT, [ReasonCode.REGIME_PERMITTED_BEAR.value]
            elif proposed_signal == SignalDirection.LONG:
                if policy == "filter":
                    return SignalDirection.FLAT, [ReasonCode.REGIME_BLOCKED_COUNTER_TREND.value]
                elif policy == "penalize":
                    elevated_long_threshold = self.config.long_threshold * 1.5  # e.g. +0.20 -> +0.30
                    elevated_confidence = self.config.min_confidence * self.config.sideways_confidence_multiplier
                    if ensemble_score >= elevated_long_threshold and confidence >= elevated_confidence:
                        return SignalDirection.LONG, [ReasonCode.REGIME_PERMITTED_BEAR.value]
                    else:
                        return SignalDirection.FLAT, [ReasonCode.REGIME_ELEVATED_THRESHOLD_REJECTED.value]
                else:  # "allow"
                    return SignalDirection.LONG, [ReasonCode.REGIME_PERMITTED_BEAR.value]

        # 3. Sideways / High Volatility Regimes: SIDEWAYS_NEUTRAL, HIGH_VOLATILITY
        if "SIDEWAYS" in reg_upper or "HIGH_VOLATILITY" in reg_upper:
            required_confidence = self.config.min_confidence * self.config.sideways_confidence_multiplier
            if confidence < required_confidence:
                return SignalDirection.FLAT, [ReasonCode.REGIME_SIDEWAYS_FILTER.value]
            else:
                return proposed_signal, []

        return proposed_signal, [ReasonCode.REGIME_UNKNOWN_DEFAULT.value]
