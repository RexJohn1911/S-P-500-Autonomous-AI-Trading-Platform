"""
Signal Engine and Alpha Strategy Synthesis Engine (Phase 10).
Exports core schemas, adapters, ensemble combiners, regime gating, evaluator, and SignalEngine orchestrator.
"""

from backend.app.strategy.adapters import (
    BasePredictionAdapter,
    BinaryClassificationAdapter,
    DirectPredictionAdapter,
)
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
from backend.app.strategy.service import SignalEngine
from backend.app.strategy.storage import SignalStorage

__all__ = [
    "SignalDirection",
    "ReasonCode",
    "StandardizedPrediction",
    "SignalCandidate",
    "SignalEngineConfig",
    "SignalEvaluationReport",
    "BasePredictionAdapter",
    "BinaryClassificationAdapter",
    "DirectPredictionAdapter",
    "SignalEnsemble",
    "RegimeGate",
    "SignalEvaluator",
    "SignalStorage",
    "SignalEngine",
]
