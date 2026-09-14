"""
Signal Evaluation Module.
Provides diagnostic evaluation of directional signal quality, coverage, precision, recall, and forward return distributions.
NOTE: This is diagnostic signal evaluation, NOT a trading backtest or P&L execution simulation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.strategy.schemas import SignalCandidate, SignalDirection, SignalEvaluationReport


class SignalEvaluator:
    """
    Evaluator for analyzing statistical signal performance, directional alignment, and signal distributions.
    """

    @staticmethod
    def evaluate(
        signals: List[SignalCandidate],
        future_returns: Optional[Union[np.ndarray, List[float]]] = None,
        realized_directions: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> SignalEvaluationReport:
        """
        Evaluate a sequence of SignalCandidate objects.

        Args:
            signals: List of SignalCandidate instances to evaluate.
            future_returns: Optional 1D array of realized forward returns aligned with signals.
            realized_directions: Optional 1D array of realized binary directions (1 = up, 0 = down).

        Returns:
            SignalEvaluationReport with comprehensive signal diagnostics.
        """
        total_signals = len(signals)
        if total_signals == 0:
            return SignalEvaluationReport(
                total_signals=0,
                long_count=0,
                short_count=0,
                flat_count=0,
                long_percentage=0.0,
                short_percentage=0.0,
                flat_percentage=0.0,
            )

        long_signals = [s for s in signals if s.signal == SignalDirection.LONG]
        short_signals = [s for s in signals if s.signal == SignalDirection.SHORT]
        flat_signals = [s for s in signals if s.signal == SignalDirection.FLAT]

        long_count = len(long_signals)
        short_count = len(short_signals)
        flat_count = len(flat_signals)

        long_pct = long_count / total_signals
        short_pct = short_count / total_signals
        flat_pct = flat_count / total_signals

        mean_confidence = float(np.mean([s.confidence for s in signals]))
        mean_agreement = float(np.mean([s.model_agreement for s in signals]))

        start_dt = min(s.timestamp for s in signals)
        end_dt = max(s.timestamp for s in signals)

        # Forward return statistics per signal type
        mean_ret_long: Optional[float] = None
        mean_ret_short: Optional[float] = None
        mean_ret_flat: Optional[float] = None
        return_spread: Optional[float] = None

        if future_returns is not None:
            returns_arr = np.asarray(future_returns).flatten()
            if len(returns_arr) == total_signals:
                long_indices = [i for i, s in enumerate(signals) if s.signal == SignalDirection.LONG]
                short_indices = [i for i, s in enumerate(signals) if s.signal == SignalDirection.SHORT]
                flat_indices = [i for i, s in enumerate(signals) if s.signal == SignalDirection.FLAT]

                if long_indices:
                    mean_ret_long = float(np.nanmean(returns_arr[long_indices]))
                if short_indices:
                    mean_ret_short = float(np.nanmean(returns_arr[short_indices]))
                if flat_indices:
                    mean_ret_flat = float(np.nanmean(returns_arr[flat_indices]))

                if mean_ret_long is not None and mean_ret_short is not None:
                    return_spread = mean_ret_long - mean_ret_short

        # Directional accuracy / precision / recall / F1
        directional_accuracy: Optional[float] = None
        long_precision: Optional[float] = None
        long_recall: Optional[float] = None
        long_f1: Optional[float] = None
        short_precision: Optional[float] = None
        short_recall: Optional[float] = None
        short_f1: Optional[float] = None

        # Derive realized direction from future_returns if not directly provided
        if realized_directions is None and future_returns is not None:
            returns_arr = np.asarray(future_returns).flatten()
            if len(returns_arr) == total_signals:
                realized_directions = (returns_arr > 0.0).astype(int)

        if realized_directions is not None:
            dirs = np.asarray(realized_directions).flatten()
            if len(dirs) == total_signals:
                # Active directional signals (excluding FLAT)
                active_pairs = [(s.signal, dirs[i]) for i, s in enumerate(signals) if s.signal != SignalDirection.FLAT]
                if active_pairs:
                    correct = sum(
                        1 for sig, d in active_pairs
                        if (sig == SignalDirection.LONG and d == 1) or (sig == SignalDirection.SHORT and d == 0)
                    )
                    directional_accuracy = float(correct / len(active_pairs))

                # Binary classification metrics for LONG (positive class = 1)
                actual_positives = sum(1 for d in dirs if d == 1)
                actual_negatives = sum(1 for d in dirs if d == 0)

                tp_long = sum(1 for i, s in enumerate(signals) if s.signal == SignalDirection.LONG and dirs[i] == 1)
                fp_long = sum(1 for i, s in enumerate(signals) if s.signal == SignalDirection.LONG and dirs[i] == 0)
                fn_long = sum(1 for i, s in enumerate(signals) if s.signal != SignalDirection.LONG and dirs[i] == 1)

                long_precision = float(tp_long / (tp_long + fp_long)) if (tp_long + fp_long) > 0 else 0.0
                long_recall = float(tp_long / (tp_long + fn_long)) if (tp_long + fn_long) > 0 else 0.0
                long_f1 = (
                    float(2.0 * long_precision * long_recall / (long_precision + long_recall))
                    if (long_precision + long_recall) > 0 else 0.0
                )

                # Binary classification metrics for SHORT (positive class = 0 / down)
                tp_short = sum(1 for i, s in enumerate(signals) if s.signal == SignalDirection.SHORT and dirs[i] == 0)
                fp_short = sum(1 for i, s in enumerate(signals) if s.signal == SignalDirection.SHORT and dirs[i] == 1)
                fn_short = sum(1 for i, s in enumerate(signals) if s.signal != SignalDirection.SHORT and dirs[i] == 0)

                short_precision = float(tp_short / (tp_short + fp_short)) if (tp_short + fp_short) > 0 else 0.0
                short_recall = float(tp_short / (tp_short + fn_short)) if (tp_short + fn_short) > 0 else 0.0
                short_f1 = (
                    float(2.0 * short_precision * short_recall / (short_precision + short_recall))
                    if (short_precision + short_recall) > 0 else 0.0
                )

        return SignalEvaluationReport(
            total_signals=total_signals,
            long_count=long_count,
            short_count=short_count,
            flat_count=flat_count,
            long_percentage=long_pct,
            short_percentage=short_pct,
            flat_percentage=flat_pct,
            directional_accuracy=directional_accuracy,
            long_precision=long_precision,
            long_recall=long_recall,
            long_f1=long_f1,
            short_precision=short_precision,
            short_recall=short_recall,
            short_f1=short_f1,
            mean_forward_return_long=mean_ret_long,
            mean_forward_return_short=mean_ret_short,
            mean_forward_return_flat=mean_ret_flat,
            return_spread=return_spread,
            mean_confidence=mean_confidence,
            mean_agreement=mean_agreement,
            evaluation_period_start=start_dt,
            evaluation_period_end=end_dt,
            metadata={
                "type": "diagnostic_signal_evaluation",
                "disclaimer": "Diagnostic signal quality metrics only. Does not simulate execution, fills, slippage, or portfolio P&L.",
            },
        )
