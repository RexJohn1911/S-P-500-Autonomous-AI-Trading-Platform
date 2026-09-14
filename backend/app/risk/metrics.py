"""
Risk Metrics Calculation Subsystem (Phase 12 Hardening v1.1).
Calculates deterministic portfolio exposure, leverage, point-in-time volatility,
empirical correlation, turnover, and capital notionals with strict zero look-ahead leakage protection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.risk.schemas import PortfolioCapitalContext, RiskContext, RiskMetrics


class RiskMetricsCalculator:
    """
    Calculates portfolio-level risk metrics deterministically.
    
    SCIENTIFIC INTEGRITY RULES:
    1. If point-in-time data is unavailable for higher-order metrics (volatility, correlation, drawdown),
       returns None and explicitly records the metric in unavailable_metrics.
    2. Zero look-ahead: Uses only return observations <= timestamp t provided in RiskContext.
    3. Leverage is defined explicitly as: gross_notional / portfolio_equity = sum(|weight|).
    4. Never manufactures or fabricates financial estimates.
    """

    @classmethod
    def calculate(
        cls,
        targets: List[PortfolioTarget],
        context: Optional[RiskContext] = None,
    ) -> RiskMetrics:
        """
        Compute risk metrics for the given portfolio targets and point-in-time context.
        """
        if not targets:
            equity = context.capital.equity if (context and context.capital) else None
            return RiskMetrics(
                gross_exposure=0.0,
                net_exposure=0.0,
                total_long_weight=0.0,
                total_short_weight=0.0,
                active_position_count=0,
                max_observed_position_weight=0.0,
                leverage=0.0,
                portfolio_equity=equity,
                gross_notional=0.0 if equity is not None else None,
                net_notional=0.0 if equity is not None else None,
                portfolio_volatility=None,
                average_correlation=None,
                turnover=0.0 if (context and context.previous_weights is not None) else None,
                highly_correlated_pairs=[],
                drawdown_limit_status="UNAVAILABLE",
                daily_loss_limit_status="UNAVAILABLE",
                unavailable_metrics=["portfolio_volatility", "average_correlation"],
            )

        # Basic exposure calculations
        weights = [float(t.target_weight) for t in targets]
        symbols = [t.symbol for t in targets]

        gross_exposure = float(sum(abs(w) for w in weights))
        net_exposure = float(sum(weights))
        total_long_weight = float(sum(w for w in weights if w > 0))
        total_short_weight = float(sum(abs(w) for w in weights if w < 0))
        active_positions = [t for t in targets if abs(t.target_weight) > 1e-7]
        active_position_count = len(active_positions)
        max_observed_weight = float(max(abs(w) for w in weights)) if weights else 0.0

        # Capital & Notional calculations
        portfolio_equity: Optional[float] = None
        gross_notional: Optional[float] = None
        net_notional: Optional[float] = None
        if context and context.capital:
            portfolio_equity = float(context.capital.equity)
            gross_notional = float(gross_exposure * portfolio_equity)
            net_notional = float(net_exposure * portfolio_equity)

        # Leverage: gross_notional / portfolio_equity = sum(|weight|)
        leverage = float(gross_exposure)

        unavailable_metrics: List[str] = []

        # Volatility calculation (strictly point-in-time from context)
        portfolio_volatility, vol_unavailable = cls._calculate_portfolio_volatility(
            active_positions, context
        )
        if vol_unavailable:
            unavailable_metrics.append("portfolio_volatility")

        # Correlation & Highly Correlated Pairs calculation
        average_correlation, high_corr_pairs, corr_unavailable = cls._calculate_correlation_diagnostics(
            active_positions, context
        )
        if corr_unavailable:
            unavailable_metrics.append("average_correlation")

        # Turnover calculation (if previous weights are provided)
        turnover = cls._calculate_turnover(targets, context)

        return RiskMetrics(
            gross_exposure=round(gross_exposure, 6),
            net_exposure=round(net_exposure, 6),
            total_long_weight=round(total_long_weight, 6),
            total_short_weight=round(total_short_weight, 6),
            active_position_count=active_position_count,
            max_observed_position_weight=round(max_observed_weight, 6),
            leverage=round(leverage, 6),
            portfolio_equity=round(portfolio_equity, 2) if portfolio_equity is not None else None,
            gross_notional=round(gross_notional, 2) if gross_notional is not None else None,
            net_notional=round(net_notional, 2) if net_notional is not None else None,
            portfolio_volatility=round(portfolio_volatility, 6) if portfolio_volatility is not None else None,
            average_correlation=round(average_correlation, 6) if average_correlation is not None else None,
            turnover=round(turnover, 6) if turnover is not None else None,
            highly_correlated_pairs=high_corr_pairs,
            drawdown_limit_status="UNAVAILABLE",
            daily_loss_limit_status="UNAVAILABLE",
            unavailable_metrics=sorted(unavailable_metrics),
        )

    @classmethod
    def _calculate_portfolio_volatility(
        cls,
        active_targets: List[PortfolioTarget],
        context: Optional[RiskContext],
    ) -> Tuple[Optional[float], bool]:
        """
        Calculate annualized portfolio volatility: sqrt(w^T * Sigma * w) * sqrt(252).
        Returns (volatility, is_unavailable_bool).
        """
        if not active_targets:
            return 0.0, False

        if context is None or not context.historical_returns:
            return None, True

        hist_returns = context.historical_returns
        active_symbols = [t.symbol for t in active_targets]
        active_weights = np.array([t.target_weight for t in active_targets], dtype=float)

        for sym in active_symbols:
            if sym not in hist_returns or not hist_returns[sym]:
                return None, True

        lengths = [len(hist_returns[sym]) for sym in active_symbols]
        min_len = min(lengths)
        if min_len < 5 or len(set(lengths)) > 1:
            return None, True

        return_matrix = np.column_stack([hist_returns[sym] for sym in active_symbols])

        if np.any(np.isnan(return_matrix)) or np.any(np.isinf(return_matrix)):
            return None, True

        cov_matrix = np.cov(return_matrix, rowvar=False)
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]])

        port_variance = float(np.dot(active_weights.T, np.dot(cov_matrix, active_weights)))
        if port_variance < 0.0:
            port_variance = 0.0

        port_volatility = float(np.sqrt(port_variance) * np.sqrt(252))
        return port_volatility, False

    @classmethod
    def _calculate_correlation_diagnostics(
        cls,
        active_targets: List[PortfolioTarget],
        context: Optional[RiskContext],
        threshold: float = 0.80,
    ) -> Tuple[Optional[float], List[Dict[str, Any]], bool]:
        """
        Calculate average off-diagonal pairwise correlation and identify highly correlated pairs.
        Returns (avg_correlation, high_corr_pairs, is_unavailable_bool).
        """
        if len(active_targets) < 2:
            return None, [], True

        if context is None or not context.historical_returns:
            return None, [], True

        hist_returns = context.historical_returns
        active_symbols = [t.symbol for t in active_targets]

        for sym in active_symbols:
            if sym not in hist_returns or not hist_returns[sym]:
                return None, [], True

        lengths = [len(hist_returns[sym]) for sym in active_symbols]
        min_len = min(lengths)
        if min_len < 5 or len(set(lengths)) > 1:
            return None, [], True

        return_matrix = np.column_stack([hist_returns[sym] for sym in active_symbols])
        if np.any(np.isnan(return_matrix)) or np.any(np.isinf(return_matrix)):
            return None, [], True

        corr_matrix = np.corrcoef(return_matrix, rowvar=False)
        if np.any(np.isnan(corr_matrix)) or np.any(np.isinf(corr_matrix)):
            return None, [], True

        n = len(active_symbols)
        off_diag_mask = ~np.eye(n, dtype=bool)
        avg_corr = float(np.mean(corr_matrix[off_diag_mask]))

        # High correlation pairs detection
        high_corr_pairs: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                c_val = float(corr_matrix[i, j])
                if abs(c_val) >= threshold:
                    high_corr_pairs.append({
                        "symbol_1": active_symbols[i],
                        "symbol_2": active_symbols[j],
                        "correlation": round(c_val, 4),
                    })

        return avg_corr, high_corr_pairs, False

    @classmethod
    def _calculate_turnover(
        cls,
        targets: List[PortfolioTarget],
        context: Optional[RiskContext],
    ) -> Optional[float]:
        """
        Calculate portfolio turnover relative to previous weights: sum(|w_target - w_prev|).
        Returns None if previous weights are unavailable.
        """
        if context is None or context.previous_weights is None:
            return None

        target_weights = {t.symbol: float(t.target_weight) for t in targets}
        prev_weights = {k: float(v) for k, v in context.previous_weights.items()}

        all_symbols = set(target_weights.keys()) | set(prev_weights.keys())
        turnover = sum(
            abs(target_weights.get(sym, 0.0) - prev_weights.get(sym, 0.0))
            for sym in all_symbols
        )
        return float(turnover)
