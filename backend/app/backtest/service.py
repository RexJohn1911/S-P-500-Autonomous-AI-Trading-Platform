"""
Backtesting Service Orchestrator (Phase 13 Upgrade v1.2).
High-level service for executing end-to-end strategy pipeline backtests,
connecting Signals, Portfolio Construction, Risk Engine, Walk-Forward Evaluation,
and Historical Simulation.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from backend.app.backtest.benchmark import BenchmarkEvaluator
from backend.app.backtest.engine import BacktestEngine
from backend.app.backtest.reporting import BacktestReporter
from backend.app.backtest.schemas import (
    BacktestConfig,
    BacktestResult,
    BenchmarkMetrics,
    DividendEvent,
    SplitEvent,
    WalkForwardFoldConfig,
    WalkForwardResult,
)
from backend.app.backtest.storage import BacktestStorage
from backend.app.backtest.walkforward import WalkForwardConfig, WalkForwardEngine
from backend.app.data.models import BarData
from backend.app.portfolio.schemas import PortfolioTarget
from backend.app.portfolio.service import PortfolioConstructionService
from backend.app.risk.engine import RiskAdjustmentEngine
from backend.app.risk.schemas import PortfolioCapitalContext, RiskAdjustedTarget, RiskContext, RiskEngineConfig
from backend.app.strategy.schemas import SignalCandidate
from backend.app.strategy.service import SignalEngine

logger = logging.getLogger(__name__)


class BacktestService:
    """
    Primary service for running research-grade point-in-time quantitative backtests.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        storage: Optional[BacktestStorage] = None,
        engine: Optional[BacktestEngine] = None,
    ):
        self.config = config or BacktestConfig()
        self.storage = storage or BacktestStorage()
        self.engine = engine or BacktestEngine(config=self.config)

    def run_backtest(
        self,
        market_data: Dict[str, List[BarData]],
        target_series: Optional[Dict[datetime, Union[Dict[str, float], List[RiskAdjustedTarget], List[PortfolioTarget]]]] = None,
        target_generator: Optional[Callable[[datetime, Dict[str, BarData], Any], Dict[str, float]]] = None,
        corporate_actions: Optional[Dict[datetime, List[Union[DividendEvent, SplitEvent]]]] = None,
        benchmark_prices: Optional[Dict[str, float]] = None,
        save_artifacts: bool = False,
        backtest_id: Optional[str] = None,
    ) -> BacktestResult:
        """
        Execute backtest simulation and optionally persist artifacts to disk.
        """
        result = self.engine.run(
            market_data=market_data,
            target_series=target_series,
            target_generator=target_generator,
            corporate_actions=corporate_actions,
            benchmark_prices=benchmark_prices,
            backtest_id=backtest_id,
        )

        if save_artifacts:
            saved_dir = self.storage.save_result(result)
            logger.info("Saved backtest artifacts for ID %s to %s", result.backtest_id, saved_dir)

        return result

    def run_integrated_pipeline_backtest(
        self,
        market_data: Dict[str, List[BarData]],
        signal_generator: Callable[[datetime, Dict[str, BarData]], List[SignalCandidate]],
        portfolio_service: Optional[PortfolioConstructionService] = None,
        risk_config: Optional[RiskEngineConfig] = None,
        corporate_actions: Optional[Dict[datetime, List[Union[DividendEvent, SplitEvent]]]] = None,
        benchmark_prices: Optional[Dict[str, float]] = None,
        save_artifacts: bool = False,
        backtest_id: Optional[str] = None,
    ) -> BacktestResult:
        """
        Execute end-to-end pipeline backtest:
        At bar t:
          1. signal_generator(t, bars_at_t) -> List[SignalCandidate]
          2. portfolio_service.construct(signals, timestamp=t) -> PortfolioConstructionResult
          3. risk_context = RiskContext(as_of_timestamp=t, capital=portfolio_capital_at_t, ...)
          4. RiskAdjustmentEngine.adjust(targets, risk_config, context=risk_context) -> List[RiskAdjustedTarget]
          5. Schedule adjusted targets for execution at bar t+1
        """
        p_service = portfolio_service or PortfolioConstructionService()
        r_config = risk_config or RiskEngineConfig()

        # Build target series using chronological simulation
        timestamp_set = set()
        for sym, bars in market_data.items():
            for b in bars:
                if self.config.start_date and b.timestamp < self.config.start_date:
                    continue
                if self.config.end_date and b.timestamp > self.config.end_date:
                    continue
                timestamp_set.add(b.timestamp)

        sorted_timestamps = sorted(timestamp_set)
        if not sorted_timestamps:
            raise ValueError("No valid bars in specified backtest date range")

        bar_lookup: Dict[datetime, Dict[str, BarData]] = {}
        for sym, bars in market_data.items():
            for b in bars:
                if b.timestamp not in bar_lookup:
                    bar_lookup[b.timestamp] = {}
                bar_lookup[b.timestamp][sym] = b

        target_series: Dict[datetime, List[RiskAdjustedTarget]] = {}

        # Target generator callback that respects point-in-time information
        for ts in sorted_timestamps:
            current_bars = bar_lookup.get(ts, {})
            # 1. Signals generated at t
            signals = signal_generator(ts, current_bars)
            if not signals:
                continue

            # 2. Portfolio construction at t
            p_result = p_service.construct(signals=signals, timestamp=ts)

            # 3. Risk Engine adjustment at t
            known_prices = {s: b.close for s, b in current_bars.items() if b.close > 0}
            risk_context = RiskContext(
                as_of_timestamp=ts,
                asset_prices=known_prices,
            )

            adj_targets, assessment, audit, rejected, was_adj, status = RiskAdjustmentEngine.adjust(
                targets=p_result.targets,
                config=r_config,
                context=risk_context,
            )

            target_series[ts] = adj_targets

        return self.run_backtest(
            market_data=market_data,
            target_series=target_series,
            corporate_actions=corporate_actions,
            benchmark_prices=benchmark_prices,
            save_artifacts=save_artifacts,
            backtest_id=backtest_id,
        )

    def run_walk_forward_backtest(
        self,
        market_data: Dict[str, List[BarData]],
        walk_forward_config: WalkForwardConfig,
        fold_pipeline_fn: Callable[
            [int, WalkForwardFoldConfig, Dict[str, List[BarData]], Optional[Dict[str, List[BarData]]], Dict[str, List[BarData]]],
            BacktestResult
        ],
        walk_forward_id: Optional[str] = None,
    ) -> WalkForwardResult:
        """
        Execute sequential, out-of-sample walk-forward backtest.
        """
        wf_engine = WalkForwardEngine(config=walk_forward_config)
        return wf_engine.run(
            market_data=market_data,
            fold_pipeline_fn=fold_pipeline_fn,
            walk_forward_id=walk_forward_id,
        )

    def generate_report(self, result: Union[BacktestResult, WalkForwardResult]) -> str:
        """Generate markdown summary report."""
        if isinstance(result, WalkForwardResult):
            return BacktestReporter.generate_walk_forward_markdown_report(result)
        return BacktestReporter.generate_markdown_report(result)
