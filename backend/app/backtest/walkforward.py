"""
Walk-Forward Evaluation Subsystem (Phase 13 Upgrade v1.2).
Provides reusable, zero-lookahead walk-forward evaluation across sequential chronological folds
with train-only preprocessing isolation, strict fold boundary enforcement, and out-of-sample concatenation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

from backend.app.backtest.engine import BacktestEngine
from backend.app.backtest.metrics import BacktestMetricsCalculator
from backend.app.backtest.schemas import (
    BacktestConfig,
    BacktestResult,
    BacktestSummary,
    DrawdownPoint,
    EquityPoint,
    ExecutionRecord,
    PerformanceMetrics,
    PortfolioState,
    TradeRecord,
    WalkForwardFoldConfig,
    WalkForwardFoldResult,
    WalkForwardResult,
)
from backend.app.data.models import BarData

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """
    Configuration parameters for walk-forward evaluation.
    """
    train_window_bars: int  # e.g., 252 bars (~1 year)
    test_window_bars: int  # e.g., 63 bars (~1 quarter)
    val_window_bars: int = 0  # Optional validation window (0 if none)
    step_bars: Optional[int] = None  # Step forward distance (defaults to test_window_bars)
    expanding_window: bool = False  # If True, train window expands from start; if False, rolls
    initial_capital: float = 100_000.0
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0005

    def __post_init__(self):
        if self.train_window_bars <= 0:
            raise ValueError(f"train_window_bars must be > 0, got {self.train_window_bars}")
        if self.test_window_bars <= 0:
            raise ValueError(f"test_window_bars must be > 0, got {self.test_window_bars}")
        if self.val_window_bars < 0:
            raise ValueError(f"val_window_bars must be >= 0, got {self.val_window_bars}")
        if self.step_bars is None:
            self.step_bars = self.test_window_bars
        elif self.step_bars <= 0:
            raise ValueError(f"step_bars must be > 0, got {self.step_bars}")


class WalkForwardEngine:
    """
    Executes sequential, point-in-time-isolated walk-forward backtests.
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_folds(self, timestamps: List[datetime]) -> List[WalkForwardFoldConfig]:
        """
        Generate chronological, non-leaking fold configurations.
        """
        n_total = len(timestamps)
        req_window = self.config.train_window_bars + self.config.val_window_bars + self.config.test_window_bars

        if n_total < req_window:
            raise ValueError(
                f"Insufficient market data bars ({n_total}) for required walk-forward window ({req_window})"
            )

        folds: List[WalkForwardFoldConfig] = []
        fold_idx = 0
        current_train_start_idx = 0

        while True:
            train_end_idx = current_train_start_idx + self.config.train_window_bars - 1
            val_start_idx = train_end_idx + 1 if self.config.val_window_bars > 0 else None
            val_end_idx = (
                val_start_idx + self.config.val_window_bars - 1
                if val_start_idx is not None
                else None
            )

            test_start_idx = (val_end_idx + 1) if val_end_idx is not None else (train_end_idx + 1)
            test_end_idx = test_start_idx + self.config.test_window_bars - 1

            if test_end_idx >= n_total:
                # If cannot form a full test window, break
                break

            fold_cfg = WalkForwardFoldConfig(
                fold_idx=fold_idx,
                train_start=timestamps[current_train_start_idx],
                train_end=timestamps[train_end_idx],
                val_start=timestamps[val_start_idx] if val_start_idx is not None else None,
                val_end=timestamps[val_end_idx] if val_end_idx is not None else None,
                test_start=timestamps[test_start_idx],
                test_end=timestamps[test_end_idx],
            )
            folds.append(fold_cfg)

            # Advance window
            fold_idx += 1
            step = self.config.step_bars or self.config.test_window_bars
            if not self.config.expanding_window:
                current_train_start_idx += step
            else:
                # In expanding window, train_start stays 0, train_window_bars increases
                # Here we adjust the next slice by increasing train_window_bars
                self.config.train_window_bars += step

        if not folds:
            raise ValueError("No valid walk-forward folds could be constructed with the given configuration")

        return folds

    def run(
        self,
        market_data: Dict[str, List[BarData]],
        fold_pipeline_fn: Callable[
            [int, WalkForwardFoldConfig, Dict[str, List[BarData]], Optional[Dict[str, List[BarData]]], Dict[str, List[BarData]]],
            BacktestResult
        ],
        walk_forward_id: Optional[str] = None,
        evaluation_timestamp: Optional[datetime] = None,
    ) -> WalkForwardResult:
        """
        Execute walk-forward pipeline across all generated folds.
        `fold_pipeline_fn(fold_idx, fold_config, train_data, val_data, test_data)` returns out-of-sample BacktestResult.
        """
        if not market_data:
            raise ValueError("market_data cannot be empty")

        wf_id = walk_forward_id or f"wf_{uuid.uuid4().hex[:10]}"
        eval_time = evaluation_timestamp or datetime.now(timezone.utc)

        # 1. Extract and sort all unique timestamps
        timestamp_set = set()
        for bars in market_data.values():
            for b in bars:
                timestamp_set.add(b.timestamp)
        sorted_timestamps = sorted(timestamp_set)

        # 2. Generate folds
        folds_config = self.generate_folds(sorted_timestamps)

        fold_results: List[WalkForwardFoldResult] = []
        all_oos_equity_points: List[EquityPoint] = []
        all_oos_trades: List[TradeRecord] = []
        all_oos_executions: List[ExecutionRecord] = []
        all_oos_portfolio_states: List[PortfolioState] = []

        cumulative_capital = float(self.config.initial_capital)

        for fold_cfg in folds_config:
            # Slice point-in-time data for train, validation, test
            train_data: Dict[str, List[BarData]] = {}
            val_data: Optional[Dict[str, List[BarData]]] = {} if fold_cfg.val_start else None
            test_data: Dict[str, List[BarData]] = {}

            for sym, bars in market_data.items():
                train_data[sym] = [b for b in bars if fold_cfg.train_start <= b.timestamp <= fold_cfg.train_end]
                if fold_cfg.val_start and fold_cfg.val_end and val_data is not None:
                    val_data[sym] = [b for b in bars if fold_cfg.val_start <= b.timestamp <= fold_cfg.val_end]
                if fold_cfg.test_start and fold_cfg.test_end:
                    test_data[sym] = [b for b in bars if fold_cfg.test_start <= b.timestamp <= fold_cfg.test_end]

            # Execute user's fold pipeline function (trains models strictly on train_data, tests on test_data)
            oos_result = fold_pipeline_fn(
                fold_cfg.fold_idx,
                fold_cfg,
                train_data,
                val_data,
                test_data,
            )

            fold_res = WalkForwardFoldResult(
                fold_idx=fold_cfg.fold_idx,
                fold_config=fold_cfg,
                out_of_sample_metrics=oos_result.metrics,
                out_of_sample_result=oos_result,
            )
            fold_results.append(fold_res)

            # Accumulate out-of-sample history
            all_oos_equity_points.extend(oos_result.equity_curve)
            all_oos_trades.extend(oos_result.trades)
            all_oos_executions.extend(oos_result.executions)
            all_oos_portfolio_states.extend(oos_result.portfolio_history)

        # 3. Build overall stitched Out-of-Sample Result
        # Sort and deduplicate stitched equity points by timestamp
        unique_eq_dict: Dict[datetime, EquityPoint] = {}
        for ep in all_oos_equity_points:
            unique_eq_dict[ep.timestamp] = ep

        stitched_equity_curve = [unique_eq_dict[k] for k in sorted(unique_eq_dict.keys())]

        overall_metrics = BacktestMetricsCalculator.compute_all_metrics(
            equity_curve=stitched_equity_curve,
            trades=all_oos_trades,
            executions=all_oos_executions,
            initial_capital=self.config.initial_capital,
        )

        base_config = BacktestConfig(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
        )

        summary = BacktestSummary(
            backtest_id=wf_id,
            timestamp=eval_time,
            start_date=stitched_equity_curve[0].timestamp if stitched_equity_curve else sorted_timestamps[0],
            end_date=stitched_equity_curve[-1].timestamp if stitched_equity_curve else sorted_timestamps[-1],
            total_bars=len(stitched_equity_curve),
            symbols=sorted(list(market_data.keys())),
            initial_capital=self.config.initial_capital,
            final_equity=stitched_equity_curve[-1].equity if stitched_equity_curve else self.config.initial_capital,
            total_return=overall_metrics.total_return,
            annualized_return=overall_metrics.annualized_return,
            sharpe_ratio=overall_metrics.sharpe_ratio,
            max_drawdown=overall_metrics.max_drawdown,
            total_trades=overall_metrics.total_trades,
            win_rate=overall_metrics.win_rate,
        )

        concat_result = BacktestResult(
            backtest_id=wf_id,
            timestamp=eval_time,
            config=base_config,
            summary=summary,
            metrics=overall_metrics,
            equity_curve=stitched_equity_curve,
            trades=all_oos_trades,
            executions=all_oos_executions,
            portfolio_history=all_oos_portfolio_states,
            metadata={
                "walk_forward_id": wf_id,
                "total_folds": len(fold_results),
                "evaluation_type": "walk_forward_out_of_sample",
            },
        )

        return WalkForwardResult(
            walk_forward_id=wf_id,
            timestamp=eval_time,
            config=base_config,
            folds=fold_results,
            concatenated_result=concat_result,
            summary=summary,
            overall_metrics=overall_metrics,
            provenance={
                "walk_forward_engine_version": "wf-v1.2",
                "train_window_bars": self.config.train_window_bars,
                "val_window_bars": self.config.val_window_bars,
                "test_window_bars": self.config.test_window_bars,
                "step_bars": self.config.step_bars,
                "expanding_window": self.config.expanding_window,
            },
        )
