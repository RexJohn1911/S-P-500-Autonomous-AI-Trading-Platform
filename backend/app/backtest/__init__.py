"""
Phase 13 Backtesting Engine Package (Upgrade v1.2 Research-Grade Framework).
Deterministic, point-in-time-correct historical simulation and walk-forward evaluation engine.
"""

from backend.app.backtest.benchmark import BenchmarkEvaluator
from backend.app.backtest.costs import CostModel
from backend.app.backtest.engine import BacktestEngine
from backend.app.backtest.execution import SimulatedExecutionEngine
from backend.app.backtest.metrics import BacktestMetricsCalculator
from backend.app.backtest.portfolio import SimulatedPortfolio
from backend.app.backtest.reporting import BacktestReporter
from backend.app.backtest.schemas import (
    AdjustmentStatus,
    BacktestConfig,
    BacktestResult,
    BacktestSummary,
    BenchmarkMetrics,
    CostBreakdown,
    DataSourceType,
    DatasetProvenance,
    DividendEvent,
    DrawdownEpisode,
    DrawdownPoint,
    EquityPoint,
    ExecutionPriceType,
    ExecutionRecord,
    ExposurePoint,
    OrderIntent,
    OrderSide,
    PerformanceMetrics,
    PeriodPerformance,
    PortfolioState,
    PositionSide,
    PositionState,
    SplitEvent,
    StatisticalDiagnostics,
    TradeRecord,
    WalkForwardFoldConfig,
    WalkForwardFoldResult,
    WalkForwardResult,
)
from backend.app.backtest.service import BacktestService
from backend.app.backtest.storage import BacktestStorage
from backend.app.backtest.walkforward import WalkForwardConfig, WalkForwardEngine

__all__ = [
    "AdjustmentStatus",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetricsCalculator",
    "BacktestReporter",
    "BacktestResult",
    "BacktestService",
    "BacktestStorage",
    "BacktestSummary",
    "BenchmarkEvaluator",
    "BenchmarkMetrics",
    "CostBreakdown",
    "CostModel",
    "DataSourceType",
    "DatasetProvenance",
    "DividendEvent",
    "DrawdownEpisode",
    "DrawdownPoint",
    "EquityPoint",
    "ExecutionPriceType",
    "ExecutionRecord",
    "ExposurePoint",
    "OrderIntent",
    "OrderSide",
    "PerformanceMetrics",
    "PeriodPerformance",
    "PortfolioState",
    "PositionSide",
    "PositionState",
    "SimulatedExecutionEngine",
    "SimulatedPortfolio",
    "SplitEvent",
    "StatisticalDiagnostics",
    "TradeRecord",
    "WalkForwardConfig",
    "WalkForwardEngine",
    "WalkForwardFoldConfig",
    "WalkForwardFoldResult",
    "WalkForwardResult",
]
