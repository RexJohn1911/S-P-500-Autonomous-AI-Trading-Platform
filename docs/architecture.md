# System Architecture Specification

## 1. Architectural Philosophy
The S&P 500 Autonomous AI Trading Platform is designed around three foundational principles:
1. **Strict Temporal Causality (Zero Lookahead Bias)**: Every component operates strictly point-in-time. Any computation at timestamp $t$ has zero access to data from $t' > t$.
2. **Execution Decoupling**: Trading signals, portfolio allocations, and risk adjustments are computed independently of the execution layer. The same strategy pipeline can drive backtesting, simulated paper trading, or live broker execution without altering a single line of strategy code.
3. **Single-Worker Process Integrity**: The autonomous loop state machine, scheduler, Global Kill Switch, in-memory event bus, and reconciliation watchdog operate under a single ASGI worker process (`WORKERS_COUNT=1`) to prevent duplicated orders, race conditions, and out-of-sync state across concurrent processes.

---

## 2. End-to-End Pipeline Data Flow

```
[ Market Data Providers (Alpaca / Yahoo / Mock) ]
                       ↓
[ Market Data Layer (app/data) ]
   • Schema normalization (BarData, QuoteData, MacroData)
   • Data validation, anomaly detection, UTC normalization
                       ↓
[ Feature Engineering (app/features) ]
   • 25+ statistical & technical indicators
   • Point-in-time window calculations
                       ↓
      +----------------+----------------+
      ↓                                 ↓
[ Machine Learning Models ]   [ Regime Detection Engine ]
   • Baseline ML (XGB, LGBM)     • K-Means, GMM, HMM
   • Deep Learning (Transformer) • Semantic Trend/Volatility States
      ↓                                 ↓
      +----------------+----------------+
                       ↓
[ Signal Engine & Ensemble (app/strategy) ]
   • Standardized predictions & directional scores
   • Weighted ensemble combination & model agreement
   • Regime gating & reason code assignment
                       ↓
[ Portfolio Construction (app/portfolio) ]
   • Long-only / long-short target allocation
   • Position capping, min-weight filtering, ranking
                       ↓
[ Risk Engine (app/risk) ]
   • Fail-closed constraint checks (hard limits)
   • Covariance / volatility estimation
   • Target weight adjustments & notional calculation
                       ↓
[ Execution Dispatcher ]
   • Backtesting Engine (app/backtest) -> Research simulation
   • Paper Trading Engine (app/paper_trading) -> FIFO accounting ledger
   • Broker Adapter (app/broker) -> Live/Sandbox REST execution
                       ↓
[ Monitoring & Safety Layer (app/monitoring) ]
   • 13-component health telemetry
   • Pre-flight 12-point safety gates
   • Double-entry reconciliation watchdog
   • Global Kill Switch circuit breaker
                       ↓
[ Web Dashboard API & UI (app/api + frontend) ]
   • 17 real-time monitoring views
   • Read-heavy safety architecture
```

---

## 3. Subsystem Breakdown

### 3.1 Market Data Layer (`app/data`)
- Normalizes raw vendor data into immutable dataclasses: `BarData`, `QuoteData`, `TickData`, `MacroData`.
- Enforces strict timezone normalization to UTC (`ensure_utc`), with America/New_York market timestamp properties.
- Storage partitioned cleanly in `data/raw/{symbol}/{timeframe}/bars.json`.

### 3.2 Feature Engineering (`app/features`)
- Modular feature generators implementing `BaseFeatureGenerator`:
  - Returns: `return_1d`, `return_5d`, `return_20d`
  - Volatility: `rolling_volatility_20d`, `atr_14`
  - Trend: `price_vs_sma_20`, `price_vs_sma_50`, `price_vs_sma_200`, `macd`, `rsi_14`
  - Volume: `volume_zscore_20`
  - Market Relative: `sp500_relative_return_20d`, `beta_60d`

### 3.3 Strategy & Signal Engine (`app/strategy`)
- Normalizes multi-model probability outputs into `StandardizedPrediction`.
- Dynamic ensemble aggregation with configurable weights and agreement thresholds (`min_agreement = 0.50`).
- Generates `SignalCandidate` objects with explainability reason codes.

### 3.4 Portfolio & Risk Subsystems (`app/portfolio`, `app/risk`)
- `PortfolioConstructionService`: Optimizes weights according to signal confidence, subject to gross exposure and max position constraints.
- `RiskEngineService`: Hard constraint evaluation enforcing max position weights, gross/net leverage, and point-in-time covariance diagnostics. Returns `RiskAdjustedTarget`.

### 3.5 Execution Engines (`app/backtest`, `app/paper_trading`, `app/broker`)
- `BacktestingEngine`: Walk-forward simulation with next-bar execution causality, modeled transaction costs, borrow fees, and corporate actions.
- `PaperTradingService`: Simulated execution broker with FIFO realization, cash accounting, and persistence.
- `BrokerAdapter`: Normalized broker abstraction for Alpaca API integration with idempotent submission.

### 3.6 Monitoring & Safety Subsystem (`app/monitoring`)
- `GlobalKillSwitch`: Persistent emergency stop mechanism.
- `SystemWatchdog`: 13-component health aggregation and heartbeat tracker.
- `PaperReconciliationEngine`: Real-time ledger discrepancy watchdog.
