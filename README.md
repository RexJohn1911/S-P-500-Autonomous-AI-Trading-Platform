# S&P 500 Autonomous AI Trading Platform

[![CI / Regression Tests](https://img.shields.io/badge/tests-372%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)]()
[![Frontend](https://img.shields.io/badge/react-19%20%7C%20vite-61dafb.svg)]()
[![Status](https://img.shields.io/badge/release-v1.0.0%20(Audited)-blueviolet.svg)]()
[![License](https://img.shields.io/badge/license-MIT-informational.svg)]()

> **Research-Grade Autonomous Quantitative Trading Platform** designed for S&P 500 equities research, point-in-time machine learning, market regime detection, multi-model ensemble forecasting, risk-constrained portfolio construction, research-grade backtesting, simulated paper trading, broker abstraction (Alpaca integration), fail-closed safety monitoring, and a real subsystem-backed web dashboard.

---

## 1. Overview
The **S&P 500 Autonomous AI Trading Platform** is an institutional-grade quantitative trading and research system built to execute systematic trading strategies across equities with strict causal invariance (zero lookahead bias), double-entry accounting integrity, fail-closed risk management, and process-isolated deployment.

The platform provides a unified pipeline spanning market data ingestion to automated order lifecycle execution, with complete decoupling between strategy research, simulated paper trading, and broker-gated execution.

> [!NOTE]
> **Universe & Data Scope**: The bundled reproducible market dataset focuses on validated daily bars for **AAPL** (2023–2025) and benchmark **SPY** (`data/raw/` and `data/processed/`). The system architecture is built to support the full 500-constituent S&P 500 universe when connected to external market data provider feeds (Alpaca, Polygon.io).

---

## 2. Problem Statement
Many algorithmic trading projects suffer from critical structural flaws:
1. **Lookahead Bias / Data Leakage**: Future prices leaking into historical feature engineering or preprocessing scalers.
2. **Execution Realism Deficits**: Ignoring slippage, bid-ask spreads, liquidity impact, borrow financing fees, and corporate actions (dividends/splits).
3. **Safety & Risk Neglect**: Lacking process-level circuit breakers, real-time ledger reconciliation, and persistent kill switches.
4. **Execution Coupling**: Entangling trading strategy logic with specific broker SDKs, preventing safe offline testing.
5. **Fabricated Telemetry**: Dashboards returning hard-coded mock numbers rather than real subsystem runtime states.

---

## 3. Solution & Readiness Hierarchy

| Subsystem / Capability | Status Level | Description & Boundary |
|---|---|---|
| **Quantitative Research & Feature Pipeline** | **IMPLEMENTED & UNIT/INTEGRATION TESTED** | 25+ features, walk-forward chronological splits, 7 ML models, 4 regime detectors with 372 automated regression tests passing. |
| **Simulated Paper Trading Engine** | **PAPER-TESTED & PERSISTENCE VALIDATED** | FIFO lot accounting, friction modeling (slippage, spread, commission), local JSON/Parquet persistence, and ledger reconciliation. |
| **Autonomous Loop & Safety Watchdog** | **PAPER-TESTED & VALIDATED** | 14-stage execution cycle, 12-point pre-flight safety gates, and global persistent kill switch. Single-worker execution verified. |
| **Web Dashboard Telemetry** | **REAL SUBSYSTEM-BACKED** | Real-time REST endpoints querying actual persisted states with explicit unavailable/degraded reporting (zero fabricated mock data). |
| **Container & Deployment Configuration** | **PRODUCTION-READY** | Multi-stage Dockerfile, Docker Compose (PostgreSQL, Redis), pre-flight audit script with strict exit codes, and secret redaction. |
| **Live Broker Verification (Alpaca)** | **GATED & PENDING CREDENTIALS** | Bounded read-only health/account probe (`verify_live_connection`) prevents false connected reports; strictly blocks LIVE execution without verified production credentials. |

---

## 4. Key Features

- **Data Ingestion & Validation (Phases 04–05)**: Daily and intraday OHLCV bars, quotes, macro indicators (VIX, 10Y Treasury), and corporate action tracking with strict schema validation and UTC normalization.
- **Quantitative Feature Engine (Phase 06)**: 25+ technical and statistical features (returns, ATR, rolling volatility, trend indicators, volume Z-scores, market-relative beta) with zero-leakage guarantees.
- **Machine Learning & Deep Learning (Phases 07–08)**: Scikit-Learn baseline models, LightGBM, XGBoost, and PyTorch/Keras deep learning architectures with walk-forward chronological validation.
- **Market Regime Detection (Phase 09)**: Unsupervised clustering (K-Means, GMM) and dynamic state transitions (HMM) categorizing Bull Trend, Bear Trend, and Sideways regimes.
- **Signal Engine & Ensemble (Phase 10)**: Weighted ensemble aggregation with dynamic model agreement thresholds, regime gating, and explainable reason codes (`BULLISH_ENSEMBLE_SCORE`, `MODEL_DISAGREEMENT`, `REGIME_FILTERED`).
- **Portfolio Construction (Phase 11)**: Long-only and long-short target allocation with position capping, min-weight filtering, and deterministic ranking.
- **Risk Engine (Phase 12)**: Hard limit enforcement, position sizing adjustments, point-in-time covariance matrix calculation, and capital ratio verification.
- **Research-Grade Backtesting (Phase 13)**: Point-in-time simulation with next-bar execution, directional slippage, commission models, borrow cost financing, dividend adjustments, and benchmark comparison (SPY).
- **Paper Trading Engine (Phase 14)**: Simulated broker execution with FIFO multi-lot realization, transaction friction modeling, position tracking, and disk persistence.
- **Autonomous Trading Loop (Phase 15)**: Multi-stage state machine orchestrating end-to-end trading cycles with crash recovery, checkpointing, and scheduler deduplication.
- **Broker Abstraction & Live Adapter (Phases 16–17)**: Normalized broker interfaces, capability mapping, order idempotency, and Alpaca REST API integration with paper/live isolation.
- **Monitoring & Safety Subsystem (Phase 18)**: Multi-component health aggregation, heartbeat tracking, 12-point pre-flight safety gates, and persistent Global Kill Switch.
- **Web Dashboard (Phase 19)**: 17 dedicated real-time views backed by real subsystem telemetry.
- **Production Deployment (Phase 20)**: Multi-stage Docker containerization, Docker Compose orchestration (PostgreSQL 16, Redis 7), single-worker stateful process architecture, and sanitized health probes (`/health/live`, `/health/ready`, `/info`).
- **End-to-End Technical Validation (Phase 21)**: 20 cross-phase validation tests covering leakage invariance, accounting identities, order idempotency, and 18 failure-injection scenarios.

---

## 5. System Architecture

```
                                +-----------------------------------+
                                |     Market Data Layer (P04/P05)   |
                                +-----------------------------------+
                                                  ↓
                                +-----------------------------------+
                                |    Feature Engineering (P06)      |
                                +-----------------------------------+
                                                  ↓
                     +----------------------------+----------------------------+
                     |                                                         |
                     ↓                                                         ↓
     +-------------------------------+                         +-------------------------------+
     |   AI/ML Models (P07/P08)      |                         |  Regime Detection (P09)       |
     |  XGB, LGBM, MLP, LSTM, Trans  |                         |    K-Means, GMM, HMM          |
     +-------------------------------+                         +-------------------------------+
                     |                                                         |
                     +----------------------------+----------------------------+
                                                  ↓
                                +-----------------------------------+
                                |      Signal Engine (P10)          |
                                +-----------------------------------+
                                                  ↓
                                +-----------------------------------+
                                |   Portfolio Construction (P11)    |
                                +-----------------------------------+
                                                  ↓
                                +-----------------------------------+
                                |       Risk Engine (P12)           |
                                +-----------------------------------+
                                                  ↓
                     +----------------------------+----------------------------+
                     |                                                         |
                     ↓                                                         ↓
     +-------------------------------+                         +-------------------------------+
     | Research Backtesting (P13)    |                         |  Autonomous Loop (P15)        |
     | Walk-Forward Historical Sim   |                         |  Checkpointing & Scheduler    |
     +-------------------------------+                         +-------------------------------+
                     |                                                         |
                     ↓                                                         ↓
     +-------------------------------+                         +-------------------------------+
     | Simulated Broker (P14)        |                         |  Broker Abstraction (P16/P17) |
     | In-Memory Matching & Fills    |                         |  Paper & Alpaca REST Gateway  |
     +-------------------------------+                         +-------------------------------+
                                                                               |
                                                                               ↓
                                                               +-------------------------------+
                                                               | Monitoring & Watchdog (P18)   |
                                                               | Global Kill Switch & Ledger   |
                                                               +-------------------------------+
                                                                               |
                                                                               ↓
                                                               +-------------------------------+
                                                               |   Web Dashboard (P19)         |
                                                               |   React 19 Subsystem Telemetry|
                                                               +-------------------------------+
```

---

## 6. Machine Learning Ensemble Architecture

| Model Architecture | Implementation Type | Feature Inputs | Horizon | Validation Scheme |
|---|---|---|---|---|
| **Logistic Regression** | Linear Baseline + L2 Reg | 16 Normalized Features | 5 Days | Chronological Purged Split |
| **Random Forest** | Bagged Decision Trees | 16 Normalized Features | 5 Days | Chronological Purged Split |
| **XGBoost** | Gradient Boosted Trees | 16 Technical & Regime Features | 5 Days | Walk-Forward TimeSeriesSplit |
| **LightGBM** | Histogram-Based GBDT | 16 Technical & Regime Features | 5 Days | Walk-Forward TimeSeriesSplit |
| **Feed-Forward MLP** | 2-Layer Dense Neural Net | 16 Scaled Features | 5 Days | Chronological Purged Split |
| **LSTM Temporal** | Recurrent Sequence Model | 15-Day Historical Sequences | 5 Days | Chronological Purged Split |
| **Multi-Head Transformer** | Self-Attention Network | 15-Day Historical Sequences | 5 Days | Chronological Purged Split |

---

## 7. Backtesting & Empirical Performance (Phase 13)

The research-grade backtesting engine models realistic market microstructure frictions:
- **Historical AAPL Single-Asset Point-in-Time Benchmark Replay**:
  - Period: `2023-01-02` to `2025-01-01` (523 daily bars)
  - Initial Capital: **$100,000.00**
  - Final Equity: **$101,599.79** (Net P&L: **+$1,599.79**)

---

## 8. Safety & Monitoring Architecture (Phase 18)

1. **Global Kill Switch**: Centralized thread-safe circuit breaker. When triggered, all new order submissions are halted immediately across all subsystems. State persists across restarts in `models/monitoring/kill_switch.json`.
2. **Pre-Flight Safety Gates**: 12 deterministic checks evaluated before every trading cycle (configuration validity, storage accessibility, market data freshness, model health, risk limits, broker heartbeat).
3. **Double-Entry Reconciliation**: Automated watchdog cross-referencing broker reported positions/cash against local transaction ledgers to catch execution drift.
4. **Append-Only Audit Trail**: Every signal, order, fill, risk modification, and incident is recorded with ISO-8601 timestamps and recursive credential redaction.

---

## 9. Web Dashboard (Phase 19)

A responsive, dark-mode single-page application built with **React 19**, **TypeScript**, and **Vite**, featuring 17 dedicated views backed directly by real subsystem states:
- `/overview`: High-level system vitals, execution mode badge, and operational metrics
- `/health`: Detailed 13-component health telemetry and heartbeat status
- `/market-data`: Ingested bar browser, quotes, and data provenance
- `/models`: Loaded model versions, training metadata, and validation metrics
- `/signals`: Active directional candidates, model agreements, and reason codes
- `/portfolio`: Target weights, allocated cash, and asset distributions
- `/risk`: Real-time risk checks, exposure limits, and covariance diagnostics
- `/orders`: Full order lifecycle states (Created, Submitted, Filled, Rejected)
- `/executions`: Executed trade fills, transaction costs, and realized P&L
- `/broker`: Broker connection status, capabilities, and account balances
- `/autonomous-loop`: 14-stage loop visualization, checkpoint status, and scheduler
- `/alerts`: Active warning and error alerts with acknowledgment controls
- `/incidents`: High-severity incidents and resolution runbooks
- `/kill-switch`: Global Kill Switch status, trigger modal, and safe reset controls
- `/reconciliation`: Real-time position/cash ledger reconciliation reports
- `/audit`: Searchable, append-only security and trading audit logs
- `/backtest`: Interactive research analytics, equity curves, and drawdowns

---

## 10. Technology Stack

- **Backend**: Python 3.13, FastAPI, Uvicorn, Pydantic v2
- **Data & Computation**: NumPy, Pandas, Scikit-Learn, LightGBM, XGBoost, TensorFlow/Keras
- **Frontend**: React 19, TypeScript, Vite, Vanilla CSS Design System (Dark Glassmorphism)
- **Database & Storage**: PostgreSQL 16, Redis 7, JSON/Parquet local partitioned storage
- **Containerization**: Docker (Multi-stage build), Docker Compose
- **Testing & Tooling**: Pytest, Pytest-Asyncio, Oxlint

---

## 11. Project Structure

```
S&P 500 AI BOT/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI REST endpoints & Dashboard API
│   │   ├── autonomous/      # Autonomous trading loop controller & scheduler
│   │   ├── backtest/        # Research-grade historical backtesting engine
│   │   ├── broker/          # Broker abstraction layer & Alpaca adapter
│   │   ├── config/          # Pydantic settings & validation
│   │   ├── data/            # Market data ingestion, providers, & validation
│   │   ├── features/        # Quantitative feature engineering library
│   │   ├── models/          # Baseline ML & deep learning architectures
│   │   ├── monitoring/      # Health aggregation, watchdog, & Global Kill Switch
│   │   ├── paper_trading/   # Simulated paper trading broker & accounting
│   │   ├── portfolio/       # Portfolio construction & allocation optimizer
│   │   ├── regime/          # Statistical market regime detection (K-Means/GMM/HMM)
│   │   ├── risk/            # Risk engine & constraint enforcement
│   │   ├── strategy/        # Signal engine, adapters, & ensemble aggregation
│   │   └── main.py          # FastAPI application entrypoint & static serving
│   ├── scripts/             # Deployment check & smoke test scripts
│   └── tests/               # 20 test suites (367 passing tests)
├── frontend/
│   ├── src/                 # React 19 + TypeScript dashboard components
│   ├── index.html           # Single-page application entrypoint
│   ├── .env.example         # Frontend API configuration template
│   └── vite.config.ts       # Vite build & dev proxy configuration
├── data/                    # Partitioned raw & processed market data
├── models/                  # Trained model checkpoints & monitoring state
├── docs/                    # Complete institutional project documentation
├── Dockerfile               # Multi-stage production container build
├── docker-compose.prod.yml  # Production container orchestration
├── .env.example             # Documented environment template
└── pyproject.toml           # Python package dependencies & configuration
```

---

## 12. Getting Started & Development

### Prerequisites
- Python 3.13+
- Node.js 20+ & npm
- Git

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/example/sp500-autonomous-trading-platform.git
   cd sp500-autonomous-trading-platform
   ```

2. **Set up Python Virtual Environment**:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Install Frontend Dependencies & Build Bundle**:
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```

5. **Run Pre-Flight Deployment Validation**:
   ```bash
   python backend/scripts/deploy_check.py
   ```

6. **Start the Unified Backend & Dashboard Server**:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
   ```
   Open `http://localhost:8000` to view the Web Dashboard.

### Local Frontend Development (Hot Reloading)

When modifying frontend components independently, run the Vite development server:
```bash
cd frontend
npm run dev
```
The development server runs at `http://localhost:5173` and automatically proxies all `/api` calls to `http://localhost:8000`. To point to a custom API host, set `VITE_API_BASE_URL` in `frontend/.env`.

---

## 13. Operational Scripts & Runtime Workflows

The platform provides dedicated, self-contained runtime scripts for paper execution, research backtesting, and deployment verification:

```bash
# 1. Execute a single safe autonomous paper trading cycle on validated local data
# Persists session, orders, fills, reconciliation, cycle checkpoint, and model telemetry
python backend/scripts/run_paper_cycle.py

# 2. Execute and persist reproducible historical backtest summary under backtests/
python backend/scripts/run_backtest.py

# 3. Run pre-flight deployment check across environments
# Development mode:
python backend/scripts/deploy_check.py --env development

# Production mode audit (fails fast if PostgreSQL/Redis/Secret are unconfigured):
python backend/scripts/deploy_check.py --env production
```

---

## 14. Testing & Verification

The platform is covered by a comprehensive regression suite of **372 tests**:

```bash
# Run complete test suite (372 passed)
./.venv/bin/python -m pytest backend/tests -v

# Run Phase 21 technical system validation suite
./.venv/bin/python -m pytest backend/tests/test_phase21_validation.py -v

# Run frontend linter (0 errors, 0 warnings)
cd frontend && npm run lint && cd ..

# Run frontend production build
cd frontend && npm run build && cd ..

# Run pre-flight deployment check
python backend/scripts/deploy_check.py
```

### Upstream Third-Party Deprecation Warnings
The codebase produces zero test failures and zero internal application warnings. For transparency, running pytest outputs known third-party library deprecation notices from the scientific Python ecosystem:
- `keras/src/backend/tensorflow/core.py:172`: NumPy 2.0 `__array__` protocol copy keyword deprecation.
- `hmmlearn/utils.py:27`: NumPy 2.5 ndarray shape assignment deprecation.
- `lightgbm/sklearn.py:1106`: LightGBM `eval_set` argument deprecation in favor of `eval_X`/`eval_y`.
- `fastapi/testclient.py:1`: Starlette TestClient `httpx` vs `httpx2` deprecation notice.

---

## 15. Important Safety Notice & Disclaimer

> [!WARNING]
> **DISCLAIMER**: This software is provided strictly for academic research, educational demonstrations, and quantitative modeling purposes.
> - **NO FINANCIAL ADVICE**: Nothing in this repository constitutes investment, financial, legal, or tax advice.
> - **ZERO REAL-MONEY GUARANTEES**: Algorithmic models and historical backtests do not guarantee future profitability in live markets.
> - **FAIL-CLOSED PAPER TRADING**: The system is pre-configured in `PAPER` mode by default. Real-money live execution was NOT conducted during development or validation.

---

## 16. License
Distributed under the MIT License. See `LICENSE` for more information.
