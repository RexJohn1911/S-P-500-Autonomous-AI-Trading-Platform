# Web Dashboard Specification

## 1. Overview
The Web Dashboard (`frontend/`) is a modern, responsive Single-Page Application (SPA) built with **React 19**, **TypeScript**, and **Vite**, offering comprehensive observability into the trading platform.

---

## 2. 17 Dedicated Views

1. **Overview (`/overview`)**: Real-time equity curve, active allocations, quick stats, and persistent execution mode banner (`PAPER` vs `LIVE`).
2. **System Health (`/health`)**: Aggregated health status across 13 subsystems and heartbeat freshness.
3. **Market Data (`/market-data`)**: Historical and streaming bar inspector, quote spreads, and dataset provenance.
4. **AI Models (`/models`)**: Loaded model artifacts, training metadata, and validation metrics.
5. **Signals (`/signals`)**: Active trade candidates, confidence scores, model votes, and explainability reason codes.
6. **Portfolio (`/portfolio`)**: Target weights, active vs target positioning, and asset distribution.
7. **Risk Engine (`/risk`)**: Real-time risk limit evaluations, hard constraint status, and covariance diagnostics.
8. **Orders (`/orders`)**: Complete order lifecycle table with status filtering.
9. **Executions (`/executions`)**: Historical fill records, realized P&L, commissions, and execution slippage.
10. **Broker (`/broker`)**: Broker connection diagnostics, buying power, and cash balances.
11. **Autonomous Loop (`/autonomous-loop`)**: 14-stage loop visualization, scheduler state, and checkpoint telemetry.
12. **Alerts (`/alerts`)**: Warning and error alerts with acknowledgment controls.
13. **Incidents (`/incidents`)**: Critical incidents, severity ratings, and resolution workflows.
14. **Kill Switch (`/kill-switch`)**: Dedicated Global Kill Switch control panel with interactive trigger modal and safe reset verification.
15. **Reconciliation (`/reconciliation`)**: Ledger discrepancy reports comparing broker vs local accounts.
16. **Audit Trail (`/audit`)**: Searchable, append-only security and trading event log.
17. **Backtest (`/backtest`)**: Research backtesting analytics, equity charts, drawdowns, and Sharpe ratios.

---

## 3. Safety & Design Standards
- **Read-Heavy Architecture**: Dashboard observes state; backend remains sole authoritative decision maker.
- **Zero Exposed Secrets**: All sensitive API keys and credentials are sanitized before leaving the backend API boundary.
- **Dark Glassmorphic UI**: High-contrast, clean typography, smooth transitions, and responsive layout.
