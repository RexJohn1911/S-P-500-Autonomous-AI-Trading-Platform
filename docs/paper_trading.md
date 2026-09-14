# Paper Trading Engine Specification

## 1. Overview
The Paper Trading Engine (`backend/app/paper_trading`) provides realistic simulated trading execution with double-entry accounting and persistent ledger state.

---

## 2. Simulated Broker Architecture

- **Account & Cash Accounting**:
  - Double-entry cash and position tracking (`PaperAccount`).
  - Strict accounting invariant: $Equity = Cash + Long\,Market\,Value + Short\,Collateral$.
- **Order Lifecycle States**:
  - `CREATED` $\rightarrow$ `SUBMITTED` $\rightarrow$ `ACCEPTED` $\rightarrow$ `FILLED` (or `REJECTED` / `CANCELLED`).
- **FIFO Multi-Lot Realization**:
  - Realizes P&L on partial exits and full liquidations using First-In-First-Out accounting.
- **Transaction Costs & Slippage**:
  - Simulated commissions (5 bps), half-spread (2 bps), and directional slippage (5 bps).
- **Idempotent Order Handling**:
  - Duplicate submissions with identical `client_order_id` safely return the existing order without creating duplicate positions.
- **Reconciliation Engine**:
  - Continuous watchdog verifying that in-memory positions match historical execution logs and cash balances.
