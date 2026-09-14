# Backtesting Engine Specification

## 1. Overview & Research-Grade Standards
The Backtesting Engine (`backend/app/backtest`) provides an institutional historical simulation framework adhering to strict point-in-time causality and market microstructure realism.

---

## 2. Key Microstructure Modeling Features

1. **Next-Bar Execution**: Signals formed at bar $t$ close are executed at bar $t+1$ open, completely eliminating same-bar execution lookahead.
2. **Transaction Costs**:
   - Fixed commission: `$0.005` per share (minimum `$1.00`).
   - Bid-ask half-spread: Modeled percentage cost on order entry/exit.
   - Directional slippage: Linear or volatility-adjusted execution drag.
   - Market impact: Volume-proportional slippage on large order notionals.
3. **Borrow Financing Fees**: Daily borrow interest accrued on overnight short positions.
4. **Corporate Actions**: Seamless cash dividend payments credited to cash balances and stock split share/price ratio adjustments.
5. **Walk-Forward Cross Validation**: Anchored and rolling walk-forward simulation windows to evaluate out-of-sample stability.

---

## 3. Validated Historical Experiment Results

### Experiment A: Single-Asset AAPL SMA Point-in-Time Simulation
- **Dataset**: `data/raw/AAPL/1Day/bars.json` (523 daily bars)
- **Date Range**: `2023-01-02` to `2025-01-01`
- **Initial Capital**: `$100,000.00`
- **Final Equity**: **$101,599.79**
- **Net P&L**: **+$1,599.79**
- **Reconciliation Status**: 100% matched across all 523 bars

### Experiment B: Multi-Asset S&P 500 Baseline Simulation
- **Initial Capital**: `$100,000.00`
- **Final Equity**: **$110,659.88**
- **Net P&L**: **+$10,659.88**

> **Notice**: Historical backtest results are research artifacts demonstrating pipeline integrity and do not guarantee future performance in live markets.
