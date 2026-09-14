# Trading System Specification

The quantitative trading pipeline transforms market data into executed trades through six modular stages.

---

## 1. Feature Engineering (Phase 06)
Calculates causal point-in-time quantitative features across multiple technical windows:
- **Return Series**: 1-day, 5-day, and 20-day percentage returns.
- **Volatility & ATR**: 20-day rolling standard deviation and 14-day Average True Range.
- **Trend Diagnostics**: Moving averages (SMA 20, 50, 200), MACD line, signal line, histogram, and RSI (14-period).
- **Volume Metrics**: 20-day rolling volume Z-scores.
- **Market & Sector Relative**: Relative performance vs SPY and rolling 60-day equity beta.

---

## 2. Signal Engine & Ensemble (Phase 10)
Synthesizes predictions from baseline ML (XGBoost, LightGBM) and deep learning models into discrete directional candidate signals:
- **Standardized Predictions**: Probabilities and directional scores normalized to $[-1.0, +1.0]$.
- **Ensemble Combination**: Weighted combination checking minimum active model count (`min_active_models = 1`) and consensus (`min_agreement = 0.50`).
- **Regime Gating**: Bull regime permits LONG only (filters counter-trend shorts), Bear regime restricts aggressive long exposure, Sideways regime applies confidence penalties.
- **Explainability**: Outputs structured `ReasonCode` values (e.g. `BULLISH_ENSEMBLE_SCORE`, `MODEL_DISAGREEMENT`, `REGIME_FILTERED`).

---

## 3. Portfolio Construction (Phase 11)
Allocates capital across active signal candidates:
- **Allocation Modes**: Supports `LONG_ONLY` (default) and `LONG_SHORT`.
- **Constraint Enforcement**:
  - `max_position_weight`: Caps individual symbol exposure (default: 20%).
  - `min_position_weight`: Prunes negligible allocations (default: 5%).
  - `max_positions`: Limits portfolio cardinality (default: 10).
- **Deterministic Ranking**: Sorts candidate signals by confidence and strength with deterministic tie-breaking.

---

## 4. Risk Engine (Phase 12)
Validates and adjusts portfolio target allocations under strict fail-closed constraints:
- **Hard Exposure Limits**: Max position weight, gross leverage, net leverage.
- **Point-in-Time Covariance**: Calculates asset covariance and portfolio volatility using historical lookback windows strictly before timestamp $t$.
- **Adjusted Targets**: Emits `RiskAdjustedTarget` objects with calculated target shares and notionals.

---

## 5. Execution Decoupling
The same `RiskAdjustedTarget` specifications are consumed uniformly across all three execution targets:
- **Backtesting Simulation (`app/backtest`)**
- **Paper Trading Broker (`app/paper_trading`)**
- **Live/Sandbox Broker (`app/broker`)**
