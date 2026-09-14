# Risk Management & Engine Specification

## 1. Risk Philosophy: Fail-Closed Protection
The Risk Management Engine (`backend/app/risk`) acts as an authoritative, non-bypassable safety layer. If market conditions, portfolio proposals, or data feeds violate configured risk parameters or become indeterminate, the system fails closed (rejects or down-weights positions).

---

## 2. Risk Checks & Limits

- **Maximum Position Weight**: Limits individual symbol allocation (default: 20% of total equity).
- **Maximum Gross Exposure**: Caps aggregate long + short notional to total equity (default: 100% / no leverage).
- **Maximum Net Exposure**: Limits directional bias in long-short configurations.
- **Maximum Portfolio Cardinality**: Caps simultaneous active positions to prevent fragmentation (default: 10).
- **Point-in-Time Covariance Estimation**: Computes historical covariance matrices using only data available prior to evaluation timestamp $t$.
- **High Correlation Diagnostic**: Flags asset pairs with correlation $> 0.85$ to prevent excessive concentration in correlated names.

---

## 3. Evaluation Lifecycle

```
[ Proposed Portfolio Targets ]
              ↓
[ Risk Limit Evaluator ]
   • Checks hard limits (position weight, gross exposure, leverage)
   • Flags soft limit warnings (volatility, correlation, turnover)
              ↓
[ Risk Adjustment Engine ]
   • Scales down oversized targets deterministically
   • Prunes rejected symbols
   • Emits RiskAdjustmentRecord audit entries
              ↓
[ Emitted RiskEngineResult ]
   • Adjusted targets with validated notionals and share counts
   • Cannot be bypassed by Dashboard, Autonomous Loop, or Broker
```
