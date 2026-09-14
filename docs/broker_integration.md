# Broker Integration Specification

## 1. Broker Abstraction Layer (Phase 16)
The broker abstraction layer decouples quantitative trading algorithms from specific broker APIs:
- **Base Broker Interface (`BaseBroker`)**: Normalized methods for account queries, position retrieval, order submission, cancellation, and execution reporting.
- **Normalized Data Contracts**: `BrokerOrder`, `BrokerExecution`, `BrokerAccount`, `BrokerPosition`.
- **Capability Mapping**: Dynamically queries broker feature support (e.g. fractional shares, extended hours, short selling, bracket orders).

---

## 2. Alpaca Live / Sandbox Adapter (Phase 17)
- **REST Client Integration**: Interacts with Alpaca Paper & Live API endpoints with rate-limit retry backoffs.
- **Idempotency Safeguards**: Maps internal client order IDs to Alpaca client order identifiers.
- **Strict Paper/Live Isolation**:
  - `EXECUTION_MODE=PAPER` is default and completely isolated.
  - `EXECUTION_MODE=LIVE` requires explicit environment configuration and 12-point pre-flight validation.

> **Validation Status**: Real-money live brokerage orders were NOT submitted or executed during platform testing. Live trading was verified strictly via paper/sandbox configurations and mocked response adapters.
