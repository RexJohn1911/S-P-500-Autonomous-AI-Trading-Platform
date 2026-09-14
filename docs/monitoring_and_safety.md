# Monitoring & Safety Subsystem

## 1. Safety Architecture Overview
The Monitoring & Safety Subsystem (`backend/app/monitoring`) provides centralized operational observability, proactive risk gating, and instant emergency halt capabilities.

---

## 2. Global Kill Switch

```
[ Incident / Anomaly Detected ] ────┐
                                    ↓
[ Manual Operator Trigger ] ────→ [ Global Kill Switch ]
                                    ↓
                       [ State = TRIGGERED ]
                                    ↓
               +--------------------+--------------------+
               ↓                                         ↓
   [ All New Orders Halted ]              [ Persistent State Saved to Disk ]
               ↓                                         ↓
   [ Incident Logged in Audit ]            [ Operator Resolution Required ]
               ↓                                         ↓
   [ Safety Gate Verification ] ─────────→ [ Safe Reset to ARMED ]
```

- **Thread-Safe & Centralized**: Implemented with reentrant locks (`RLock`).
- **Disk Persistence**: Persists state across service restarts in `models/monitoring/kill_switch_state.json`.
- **Safe Reset Verification**: Requires verified operator identity and passing health checks.

---

## 3. Subsystem Health Aggregation
Continuously monitors 13 platform components:
1. Market Data Feed
2. Feature Engine
3. Model Ingestion
4. Regime Detection
5. Signal Engine
6. Portfolio Construction
7. Risk Engine
8. Broker Connectivity
9. Paper Trading Ledger
10. Autonomous Loop
11. Storage Subsystem
12. Database Connectivity
13. Redis Cache

Status aggregates deterministically into `HEALTHY`, `DEGRADED`, `CRITICAL`, or `UNKNOWN`. If any critical component fails, trading is automatically halted.
