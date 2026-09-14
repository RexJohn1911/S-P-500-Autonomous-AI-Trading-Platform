# System Validation Specification

## 1. Scope & Verification Matrix
System validation is codified in `backend/tests/test_phase21_validation.py` (20 cross-phase tests) and the complete regression suite (366 tests).

---

## 2. Cross-Phase Invariant Tests

1. **Roadmap Integrity**: Verifies exact 23-phase sequential ordering (Phases 00–22).
2. **Configuration Integrity**: Enforces fail-closed paper mode defaults and validates production settings.
3. **Model Artifact Integrity**: Verifies feature outputs are numeric, finite, and bounded.
4. **Dataset Provenance**: Verifies dataset metadata and symbol partition storage.
5. **Data Leakage Invariance**: Verifies that mutating future prices at $t' > t$ does not alter historical features at $t$.
6. **Chronological Ordering**: Enforces strictly monotonic bar timestamps.
7. **Accounting Invariants**: Verifies $Equity = Cash + Market\,Value$ and FIFO realization.
8. **Portfolio Constraints**: Verifies position capping (20%), min-weight pruning, and gross exposure limits.
9. **Risk Engine Fail-Closed Enforcement**: Verifies oversized allocations are scaled down or blocked.
10. **Signal Engine Determinism**: Verifies identical inputs yield identical directional outputs and reason codes.
11. **Autonomous Loop Controller**: Verifies state transitions across `CREATED`, `READY`, `RUNNING`, `PAUSED`, `STOPPED`.
12. **Order Idempotency**: Verifies duplicate `client_order_id` submissions return the existing order.
13. **Reconciliation Watchdog**: Verifies ledger tracking matches simulated broker positions.
14. **Kill Switch Order Blocking**: Verifies `TRIGGERED` state unconditionally halts order submission.
15. **Kill Switch Persistence**: Verifies kill switch state survives process restart.
16. **Paper/Live Separation**: Verifies paper trading operates purely locally without external broker calls.
17. **Secret Redaction**: Verifies sensitive keys are masked in config, logs, and audit trails.
18. **Dashboard API Security**: Verifies `/info` and API responses leak zero credentials.
19. **Deployment Probes**: Verifies `/health/live` and `/health/ready` endpoints respond correctly.
20. **Historical End-to-End Replay**: Verifies complete data $\rightarrow$ feature $\rightarrow$ signal $\rightarrow$ portfolio $\rightarrow$ risk $\rightarrow$ execution workflow.
