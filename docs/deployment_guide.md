# S&P 500 Autonomous AI Trading Platform — Production Deployment Guide

## 1. Deployment Architecture Overview

The S&P 500 Autonomous AI Trading System utilizes a **unified, containerized, stateful production architecture** designed for high reliability, deterministic execution, and fail-closed safety.

```
+-------------------------------------------------------------------------+
|                               Browser Client                            |
+-------------------------------------------------------------------------+
                                    |
                            HTTPS / Port 8000
                                    |
+-------------------------------------------------------------------------+
|                    FastAPI Unified Production Container                 |
|                                                                         |
|  [Static SPA Assets (frontend/dist)]  <--->  [/api/dashboard/* Endpoints]|
|  [Liveness Probe (/health/live)]     <--->  [Readiness (/health/ready)] |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                  Single-Worker Stateful Process                   |  |
|  |                                                                   |  |
|  |  * Autonomous Trading Loop (Phase 15 State Machine & Scheduler)   |  |
|  |  * System Watchdog & Health Aggregator (Phase 18)                 |  |
|  |  * Global Emergency Kill Switch (Phase 18 In-Memory + Disk Lock)  |  |
|  |  * 12-Point Safety Gate Evaluator (Phase 18)                      |  |
|  |  * Broker Abstraction & Gateway (Phase 16 & 17)                   |  |
|  |  * Risk Engine (Phase 12 Hardened Constraints)                    |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
         |                                           |
         v                                           v
+-----------------------+                 +-------------------------------+
|  Persistent Volumes   |                 |  External Broker & Providers  |
|  - /app/data          |                 |  - Alpaca Paper / Live API    |
|  - /app/models        |                 |  - Market Data Feeds          |
|  - /app/backtests     |                 |  (Direct client calls BLOCKED)|
|  - /app/monitoring    |                 +-------------------------------+
+-----------------------+
```

---

## 2. Stateful Single-Worker Rationale

> [!IMPORTANT]
> **Production Worker Count**: The backend ASGI server MUST run with `WORKERS_COUNT=1` (`workers=1`).

Running multiple worker processes (e.g. `uvicorn --workers 4`) would create concurrent copies of the autonomous loop scheduler, out-of-sync in-memory kill switches, duplicate order generation, and racing checkpoint locks. State coordination is strictly unified within a single worker process backed by durable file checkpoints and SQLite/Parquet/JSON persistence.

---

## 3. Environment Configuration Model

The platform distinguishes between four mutually exclusive environments:

| Environment | Purpose | `EXECUTION_MODE` | Default Host / Port |
|---|---|---|---|
| `DEVELOPMENT` | Local research & UI dev | `PAPER` | `0.0.0.0:8000` |
| `TESTING` | Automated Pytest regression | `PAPER` | In-memory / ephemeral |
| `PAPER` | Automated paper simulation | `PAPER` | `0.0.0.0:8000` (Docker) |
| `PRODUCTION` | Live broker execution | `LIVE` (or `PAPER`) | `0.0.0.0:8000` (Container) |

### Environment Variables Reference

| Variable | Description | Default | Production Requirement |
|---|---|---|---|
| `ENVIRONMENT` | Deployment environment | `development` | Set to `production` |
| `EXECUTION_MODE` | Trading mode (`PAPER` or `LIVE`) | `PAPER` | Default `PAPER`; explicitly set `LIVE` only when approved |
| `WORKERS_COUNT` | ASGI worker count | `1` | Strictly `1` |
| `SECRET_KEY` | Session signing key | Dev default | Cryptographically random 64-char string |
| `ALLOWED_ORIGINS` | Permitted CORS origins | Localhost origins | Explicit list of trusted domains |
| `SERVE_STATIC_FRONTEND`| Serve compiled React SPA | `True` | `True` |
| `STATIC_DIR` | Compiled frontend directory | `frontend/dist` | Path to production assets |
| `ALPACA_API_KEY` | Alpaca API key | `None` | Runtime injected |
| `ALPACA_API_SECRET` | Alpaca API secret | `None` | Runtime injected |
| `ALPACA_BASE_URL` | Alpaca API URL | `https://api.alpaca.markets` | `https://paper-api.alpaca.markets` (Paper) |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` | Production database URL |
| `REDIS_URL` | Redis cache connection string | `redis://...` | Production Redis URL |

---

## 4. Secret Management & Zero-Leakage Policy

1. **No Committed Secrets**: Never commit `.env` or files containing credentials.
2. **Runtime Injection**: Inject API keys and secrets via environment variables at container launch.
3. **Automatic Redaction**: All API responses, log statements, audit records, and `/info` endpoints automatically redact sensitive keys (`SECRET_KEY`, `BROKER_SECRET_KEY`, `ALPACA_API_SECRET`).
4. **Zero Frontend Exposure**: The React frontend bundle contains no broker secrets or direct broker communication code.

---

## 5. Health Checks & Diagnostic Probes

### Liveness Probe (`GET /health/live`)
- **Purpose**: Verifies that the ASGI container process is responsive.
- **Expected Response**: `200 OK`
```json
{
  "status": "alive",
  "timestamp": "2026-09-08T12:00:00Z",
  "environment": "production",
  "workers": 1
}
```

### Readiness Probe (`GET /health/ready`)
- **Purpose**: Verifies that configuration is valid, storage volumes are writable, and monitoring subsystem is initialized.
- **Expected Response**: `200 OK` (or `503 Service Unavailable` if storage or config fails)
```json
{
  "status": "ready",
  "timestamp": "2026-09-08T12:00:00Z",
  "monitoring_initialized": true,
  "execution_mode": "PAPER",
  "environment": "production",
  "kill_switch_state": "ARMED"
}
```

### Aggregate Health Probe (`GET /health`)
- **Purpose**: Provides aggregate health from Phase 18 Watchdog across all 13 components.

### Deployment Info (`GET /info`)
- **Purpose**: Safe version metadata for CI/CD and monitoring dashboards.

---

## 6. Local Production-Like Deployment

To build and run the production environment locally:

### Step 1: Build the Production Frontend
```bash
cd frontend
npm ci
npm run build
cd ..
```

### Step 2: Run Pre-Flight Verification
```bash
python backend/scripts/deploy_check.py
```
> [!IMPORTANT]
> `deploy_check.py` performs rigorous pre-flight validation and returns an exit code of `0` on success and `1` on failure:
> - Validates Python 3.11+ runtime and single-worker policy (`WORKERS_COUNT=1`).
> - Validates recursive secret redaction in configuration.
> - Verifies persistent volume writability (`data`, `models`, `backtests`, `models/monitoring`, `models/paper_trading`, `models/autonomous`).
> - Probes PostgreSQL (port 5432) and Redis (port 6379) reachability (strict requirement in `PRODUCTION` mode).
> - Confirms frontend production bundle (`frontend/dist/index.html`) presence.
> - Validates production secret rotation and distinguishes `PAPER-ready` from `LIVE-ready` (LIVE execution is blocked without verified credentials).

### Step 3: Local Frontend Hot-Reloading Development (Optional)
When developing the UI independently from the container, run:
```bash
cd frontend
npm run dev
```
The Vite development server runs at `http://localhost:5173` and automatically proxies `/api` requests to `http://localhost:8000`. You can configure a custom API endpoint using `VITE_API_BASE_URL` in `frontend/.env`.

### Step 4: Launch Production Docker Stack
```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

### Step 5: Run Smoke Tests
```bash
python backend/scripts/smoke_test.py http://localhost:8000
```

---

## 7. Restart Safety & Crash Recovery

1. **State Preservation**: The autonomous loop writes checkpoints to `/app/models/autonomous/{session_id}/checkpoints.json`.
2. **Kill Switch Persistence**: If triggered, the kill switch state is durably saved to disk and automatically restored to `TRIGGERED` upon process restart.
3. **Reconciliation on Boot**: Upon startup, the autonomous loop and broker gateway verify order IDs and positions against the broker before resuming execution cycles.

---

## 8. Rollback Runbook

If a deployment must be rolled back:

1. **Stop Current Version**:
   ```bash
   docker-compose -f docker-compose.prod.yml down
   ```
2. **Verify Persistent Volumes**: Persistent state in `trading_data`, `trading_models`, and `trading_backtests` is preserved across container restarts.
3. **Checkout Previous Image Tag**:
   ```bash
   docker pull your-registry/sp500-trading:previous-stable-tag
   ```
4. **Start Previous Version**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```
5. **Verify System Health**:
   ```bash
   python backend/scripts/smoke_test.py http://localhost:8000
   ```
