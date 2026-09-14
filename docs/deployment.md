# Deployment & Containerization Specification

## 1. Single-Worker Stateful Architecture
The backend application is strictly configured to execute under a single ASGI worker process (`WORKERS_COUNT=1`).

### Rationale:
The autonomous loop scheduler, Global Kill Switch circuit breaker, in-memory event bus, real-time reconciliation watchdog, and order routing state machines are process-local. Running multiple concurrent worker processes would create duplicated autonomous loops, racing order submissions, and split-brain kill switch states.

---

## 2. Multi-Stage Container Build

The root `Dockerfile` utilizes a two-stage build:
- **Stage 1 (Frontend Builder)**: Node 20 environment compiles the React 19 + TypeScript frontend into static production assets (`frontend/dist`).
- **Stage 2 (Runtime Server)**: Python 3.13-slim environment embeds the compiled frontend and serves both the FastAPI REST API and static UI with client-side SPA fallback.
- **Security**: Runs under an unprivileged non-root user (`appuser:10001`).

---

## 3. Production Compose Topology (`docker-compose.prod.yml`)

- **Web / API Service**: FastAPI application running on port 8000.
- **Database**: PostgreSQL 16 Alpine with persistent volume `sp500_trading_data`.
- **Cache / Message Queue**: Redis 7 Alpine.
- **Health Probes**:
  - `GET /health/live`: Liveness probe.
  - `GET /health/ready`: Readiness probe verifying storage access and monitoring initialization.
