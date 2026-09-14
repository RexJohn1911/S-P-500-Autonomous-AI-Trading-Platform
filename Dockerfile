# =========================================================================
# Multi-Stage Production Dockerfile for S&P 500 AI Trading Platform (Phase 20)
# Stage 1: Build Production Frontend Assets (React + Vite + TypeScript)
# Stage 2: Minimal, Non-Root Python 3.13 Runtime with Embedded SPA Serving
# =========================================================================

# --- Stage 1: Frontend Build ---
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Production ASGI Runtime ---
FROM python:3.13-slim AS production

WORKDIR /app

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    EXECUTION_MODE=PAPER \
    WORKERS_COUNT=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    SERVE_STATIC_FRONTEND=True \
    STATIC_DIR=frontend/dist

# Install runtime system utilities and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN groupadd -r -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -d /app -s /sbin/nologin appuser

# Install Python application dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application source
COPY . /app

# Copy compiled frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Establish persistent storage directories with non-root ownership
RUN mkdir -p /app/data /app/models /app/backtests /app/models/monitoring && \
    chown -R appuser:appgroup /app

# Switch to unprivileged user
USER appuser

# Expose API and Dashboard port
EXPOSE 8000

# Docker Healthcheck Probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Production single-worker entrypoint (preserves stateful autonomous loop & kill switch)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
