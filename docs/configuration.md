# Configuration & Environment Guide

The platform uses Pydantic Settings (`backend/app/config/settings.py`) for type-safe environment variable management, runtime secret redaction, and strict production validation.

---

## Environment Variables Reference

| Category | Variable | Default | Description |
|---|---|---|---|
| **App Environment** | `ENVIRONMENT` | `development` | `development`, `staging`, `production` |
| | `EXECUTION_MODE` | `PAPER` | `PAPER` (simulated default) or `LIVE` |
| | `WORKERS_COUNT` | `1` | Strictly `1` for stateful single-worker architecture |
| | `SECRET_KEY` | `dev-secret-...` | JWT signing secret (must rotate in production) |
| **API & Networking** | `API_HOST` | `0.0.0.0` | Bind address for FastAPI server |
| | `API_PORT` | `8000` | Port for FastAPI server |
| | `ALLOWED_ORIGINS` | `["http://localhost:8000"]` | CORS origins (no wildcards in production) |
| **Broker Config** | `BROKER_PROVIDER` | `PAPER` | `PAPER` or `ALPACA` |
| | `ALPACA_API_KEY` | `None` | Alpaca API key ID (runtime injected) |
| | `ALPACA_API_SECRET` | `None` | Alpaca Secret key (runtime injected) |
| | `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Alpaca API endpoint URL |
| **Persistence** | `DATABASE_URL` | `postgresql://...` | PostgreSQL connection URL |
| | `REDIS_URL` | `redis://...` | Redis cache and message queue URL |
| | `MONITORING_STORAGE_DIR` | `models/monitoring` | Path for persistent kill switch & incident logs |
| **Risk Limits** | `RISK_MAX_POSITION_WEIGHT` | `0.20` | Max allocation per symbol (20%) |
| | `RISK_MAX_GROSS_EXPOSURE` | `1.00` | Maximum gross portfolio leverage (100%) |
| | `RISK_MAX_NET_EXPOSURE` | `1.00` | Maximum net portfolio exposure (100%) |
| **Signal Engine** | `SIGNAL_LONG_THRESHOLD` | `0.20` | Minimum score for LONG signal |
| | `SIGNAL_SHORT_THRESHOLD` | `-0.20` | Maximum score for SHORT signal |
| | `SIGNAL_MIN_CONFIDENCE` | `0.25` | Minimum confidence threshold |
| | `SIGNAL_MIN_AGREEMENT` | `0.50` | Minimum ensemble model agreement |

---

## Security & Secret Protection

1. **Secret Masking (`Settings.to_safe_dict()`)**:
   All sensitive fields (keys containing `key`, `secret`, `password`, `token`, `auth`, `credential`, `private`) are automatically replaced with `"REDACTED"` before serialization.
2. **Production Pre-Flight Validator (`Settings.validate_production_settings()`)**:
   When `ENVIRONMENT=production`:
   - Rejects default dev `SECRET_KEY`.
   - Rejects wildcard CORS (`ALLOWED_ORIGINS=["*"]`).
   - Validates presence of broker API credentials when `EXECUTION_MODE=LIVE`.
