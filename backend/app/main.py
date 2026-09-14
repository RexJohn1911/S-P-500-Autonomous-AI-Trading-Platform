"""
S&P 500 Autonomous AI Trading System - Core Production API Entrypoint (Phase 20)
Provides production ASGI server endpoints: liveness, readiness, health, info, API routing,
and SPA static asset delivery. Single-worker execution is enforced for state consistency.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.dashboard import router as dashboard_router, get_monitoring_service
from backend.app.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Institutional-grade autonomous AI quantitative trading platform.",
    version=settings.VERSION,
    docs_url="/docs" if settings.ENVIRONMENT.value != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT.value != "production" else None,
)

# Configure CORS with explicit allowed origins
cors_origins = settings.ALLOWED_ORIGINS
if settings.ENVIRONMENT.value == "development" and "*" not in cors_origins:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True if cors_origins != ["*"] else False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Include Dashboard API Router
app.include_router(dashboard_router)


# =========================================================================
# Liveness & Readiness Health Probes
# =========================================================================

@app.get("/health/live", tags=["Health"])
async def liveness_probe():
    """Liveness probe: verifies the ASGI worker process is responsive."""
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.ENVIRONMENT.value,
        "workers": settings.WORKERS_COUNT,
    }


@app.get("/health/ready", tags=["Health"])
async def readiness_probe():
    """Readiness probe: validates critical component initialization and storage."""
    is_valid_cfg, config_issues = settings.validate_production_settings()
    if not is_valid_cfg:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "Configuration validation failure",
                "issues": config_issues,
            },
        )

    # Check persistence paths accessibility
    storage_dirs = ["data", "models", "backtests", settings.MONITORING_STORAGE_DIR]
    unwritable = []
    for s_dir in storage_dirs:
        try:
            p = Path(s_dir)
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            unwritable.append(f"{s_dir}: {str(e)}")

    if unwritable:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "Storage access failure",
                "errors": unwritable,
            },
        )

    # Verify monitoring subsystem is initialized
    monitoring = get_monitoring_service(settings)
    monitoring_healthy = monitoring is not None

    return {
        "status": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monitoring_initialized": monitoring_healthy,
        "execution_mode": settings.EXECUTION_MODE.value,
        "environment": settings.ENVIRONMENT.value,
        "kill_switch_state": monitoring.kill_switch.get_snapshot().state.value,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Aggregate health check endpoint providing comprehensive runtime telemetry."""
    monitoring = get_monitoring_service(settings)
    health_snap = monitoring.get_system_health()

    return {
        "status": "healthy" if health_snap.is_trading_permitted else health_snap.status.value,
        "system_status": health_snap.status.value,
        "is_trading_permitted": health_snap.is_trading_permitted,
        "active_incidents": health_snap.active_incident_count,
        "kill_switch_triggered": health_snap.kill_switch_triggered,
        "environment": settings.ENVIRONMENT.value,
        "execution_mode": settings.EXECUTION_MODE.value,
        "database_configured": bool(settings.DATABASE_URL),
        "redis_configured": bool(settings.REDIS_URL),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/info", tags=["System"])
async def system_info():
    """Deployment metadata endpoint exposing version and build info without secrets."""
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "execution_mode": settings.EXECUTION_MODE.value,
        "build_timestamp": settings.BUILD_TIMESTAMP or "N/A",
        "git_commit": settings.GIT_COMMIT or "N/A",
        "broker_provider": settings.BROKER_PROVIDER,
        "data_provider": settings.DATA_PROVIDER,
        "workers_count": settings.WORKERS_COUNT,
        "static_frontend_serving": settings.SERVE_STATIC_FRONTEND,
    }


# =========================================================================
# Root & Static Frontend Serving & SPA Routing Fallback
# =========================================================================

static_dir_path = Path(settings.STATIC_DIR)
assets_dir_path = static_dir_path / "assets"

if settings.SERVE_STATIC_FRONTEND and static_dir_path.exists() and assets_dir_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir_path)), name="static_assets")


@app.get("/", tags=["System"])
async def root_endpoint(request: Request):
    """Root endpoint returning system JSON by default, or serving SPA HTML for direct browser navigation."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        index_file = static_dir_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))

    return {
        "system": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT.value,
        "execution_mode": settings.EXECUTION_MODE.value,
        "version": settings.VERSION,
        "status": "online",
        "dashboard_api": "/api/dashboard/overview",
    }


if settings.SERVE_STATIC_FRONTEND and static_dir_path.exists():
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa_fallback(full_path: str):
        """Catch-all route handler for Single Page Application routing."""
        # Never intercept API, health, docs, or asset routes
        if (
            full_path.startswith("api/")
            or full_path.startswith("health")
            or full_path.startswith("info")
            or full_path.startswith("docs")
            or full_path.startswith("openapi.json")
        ):
            raise HTTPException(status_code=404, detail="Not Found")

        # Check if direct file exists in static dir
        target_file = static_dir_path / full_path
        if target_file.is_file():
            return FileResponse(str(target_file))

        # Fallback to SPA index.html for client-side routing
        index_file = static_dir_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))

        raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.WORKERS_COUNT,
        reload=(settings.ENVIRONMENT.value == "development"),
    )
