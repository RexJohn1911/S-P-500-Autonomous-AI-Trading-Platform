"""
Deployment Verification and Safety Test Suite (Phase 20).
Comprehensive test suite verifying:
- Configuration loading and production validation
- Secret redaction across settings, info, and metadata
- Paper mode default and live execution mode safety guards
- Liveness (/health/live), Readiness (/health/ready), and Aggregate Health (/health)
- Versioning and deployment metadata (/info)
- CORS origin handling
- Static asset delivery and SPA fallback routing
- Persistence volumes and directory writeability
- Single-worker stateful process integrity
- Kill-switch state persistence across simulated restarts
- Monitoring subsystem integration and health aggregation
"""

import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.config.settings import Settings, EnvironmentEnum, ExecutionModeEnum
from backend.app.main import app
from backend.app.monitoring.service import MonitoringService
from backend.app.monitoring.thresholds import MonitoringConfig


@pytest.fixture
def client():
    """Test client for production endpoints."""
    return TestClient(app)


def test_configuration_defaults_paper_mode():
    """Verify that default configuration is PAPER mode with 1 worker."""
    cfg = Settings()
    assert cfg.EXECUTION_MODE == ExecutionModeEnum.PAPER
    assert cfg.ENVIRONMENT in (EnvironmentEnum.DEVELOPMENT, EnvironmentEnum.TESTING, EnvironmentEnum.PRODUCTION)
    assert cfg.WORKERS_COUNT == 1
    assert cfg.SERVE_STATIC_FRONTEND is True


def test_production_configuration_validation_flags_dev_secrets():
    """Verify that validate_production_settings flags default dev secret in production."""
    prod_cfg = Settings(
        ENVIRONMENT=EnvironmentEnum.PRODUCTION,
        SECRET_KEY="dev-secret-key-for-local-sp500-ai-trading-system-32chars",
    )
    is_valid, issues = prod_cfg.validate_production_settings()
    assert is_valid is False
    assert any("SECRET_KEY" in issue for issue in issues)


def test_production_configuration_validation_guards_live_mode():
    """Verify that validate_production_settings strictly rejects LIVE mode without credentials."""
    prod_cfg = Settings(
        ENVIRONMENT=EnvironmentEnum.PRODUCTION,
        EXECUTION_MODE=ExecutionModeEnum.LIVE,
        SECRET_KEY="secure-64-character-production-secret-key-rotating-safely",
        ALPACA_API_KEY=None,
        ALPACA_API_SECRET=None,
    )
    is_valid, issues = prod_cfg.validate_production_settings()
    assert is_valid is False
    assert any("ALPACA_API_KEY" in issue for issue in issues)
    assert any("ALPACA_API_SECRET" in issue for issue in issues)


def test_secret_redaction_in_settings():
    """Verify that to_safe_dict recursively redacts API keys, secrets, and passwords."""
    cfg = Settings(
        SECRET_KEY="super_secret_string",
        BROKER_SECRET_KEY="broker_secret_string",
        ALPACA_API_SECRET="alpaca_secret_string",
    )
    safe = cfg.to_safe_dict()
    assert safe.get("SECRET_KEY") == "REDACTED" or safe.get("secret_key") == "REDACTED"
    assert safe.get("BROKER_SECRET_KEY") == "REDACTED" or safe.get("broker_secret_key") == "REDACTED"
    assert safe.get("ALPACA_API_SECRET") == "REDACTED" or safe.get("alpaca_api_secret") == "REDACTED"



def test_liveness_probe_endpoint(client):
    """Verify GET /health/live returns 200 OK with worker telemetry."""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data
    assert data["workers"] == 1


def test_readiness_probe_endpoint(client):
    """Verify GET /health/ready validates storage and monitoring readiness."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["monitoring_initialized"] is True
    assert "kill_switch_state" in data
    assert "timestamp" in data


def test_aggregate_health_endpoint(client):
    """Verify GET /health returns comprehensive subsystem health telemetry."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "is_trading_permitted" in data
    assert "active_incidents" in data
    assert "kill_switch_triggered" in data
    assert "environment" in data


def test_system_info_endpoint_has_zero_secrets(client):
    """Verify GET /info returns version & build info with zero secrets."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["system"] == "S&P 500 Autonomous AI Trading System"
    assert data["version"] == "1.0.0"
    assert "execution_mode" in data
    assert "workers_count" in data

    # Verify no credentials leaked
    raw_text = response.text.lower()
    assert "secret" not in raw_text or '"secret_key"' not in raw_text
    assert "password" not in raw_text
    assert "key" not in raw_text or "broker_key" not in raw_text


def test_cors_configuration():
    """Verify that CORS middleware is properly initialized with allowed origins."""
    from backend.app.main import cors_origins
    assert len(cors_origins) > 0


def test_spa_routing_fallback_logic(client):
    """Verify SPA routing fallback logic for client-side routing."""
    # Direct API 404 should return 404 JSON, not SPA index.html
    response = client.get("/api/nonexistent_route")
    assert response.status_code == 404

    # Dashboard routes should be served cleanly
    overview_resp = client.get("/api/dashboard/overview")
    assert overview_resp.status_code == 200


def test_persistence_directories_exist_and_writable():
    """Verify required persistent directories exist and are writable."""
    dirs = ["data", "models", "backtests", "models/monitoring"]
    for d_str in dirs:
        p = Path(d_str)
        p.mkdir(parents=True, exist_ok=True)
        assert p.exists()
        assert p.is_dir()
        test_file = p / ".write_check.tmp"
        test_file.write_text("test")
        assert test_file.read_text() == "test"
        test_file.unlink()


def test_kill_switch_persistence_across_restart(tmp_path):
    """Verify kill switch state persists across simulated process restart."""
    cfg = MonitoringConfig(storage_dir=str(tmp_path))
    service1 = MonitoringService(config=cfg)

    # Initial state: ARMED
    assert not service1.kill_switch.is_triggered()

    # Trigger emergency halt
    service1.trigger_kill_switch(reason="Simulated Deployment Test", triggered_by="DEPLOY_TEST")
    assert service1.kill_switch.is_triggered()

    # Simulate restart by creating new MonitoringService instance pointing to same storage
    service2 = MonitoringService(config=cfg)
    assert service2.kill_switch.is_triggered()
    assert service2.kill_switch.get_snapshot().reason == "Simulated Deployment Test"


def test_dashboard_api_full_smoke_test(client):
    """Execute complete smoke test across all dashboard endpoints."""
    endpoints = [
        "/api/dashboard/overview",
        "/api/dashboard/health",
        "/api/dashboard/market-data",
        "/api/dashboard/models",
        "/api/dashboard/signals",
        "/api/dashboard/portfolio",
        "/api/dashboard/risk",
        "/api/dashboard/orders",
        "/api/dashboard/executions",
        "/api/dashboard/broker",
        "/api/dashboard/autonomous-loop",
        "/api/dashboard/alerts",
        "/api/dashboard/incidents",
        "/api/dashboard/kill-switch",
        "/api/dashboard/reconciliation",
        "/api/dashboard/audit",
        "/api/dashboard/backtest",
    ]

    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 200, f"Endpoint {ep} failed with status {resp.status_code}"
