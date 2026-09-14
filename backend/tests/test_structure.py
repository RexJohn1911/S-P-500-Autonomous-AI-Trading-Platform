"""
Verification tests for project structure, module imports, and configuration.
"""

import importlib
from pathlib import Path
from fastapi.testclient import TestClient


def test_app_packages_importable():
    """Verify that all core architectural modules in backend/app import cleanly."""
    packages = [
        "backend.app.api",
        "backend.app.config",
        "backend.app.data",
        "backend.app.features",
        "backend.app.models",
        "backend.app.strategy",
        "backend.app.portfolio",
        "backend.app.risk",
        "backend.app.execution",
        "backend.app.backtest",
        "backend.app.monitoring",
        "backend.app.main",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_directory_structure_exists():
    """Verify all mandated directories and placeholders exist."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    expected_paths = [
        base_dir / "backend" / "app" / "api",
        base_dir / "backend" / "app" / "config",
        base_dir / "backend" / "app" / "data",
        base_dir / "backend" / "app" / "features",
        base_dir / "backend" / "app" / "models",
        base_dir / "backend" / "app" / "strategy",
        base_dir / "backend" / "app" / "portfolio",
        base_dir / "backend" / "app" / "risk",
        base_dir / "backend" / "app" / "execution",
        base_dir / "backend" / "app" / "backtest",
        base_dir / "backend" / "app" / "monitoring",
        base_dir / "backend" / "tests",
        base_dir / "backend" / "scripts",
        base_dir / "backend" / "requirements.txt",
        base_dir / "frontend" / "src" / "components",
        base_dir / "frontend" / "src" / "pages",
        base_dir / "frontend" / "src" / "charts",
        base_dir / "frontend" / "src" / "services",
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "data" / "features",
        base_dir / "models" / "trained",
        base_dir / "models" / "metadata",
        base_dir / "backtests",
        base_dir / "configs",
        base_dir / "docker",
        base_dir / "docs",
        base_dir / ".env.example",
        base_dir / "docker-compose.yml",
        base_dir / "README.md",
        base_dir / "LICENSE",
    ]
    for path in expected_paths:
        assert path.exists(), f"Expected path does not exist: {path}"


def test_fastapi_backend_endpoints():
    """Verify backend main FastAPI app initializes and handles requests."""
    from backend.app.main import app
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert root_resp.json()["status"] == "online"

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"
