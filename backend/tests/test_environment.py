"""
Verification tests for development environment, ML imports, config, and FastAPI API.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config.settings import get_settings, ExecutionModeEnum, EnvironmentEnum


def test_ml_and_data_imports():
    """Verify that all core scientific, numerical and ML libraries load properly."""
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import xgboost as xgb
    import lightgbm as lgb
    import sqlalchemy
    import redis

    arr = np.array([1.0, 2.0, 3.0])
    assert arr.mean() == 2.0

    df = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "price": [180.5, 420.0]})
    assert len(df) == 2

    assert xgb.__version__ is not None
    assert lgb.__version__ is not None
    assert sklearn.__version__ is not None


def test_configuration_loading():
    """Verify settings load defaults and parse environment correctly."""
    settings = get_settings()
    assert settings.PROJECT_NAME == "S&P 500 Autonomous AI Trading System"
    assert settings.EXECUTION_MODE in [ExecutionModeEnum.PAPER, ExecutionModeEnum.LIVE]
    assert settings.MAX_POSITION_PCT == 0.05
    assert settings.MAX_SECTOR_PCT == 0.25
    assert settings.MAX_PORTFOLIO_DRAWDOWN_PCT == 0.06


def test_api_root_and_health():
    """Verify FastAPI application starts and handles root and /health routes."""
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    root_data = root_resp.json()
    assert root_data["status"] == "online"
    assert root_data["execution_mode"] == "PAPER"

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
