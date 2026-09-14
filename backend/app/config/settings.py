"""
Configuration and Environment Settings Module
Provides centralized, validated configuration for the trading platform using Pydantic Settings.
"""

from enum import Enum
from functools import lru_cache
from typing import Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionModeEnum(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class EnvironmentEnum(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core Application Settings
    PROJECT_NAME: str = "S&P 500 Autonomous AI Trading System"
    ENVIRONMENT: EnvironmentEnum = EnvironmentEnum.DEVELOPMENT
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    WORKERS_COUNT: int = 1  # Stateful single-process architecture for autonomous loop & safety watchdog
    SECRET_KEY: str = "dev-secret-key-for-local-sp500-ai-trading-system-32chars"
    VERSION: str = "1.0.0"
    BUILD_TIMESTAMP: Optional[str] = None
    GIT_COMMIT: Optional[str] = None

    # CORS & Networking Configuration
    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8000",
        ]
    )

    # Static Frontend Serving Configuration
    SERVE_STATIC_FRONTEND: bool = True
    STATIC_DIR: str = "frontend/dist"

    # Trading Execution Configuration
    EXECUTION_MODE: ExecutionModeEnum = ExecutionModeEnum.PAPER

    # Broker API Credentials
    BROKER_API_KEY: str = "dev_paper_broker_key"
    BROKER_SECRET_KEY: str = "dev_paper_broker_secret"
    BROKER_BASE_URL: str = "https://paper-api.alpaca.markets"

    # Market Data Providers
    DATA_PROVIDER: str = "alpaca"
    DATA_PROVIDER_API_KEY: str = "dev_data_provider_key"

    # Database & Persistence (PostgreSQL)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sp500_trading_db"
    DATABASE_SYNC_URL: str = "postgresql://postgres:postgres@localhost:5432/sp500_trading_db"

    # Cache & Real-Time State (Redis)
    REDIS_URL: str = "redis://localhost:6379/0"


    # Risk Parameters & Circuit Breakers
    MAX_POSITION_PCT: float = Field(default=0.05, ge=0.001, le=1.0)
    MAX_SECTOR_PCT: float = Field(default=0.25, ge=0.01, le=1.0)
    MAX_PORTFOLIO_DRAWDOWN_PCT: float = Field(default=0.06, ge=0.01, le=1.0)
    MAX_DAILY_DRAWDOWN_PCT: float = Field(default=0.02, ge=0.005, le=1.0)
    DEFAULT_STOP_LOSS_PCT: float = Field(default=0.02, ge=0.001, le=1.0)
    DEFAULT_TAKE_PROFIT_PCT: float = Field(default=0.05, ge=0.001, le=1.0)

    # Data Validation & Quality Parameters
    VALIDATION_MAX_PRICE_JUMP_PCT: float = Field(default=0.35, ge=0.01, le=5.0)
    VALIDATION_OUTLIER_ZSCORE_WARNING: float = Field(default=6.0, ge=1.0)
    VALIDATION_OUTLIER_ZSCORE_ERROR: float = Field(default=20.0, ge=5.0)
    VALIDATION_SESSION_MODE: str = Field(default="REGULAR_HOURS")
    VALIDATION_DEDUP_POLICY: str = Field(default="KEEP_LAST")

    # Feature Engineering Parameters
    FEATURE_RETURN_WINDOWS: list[int] = Field(default_factory=lambda: [1, 5, 20])
    FEATURE_VOLATILITY_WINDOW: int = Field(default=20, ge=2)
    FEATURE_ATR_WINDOW: int = Field(default=14, ge=2)
    FEATURE_TREND_WINDOWS: list[int] = Field(default_factory=lambda: [20, 50, 200])
    FEATURE_VOLUME_ZSCORE_WINDOW: int = Field(default=20, ge=2)
    FEATURE_BETA_WINDOW: int = Field(default=60, ge=5)
    FEATURE_RELATIVE_WINDOW: int = Field(default=20, ge=2)

    # Baseline Machine Learning Model Parameters
    ML_TARGET_FORWARD_HORIZON: int = Field(default=5, ge=1)
    ML_TARGET_RETURN_THRESHOLD: float = Field(default=0.0)
    ML_TRAIN_RATIO: float = Field(default=0.6, ge=0.1, le=0.9)
    ML_VALIDATION_RATIO: float = Field(default=0.2, ge=0.05, le=0.5)
    ML_TEST_RATIO: float = Field(default=0.2, ge=0.05, le=0.5)
    ML_RANDOM_SEED: int = Field(default=42)
    ML_LOGISTIC_C: float = Field(default=1.0, ge=0.001)
    ML_RF_N_ESTIMATORS: int = Field(default=100, ge=10)
    ML_RF_MAX_DEPTH: Optional[int] = Field(default=5)
    ML_XGB_N_ESTIMATORS: int = Field(default=100, ge=10)
    ML_XGB_LEARNING_RATE: float = Field(default=0.05, ge=0.001, le=1.0)
    ML_XGB_MAX_DEPTH: int = Field(default=3, ge=1)
    ML_LGBM_N_ESTIMATORS: int = Field(default=100, ge=10)
    ML_LGBM_LEARNING_RATE: float = Field(default=0.05, ge=0.001, le=1.0)
    ML_MODEL_VERSION: str = Field(default="baseline-v1")

    # Advanced AI Model Parameters (Phase 08)
    ML_ADVANCED_ENABLED: bool = Field(default=True)
    ML_NEURAL_RANDOM_SEED: int = Field(default=42)
    ML_NEURAL_EPOCHS: int = Field(default=30, ge=1)
    ML_NEURAL_BATCH_SIZE: int = Field(default=32, ge=1)
    ML_NEURAL_LEARNING_RATE: float = Field(default=0.001, ge=1e-5, le=1.0)
    ML_NEURAL_DROPOUT: float = Field(default=0.2, ge=0.0, le=0.8)
    ML_MLP_HIDDEN_UNITS: list[int] = Field(default_factory=lambda: [64, 32])
    ML_LSTM_UNITS: int = Field(default=32, ge=4)
    ML_SEQUENCE_LENGTH: int = Field(default=15, ge=2)
    ML_TRANSFORMER_HEADS: int = Field(default=2, ge=1)
    ML_TRANSFORMER_KEY_DIM: int = Field(default=16, ge=4)
    ML_TRANSFORMER_FF_DIM: int = Field(default=32, ge=4)
    ML_TRANSFORMER_LAYERS: int = Field(default=1, ge=1)
    ML_EARLY_STOPPING_PATIENCE: int = Field(default=5, ge=1)

    # Market Regime Detection Parameters (Phase 09)
    REGIME_DEFAULT_DETECTOR: str = Field(default="kmeans")
    REGIME_N_CLUSTERS: int = Field(default=3, ge=2, le=10)
    REGIME_RANDOM_SEED: int = Field(default=42)
    REGIME_FEATURES: list[str] = Field(
        default_factory=lambda: [
            "return_20d",
            "rolling_volatility_20d",
            "price_vs_sma_50",
            "price_vs_sma_200",
            "vix_level",
            "vix_change_5d",
        ]
    )
    REGIME_MIN_SAMPLES: int = Field(default=30, ge=10)
    REGIME_SCALE_FEATURES: bool = Field(default=True)
    REGIME_HMM_ENABLED: bool = Field(default=True)
    REGIME_VERSION: str = Field(default="regime-v1")

    # Signal Engine Parameters (Phase 10)
    SIGNAL_LONG_THRESHOLD: float = Field(default=0.20, ge=-1.0, le=1.0)
    SIGNAL_SHORT_THRESHOLD: float = Field(default=-0.20, ge=-1.0, le=1.0)
    SIGNAL_MIN_CONFIDENCE: float = Field(default=0.25, ge=0.0, le=1.0)
    SIGNAL_MIN_ACTIVE_MODELS: int = Field(default=1, ge=1)
    SIGNAL_MIN_AGREEMENT: float = Field(default=0.50, ge=0.0, le=1.0)
    SIGNAL_DEFAULT_HORIZON: int = Field(default=5, ge=1)
    SIGNAL_MODEL_WEIGHTS: dict[str, float] = Field(
        default_factory=lambda: {
            "logistic_regression": 0.10,
            "random_forest": 0.15,
            "xgboost": 0.20,
            "lightgbm": 0.20,
            "mlp": 0.10,
            "lstm": 0.10,
            "transformer": 0.15,
        }
    )
    SIGNAL_REGIME_GATING_ENABLED: bool = Field(default=True)
    SIGNAL_COUNTER_TREND_POLICY: str = Field(default="filter")
    SIGNAL_SIDEWAYS_CONFIDENCE_MULTIPLIER: float = Field(default=1.30, ge=1.0)
    SIGNAL_VERSION: str = Field(default="signal-v1")

    # Portfolio Construction Parameters (Phase 11)
    PORTFOLIO_MODE: str = Field(default="long_only")
    PORTFOLIO_MAX_POSITION_WEIGHT: float = Field(default=0.10, ge=0.001, le=1.0)
    PORTFOLIO_MIN_POSITION_WEIGHT: float = Field(default=0.01, ge=0.0, le=1.0)
    PORTFOLIO_MAX_POSITIONS: int = Field(default=20, ge=1)
    PORTFOLIO_MAX_GROSS_EXPOSURE: float = Field(default=1.00, ge=0.01, le=2.0)
    PORTFOLIO_MAX_NET_EXPOSURE: float = Field(default=1.00, ge=0.01, le=2.0)
    PORTFOLIO_VERSION: str = Field(default="portfolio-v1")

    # Risk Engine Parameters (Phase 12 Hardening v1.1)
    RISK_ENGINE_ENABLED: bool = Field(default=True)
    RISK_MAX_POSITION_WEIGHT: float = Field(default=0.10, ge=0.001, le=1.0)
    RISK_MAX_GROSS_EXPOSURE: float = Field(default=1.00, ge=0.01, le=2.0)
    RISK_MAX_NET_EXPOSURE: float = Field(default=1.00, ge=0.01, le=2.0)
    RISK_MAX_LONG_EXPOSURE: float = Field(default=1.00, ge=0.01, le=2.0)
    RISK_MAX_SHORT_EXPOSURE: float = Field(default=1.00, ge=0.01, le=2.0)
    RISK_MAX_POSITIONS: int = Field(default=20, ge=1)
    RISK_MAX_LEVERAGE: float = Field(default=1.00, ge=0.01, le=5.0)
    RISK_VOLATILITY_LIMIT: Optional[float] = Field(default=None)
    RISK_CORRELATION_LIMIT: Optional[float] = Field(default=None)
    RISK_TURNOVER_LIMIT: Optional[float] = Field(default=None)
    RISK_MAX_DAILY_LOSS: Optional[float] = Field(default=None)
    RISK_MAX_DRAWDOWN: Optional[float] = Field(default=None)
    RISK_ADJUSTMENT_ENABLED: bool = Field(default=True)
    RISK_VERSION: str = Field(default="risk-v1.1")

    # Backtest Engine Parameters (Phase 13)
    BACKTEST_INITIAL_CAPITAL: float = Field(default=100000.0, gt=0.0)
    BACKTEST_BASE_CURRENCY: str = Field(default="USD")
    BACKTEST_COMMISSION_RATE: float = Field(default=0.0005, ge=0.0)
    BACKTEST_SLIPPAGE_RATE: float = Field(default=0.0005, ge=0.0)
    BACKTEST_MINIMUM_TRADE_NOTIONAL: float = Field(default=10.0, ge=0.0)
    BACKTEST_BENCHMARK_SYMBOL: str = Field(default="SPY")
    BACKTEST_EXECUTION_PRICE_TYPE: str = Field(default="next_open")
    BACKTEST_RISK_FREE_RATE: float = Field(default=0.0, ge=0.0)
    BACKTEST_VERSION: str = Field(default="backtest-v1")

    # Paper Trading Parameters (Phase 14)
    PAPER_TRADING_ENABLED: bool = Field(default=True)
    PAPER_INITIAL_CAPITAL: float = Field(default=100000.0, gt=0.0)
    PAPER_BASE_CURRENCY: str = Field(default="USD")
    PAPER_COMMISSION_RATE: float = Field(default=0.0005, ge=0.0)
    PAPER_SLIPPAGE_RATE: float = Field(default=0.0005, ge=0.0)
    PAPER_SPREAD_RATE: float = Field(default=0.0002, ge=0.0)
    PAPER_MARKET_IMPACT_COEFF: float = Field(default=0.0, ge=0.0)
    PAPER_BORROW_RATE: float = Field(default=0.0, ge=0.0)
    PAPER_MIN_ORDER_NOTIONAL: float = Field(default=10.0, ge=0.0)
    PAPER_MIN_POSITION_DELTA: float = Field(default=0.001, ge=0.0)
    PAPER_MAX_OPEN_ORDERS: int = Field(default=100, ge=1)
    PAPER_EXECUTION_MODE: str = Field(default="simulated")
    PAPER_IDEMPOTENCY_ENABLED: bool = Field(default=True)
    PAPER_SESSION_VERSION: str = Field(default="paper-v1.0")

    # Autonomous Trading Loop Parameters (Phase 15)
    AUTONOMOUS_LOOP_ENABLED: bool = Field(default=True)
    AUTONOMOUS_SCHEDULE_INTERVAL: str = Field(default="daily")
    AUTONOMOUS_MAX_STALE_TOLERANCE_SECONDS: float = Field(default=86400.0 * 5, ge=0.0)
    AUTONOMOUS_MISSING_SYMBOL_POLICY: str = Field(default="FAIL_CLOSED")
    AUTONOMOUS_MAX_RETRY_ATTEMPTS: int = Field(default=3, ge=0)
    AUTONOMOUS_RETRY_BACKOFF_BASE_SECONDS: float = Field(default=1.0, ge=0.0)
    AUTONOMOUS_AUTO_CHECKPOINT_ENABLED: bool = Field(default=True)
    AUTONOMOUS_VERSION: str = Field(default="autonomous-v1.0")

    # Broker Abstraction Parameters (Phase 16)
    BROKER_PROVIDER: str = Field(default="PAPER")
    BROKER_EXECUTION_MODE: str = Field(default="PAPER")
    BROKER_ENVIRONMENT: str = Field(default="PAPER")
    BROKER_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0.0)
    BROKER_MAX_RETRIES: int = Field(default=3, ge=0)
    BROKER_REQUEST_ID_PREFIX: str = Field(default="req_")
    BROKER_HEALTHCHECK_ENABLED: bool = Field(default=True)
    BROKER_VERSION: str = Field(default="broker-v1.0")

    # Live Broker Integration Parameters (Phase 17 - Alpaca)
    ALPACA_API_KEY: Optional[str] = Field(default=None)
    ALPACA_API_SECRET: Optional[str] = Field(default=None)
    ALPACA_BASE_URL: str = Field(default="https://api.alpaca.markets")
    ALPACA_PAPER_URL: str = Field(default="https://paper-api.alpaca.markets")
    ALPACA_RATE_LIMIT_PER_MINUTE: int = Field(default=200, ge=1)

    # Monitoring & Safety Parameters (Phase 18)
    MONITORING_ENABLED: bool = Field(default=True)
    MONITORING_HEARTBEAT_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0.0)
    MONITORING_MAX_CYCLE_DURATION_SECONDS: float = Field(default=120.0, gt=0.0)
    MONITORING_MAX_CONSECUTIVE_FAILURES: int = Field(default=3, ge=1)
    MONITORING_MAX_BROKER_LATENCY_MS: float = Field(default=5000.0, gt=0.0)
    MONITORING_MAX_DATA_STALENESS_SECONDS: float = Field(default=86400.0 * 5, ge=0.0)
    MONITORING_MAX_RECONCILIATION_AGE_SECONDS: float = Field(default=300.0, gt=0.0)
    MONITORING_ALERT_DEDUP_WINDOW_SECONDS: float = Field(default=60.0, ge=0.0)
    MONITORING_AUTO_KILL_SWITCH_ENABLED: bool = Field(default=True)
    MONITORING_STORAGE_DIR: str = Field(default="models/monitoring")
    MONITORING_VERSION: str = Field(default="monitoring-v1.0")


    def model_post_init(self, __context: Any) -> None:
        # Guarantee single authoritative execution mode across legacy alias fields
        object.__setattr__(self, "BROKER_EXECUTION_MODE", self.EXECUTION_MODE.value)
        if self.EXECUTION_MODE == ExecutionModeEnum.PAPER and self.BROKER_PROVIDER == "PAPER":
            object.__setattr__(self, "BROKER_ENVIRONMENT", "PAPER")
        elif self.EXECUTION_MODE == ExecutionModeEnum.LIVE:
            object.__setattr__(self, "BROKER_ENVIRONMENT", "LIVE")

    def to_safe_dict(self) -> dict:
        """Return configuration as dictionary with all secrets recursively redacted."""
        d = self.model_dump()
        sensitive_keywords = ["key", "secret", "password", "token", "auth", "credential", "private"]
        safe_dict = {}
        for k, v in d.items():
            if any(kw in k.lower() for kw in sensitive_keywords):
                safe_dict[k] = "REDACTED" if v else None
            else:
                safe_dict[k] = v
        return safe_dict

    def validate_production_settings(self) -> tuple[bool, list[str]]:
        """Validate production readiness and return (is_valid, list_of_issues)."""
        issues = []
        if self.ENVIRONMENT == EnvironmentEnum.PRODUCTION:
            if not self.SECRET_KEY or "dev-secret" in self.SECRET_KEY.lower() or len(self.SECRET_KEY) < 32:
                issues.append("SECRET_KEY must be securely rotated in production (min 32 chars, non-default).")
            if self.EXECUTION_MODE == ExecutionModeEnum.LIVE:
                if not self.ALPACA_API_KEY or "dev" in self.ALPACA_API_KEY.lower():
                    issues.append("ALPACA_API_KEY must be provided for LIVE execution mode.")
                if not self.ALPACA_API_SECRET or "dev" in self.ALPACA_API_SECRET.lower():
                    issues.append("ALPACA_API_SECRET must be provided for LIVE execution mode.")
                if self.BROKER_PROVIDER.upper() != "ALPACA":
                    issues.append(f"BROKER_PROVIDER must be ALPACA for LIVE mode (found '{self.BROKER_PROVIDER}').")
        return len(issues) == 0, issues


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()




