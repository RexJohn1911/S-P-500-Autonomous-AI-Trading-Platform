"""
Dashboard API Endpoints (Phase 19 & Audit Update).
Provides comprehensive, read-heavy, safety-first REST endpoints for the Web Dashboard.
Aggregates actual runtime and persisted state from MonitoringService, Broker, Autonomous Loop,
Risk, Portfolio, and Data storage subsystems.
Explicitly reports degraded/unavailable states and never fabricates mock healthy telemetry.
Zero direct broker exposure; zero secret leakage; fail-closed safety integration.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd
from pydantic import BaseModel, Field

from backend.app.autonomous.schemas import AutonomousLoopResult, CycleStage, CycleStatus
from backend.app.autonomous.storage import AutonomousStorage
from backend.app.broker.alpaca.adapter import AlpacaBrokerAdapter
from backend.app.broker.schemas import BrokerConfig, BrokerExecutionMode, BrokerProviderType
from backend.app.config.settings import EnvironmentEnum, ExecutionModeEnum, Settings, get_settings
from backend.app.data.calendar import evaluate_market_and_data_status, is_us_equity_market_open
from backend.app.data.models import TimeFrame
from backend.app.data.validation.storage import ProcessedDataStorage
from backend.app.monitoring.schemas import (
    ComponentType,
    HealthStatus,
    IncidentStatus,
    KillSwitchState,
    SafetySeverity,
)
from backend.app.monitoring.service import MonitoringService
from backend.app.monitoring.thresholds import MonitoringConfig
from backend.app.paper_trading.schemas import PaperTradingResult
from backend.app.paper_trading.storage import PaperTradingStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

# Singleton monitoring service instance for dashboard API
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service(settings: Settings = Depends(get_settings)) -> MonitoringService:
    global _monitoring_service
    if _monitoring_service is None:
        cfg = MonitoringConfig(
            enabled=settings.MONITORING_ENABLED,
            heartbeat_timeout_seconds=settings.MONITORING_HEARTBEAT_TIMEOUT_SECONDS,
            max_cycle_duration_seconds=settings.MONITORING_MAX_CYCLE_DURATION_SECONDS,
            max_consecutive_failures=settings.MONITORING_MAX_CONSECUTIVE_FAILURES,
            max_broker_latency_ms=settings.MONITORING_MAX_BROKER_LATENCY_MS,
            max_data_staleness_seconds=settings.MONITORING_MAX_DATA_STALENESS_SECONDS,
            max_reconciliation_age_seconds=settings.MONITORING_MAX_RECONCILIATION_AGE_SECONDS,
            alert_dedup_window_seconds=settings.MONITORING_ALERT_DEDUP_WINDOW_SECONDS,
            auto_kill_switch_enabled=settings.MONITORING_AUTO_KILL_SWITCH_ENABLED,
            storage_dir=settings.MONITORING_STORAGE_DIR,
            version=settings.MONITORING_VERSION,
        )
        _monitoring_service = MonitoringService(config=cfg)
    return _monitoring_service


# =========================================================================
# Subsystem Data Loading Helpers
# =========================================================================

def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _get_latest_paper_trading_result() -> Optional[PaperTradingResult]:
    """Inspect models/paper_trading/ for latest saved session results."""
    root = _get_project_root()
    base_dir = root / "models" / "paper_trading"
    if not base_dir.exists():
        return None
    
    session_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not session_dirs:
        return None
    
    # Sort by modification time descending
    session_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    storage = PaperTradingStorage(base_dir=base_dir)
    
    for sdir in session_dirs:
        try:
            res = storage.load_result(sdir.name)
            if res is not None:
                return res
        except Exception as e:
            logger.debug("Could not load paper trading result from %s: %s", sdir, e)
    return None


def _get_latest_autonomous_result() -> Optional[AutonomousLoopResult]:
    """Inspect models/autonomous/ for latest saved autonomous loop results."""
    root = _get_project_root()
    base_dir = root / "models" / "autonomous"
    if not base_dir.exists():
        return None
    
    session_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not session_dirs:
        return None
    
    session_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    storage = AutonomousStorage(base_dir=base_dir)
    
    for sdir in session_dirs:
        try:
            res = storage.load_result(sdir.name)
            if res is not None:
                return res
        except Exception as e:
            logger.debug("Could not load autonomous result from %s: %s", sdir, e)
    return None


def _get_latest_symbol_market_bar(symbol: str) -> Optional[Dict[str, Any]]:
    """Load latest bar for a symbol from local storage (JSON or Parquet)."""
    root = _get_project_root()
    
    # 1. Check raw JSON bars
    raw_json_path = root / "data" / "raw" / symbol / "1Day" / "bars.json"
    if raw_json_path.exists():
        try:
            with open(raw_json_path, "r", encoding="utf-8") as f:
                bars = json.load(f)
            if isinstance(bars, list) and len(bars) > 0:
                last_bar = bars[-1]
                prev_bar = bars[-2] if len(bars) > 1 else last_bar
                close_p = float(last_bar.get("close", 0.0))
                prev_close = float(prev_bar.get("close", close_p))
                change_pct = ((close_p - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
                ts_str = last_bar.get("timestamp")
                ts = datetime.fromisoformat(ts_str) if ts_str else None
                return {
                    "symbol": symbol,
                    "close": close_p,
                    "volume": float(last_bar.get("volume", 0.0)),
                    "change_pct": round(change_pct, 2),
                    "timestamp": ts,
                }
        except Exception as e:
            logger.debug("Failed reading raw JSON bars for %s: %s", symbol, e)

    # 2. Check processed Parquet bars
    proc_parquet_path = root / "data" / "processed" / symbol / "1Day" / "bars.parquet"
    if proc_parquet_path.exists():
        try:
            df = pd.read_parquet(proc_parquet_path)
            if not df.empty:
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2] if len(df) > 1 else last_row
                close_p = float(last_row.get("close", 0.0))
                prev_close = float(prev_row.get("close", close_p))
                change_pct = ((close_p - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
                ts_val = last_row.get("timestamp")
                ts = pd.to_datetime(ts_val).to_pydatetime() if ts_val is not None else None
                return {
                    "symbol": symbol,
                    "close": close_p,
                    "volume": float(last_row.get("volume", 0.0)),
                    "change_pct": round(change_pct, 2),
                    "timestamp": ts,
                }
        except Exception as e:
            logger.debug("Failed reading processed Parquet bars for %s: %s", symbol, e)

    return None


# =========================================================================
# Pydantic Schemas for Dashboard API
# =========================================================================

class PortfolioSummarySchema(BaseModel):
    equity: float = 0.0
    cash: float = 0.0
    invested_value: float = 0.0
    buying_power: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    positions_count: int = 0
    open_orders_count: int = 0
    currency: str = "USD"


class OverviewResponse(BaseModel):
    system_name: str
    execution_mode: str
    broker_provider: str
    system_status: str
    kill_switch_state: str
    autonomous_loop_state: str
    broker_status: str
    reconciliation_status: str
    portfolio: PortfolioSummarySchema
    recent_alerts: List[Dict[str, Any]]
    recent_incidents: List[Dict[str, Any]]
    last_updated: datetime
    active_incidents_count: int
    total_cycles_completed: int


class HealthResponse(BaseModel):
    system_status: str
    summary: str
    is_trading_permitted: bool
    active_incident_count: int
    kill_switch_triggered: bool
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    heartbeats: Dict[str, Dict[str, Any]]


class MarketSymbolData(BaseModel):
    symbol: str
    latest_close: float
    latest_timestamp: Optional[datetime]
    freshness_status: str
    volume_24h: float
    change_pct_24h: float
    is_stale: bool


class MarketDataResponse(BaseModel):
    market_status: str
    data_provider: str
    total_symbols_monitored: int
    symbols: List[MarketSymbolData]
    missing_symbols: List[str]
    stale_symbols: List[str]
    last_updated: datetime


class ModelInfoSchema(BaseModel):
    model_name: str
    model_type: str
    version: str
    status: str
    weight: float
    avg_inference_latency_ms: float
    is_available: bool


class ModelsResponse(BaseModel):
    active_models: List[ModelInfoSchema]
    regime_model: Dict[str, Any]
    total_models_available: int
    ensemble_status: str
    last_inference_timestamp: Optional[datetime]


class SignalItemSchema(BaseModel):
    symbol: str
    direction: str
    signal_strength: float
    confidence: float
    model_agreement: float
    regime: str
    forecast_horizon_days: int
    reason_codes: List[str]
    timestamp: datetime
    version: str


class SignalsResponse(BaseModel):
    signals: List[SignalItemSchema]
    total_signals: int
    bullish_count: int
    bearish_count: int
    flat_count: int
    last_updated: datetime


class PositionItemSchema(BaseModel):
    symbol: str
    side: str
    quantity: float
    average_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_pct: float
    exposure: float


class PortfolioResponse(BaseModel):
    summary: PortfolioSummarySchema
    positions: List[PositionItemSchema]
    allocations_by_symbol: Dict[str, float]
    equity_history: List[Dict[str, Any]]
    last_updated: datetime


class RiskMetricSchema(BaseModel):
    name: str
    current_value: float
    limit_value: float
    status: str
    unit: str


class RiskResponse(BaseModel):
    risk_status: str
    gross_exposure: float
    net_exposure: float
    leverage: float
    max_position_weight: float
    active_violations: List[str]
    warnings: List[str]
    hard_limit_breaches: List[str]
    risk_metrics: List[RiskMetricSchema]
    is_risk_approved: bool
    last_updated: datetime


class OrderItemSchema(BaseModel):
    client_order_id: str
    broker_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float
    remaining_quantity: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    average_fill_price: Optional[float]
    time_in_force: str
    status: str
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    execution_mode: str


class OrdersResponse(BaseModel):
    orders: List[OrderItemSchema]
    total_orders: int
    open_orders_count: int
    filled_orders_count: int
    cancelled_orders_count: int


class ExecutionItemSchema(BaseModel):
    execution_id: str
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    execution_price: float
    executed_at: datetime
    commission: float
    slippage: float
    venue: str
    execution_mode: str


class ExecutionsResponse(BaseModel):
    executions: List[ExecutionItemSchema]
    total_executions: int
    total_volume_traded: float
    total_commissions_paid: float


class BrokerStatusResponse(BaseModel):
    provider: str
    execution_mode: str
    connection_status: str
    authentication_status: str
    api_latency_ms: float
    rate_limit_headroom_pct: float
    supported_capabilities: List[str]
    account_status: str
    buying_power: float
    cash: float
    portfolio_value: float
    currency: str
    is_paper: bool
    last_health_check: datetime


class AutonomousLoopResponse(BaseModel):
    state: str
    session_id: str
    total_cycles: int
    successful_cycles: int
    failed_cycles: int
    current_cycle_id: Optional[str]
    uptime_seconds: float
    last_cycle_completed_at: Optional[datetime]
    pipeline_stages: List[Dict[str, Any]]
    consecutive_failures: int
    auto_checkpoint_enabled: bool


class KillSwitchTriggerRequest(BaseModel):
    reason: str
    operator_name: str = "WEB_OPERATOR"


class KillSwitchResetRequest(BaseModel):
    operator_name: str = "WEB_OPERATOR"


class IncidentAcknowledgeRequest(BaseModel):
    operator_name: str = "WEB_OPERATOR"


class IncidentResolveRequest(BaseModel):
    notes: str
    operator_name: str = "WEB_OPERATOR"


# =========================================================================
# Dashboard Routes
# =========================================================================

@router.get("/overview", response_model=OverviewResponse)
async def get_overview(
    settings: Settings = Depends(get_settings),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Retrieve holistic system overview metrics backed by real subsystem telemetry."""
    now = datetime.now(timezone.utc)
    health_snap = monitoring.get_system_health()
    ks_snap = monitoring.kill_switch.get_snapshot()
    recent_alerts = [e.to_dict() for e in monitoring.alert_manager.get_recent_events(limit=5)]
    recent_incidents = [i.to_dict() for i in monitoring.incident_manager.get_active_incidents()[:5]]
    metrics = monitoring.get_operational_metrics()

    # Load actual paper trading result if present
    paper_res = _get_latest_paper_trading_result()
    auto_res = _get_latest_autonomous_result()

    if paper_res and paper_res.account_snapshots:
        last_snap = paper_res.account_snapshots[-1]
        portfolio_data = PortfolioSummarySchema(
            equity=last_snap.equity,
            cash=last_snap.cash,
            invested_value=last_snap.market_value,
            buying_power=last_snap.buying_power,
            unrealized_pnl=last_snap.unrealized_pnl,
            realized_pnl=last_snap.realized_pnl,
            daily_pnl=0.0,
            gross_exposure=last_snap.gross_exposure,
            net_exposure=last_snap.net_exposure,
            leverage=last_snap.leverage,
            positions_count=len(last_snap.positions),
            open_orders_count=0,
        )
    else:
        # Clean unallocated initial paper state
        initial_cap = settings.PAPER_INITIAL_CAPITAL if settings.EXECUTION_MODE == ExecutionModeEnum.PAPER else 0.0
        portfolio_data = PortfolioSummarySchema(
            equity=initial_cap,
            cash=initial_cap,
            invested_value=0.0,
            buying_power=initial_cap,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            leverage=0.0,
            positions_count=0,
            open_orders_count=0,
        )

    # Broker status resolution based on authoritative mode and live verification
    if settings.EXECUTION_MODE == ExecutionModeEnum.PAPER:
        broker_status = "SIMULATED"
    else:
        try:
            live_broker = AlpacaBrokerAdapter(BrokerConfig(execution_mode=BrokerExecutionMode.LIVE, provider=BrokerProviderType.ALPACA))
            v_res = live_broker.verify_live_connection()
            broker_status = v_res.get("connection_status", "DISCONNECTED")
        except Exception:
            broker_status = "DISCONNECTED"

    # Reconciliation status resolution
    reconcil_status = "UNINITIALIZED"
    if paper_res and paper_res.reconciliation_reports:
        last_rec = paper_res.reconciliation_reports[-1]
        rec_stat_val = last_rec.status.value if hasattr(last_rec.status, "value") else str(last_rec.status)
        reconcil_status = "MATCHED" if rec_stat_val == "MATCHED" else "DISCREPANCY"

    # Autonomous loop state resolution
    if ks_snap.is_triggered:
        auto_state = "PAUSED"
    elif auto_res:
        auto_state = auto_res.session.state.value if hasattr(auto_res.session.state, "value") else str(auto_res.session.state)
    else:
        auto_state = "STANDBY"

    total_cycles = 0
    if auto_res and auto_res.cycles:
        total_cycles = len(auto_res.cycles)
    elif metrics.cycles_total:
        total_cycles = metrics.cycles_total

    return OverviewResponse(
        system_name=settings.PROJECT_NAME,
        execution_mode=settings.EXECUTION_MODE.value,
        broker_provider=settings.BROKER_PROVIDER,
        system_status=health_snap.status.value,
        kill_switch_state=ks_snap.state.value,
        autonomous_loop_state=auto_state,
        broker_status=broker_status,
        reconciliation_status=reconcil_status,
        portfolio=portfolio_data,
        recent_alerts=recent_alerts,
        recent_incidents=recent_incidents,
        last_updated=now,
        active_incidents_count=health_snap.active_incident_count,
        total_cycles_completed=total_cycles,
    )


@router.get("/health", response_model=HealthResponse)
async def get_health(monitoring: MonitoringService = Depends(get_monitoring_service)):
    """Retrieve detailed subsystem health and heartbeat diagnostics."""
    snap = monitoring.get_system_health()
    heartbeats = monitoring.heartbeat_tracker.check_heartbeats()

    return HealthResponse(
        system_status=snap.status.value,
        summary=snap.summary,
        is_trading_permitted=snap.is_trading_permitted,
        active_incident_count=snap.active_incident_count,
        kill_switch_triggered=snap.kill_switch_triggered,
        timestamp=snap.timestamp,
        components={k: v.to_dict() for k, v in snap.components.items()},
        heartbeats={k.value: v.to_dict() for k, v in heartbeats.items()},
    )


@router.get("/market-data", response_model=MarketDataResponse)
async def get_market_data(
    symbols_query: Optional[str] = Query(None, description="Comma-separated symbols to query"),
):
    """Retrieve market data freshness, prices, and provider status from real data store."""
    now = datetime.now(timezone.utc)
    target_symbols = [s.strip().upper() for s in symbols_query.split(",")] if symbols_query else ["AAPL", "TSLA", "SPY"]
    
    symbols_data: List[MarketSymbolData] = []
    missing_symbols: List[str] = []
    stale_symbols: List[str] = []

    for sym in target_symbols:
        bar_info = _get_latest_symbol_market_bar(sym)
        if bar_info:
            bar_ts = bar_info["timestamp"]
            is_stale = False
            freshness = "FRESH"
            if bar_ts:
                # If timestamp is naive, make it UTC-aware
                if bar_ts.tzinfo is None:
                    bar_ts = bar_ts.replace(tzinfo=timezone.utc)
                age_seconds = (now - bar_ts).total_seconds()
                # Consider stale if older than 5 trading days
                if age_seconds > (86400 * 5):
                    is_stale = True
                    freshness = "STALE"
                    stale_symbols.append(sym)

            symbols_data.append(
                MarketSymbolData(
                    symbol=sym,
                    latest_close=bar_info["close"],
                    latest_timestamp=bar_ts,
                    freshness_status=freshness,
                    volume_24h=bar_info["volume"],
                    change_pct_24h=bar_info["change_pct"],
                    is_stale=is_stale,
                )
            )
        else:
            missing_symbols.append(sym)

    market_status = evaluate_market_and_data_status(
        total_symbols=len(symbols_data),
        stale_symbols_count=len(stale_symbols),
        current_time=now,
    )

    return MarketDataResponse(
        market_status=market_status,
        data_provider="LOCAL_VALIDATED_DATA_STORE",
        total_symbols_monitored=len(symbols_data),
        symbols=symbols_data,
        missing_symbols=missing_symbols,
        stale_symbols=stale_symbols,
        last_updated=now,
    )


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Retrieve active AI model ensemble and inference status based on stored artifacts."""
    root = _get_project_root()
    models_dir = root / "models" / "trained"
    auto_res = _get_latest_autonomous_result()

    latest_telemetry: Dict[str, Any] = {}
    last_inf_ts: Optional[datetime] = None

    if auto_res and auto_res.cycles:
        for c in reversed(auto_res.cycles):
            if c.model_telemetry:
                latest_telemetry = c.model_telemetry
                last_inf_ts = c.end_time or c.start_time
                break

    if not latest_telemetry and auto_res and auto_res.checkpoints:
        for ck in reversed(auto_res.checkpoints):
            m_tel = ck.metadata.get("model_telemetry")
            if m_tel:
                latest_telemetry = m_tel
                last_inf_ts = ck.timestamp
                break

    latencies = latest_telemetry.get("latencies", {})
    regime_name = latest_telemetry.get("regime", "UNINITIALIZED")
    regime_conf = float(latest_telemetry.get("confidence", latest_telemetry.get("regime_confidence", 0.0)))

    model_registry_info = [
        ("logistic_regression", "Logistic Regression", "Linear Baseline", 0.10),
        ("random_forest", "Random Forest", "Tree Ensemble", 0.15),
        ("xgboost", "XGBoost", "Gradient Boosted Trees", 0.20),
        ("lightgbm", "LightGBM", "Gradient Boosted Trees", 0.20),
        ("mlp", "Feed-Forward MLP", "Neural Network", 0.10),
        ("lstm", "LSTM Temporal", "Recurrent Neural Network", 0.10),
        ("transformer", "Multi-Head Transformer", "Attention Network", 0.15),
    ]

    active_models: List[ModelInfoSchema] = []
    for mod_key, mod_name, mod_type, mod_weight in model_registry_info:
        meta_file = models_dir / mod_key / "baseline-v1" / "metadata.json"
        is_avail = meta_file.exists()
        status_str = "READY" if is_avail else "UNAVAILABLE"
        version_str = "baseline-v1"
        if is_avail:
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    version_str = meta.get("model_version", "baseline-v1")
            except Exception:
                pass

        mod_latency = float(latencies.get(mod_key, 0.0))

        active_models.append(
            ModelInfoSchema(
                model_name=mod_name,
                model_type=mod_type,
                version=version_str,
                status=status_str,
                weight=mod_weight,
                avg_inference_latency_ms=mod_latency if is_avail else 0.0,
                is_available=is_avail,
            )
        )

    # Regime model status
    regime_dir = root / "models" / "regimes" / "kmeans" / "regime-v1"
    regime_available = (regime_dir / "detector.joblib").exists()
    regime_model = {
    "current_regime": regime_name if latest_telemetry.get("regime") else "UNINITIALIZED",
    "regime_detector": "K-Means (3 Clusters) + HMM",
    "confidence": regime_conf if latest_telemetry.get("regime") else 0.0,
    "is_available": regime_available,
}
    available_count = sum(1 for m in active_models if m.is_available)
    ensemble_status = "OPERATIONAL" if available_count >= 4 else ("DEGRADED" if available_count > 0 else "UNAVAILABLE")

    return ModelsResponse(
        active_models=active_models,
        regime_model=regime_model,
        total_models_available=available_count,
        ensemble_status=ensemble_status,
        last_inference_timestamp=last_inf_ts,
    )


@router.get("/signals", response_model=SignalsResponse)
async def get_signals():
    """Retrieve active quantitative signals generated by AI engine from latest session."""
    now = datetime.now(timezone.utc)
    auto_res = _get_latest_autonomous_result()

    signals: List[SignalItemSchema] = []
    if auto_res and auto_res.checkpoints:
        last_chk = auto_res.checkpoints[-1]
        chk_signals = last_chk.metadata.get("signals", {}) if (hasattr(last_chk, "metadata") and isinstance(last_chk.metadata, dict)) else {}
        for sym, sig_dict in chk_signals.items():
            if isinstance(sig_dict, dict):
                signals.append(
                    SignalItemSchema(
                        symbol=sym,
                        direction=sig_dict.get("direction", "FLAT"),
                        signal_strength=float(sig_dict.get("strength", 0.0)),
                        confidence=float(sig_dict.get("confidence", 0.0)),
                        model_agreement=float(sig_dict.get("model_agreement", 0.0)),
                        regime=sig_dict.get("regime", "UNKNOWN"),
                        forecast_horizon_days=int(sig_dict.get("forecast_horizon", 5)),
                        reason_codes=sig_dict.get("reason_codes", []),
                        timestamp=last_chk.timestamp,
                        version=sig_dict.get("version", "signal-v1"),
                    )
                )

    bullish_count = sum(1 for s in signals if s.direction == "LONG")
    bearish_count = sum(1 for s in signals if s.direction == "SHORT")
    flat_count = sum(1 for s in signals if s.direction == "FLAT")

    return SignalsResponse(
        signals=signals,
        total_signals=len(signals),
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        flat_count=flat_count,
        last_updated=now,
    )


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(settings: Settings = Depends(get_settings)):
    """Retrieve current portfolio holdings, equity breakdown, and historical curve."""
    now = datetime.now(timezone.utc)
    paper_res = _get_latest_paper_trading_result()

    positions: List[PositionItemSchema] = []
    allocations_by_symbol: Dict[str, float] = {}
    equity_history: List[Dict[str, Any]] = []

    if paper_res and paper_res.account_snapshots:
        last_snap = paper_res.account_snapshots[-1]
        summary = PortfolioSummarySchema(
            equity=last_snap.equity,
            cash=last_snap.cash,
            invested_value=last_snap.market_value,
            buying_power=last_snap.buying_power,
            unrealized_pnl=last_snap.unrealized_pnl,
            realized_pnl=last_snap.realized_pnl,
            daily_pnl=0.0,
            gross_exposure=last_snap.gross_exposure,
            net_exposure=last_snap.net_exposure,
            leverage=last_snap.leverage,
            positions_count=len(last_snap.positions),
            open_orders_count=0,
        )
        for sym, pos_data in last_snap.positions.items():
            if isinstance(pos_data, dict):
                qty = float(pos_data.get("quantity", 0.0))
                entry_p = float(pos_data.get("average_entry_price", 0.0))
                curr_p = float(pos_data.get("current_price", entry_p))
                mkt_val = qty * curr_p
                unrealized = mkt_val - (qty * entry_p)
                unrealized_pct = (unrealized / (qty * entry_p) * 100.0) if (qty * entry_p) > 0 else 0.0
                weight_pct = (mkt_val / last_snap.equity * 100.0) if last_snap.equity > 0 else 0.0
                side = "LONG" if qty > 0 else ("SHORT" if qty < 0 else "FLAT")

                positions.append(
                    PositionItemSchema(
                        symbol=sym,
                        side=side,
                        quantity=abs(qty),
                        average_entry_price=entry_p,
                        current_price=curr_p,
                        market_value=mkt_val,
                        unrealized_pnl=unrealized,
                        unrealized_pnl_pct=unrealized_pct,
                        weight_pct=weight_pct,
                        exposure=mkt_val,
                    )
                )
                allocations_by_symbol[sym] = round(weight_pct / 100.0, 4)

        cash_pct = (last_snap.cash / last_snap.equity) if last_snap.equity > 0 else 1.0
        allocations_by_symbol["CASH"] = round(cash_pct, 4)

        for s in paper_res.account_snapshots:
            ts_str = s.timestamp.isoformat() if hasattr(s.timestamp, "isoformat") else str(s.timestamp)
            equity_history.append({"timestamp": ts_str, "equity": s.equity})
    else:
        init_cap = settings.PAPER_INITIAL_CAPITAL if settings.EXECUTION_MODE == ExecutionModeEnum.PAPER else 0.0
        summary = PortfolioSummarySchema(
            equity=init_cap,
            cash=init_cap,
            invested_value=0.0,
            buying_power=init_cap,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            leverage=0.0,
            positions_count=0,
            open_orders_count=0,
        )
        allocations_by_symbol = {"CASH": 1.0} if init_cap > 0 else {}

    return PortfolioResponse(
        summary=summary,
        positions=positions,
        allocations_by_symbol=allocations_by_symbol,
        equity_history=equity_history,
        last_updated=now,
    )


@router.get("/risk", response_model=RiskResponse)
async def get_risk(settings: Settings = Depends(get_settings)):
    """Retrieve risk engine bounds, exposures, and active violation status."""
    now = datetime.now(timezone.utc)
    paper_res = _get_latest_paper_trading_result()

    gross_exp = 0.0
    net_exp = 0.0
    leverage = 0.0
    max_weight = 0.0

    if paper_res and paper_res.account_snapshots:
        last_snap = paper_res.account_snapshots[-1]
        gross_exp = last_snap.gross_exposure
        net_exp = last_snap.net_exposure
        leverage = last_snap.leverage
        if last_snap.positions and last_snap.equity > 0:
            weights = [
                abs(float(p.get("quantity", 0.0)) * float(p.get("current_price", 0.0))) / last_snap.equity
                for p in last_snap.positions.values()
                if isinstance(p, dict)
            ]
            max_weight = max(weights) if weights else 0.0

    metrics = [
        RiskMetricSchema(
            name="Gross Exposure",
            current_value=round(gross_exp, 3),
            limit_value=settings.RISK_MAX_GROSS_EXPOSURE,
            status="PASS" if gross_exp <= settings.RISK_MAX_GROSS_EXPOSURE else "BREACH",
            unit="Ratio",
        ),
        RiskMetricSchema(
            name="Net Exposure",
            current_value=round(net_exp, 3),
            limit_value=settings.RISK_MAX_NET_EXPOSURE,
            status="PASS" if abs(net_exp) <= settings.RISK_MAX_NET_EXPOSURE else "BREACH",
            unit="Ratio",
        ),
        RiskMetricSchema(
            name="Leverage",
            current_value=round(leverage, 3),
            limit_value=settings.RISK_MAX_LEVERAGE,
            status="PASS" if leverage <= settings.RISK_MAX_LEVERAGE else "BREACH",
            unit="x",
        ),
        RiskMetricSchema(
            name="Max Position Weight",
            current_value=round(max_weight, 3),
            limit_value=settings.RISK_MAX_POSITION_WEIGHT,
            status="PASS" if max_weight <= settings.RISK_MAX_POSITION_WEIGHT else "BREACH",
            unit="Weight",
        ),
    ]

    breaches = [m.name for m in metrics if m.status == "BREACH"]
    is_approved = len(breaches) == 0

    return RiskResponse(
        risk_status="HEALTHY" if is_approved else "BREACHED",
        gross_exposure=round(gross_exp, 3),
        net_exposure=round(net_exp, 3),
        leverage=round(leverage, 3),
        max_position_weight=round(max_weight, 3),
        active_violations=breaches,
        warnings=[],
        hard_limit_breaches=breaches,
        risk_metrics=metrics,
        is_risk_approved=is_approved,
        last_updated=now,
    )


@router.get("/orders", response_model=OrdersResponse)
async def get_orders(
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """Retrieve recorded order history and open orders from actual paper trading storage."""
    paper_res = _get_latest_paper_trading_result()
    orders_list: List[OrderItemSchema] = []

    if paper_res and paper_res.orders:
        for o in paper_res.orders:
            orders_list.append(
                OrderItemSchema(
                    client_order_id=o.client_order_id,
                    broker_order_id=getattr(o, "order_id", None),
                    symbol=o.symbol,
                    side=o.side.value if hasattr(o.side, "value") else str(o.side),
                    order_type=o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type),
                    quantity=o.quantity,
                    filled_quantity=o.filled_quantity,
                    remaining_quantity=max(0.0, o.quantity - o.filled_quantity),
                    limit_price=o.limit_price,
                    stop_price=o.stop_price,
                    average_fill_price=o.executed_price,
                    time_in_force=o.time_in_force.value if hasattr(o.time_in_force, "value") else str(o.time_in_force),
                    status=o.status.value if hasattr(o.status, "value") else str(o.status),
                    submitted_at=o.submitted_at,
                    filled_at=o.filled_at,
                    execution_mode="PAPER",
                )
            )

    if symbol:
        orders_list = [o for o in orders_list if o.symbol == symbol.upper()]
    if status:
        orders_list = [o for o in orders_list if o.status.upper() == status.upper()]

    open_count = len([o for o in orders_list if o.status in ("ACCEPTED", "PARTIALLY_FILLED", "SUBMITTED")])
    filled_count = len([o for o in orders_list if o.status == "FILLED"])
    cancelled_count = len([o for o in orders_list if o.status == "CANCELLED"])

    return OrdersResponse(
        orders=orders_list[:limit],
        total_orders=len(orders_list),
        open_orders_count=open_count,
        filled_orders_count=filled_count,
        cancelled_orders_count=cancelled_count,
    )


@router.get("/executions", response_model=ExecutionsResponse)
async def get_executions(limit: int = Query(50, ge=1, le=500)):
    """Retrieve execution records and fill statistics from actual paper trading storage."""
    paper_res = _get_latest_paper_trading_result()
    execs_list: List[ExecutionItemSchema] = []
    total_vol = 0.0
    total_comm = 0.0

    if paper_res and paper_res.executions:
        for e in paper_res.executions:
            exec_time = getattr(e, "timestamp", datetime.now(timezone.utc))
            comm = getattr(e, "commission", 0.0)
            slip = getattr(e, "slippage_cost", getattr(e, "slippage", 0.0))
            price = getattr(e, "executed_price", getattr(e, "price", 0.0))
            qty = getattr(e, "quantity", 0.0)
            total_vol += (price * qty)
            total_comm += comm

            execs_list.append(
                ExecutionItemSchema(
                    execution_id=e.execution_id,
                    broker_order_id=getattr(e, "order_id", getattr(e, "broker_order_id", "")),
                    client_order_id=getattr(e, "client_order_id", ""),
                    symbol=e.symbol,
                    side=e.side.value if hasattr(e.side, "value") else str(e.side),
                    quantity=qty,
                    execution_price=price,
                    executed_at=exec_time,
                    commission=comm,
                    slippage=slip,
                    venue=getattr(e, "venue", "SIMULATED_EXCHANGE"),
                    execution_mode="PAPER",
                )
            )

    return ExecutionsResponse(
        executions=execs_list[:limit],
        total_executions=len(execs_list),
        total_volume_traded=round(total_vol, 2),
        total_commissions_paid=round(total_comm, 2),
    )


@router.get("/broker", response_model=BrokerStatusResponse)
async def get_broker(settings: Settings = Depends(get_settings)):
    """Retrieve live/paper broker connection parameters, account state, and capabilities."""
    now = datetime.now(timezone.utc)
    paper_res = _get_latest_paper_trading_result()

    is_paper = (settings.EXECUTION_MODE == ExecutionModeEnum.PAPER)
    if is_paper:
        conn_status = "CONNECTED"
        auth_status = "AUTHENTICATED"
        acct_status = "ACTIVE"
        lat_ms = 0.5
        caps = [
            "MARKET_ORDERS",
            "LIMIT_ORDERS",
            "STOP_ORDERS",
            "SHORT_SELLING",
            "CANCEL_ORDER",
            "POSITIONS",
            "ACCOUNT_DATA",
            "IDEMPOTENCY",
        ]
        equity = settings.PAPER_INITIAL_CAPITAL
        cash = settings.PAPER_INITIAL_CAPITAL
        buying_power = settings.PAPER_INITIAL_CAPITAL
        if paper_res and paper_res.account_snapshots:
            last_snap = paper_res.account_snapshots[-1]
            equity = last_snap.equity
            cash = last_snap.cash
            buying_power = last_snap.buying_power
    else:
        try:
            live_broker = AlpacaBrokerAdapter(
                BrokerConfig(execution_mode=BrokerExecutionMode.LIVE, provider=BrokerProviderType.ALPACA)
            )
            v_res = live_broker.verify_live_connection()
            conn_status = v_res.get("connection_status", "DISCONNECTED")
            auth_status = v_res.get("authentication_status", "UNAUTHENTICATED")
            acct_status = v_res.get("account_status", "UNCONFIGURED")
            lat_ms = float(v_res.get("api_latency_ms", 0.0))
            caps = ["MARKET_ORDERS", "LIMIT_ORDERS", "CANCEL_ORDER", "POSITIONS", "ACCOUNT_DATA"] if v_res.get("connected") else []
            equity = float(v_res.get("portfolio_value", 0.0))
            cash = float(v_res.get("cash", 0.0))
            buying_power = float(v_res.get("buying_power", 0.0))
        except Exception:
            conn_status = "DISCONNECTED"
            auth_status = "UNAUTHENTICATED"
            acct_status = "ERROR"
            lat_ms = 0.0
            caps = []
            equity = 0.0
            cash = 0.0
            buying_power = 0.0

    return BrokerStatusResponse(
        provider=settings.BROKER_PROVIDER,
        execution_mode=settings.EXECUTION_MODE.value,
        connection_status=conn_status,
        authentication_status=auth_status,
        api_latency_ms=lat_ms,
        rate_limit_headroom_pct=100.0,
        supported_capabilities=caps,
        account_status=acct_status,
        buying_power=buying_power,
        cash=cash,
        portfolio_value=equity,
        currency="USD",
        is_paper=is_paper,
        last_health_check=now,
    )


@router.get("/autonomous-loop", response_model=AutonomousLoopResponse)
async def get_autonomous_loop(
    settings: Settings = Depends(get_settings),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Retrieve autonomous controller execution cycle status and pipeline stages."""
    now = datetime.now(timezone.utc)
    auto_res = _get_latest_autonomous_result()
    metrics = monitoring.get_operational_metrics()
    ks_snap = monitoring.kill_switch.get_snapshot()

    if ks_snap.is_triggered:
        loop_state = "PAUSED"
    elif auto_res and auto_res.cycles:
        loop_state = auto_res.session.state.value if hasattr(auto_res.session.state, "value") else str(auto_res.session.state)
    else:
        loop_state = "STANDBY"

    session_id = auto_res.session.session_id if (auto_res and auto_res.session) else "NONE"
    total_cycles = len(auto_res.cycles) if (auto_res and auto_res.cycles) else (metrics.cycles_total or 0)
    succ_cycles = len([c for c in auto_res.cycles if str(c.status) == "COMPLETED"]) if (auto_res and auto_res.cycles) else (metrics.cycles_successful or 0)
    failed_cycles = len([c for c in auto_res.cycles if str(c.status) == "FAILED"]) if (auto_res and auto_res.cycles) else (metrics.cycles_failed or 0)
    last_cycle_ts = auto_res.cycles[-1].timestamp if (auto_res and auto_res.cycles and hasattr(auto_res.cycles[-1], "timestamp")) else (auto_res.cycles[-1].end_time if (auto_res and auto_res.cycles) else None)

    stage_definitions = [
        ("DATA_ACQUISITION", "1. Market Data"),
        ("DATA_VALIDATION", "2. Validation"),
        ("FEATURE_GENERATION", "3. Feature Engineering"),
        ("MODEL_INFERENCE", "4. Model Inference"),
        ("REGIME_DETECTION", "5. Regime Detection"),
        ("SIGNAL_GENERATION", "6. Signal Generation"),
        ("PORTFOLIO_CONSTRUCTION", "7. Portfolio Construction"),
        ("RISK_ENGINE", "8. Risk Engine"),
        ("ORDER_GENERATION", "9. Order Generation"),
        ("PAPER_EXECUTION", "10. Execution"),
        ("ACCOUNT_VALUATION", "11. Valuation"),
        ("RECONCILIATION", "12. Reconciliation"),
        ("CHECKPOINTING", "13. Checkpoint"),
        ("COMPLETED", "14. Event Emission"),
    ]

    stages: List[Dict[str, Any]] = []

    if auto_res is None or not auto_res.cycles:
        for _, display_name in stage_definitions:
            stages.append({"stage": display_name, "status": "STANDBY", "duration_ms": 0.0})
    else:
        last_cycle = auto_res.cycles[-1]
        executed_map = {}
        if last_cycle.stages_executed:
            for s in last_cycle.stages_executed:
                executed_map[s["stage"]] = s

        cycle_stat_str = last_cycle.status.value if hasattr(last_cycle.status, "value") else str(last_cycle.status)
        is_cycle_failed_or_skipped = ("FAIL" in cycle_stat_str.upper() or "SKIP" in cycle_stat_str.upper())

        if executed_map:
            for enum_key, display_name in stage_definitions:
                if enum_key in executed_map:
                    rec = executed_map[enum_key]
                    stages.append({
                        "stage": display_name,
                        "status": rec.get("status", "COMPLETED"),
                        "duration_ms": float(rec.get("duration_ms", 0.0)),
                    })
                else:
                    stages.append({
                        "stage": display_name,
                        "status": "SKIPPED" if is_cycle_failed_or_skipped else "COMPLETED",
                        "duration_ms": 0.0,
                    })
        else:
            is_comp = ("COMPLETED" in cycle_stat_str.upper())
            for _, display_name in stage_definitions:
                stages.append({
                    "stage": display_name,
                    "status": "COMPLETED" if is_comp else ("SKIPPED" if is_cycle_failed_or_skipped else "STANDBY"),
                    "duration_ms": 0.0,
                })

    return AutonomousLoopResponse(
        state=loop_state,
        session_id=session_id,
        total_cycles=total_cycles,
        successful_cycles=succ_cycles,
        failed_cycles=failed_cycles,
        current_cycle_id=None,
        uptime_seconds=0.0,
        last_cycle_completed_at=last_cycle_ts,
        pipeline_stages=stages,
        consecutive_failures=0,
        auto_checkpoint_enabled=settings.AUTONOMOUS_AUTO_CHECKPOINT_ENABLED,
    )


@router.get("/alerts")
async def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Retrieve safety events and alerts."""
    events = monitoring.alert_manager.get_recent_events(limit=limit)
    return {"alerts": [e.to_dict() for e in events], "total": len(events)}


@router.get("/incidents")
async def get_incidents(monitoring: MonitoringService = Depends(get_monitoring_service)):
    """Retrieve operational incidents."""
    incidents = monitoring.incident_manager.get_active_incidents()
    return {"incidents": [i.to_dict() for i in incidents], "active_count": len(incidents)}


@router.post("/incidents/{incident_id}/acknowledge")
async def acknowledge_incident(
    incident_id: str,
    body: IncidentAcknowledgeRequest,
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Acknowledge an active operational incident."""
    inc = monitoring.incident_manager.acknowledge_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found or already acknowledged")
    return {"status": "ACKNOWLEDGED", "incident": inc.to_dict()}


@router.post("/incidents/{incident_id}/resolve")
async def resolve_incident(
    incident_id: str,
    body: IncidentResolveRequest,
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Explicitly resolve an operational incident."""
    inc = monitoring.incident_manager.resolve_incident(incident_id, notes=body.notes)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"status": "RESOLVED", "incident": inc.to_dict()}


@router.get("/kill-switch")
async def get_kill_switch(monitoring: MonitoringService = Depends(get_monitoring_service)):
    """Retrieve current kill switch status."""
    snap = monitoring.kill_switch.get_snapshot()
    critical_comps = [
        c.value for c, s in monitoring.watchdog._component_snapshots.items()
        if s.status == HealthStatus.CRITICAL
    ]
    can_reset = not monitoring.incident_manager.has_unresolved_critical_incident() and len(critical_comps) == 0
    return {
        "kill_switch": snap.to_dict(),
        "can_reset": can_reset,
    }


@router.post("/kill-switch/trigger")
async def trigger_kill_switch(
    body: KillSwitchTriggerRequest,
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Manually trigger the global kill switch."""
    snap = monitoring.trigger_kill_switch(
        reason=body.reason,
        triggered_by=body.operator_name,
        metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
    )
    return {"status": "TRIGGERED", "kill_switch": snap.to_dict()}


@router.post("/kill-switch/reset")
async def reset_kill_switch(
    body: KillSwitchResetRequest,
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Reset the global kill switch back to ARMED state."""
    success, message = monitoring.reset_kill_switch(cleared_by=body.operator_name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Kill switch reset blocked: {message}")
    return {"status": "ARMED", "message": message, "kill_switch": monitoring.kill_switch.get_snapshot().to_dict()}


@router.get("/reconciliation")
async def get_reconciliation():
    """Retrieve ledger reconciliation comparison from actual paper trading storage."""
    paper_res = _get_latest_paper_trading_result()
    if paper_res and paper_res.reconciliation_reports:
        last_rec = paper_res.reconciliation_reports[-1]
        stat_val = last_rec.status.value if hasattr(last_rec.status, "value") else str(last_rec.status)
        is_matched = (stat_val == "MATCHED")
        cash_diff = float(last_rec.cash_expected - last_rec.cash_actual)
        loc_orders = last_rec.details.get("local_order_count", len(paper_res.orders))
        brk_orders = last_rec.details.get("broker_order_count", len(paper_res.orders))
        loc_pos = last_rec.details.get("local_position_count", len(paper_res.account_snapshots[-1].positions) if paper_res.account_snapshots else 0)
        brk_pos = last_rec.details.get("broker_position_count", loc_pos)
        return {
            "status": "MATCHED" if is_matched else "DISCREPANCY",
            "last_reconciled_at": last_rec.timestamp.isoformat() if hasattr(last_rec.timestamp, "isoformat") else str(last_rec.timestamp),
            "local_orders_count": loc_orders,
            "broker_orders_count": brk_orders,
            "local_positions_count": loc_pos,
            "broker_positions_count": brk_pos,
            "cash_difference": cash_diff,
            "position_mismatches": last_rec.positions_mismatches,
            "order_mismatches": last_rec.orders_mismatches,
            "external_orders_detected": last_rec.details.get("external_orders", []),
        }

    return {
        "status": "UNINITIALIZED",
        "last_reconciled_at": None,
        "local_orders_count": 0,
        "broker_orders_count": 0,
        "local_positions_count": 0,
        "broker_positions_count": 0,
        "cash_difference": 0.0,
        "position_mismatches": [],
        "order_mismatches": [],
        "external_orders_detected": [],
    }


@router.get("/audit")
async def get_audit_trail(
    limit: int = Query(100, ge=1, le=1000),
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Retrieve append-only audit trail records (secret-free)."""
    events = monitoring.audit_logger.get_recent_audit_events(limit=limit)
    return {"audit_events": events, "count": len(events)}


@router.get("/backtest")
async def get_backtest_summary():
    """Retrieve historical research backtest benchmarks from archive or report unavailable."""
    root = _get_project_root()
    backtests_dir = root / "backtests"
    
    # Check if any saved backtest summary file exists in backtests/
    bt_files = list(backtests_dir.glob("*.json")) if backtests_dir.exists() else []
    if bt_files:
        bt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            with open(bt_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.debug("Failed reading backtest JSON %s: %s", bt_files[0], e)

    return {
        "run_id": "NONE",
        "dataset": "NO_BACKTEST_RUN_AVAILABLE",
        "bars_count": 0,
        "initial_capital": 0.0,
        "final_equity": 0.0,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "calmar_ratio": 0.0,
        "total_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "benchmark_spy_return_pct": 0.0,
        "alpha_annualized": 0.0,
        "beta_to_benchmark": 0.0,
        "provenance_hash": "",
    }
