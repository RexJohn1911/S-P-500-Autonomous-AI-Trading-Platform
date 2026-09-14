/**
 * TypeScript Type Definitions for S&P 500 Web Dashboard (Phase 19).
 * All types strictly align with backend API schemas in backend/app/api/dashboard.py.
 */

export type PageId =
  | 'overview'
  | 'health'
  | 'market-data'
  | 'models'
  | 'signals'
  | 'portfolio'
  | 'risk'
  | 'orders'
  | 'executions'
  | 'broker'
  | 'autonomous-loop'
  | 'alerts'
  | 'incidents'
  | 'kill-switch'
  | 'reconciliation'
  | 'audit'
  | 'backtest';

// -------------------------------------------------------------
// Portfolio Types
// -------------------------------------------------------------
export interface PositionItem {
  symbol: string;
  side: 'LONG' | 'SHORT' | string;
  quantity: number;
  average_entry: number;
  average_entry_price?: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  weight: number;
  weight_pct?: number;
  exposure: number;
}

export interface PortfolioSummary {
  equity: number;
  total_value?: number;
  cash: number;
  invested_value: number;
  buying_power: number;
  unrealized_pnl: number;
  realized_pnl: number;
  daily_pnl: number;
  gross_exposure: number;
  net_exposure: number;
  long_exposure?: number;
  short_exposure?: number;
  leverage: number;
  positions_count: number;
  open_orders_count: number;
  currency: string;
  positions?: PositionItem[];
}

export interface PortfolioData {
  summary: PortfolioSummary;
  positions: PositionItem[];
  allocations_by_symbol: Record<string, number>;
  equity_history: Array<{ timestamp: string; equity: number }>;
  last_updated: string;
}

// -------------------------------------------------------------
// Overview Types
// -------------------------------------------------------------
export interface SystemOverview {
  system_name: string;
  execution_mode: 'PAPER' | 'LIVE' | string;
  broker_provider: string;
  system_status: 'HEALTHY' | 'DEGRADED' | 'WARNING' | 'CRITICAL' | 'UNKNOWN' | string;
  kill_switch_state: 'ARMED' | 'TRIGGERED' | 'CLEARING' | string;
  autonomous_loop_state: 'RUNNING' | 'PAUSED' | 'STOPPED' | 'READY' | string;
  broker_status: string;
  reconciliation_status: string;
  portfolio: PortfolioSummary;
  recent_alerts: AlertItem[];
  recent_incidents: IncidentItem[];
  last_updated: string;
  active_incidents_count: number;
  total_cycles_completed: number;
}
export type OverviewData = SystemOverview;

// -------------------------------------------------------------
// Health Types
// -------------------------------------------------------------
export interface ComponentHealth {
  component: string;
  status: string;
  timestamp: string;
  message: string;
  latency_ms: number | null;
  details: Record<string, any>;
  version: string;
  cycle_id: string | null;
}

export interface HeartbeatItem {
  component: string;
  timestamp: string;
  interval_seconds: number;
  timeout_seconds: number;
  status: string;
  metadata: Record<string, any>;
}

export interface HealthData {
  system_status: string;
  summary: string;
  is_trading_permitted: boolean;
  active_incident_count: number;
  kill_switch_triggered: boolean;
  timestamp: string;
  components: Record<string, ComponentHealth>;
  heartbeats: Record<string, HeartbeatItem>;
}

// -------------------------------------------------------------
// Market Data Types
// -------------------------------------------------------------
export interface MarketSymbol {
  symbol: string;
  latest_close: number;
  latest_timestamp: string | null;
  freshness_status: string;
  volume_24h: number;
  change_pct_24h: number;
  is_stale: boolean;
}

export interface MarketDataResponse {
  market_status: string;
  data_provider: string;
  total_symbols_monitored: number;
  symbols: MarketSymbol[];
  missing_symbols: string[];
  stale_symbols: string[];
  last_updated: string;
}

// -------------------------------------------------------------
// Models Types
// -------------------------------------------------------------
export interface ModelInfo {
  model_name: string;
  model_type: string;
  version: string;
  status: string;
  weight: number;
  avg_inference_latency_ms: number;
  is_available: boolean;
}

export interface ModelsData {
  active_models: ModelInfo[];
  regime_model: {
    current_regime: string;
    regime_detector: string;
    confidence: number;
  };
  total_models_available: number;
  ensemble_status: string;
  last_inference_timestamp: string | null;
}

// -------------------------------------------------------------
// Signals Types
// -------------------------------------------------------------
export interface SignalItem {
  symbol: string;
  direction: 'LONG' | 'SHORT' | 'FLAT' | 'BUY' | 'SELL' | 'HOLD' | string;
  signal_strength: number;
  confidence: number;
  model_agreement: number;
  regime: string;
  forecast_horizon?: string;
  forecast_horizon_days?: number;
  reason_codes: string[];
  timestamp: string;
  version: string;
}

export interface SignalSummary {
  signals: SignalItem[];
  total_signals: number;
  bullish_count: number;
  bearish_count: number;
  flat_count: number;
  last_updated: string;
}
export type SignalsData = SignalSummary;

// -------------------------------------------------------------
// Risk Types
// -------------------------------------------------------------
export interface RiskMetric {
  name: string;
  current_value: number;
  limit_value: number;
  status: 'PASS' | 'WARNING' | 'VIOLATION' | string;
  unit: string;
}

export interface RiskSummary {
  status: string;
  risk_status?: string;
  passed: boolean;
  gross_exposure: number;
  net_exposure: number;
  long_exposure: number;
  short_exposure: number;
  leverage: number;
  max_position_weight: number;
  active_positions_count: number;
  volatility_status: string;
  correlation_status: string;
  violations: string[];
  active_violations?: string[];
  warnings: string[];
  hard_limit_breaches?: string[];
  risk_metrics?: RiskMetric[];
  is_risk_approved?: boolean;
  last_updated?: string;
}
export type RiskData = RiskSummary;

// -------------------------------------------------------------
// Orders Types
// -------------------------------------------------------------
export interface OrderItem {
  client_order_id: string;
  broker_order_id: string | null;
  symbol: string;
  side: 'BUY' | 'SELL' | 'LONG' | 'SHORT' | string;
  order_type: string;
  quantity: number;
  filled_quantity: number;
  remaining_quantity?: number;
  limit_price?: number | null;
  stop_price?: number | null;
  average_fill_price?: number | null;
  time_in_force?: string;
  status: string;
  submitted_at?: string | null;
  filled_at?: string | null;
  execution_mode: string;
}

export interface OrderSummary {
  orders: OrderItem[];
  total_orders: number;
  open_orders_count?: number;
  filled_orders_count?: number;
  cancelled_orders_count?: number;
}
export type OrdersData = OrderSummary;

// -------------------------------------------------------------
// Executions Types
// -------------------------------------------------------------
export interface ExecutionItem {
  execution_id: string;
  broker_order_id?: string;
  order_id: string;
  client_order_id?: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  execution_price?: number;
  timestamp: string;
  executed_at?: string;
  commission: number;
  slippage: number;
  venue?: string;
  execution_mode: string;
}

export interface ExecutionSummary {
  executions: ExecutionItem[];
  total_executions: number;
  total_commission: number;
  total_commissions_paid?: number;
  average_slippage_bps: number;
  total_volume_traded?: number;
}
export type ExecutionsData = ExecutionSummary;

// -------------------------------------------------------------
// Broker Types
// -------------------------------------------------------------
export interface BrokerSummary {
  provider: string;
  execution_mode: string;
  connected: boolean;
  connection_status?: string;
  authenticated: boolean;
  authentication_status?: string;
  latency_ms: number;
  api_latency_ms?: number;
  rate_limit_remaining: number;
  rate_limit_headroom_pct?: number;
  capabilities: string[];
  supported_capabilities?: string[];
  account_status: string;
  equity: number;
  cash: number;
  buying_power: number;
  portfolio_value?: number;
  currency?: string;
  is_paper?: boolean;
  last_error?: string | null;
  last_health_check?: string;
}
export type BrokerData = BrokerSummary;

// -------------------------------------------------------------
// Autonomous Loop Types
// -------------------------------------------------------------
export interface AutonomousLoopSummary {
  state: string;
  session_id?: string;
  cycle_id: string | null;
  current_cycle_id?: string | null;
  cycles_completed: number;
  total_cycles?: number;
  successful_cycles?: number;
  failed_cycles: number;
  retry_count: number;
  consecutive_failures?: number;
  cycle_duration_sec: number;
  uptime_seconds?: number;
  checkpoint_age_sec: number;
  recovery_state: string;
  last_successful_stage: string;
  recent_events: Array<{ message?: string; timestamp?: string; [key: string]: any }>;
  pipeline_stages?: Array<{ stage: string; status: string; duration_ms: number }>;
}
export type AutonomousLoopData = AutonomousLoopSummary;

// -------------------------------------------------------------
// Alerts & Incidents Types
// -------------------------------------------------------------
export interface AlertItem {
  alert_id?: string;
  event_id?: string;
  timestamp: string;
  severity: 'INFO' | 'WARNING' | 'HIGH' | 'CRITICAL' | string;
  component: string;
  event_type?: string;
  message: string;
  reason?: string;
  symbol?: string;
  resolved?: boolean;
}

export interface AlertSummary {
  alerts: AlertItem[];
  total_alerts: number;
  active_alerts: number;
  critical_alerts: number;
  resolved_alerts: number;
}

export interface IncidentItem {
  incident_id: string;
  title?: string;
  description: string;
  severity: 'INFO' | 'WARNING' | 'HIGH' | 'CRITICAL' | string;
  component: string;
  status: 'OPEN' | 'ACKNOWLEDGED' | 'MITIGATING' | 'RESOLVED' | string;
  triggering_condition?: string;
  created_at: string;
  updated_at?: string;
  resolved_at?: string | null;
  resolution_notes?: string | null;
}

export interface IncidentSummary {
  incidents: IncidentItem[];
  total_incidents: number;
  active_incidents: number;
  critical_incidents: number;
  resolved_incidents: number;
}

// -------------------------------------------------------------
// Kill Switch Types
// -------------------------------------------------------------
export interface KillSwitchSummary {
  status: string;
  is_triggered: boolean;
  trigger_reason: string | null;
  triggered_by: string | null;
  triggered_at: string | null;
  affected_systems: string[];
  current_safety_state: string;
  active_critical_incidents: number;
  reset_permitted: boolean;
  can_reset?: boolean;
}
export interface KillSwitchData {
  kill_switch: {
    state: string;
    is_triggered: boolean;
    triggered_at: string | null;
    triggered_by: string | null;
    reason: string | null;
    cleared_at: string | null;
    cleared_by: string | null;
  };
  can_reset: boolean;
}

// -------------------------------------------------------------
// Reconciliation Types
// -------------------------------------------------------------
export interface ReconciliationSummary {
  status: 'CLEAN' | 'MATCHED' | 'MISMATCH' | string;
  last_reconciliation_time: string;
  last_reconciled_at?: string;
  local_positions: number;
  broker_positions: number;
  local_positions_count?: number;
  broker_positions_count?: number;
  local_orders: number;
  broker_orders: number;
  local_orders_count?: number;
  broker_orders_count?: number;
  cash_difference: number;
  position_differences: any[];
  position_mismatches?: any[];
  order_differences: any[];
  order_mismatches?: any[];
  unknown_external_orders: number;
  external_orders_detected?: any[];
  unresolved_discrepancies: number;
}
export type ReconciliationData = ReconciliationSummary;

// -------------------------------------------------------------
// Audit Types
// -------------------------------------------------------------
export interface AuditEventItem {
  event_id: string;
  timestamp: string;
  event_type: string;
  source: string;
  actor: string;
  severity: string;
  summary: string;
  message?: string;
  cycle_id?: string | null;
  order_id?: string | null;
  metadata?: Record<string, any>;
}

export interface AuditSummary {
  events: AuditEventItem[];
  total_events: number;
}
export interface AuditData {
  audit_events: AuditEventItem[];
  count: number;
}

// -------------------------------------------------------------
// Backtest Types
// -------------------------------------------------------------
export interface BacktestRun {
  run_id: string;
  dataset: string;
  dataset_provenance: string;
  strategy: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_equity: number;
  total_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  total_trades: number;
  turnover: number;
}

export interface BacktestSummary {
  runs: BacktestRun[];
  total_runs: number;
}

export interface BacktestData {
  run_id: string;
  dataset: string;
  bars_count: number;
  initial_capital: number;
  final_equity: number;
  total_pnl: number;
  total_return_pct: number;
  cagr_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  calmar_ratio: number;
  total_trades: number;
  win_rate_pct: number;
  profit_factor: number;
  benchmark_spy_return_pct: number;
  alpha_annualized: number;
  beta_to_benchmark: number;
  provenance_hash: string;
}
