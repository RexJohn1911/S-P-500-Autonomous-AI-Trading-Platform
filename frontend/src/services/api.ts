/**
 * Dashboard API Client Service (Phase 19).
 * Exclusively queries the Backend Dashboard API at /api/dashboard.
 * Zero direct broker or third-party connections.
 */

import type {
  AlertSummary,
  AutonomousLoopSummary,
  AuditSummary,
  BacktestSummary,
  BrokerSummary,
  ExecutionSummary,
  HealthData,
  IncidentSummary,
  KillSwitchSummary,
  MarketDataResponse,
  ModelsData,
  OrderSummary,
  PortfolioSummary,
  ReconciliationSummary,
  RiskSummary,
  SignalSummary,
  SystemOverview,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/dashboard';

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      const errorBody = await res.text();
      throw new Error(`API error ${res.status}: ${errorBody || res.statusText}`);
    }

    return await res.json();
  } catch (err: any) {
    console.error(`Failed to fetch ${url}:`, err);
    throw err;
  }
}

export async function getOverview(): Promise<SystemOverview> {
  return fetchJson<SystemOverview>('/overview');
}

export async function getHealth(): Promise<HealthData> {
  return fetchJson<HealthData>('/health');
}

export async function getMarketData(): Promise<MarketDataResponse> {
  return fetchJson<MarketDataResponse>('/market-data');
}

export async function getModels(): Promise<ModelsData> {
  return fetchJson<ModelsData>('/models');
}

export async function getSignals(): Promise<SignalSummary> {
  return fetchJson<SignalSummary>('/signals');
}

export async function getPortfolio(): Promise<PortfolioSummary> {
  const data = await fetchJson<any>('/portfolio');
  const summary = data.summary || data;
  return {
    ...summary,
    equity: summary.equity ?? 0.0,
    cash: summary.cash ?? 0.0,
    invested_value: summary.invested_value ?? 0.0,
    buying_power: summary.buying_power ?? 0.0,
    unrealized_pnl: summary.unrealized_pnl ?? 0.0,
    realized_pnl: summary.realized_pnl ?? 0.0,
    daily_pnl: summary.daily_pnl ?? 0.0,
    gross_exposure: summary.gross_exposure ?? 0.0,
    net_exposure: summary.net_exposure ?? 0.0,
    leverage: summary.leverage ?? 0.0,
    positions_count: summary.positions_count ?? (data.positions?.length || 0),
    open_orders_count: summary.open_orders_count ?? 0,
    total_value: summary.equity ?? 0.0,
    long_exposure: summary.gross_exposure ?? 0.0,
    short_exposure: 0.0,
    positions: data.positions || [],
  };
}

export async function getRisk(): Promise<RiskSummary> {
  const data = await fetchJson<any>('/risk');
  return {
    status: data.risk_status ?? 'UNINITIALIZED',
    risk_status: data.risk_status ?? 'UNINITIALIZED',
    passed: data.is_risk_approved ?? false,
    gross_exposure: data.gross_exposure ?? 0.0,
    net_exposure: data.net_exposure ?? 0.0,
    long_exposure: data.gross_exposure ?? 0.0,
    short_exposure: 0.0,
    leverage: data.leverage ?? 0.0,
    max_position_weight: data.max_position_weight ?? 0.0,
    active_positions_count: data.active_positions_count ?? 0,
    volatility_status: data.risk_status === 'HEALTHY' ? 'NORMAL' : (data.risk_status || 'UNINITIALIZED'),
    correlation_status: data.risk_status === 'HEALTHY' ? 'STABLE' : (data.risk_status || 'UNINITIALIZED'),
    violations: data.active_violations || [],
    warnings: data.warnings || [],
    hard_limit_breaches: data.hard_limit_breaches || [],
    risk_metrics: data.risk_metrics || [],
    is_risk_approved: data.is_risk_approved ?? false,
  };
}

export async function getOrders(params?: { symbol?: string; status?: string }): Promise<OrderSummary> {
  const query = new URLSearchParams();
  if (params?.symbol) query.append('symbol', params.symbol);
  if (params?.status) query.append('status', params.status);
  const qStr = query.toString() ? `?${query.toString()}` : '';
  const data = await fetchJson<any>(`/orders${qStr}`);
  return {
    orders: data.orders || [],
    total_orders: data.total_orders ?? (data.orders?.length || 0),
    open_orders_count: data.open_orders_count ?? 0,
    filled_orders_count: data.filled_orders_count ?? 0,
    cancelled_orders_count: data.cancelled_orders_count ?? 0,
  };
}

export async function getExecutions(): Promise<ExecutionSummary> {
  const data = await fetchJson<any>('/executions');
  const rawExecs = data.executions || [];
  const execs = rawExecs.map((e: any) => ({
    execution_id: e.execution_id,
    broker_order_id: e.broker_order_id,
    order_id: e.broker_order_id || e.client_order_id || 'N/A',
    client_order_id: e.client_order_id,
    symbol: e.symbol,
    side: e.side,
    quantity: e.quantity ?? 0.0,
    price: e.execution_price ?? e.price ?? 0.0,
    timestamp: e.executed_at || e.timestamp || '',
    commission: e.commission ?? 0.0,
    slippage: e.slippage ?? 0.0,
    venue: e.venue || 'UNKNOWN',
    execution_mode: e.execution_mode || 'PAPER',
  }));
  return {
    executions: execs,
    total_executions: data.total_executions ?? execs.length,
    total_commission: data.total_commissions_paid ?? 0.0,
    average_slippage_bps: data.total_executions ? (data.average_slippage_bps ?? 0.0) : 0.0,
  };
}

export async function getBroker(): Promise<BrokerSummary> {
  const data = await fetchJson<any>('/broker');
  return {
    provider: data.provider || 'UNKNOWN',
    execution_mode: data.execution_mode || 'PAPER',
    connected: data.connection_status === 'CONNECTED',
    connection_status: data.connection_status || 'DISCONNECTED',
    authenticated: data.authentication_status === 'AUTHENTICATED',
    authentication_status: data.authentication_status || 'UNAUTHENTICATED',
    latency_ms: data.api_latency_ms ?? 0.0,
    rate_limit_remaining: data.rate_limit_headroom_pct != null ? Math.round((data.rate_limit_headroom_pct / 100) * 200) : 0,
    capabilities: data.supported_capabilities || [],
    account_status: data.account_status || 'UNINITIALIZED',
    equity: data.portfolio_value ?? 0.0,
    cash: data.cash ?? 0.0,
    buying_power: data.buying_power ?? 0.0,
    last_error: data.last_error || null,
  };
}

export async function getAutonomousLoop(): Promise<AutonomousLoopSummary> {
  const data = await fetchJson<any>('/autonomous-loop');
  const stages = data.pipeline_stages || [];
  return {
    state: data.state || 'STANDBY',
    cycle_id: data.session_id || 'NONE',
    cycles_completed: data.total_cycles ?? 0,
    failed_cycles: data.failed_cycles ?? 0,
    retry_count: data.consecutive_failures ?? 0,
    cycle_duration_sec: data.uptime_seconds ?? 0.0,
    checkpoint_age_sec: 0.0,
    recovery_state: 'IDLE',
    last_successful_stage: stages.length ? stages[stages.length - 1].stage : 'N/A',
    recent_events: data.recent_events || [],
    pipeline_stages: stages,
  };
}

export async function getAlerts(): Promise<AlertSummary> {
  const data = await fetchJson<any>('/alerts');
  const alerts = (data.alerts || []).map((a: any) => ({
    alert_id: a.event_id || a.alert_id || 'alt_000',
    timestamp: a.timestamp,
    severity: a.severity || 'INFO',
    component: a.component || 'SYSTEM',
    message: a.message || a.reason || 'Notice',
    resolved: false,
  }));
  return {
    alerts,
    total_alerts: alerts.length,
    active_alerts: alerts.filter((a: any) => !a.resolved).length,
    critical_alerts: alerts.filter((a: any) => a.severity === 'CRITICAL').length,
    resolved_alerts: alerts.filter((a: any) => a.resolved).length,
  };
}

export async function getIncidents(): Promise<IncidentSummary> {
  const data = await fetchJson<any>('/incidents');
  const incidents = (data.incidents || []).map((i: any) => ({
    incident_id: i.incident_id,
    severity: i.severity,
    component: i.component,
    status: i.status,
    description: i.description,
    triggering_condition: i.triggering_condition || i.reason || 'Condition breach',
    created_at: i.created_at,
    resolved_at: i.resolved_at,
  }));
  return {
    incidents,
    total_incidents: incidents.length,
    active_incidents: incidents.filter((i: any) => i.status !== 'RESOLVED').length,
    critical_incidents: incidents.filter((i: any) => i.severity === 'CRITICAL').length,
    resolved_incidents: incidents.filter((i: any) => i.status === 'RESOLVED').length,
  };
}

export async function getKillSwitch(): Promise<KillSwitchSummary> {
  const data = await fetchJson<any>('/kill-switch');
  const ks = data.kill_switch || {};
  return {
    status: ks.state || (ks.is_triggered ? 'TRIGGERED' : 'ARMED'),
    is_triggered: ks.is_triggered || false,
    trigger_reason: ks.reason,
    triggered_by: ks.triggered_by,
    triggered_at: ks.triggered_at,
    affected_systems: ks.is_triggered ? ['AUTONOMOUS_LOOP', 'ORDER_EXECUTION', 'BROKER_GATEWAY'] : [],
    current_safety_state: ks.is_triggered ? 'EMERGENCY_HALT' : 'FAIL_CLOSED_PROTECTED',
    active_critical_incidents: 0,
    reset_permitted: data.can_reset ?? true,
  };
}

export async function triggerKillSwitch(reason: string, operatorName: string = 'WEB_OPERATOR'): Promise<any> {
  return fetchJson('/kill-switch/trigger', {
    method: 'POST',
    body: JSON.stringify({ reason, operator_name: operatorName }),
  });
}

export async function resetKillSwitch(operatorName: string = 'WEB_OPERATOR'): Promise<any> {
  return fetchJson('/kill-switch/reset', {
    method: 'POST',
    body: JSON.stringify({ operator_name: operatorName }),
  });
}

export async function getReconciliation(): Promise<ReconciliationSummary> {
  const data = await fetchJson<any>('/reconciliation');
  return {
    status: data.status || 'UNINITIALIZED',
    last_reconciliation_time: data.last_reconciled_at || 'N/A',
    local_positions: data.local_positions_count ?? 0,
    broker_positions: data.broker_positions_count ?? 0,
    local_orders: data.local_orders_count ?? 0,
    broker_orders: data.broker_orders_count ?? 0,
    cash_difference: data.cash_difference ?? 0.0,
    position_differences: data.position_mismatches || [],
    order_differences: data.order_mismatches || [],
    unknown_external_orders: (data.external_orders_detected || []).length,
    unresolved_discrepancies: (data.position_mismatches || []).length + (data.order_mismatches || []).length,
  };
}

export async function getAudit(limit: number = 100): Promise<AuditSummary> {
  const data = await fetchJson<any>(`/audit?limit=${limit}`);
  const rawEvents = data.audit_events || [];
  const events = rawEvents.map((e: any) => ({
    event_id: e.event_id || 'evt_000',
    timestamp: e.timestamp,
    event_type: e.event_type || 'SYSTEM_EVENT',
    source: e.component || e.source || 'BACKEND',
    actor: e.actor || 'SYSTEM',
    severity: e.severity || 'INFO',
    summary: e.message || e.summary || 'Operational transition event',
  }));
  return {
    events,
    total_events: data.count || events.length,
  };
}

export async function getBacktest(): Promise<BacktestSummary> {
  const data = await fetchJson<any>('/backtest');
  if (!data || !data.run_id || data.run_id === 'NONE') {
    return {
      total_runs: 0,
      runs: [],
    };
  }
  return {
    total_runs: 1,
    runs: [
      {
        run_id: data.run_id,
        dataset: data.dataset || 'N/A',
        dataset_provenance: 'Local Data Store (Validated)',
        strategy: 'Multi-Model AI Alpha Ensemble',
        start_date: data.start_date || '2023-01-02',
        end_date: data.end_date || '2025-01-01',
        initial_capital: data.initial_capital ?? 100000.0,
        final_equity: data.final_equity ?? 100000.0,
        total_return: (data.total_return_pct ?? 0.0) / 100,
        sharpe_ratio: data.sharpe_ratio ?? 0.0,
        sortino_ratio: data.sortino_ratio ?? 0.0,
        max_drawdown: (data.max_drawdown_pct ?? 0.0) / 100,
        total_trades: data.total_trades ?? 0,
        turnover: data.turnover ?? 0.0,
      },
    ],
  };
}
