import React, { useEffect, useState } from 'react';
import { Activity, AlertTriangle, Bot, DollarSign, ShieldCheck, TrendingUp } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import StatusBadge from '../components/StatusBadge';
import type { OverviewData } from '../types';
import { getOverview } from '../services/api';

interface OverviewPageProps {
  data?: OverviewData | null;
  onNavigate?: (page: any) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ data: propData, onNavigate }) => {
  const [data, setData] = useState<OverviewData | null>(propData || null);
  const [loading, setLoading] = useState(!propData);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!propData) {
      getOverview()
        .then(setData)
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false));
    }
  }, [propData]);

  if (loading) return <div className="p-6 text-slate-500 font-mono text-xs">Loading System Overview...</div>;
  if (error) return <div className="p-6 text-rose-600 font-mono text-xs">Failed to load overview: {error}</div>;
  if (!data) return <div className="p-6 text-slate-500 font-mono text-xs">No overview data available.</div>;

  const p = data.portfolio;

  return (
    <div className="space-y-3.5">
      <div>
        <h1 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight">System Operations Center</h1>
        <p className="text-[11.5px] text-slate-500 mt-0.5">Real-time holistic observability over autonomous execution and safety gates</p>
      </div>

      {/* Top Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          title="Portfolio Equity"
          value={`$${p.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          subtitle={`Cash: $${p.cash.toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
          icon={DollarSign}
          trend="up"
          trendValue={`+$${p.daily_pnl.toFixed(2)} Today`}
        />
        <MetricCard
          title="Gross Exposure"
          value={`${(p.gross_exposure * 100).toFixed(1)}%`}
          subtitle={`Net: ${(p.net_exposure * 100).toFixed(1)}% | Lev: ${p.leverage.toFixed(2)}x`}
          icon={TrendingUp}
          badge="HEALTHY"
        />
        <MetricCard
          title="Autonomous Loop"
          value={data.autonomous_loop_state}
          subtitle={`Cycles: ${data.total_cycles_completed}`}
          icon={Bot}
          badge={data.autonomous_loop_state}
        />
        <MetricCard
          title="System Health"
          value={data.system_status}
          subtitle={`Incidents: ${data.active_incidents_count}`}
          icon={Activity}
          badge={data.system_status}
        />
      </div>

      {/* Primary Panels Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3.5">
        {/* Core Operational Status */}
        <div className="glass-card p-4">
          <div className="flex justify-between items-center mb-2.5 pb-2 border-b border-slate-100">
            <h3 className="text-xs font-bold text-slate-900 flex items-center space-x-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-600" />
              <span>Subsystem Readiness</span>
            </h3>
            {onNavigate && (
              <button
                onClick={() => onNavigate('health')}
                className="text-[11px] text-indigo-600 hover:text-indigo-800 font-semibold transition-colors"
              >
                View Detailed Health →
              </button>
            )}
          </div>

          <div className="space-y-2 font-mono text-[11.5px]">
            <div className="flex justify-between items-center py-1 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Execution Mode</span>
              <StatusBadge status={data.execution_mode} size="sm" />
            </div>
            <div className="flex justify-between items-center py-1 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Broker Provider & Connectivity</span>
              <div className="flex items-center space-x-1.5">
                <span className="text-slate-800 font-bold">{data.broker_provider}</span>
                <StatusBadge status={data.broker_status} size="sm" />
              </div>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Global Kill Switch</span>
              <StatusBadge status={data.kill_switch_state} size="sm" />
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-slate-500 font-sans font-medium">Ledger Reconciliation</span>
              <StatusBadge status={data.reconciliation_status} size="sm" />
            </div>
          </div>
        </div>

        {/* Active Safety Alerts & Incidents */}
        <div className="glass-card p-4">
          <div className="flex justify-between items-center mb-2.5 pb-2 border-b border-slate-100">
            <h3 className="text-xs font-bold text-slate-900 flex items-center space-x-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
              <span>Recent Safety Telemetry</span>
            </h3>
            {onNavigate && (
              <button
                onClick={() => onNavigate('alerts')}
                className="text-[11px] text-indigo-600 hover:text-indigo-800 font-semibold transition-colors"
              >
                All Alerts →
              </button>
            )}
          </div>

          {data.recent_alerts.length === 0 ? (
            <div className="text-slate-400 text-xs py-6 text-center font-sans">
              No active safety warnings. System operating nominally.
            </div>
          ) : (
            <div className="space-y-1.5">
              {data.recent_alerts.map((alt, idx) => (
                <div
                  key={alt.event_id || alt.alert_id || idx}
                  className="bg-white/85 border border-slate-200/80 rounded-lg p-2 flex justify-between items-center shadow-2xs"
                >
                  <div>
                    <div className="text-[11.5px] font-semibold text-slate-900">{alt.message}</div>
                    <div className="text-[9.5px] text-slate-500 mt-0.5 font-mono">
                      {alt.component} • {new Date(alt.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  <StatusBadge status={alt.severity} size="sm" />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OverviewPage;
