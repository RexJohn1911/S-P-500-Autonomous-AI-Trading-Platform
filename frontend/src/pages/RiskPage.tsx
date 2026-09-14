import React, { useEffect, useState } from 'react';
import { getRisk } from '../services/api';
import type { RiskSummary } from '../types';
import MetricCard from '../components/MetricCard';
import { ShieldCheck, ShieldAlert, AlertTriangle, Activity } from 'lucide-react';

const RiskPage: React.FC = () => {
  const [data, setData] = useState<RiskSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRisk()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Risk Engine telemetry...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load risk telemetry: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No risk telemetry available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Risk Engine & Safety Constraints</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Real-time risk compliance, exposure ceilings, and volatility telemetry</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard
          title="Risk Engine Status"
          value={data.status}
          subtitle={`Check Passed: ${data.passed ? 'YES' : 'NO'}`}
          icon={data.passed ? ShieldCheck : ShieldAlert}
        />
        <MetricCard
          title="Active Violations"
          value={data.violations.length}
          subtitle="Hard constraint breaches"
          icon={AlertTriangle}
          trend={data.violations.length > 0 ? 'down' : 'neutral'}
        />
        <MetricCard
          title="Max Position Weight"
          value={`${(data.max_position_weight * 100).toFixed(1)}%`}
          subtitle="Concentration threshold"
          icon={Activity}
        />
        <MetricCard
          title="Market Volatility Status"
          value={data.volatility_status}
          subtitle={`Correlation: ${data.correlation_status}`}
          icon={ShieldCheck}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="glass-card p-5">
          <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Hard Limit Constraints & Exposure</h2>
          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Gross Exposure:</span>
              <span className="text-slate-900 font-bold">{(data.gross_exposure * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Net Exposure:</span>
              <span className="text-slate-900 font-bold">{(data.net_exposure * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Long Exposure:</span>
              <span className="text-emerald-600 font-bold">{(data.long_exposure * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Short Exposure:</span>
              <span className="text-rose-600 font-bold">{(data.short_exposure * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Leverage:</span>
              <span className="text-indigo-600 font-bold">{data.leverage.toFixed(2)}x</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500 font-sans font-medium">Active Positions Count:</span>
              <span className="text-slate-900 font-bold">{data.active_positions_count}</span>
            </div>
          </div>
        </div>

        <div className="glass-card p-5">
          <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Violations & Risk Warnings</h2>
          {data.violations.length === 0 && data.warnings.length === 0 ? (
            <div className="py-12 flex flex-col items-center justify-center text-center">
              <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600 mb-2">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <p className="text-sm font-semibold text-emerald-700">All Risk Bounds Respected</p>
              <p className="text-xs text-slate-500 mt-1">No limit breaches or warning thresholds violated</p>
            </div>
          ) : (
            <div className="space-y-2.5">
              {data.violations.map((violation: string, idx: number) => (
                <div key={idx} className="p-3 bg-rose-50 border border-rose-200 rounded-lg flex items-start space-x-2.5">
                  <ShieldAlert className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-[11px] font-bold text-rose-700 uppercase tracking-wider">HARD LIMIT VIOLATION</span>
                    <p className="text-xs text-rose-900 mt-0.5 font-mono">{violation}</p>
                  </div>
                </div>
              ))}
              {data.warnings.map((warning: string, idx: number) => (
                <div key={idx} className="p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-start space-x-2.5">
                  <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-[11px] font-bold text-amber-700 uppercase tracking-wider">RISK WARNING</span>
                    <p className="text-xs text-amber-900 mt-0.5 font-mono">{warning}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RiskPage;
