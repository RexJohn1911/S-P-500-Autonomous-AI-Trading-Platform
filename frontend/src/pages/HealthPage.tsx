import React, { useEffect, useState } from 'react';
import { Activity, Heart, Shield, Zap } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import StatusBadge from '../components/StatusBadge';
import type { HealthData } from '../types';
import { getHealth } from '../services/api';

interface HealthPageProps {
  data?: HealthData | null;
}

export const HealthPage: React.FC<HealthPageProps> = ({ data: propData }) => {
  const [data, setData] = useState<HealthData | null>(propData || null);
  const [loading, setLoading] = useState(!propData);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!propData) {
      getHealth()
        .then(setData)
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false));
    }
  }, [propData]);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading health diagnostics...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load health: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No health data available.</div>;

  const componentsList = Object.values(data.components);
  const heartbeatsList = Object.values(data.heartbeats);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">System Health & Watchdog Engine</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Continuous 13-component diagnostics, heartbeat freshness, and pre-flight safety gates</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
        <MetricCard
          title="Overall System Status"
          value={data.system_status}
          subtitle={data.summary}
          badge={data.system_status}
          icon={Activity}
        />
        <MetricCard
          title="Order Dispatch Permission"
          value={data.is_trading_permitted ? 'PERMITTED' : 'BLOCKED'}
          subtitle="Pre-flight gates evaluation"
          badge={data.is_trading_permitted ? 'PASS' : 'CRITICAL'}
          icon={Shield}
        />
        <MetricCard
          title="Active Incidents"
          value={data.active_incident_count}
          subtitle="Subsystem warnings & errors"
          badge={data.active_incident_count === 0 ? 'HEALTHY' : 'WARNING'}
          icon={Zap}
        />
      </div>

      {/* Component Health Table */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">
          Subsystem Component Diagnostics ({componentsList.length} Components)
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Component</th>
                <th className="py-2.5 px-3">Health Status</th>
                <th className="py-2.5 px-3">Diagnostic Message</th>
                <th className="py-2.5 px-3">Latency</th>
                <th className="py-2.5 px-3">Last Updated</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {componentsList.map((comp) => (
                <tr key={comp.component} className="hover:bg-indigo-50/40 transition-colors">
                  <td className="py-2 px-3 font-bold text-slate-900 font-sans">{comp.component}</td>
                  <td className="py-2 px-3"><StatusBadge status={comp.status} size="sm" /></td>
                  <td className="py-2 px-3 text-slate-600 font-sans">{comp.message}</td>
                  <td className="py-2 px-3 text-indigo-600 font-semibold">
                    {comp.latency_ms !== null ? `${comp.latency_ms.toFixed(1)} ms` : '—'}
                  </td>
                  <td className="py-2 px-3 text-slate-500">{new Date(comp.timestamp).toLocaleTimeString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Heartbeat Monitors */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100 flex items-center space-x-2">
          <Heart className="w-4 h-4 text-rose-500" />
          <span>Periodic Heartbeat Monitors</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {heartbeatsList.map((hb) => (
            <div key={hb.component} className="bg-white/80 border border-slate-200/80 rounded-lg p-3 font-mono text-xs shadow-2xs">
              <div className="flex justify-between items-center mb-1.5">
                <span className="font-bold text-slate-900 font-sans">{hb.component}</span>
                <StatusBadge status={hb.status} size="sm" />
              </div>
              <div className="text-[11px] text-slate-500 space-y-0.5">
                <div>Interval: {hb.interval_seconds}s | Timeout: {hb.timeout_seconds}s</div>
                <div>Last Heartbeat: {new Date(hb.timestamp).toLocaleTimeString()}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HealthPage;
