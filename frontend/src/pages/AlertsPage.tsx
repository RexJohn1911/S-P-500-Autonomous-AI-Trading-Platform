import React, { useEffect, useState } from 'react';
import { getAlerts } from '../services/api';
import type { AlertSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { Bell, AlertTriangle, AlertOctagon, CheckCircle2 } from 'lucide-react';

const AlertsPage: React.FC = () => {
  const [data, setData] = useState<AlertSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');

  useEffect(() => {
    getAlerts()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Monitoring Alerts...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load alerts: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No alert telemetry available.</div>;

  const filteredAlerts = data.alerts.filter((a) => {
    if (severityFilter === 'ALL') return true;
    return a.severity.toUpperCase() === severityFilter;
  });

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Monitoring & Safety Alerts</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Phase 18 aggregated telemetry alerts, threshold breaches, and subsystem health warnings</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard title="Total Alerts" value={data.total_alerts} subtitle="Recorded alerts" icon={Bell} />
        <MetricCard title="Active Alerts" value={data.active_alerts} subtitle="Currently unresolved" icon={AlertTriangle} trend={data.active_alerts > 0 ? 'down' : 'neutral'} />
        <MetricCard title="Critical Alerts" value={data.critical_alerts} subtitle="High severity" icon={AlertOctagon} trend={data.critical_alerts > 0 ? 'down' : 'neutral'} />
        <MetricCard title="Resolved Alerts" value={data.resolved_alerts} subtitle="Cleared issues" icon={CheckCircle2} />
      </div>

      <div className="glass-card p-5">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3.5 pb-2 border-b border-slate-100">
          <h2 className="text-sm font-bold text-slate-900">Alerts Log</h2>
          <div className="flex items-center space-x-1.5">
            <span className="text-[11px] text-slate-500 uppercase font-bold tracking-wider mr-1">Severity:</span>
            {['ALL', 'INFO', 'WARNING', 'CRITICAL'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all ${
                  severityFilter === sev
                    ? 'bg-indigo-600 text-white shadow-xs'
                    : 'bg-white/80 text-slate-600 border border-slate-200/80 hover:bg-slate-100/80 hover:text-slate-900'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Alert ID</th>
                <th className="py-2.5 px-3">Severity</th>
                <th className="py-2.5 px-3">Component</th>
                <th className="py-2.5 px-3">Message</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {filteredAlerts.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-400 font-sans">
                    No alerts matching filter criteria.
                  </td>
                </tr>
              ) : (
                filteredAlerts.map((alert, idx) => (
                  <tr key={alert.alert_id || alert.event_id || idx} className="hover:bg-indigo-50/40 transition-colors">
                    <td className="py-2 px-3 text-slate-500">{(alert.alert_id || alert.event_id || 'ALT-001').slice(0, 12)}...</td>
                    <td className="py-2 px-3"><StatusBadge status={alert.severity} size="sm" /></td>
                    <td className="py-2 px-3 font-bold text-slate-900 font-sans">{alert.component}</td>
                    <td className="py-2 px-3 text-slate-700 font-sans">{alert.message}</td>
                    <td className="py-2 px-3"><StatusBadge status={alert.resolved ? 'RESOLVED' : 'ACTIVE'} size="sm" /></td>
                    <td className="py-2 px-3 text-slate-500">{new Date(alert.timestamp).toLocaleTimeString()}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AlertsPage;
