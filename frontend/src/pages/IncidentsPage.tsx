import React, { useEffect, useState } from 'react';
import { getIncidents } from '../services/api';
import type { IncidentSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { ShieldAlert, AlertOctagon, CheckCircle2, History } from 'lucide-react';

const IncidentsPage: React.FC = () => {
  const [data, setData] = useState<IncidentSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getIncidents()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Incidents Log...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load incidents: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No incidents data available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Incidents & Root Cause Analysis</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Track high-severity system incidents, triggering conditions, and recovery timestamps</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard title="Total Incidents" value={data.total_incidents} subtitle="Logged events" icon={History} />
        <MetricCard title="Active Incidents" value={data.active_incidents} subtitle="Unresolved" icon={ShieldAlert} trend={data.active_incidents > 0 ? 'down' : 'neutral'} />
        <MetricCard title="Critical Incidents" value={data.critical_incidents} subtitle="High impact" icon={AlertOctagon} trend={data.critical_incidents > 0 ? 'down' : 'neutral'} />
        <MetricCard title="Resolved Incidents" value={data.resolved_incidents} subtitle="Recovered systems" icon={CheckCircle2} />
      </div>

      <div className="glass-card p-5">
        <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">System Incidents</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Incident ID</th>
                <th className="py-2.5 px-3">Severity</th>
                <th className="py-2.5 px-3">Component</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Description</th>
                <th className="py-2.5 px-3">Triggering Condition</th>
                <th className="py-2.5 px-3">Created At</th>
                <th className="py-2.5 px-3">Resolved At</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {data.incidents.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-slate-400 font-sans">
                    No recorded system incidents.
                  </td>
                </tr>
              ) : (
                data.incidents.map((inc) => (
                  <tr key={inc.incident_id} className="hover:bg-indigo-50/40 transition-colors">
                    <td className="py-2 px-3 text-slate-500">{inc.incident_id.slice(0, 12)}...</td>
                    <td className="py-2 px-3"><StatusBadge status={inc.severity} size="sm" /></td>
                    <td className="py-2 px-3 font-bold text-slate-900 font-sans">{inc.component}</td>
                    <td className="py-2 px-3"><StatusBadge status={inc.status} size="sm" /></td>
                    <td className="py-2 px-3 text-slate-800 font-sans">{inc.description}</td>
                    <td className="py-2 px-3 text-slate-500 font-sans">{inc.triggering_condition || 'N/A'}</td>
                    <td className="py-2 px-3 text-slate-500">{new Date(inc.created_at).toLocaleTimeString()}</td>
                    <td className="py-2 px-3 text-slate-500">{inc.resolved_at ? new Date(inc.resolved_at).toLocaleTimeString() : 'N/A'}</td>
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

export default IncidentsPage;
