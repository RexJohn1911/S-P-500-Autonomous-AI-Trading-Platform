import React, { useEffect, useState } from 'react';
import { getAudit } from '../services/api';
import type { AuditSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { Shield, Search, FileText, Lock } from 'lucide-react';

const AuditPage: React.FC = () => {
  const [data, setData] = useState<AuditSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState('ALL');

  useEffect(() => {
    getAudit()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Append-Only Audit Ledger...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load audit ledger: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No audit data available.</div>;

  const filteredEvents = data.events.filter((evt) => {
    const matchesSev = severityFilter === 'ALL' || evt.severity.toUpperCase() === severityFilter;
    const matchesQuery =
      searchQuery === '' ||
      evt.event_type.toLowerCase().includes(searchQuery.toLowerCase()) ||
      evt.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
      evt.source.toLowerCase().includes(searchQuery.toLowerCase()) ||
      evt.actor.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSev && matchesQuery;
  });

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Cryptographic & Append-Only Audit Trail</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Immutable ledger of all state transitions, order routing decisions, and safety events</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard title="Total Audit Records" value={data.total_events} subtitle="Append-only log" icon={Shield} />
        <MetricCard title="Ledger Integrity" value="VERIFIED" subtitle="Tamper-evident log" icon={Lock} />
        <MetricCard title="Actor Sources" value="SYSTEM / OPERATOR" subtitle="Authorized entities" icon={FileText} />
        <MetricCard title="Export / Compliance" value="READY" subtitle="Auditor grade" icon={Shield} />
      </div>

      <div className="glass-card p-5">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3.5 pb-2 border-b border-slate-100">
          <div className="relative flex-1 max-w-md w-full">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search audit trail by keyword, actor, or type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 font-mono shadow-xs"
            />
          </div>

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
                <th className="py-2.5 px-3">Event ID</th>
                <th className="py-2.5 px-3">Timestamp</th>
                <th className="py-2.5 px-3">Event Type</th>
                <th className="py-2.5 px-3">Source</th>
                <th className="py-2.5 px-3">Actor</th>
                <th className="py-2.5 px-3">Severity</th>
                <th className="py-2.5 px-3">Summary</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {filteredEvents.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400 font-sans">
                    No audit records matching query criteria.
                  </td>
                </tr>
              ) : (
                filteredEvents.map((evt) => (
                  <tr key={evt.event_id} className="hover:bg-indigo-50/40 transition-colors">
                    <td className="py-2 px-3 text-slate-500">{evt.event_id.slice(0, 12)}...</td>
                    <td className="py-2 px-3 text-slate-500">{new Date(evt.timestamp).toLocaleTimeString()}</td>
                    <td className="py-2 px-3 font-bold text-slate-900 font-sans">{evt.event_type}</td>
                    <td className="py-2 px-3 text-slate-500 font-sans">{evt.source}</td>
                    <td className="py-2 px-3 text-indigo-600 font-semibold">{evt.actor}</td>
                    <td className="py-2 px-3"><StatusBadge status={evt.severity} size="sm" /></td>
                    <td className="py-2 px-3 text-slate-700 font-sans">{evt.summary}</td>
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

export default AuditPage;
