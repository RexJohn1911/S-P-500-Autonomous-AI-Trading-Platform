import React, { useEffect, useState } from 'react';
import { getReconciliation } from '../services/api';
import type { ReconciliationSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { Scale, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';

const ReconciliationPage: React.FC = () => {
  const [data, setData] = useState<ReconciliationSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getReconciliation()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Broker Reconciliation Engine...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load reconciliation state: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No reconciliation telemetry available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Broker Reconciliation Engine</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Continuous local-state vs broker-reported ledger verification and discrepancy tracking</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard
          title="Reconciliation Status"
          value={data.status}
          subtitle={`Discrepancies: ${data.unresolved_discrepancies}`}
          icon={data.status === 'CLEAN' ? CheckCircle2 : AlertTriangle}
        />
        <MetricCard
          title="Cash Difference"
          value={`$${Math.abs(data.cash_difference).toFixed(2)}`}
          subtitle="Delta between local & broker"
          icon={Scale}
        />
        <MetricCard
          title="Position Counts"
          value={`${data.local_positions} / ${data.broker_positions}`}
          subtitle="Local vs Broker"
          icon={Layers}
        />
        <MetricCard
          title="Order Counts"
          value={`${data.local_orders} / ${data.broker_orders}`}
          subtitle="Local vs Broker"
          icon={Layers}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="glass-card p-5">
          <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Reconciliation Summary & Timestamp</h2>
          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Reconciliation Health:</span>
              <StatusBadge status={data.status} size="sm" />
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Last Reconciled Timestamp:</span>
              <span className="text-slate-900">{data.last_reconciliation_time ? new Date(data.last_reconciliation_time).toLocaleString() : 'N/A'}</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Unknown / External Orders:</span>
              <span className="text-slate-700">{data.unknown_external_orders}</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Position Mismatches:</span>
              <span className="text-slate-700">{data.position_differences.length}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500 font-sans font-medium">Order Mismatches:</span>
              <span className="text-slate-700">{data.order_differences.length}</span>
            </div>
          </div>
        </div>

        <div className="glass-card p-5">
          <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Unresolved Discrepancies</h2>
          {data.position_differences.length === 0 && data.order_differences.length === 0 ? (
            <div className="py-12 flex flex-col items-center justify-center text-center">
              <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600 mb-2">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <p className="text-sm font-semibold text-emerald-700">100% Ledger Parity Confirmed</p>
              <p className="text-xs text-slate-500 mt-1">Zero position, order, or cash mismatches between local state and broker</p>
            </div>
          ) : (
            <div className="space-y-2.5">
              {data.position_differences.map((diff: any, idx: number) => (
                <div key={idx} className="p-2.5 bg-amber-50 border border-amber-200 rounded-lg text-xs font-mono text-amber-900 shadow-2xs">
                  {JSON.stringify(diff)}
                </div>
              ))}
              {data.order_differences.map((diff: any, idx: number) => (
                <div key={idx} className="p-2.5 bg-rose-50 border border-rose-200 rounded-lg text-xs font-mono text-rose-900 shadow-2xs">
                  {JSON.stringify(diff)}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ReconciliationPage;
