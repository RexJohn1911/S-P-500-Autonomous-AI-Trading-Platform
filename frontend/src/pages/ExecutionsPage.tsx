import React, { useEffect, useState } from 'react';
import { getExecutions } from '../services/api';
import type { ExecutionSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { Zap, DollarSign, Percent, BarChart2 } from 'lucide-react';

const ExecutionsPage: React.FC = () => {
  const [data, setData] = useState<ExecutionSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getExecutions()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Execution telemetry...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load executions: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No execution data available.</div>;

  const totalVol = data.executions.reduce((acc, e) => acc + e.quantity * e.price, 0);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Executions & Fills</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Granular transaction logs with slippage, commission, and execution mode tracking</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard title="Total Executions" value={data.total_executions} subtitle="Fills recorded" icon={Zap} />
        <MetricCard
          title="Executed Volume"
          value={`$${totalVol.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          subtitle="Notional traded"
          icon={DollarSign}
        />
        <MetricCard
          title="Total Commission"
          value={`$${data.total_commission.toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
          subtitle="Broker friction"
          icon={Percent}
        />
        <MetricCard
          title="Avg Slippage"
          value={`${data.average_slippage_bps.toFixed(2)} bps`}
          subtitle="Execution quality"
          icon={BarChart2}
        />
      </div>

      <div className="glass-card p-5">
        <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Execution History</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Execution ID</th>
                <th className="py-2.5 px-3">Order ID</th>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Side</th>
                <th className="py-2.5 px-3">Quantity</th>
                <th className="py-2.5 px-3">Price</th>
                <th className="py-2.5 px-3">Commission</th>
                <th className="py-2.5 px-3">Slippage</th>
                <th className="py-2.5 px-3">Mode</th>
                <th className="py-2.5 px-3">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {data.executions.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-slate-400 font-sans">
                    No executions recorded in this session.
                  </td>
                </tr>
              ) : (
                data.executions.map((exec) => (
                  <tr key={exec.execution_id} className="hover:bg-indigo-50/40 transition-colors">
                    <td className="py-2 px-3 text-slate-500">{exec.execution_id.slice(0, 12)}...</td>
                    <td className="py-2 px-3 text-slate-500">{exec.order_id.slice(0, 12)}...</td>
                    <td className="py-2 px-3 font-bold text-slate-900 font-sans">{exec.symbol}</td>
                    <td className="py-2 px-3"><StatusBadge status={exec.side} size="sm" /></td>
                    <td className="py-2 px-3">{exec.quantity.toLocaleString()}</td>
                    <td className="py-2 px-3 font-semibold text-slate-900">${exec.price.toFixed(2)}</td>
                    <td className="py-2 px-3 text-slate-700">${exec.commission.toFixed(2)}</td>
                    <td className="py-2 px-3 text-slate-700">{exec.slippage.toFixed(4)}</td>
                    <td className="py-2 px-3"><span className="text-slate-500">{exec.execution_mode}</span></td>
                    <td className="py-2 px-3 text-slate-500">{new Date(exec.timestamp).toLocaleTimeString()}</td>
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

export default ExecutionsPage;
