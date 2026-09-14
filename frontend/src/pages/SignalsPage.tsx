import React, { useEffect, useState } from 'react';
import { getSignals } from '../services/api';
import type { SignalSummary, SignalItem } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { Activity, TrendingUp, TrendingDown, Minus } from 'lucide-react';

const SignalsPage: React.FC = () => {
  const [data, setData] = useState<SignalSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterDirection, setFilterDirection] = useState<string>('ALL');

  useEffect(() => {
    getSignals()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Signal Engine data...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load signals: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No signal data available.</div>;

  const longSignals = data.signals.filter((s: SignalItem) => s.direction.toUpperCase() === 'LONG' || s.direction.toUpperCase() === 'BUY').length;
  const shortSignals = data.signals.filter((s: SignalItem) => s.direction.toUpperCase() === 'SHORT' || s.direction.toUpperCase() === 'SELL').length;
  const flatSignals = data.signals.filter((s: SignalItem) => s.direction.toUpperCase() === 'FLAT' || s.direction.toUpperCase() === 'HOLD').length;

  const filteredSignals = data.signals.filter((s: SignalItem) => {
    if (filterDirection === 'ALL') return true;
    return s.direction.toUpperCase().includes(filterDirection);
  });

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Signal Engine</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Real-time direction, confidence, and model agreement for target universe</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard title="Total Monitored" value={data.signals.length} subtitle="Active universe targets" icon={Activity} />
        <MetricCard title="Long Signals" value={longSignals} subtitle="Bullish conviction" icon={TrendingUp} />
        <MetricCard title="Short Signals" value={shortSignals} subtitle="Bearish conviction" icon={TrendingDown} />
        <MetricCard title="Neutral / Flat" value={flatSignals} subtitle="Zero conviction" icon={Minus} />
      </div>

      <div className="glass-card p-5">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3.5 pb-2 border-b border-slate-100">
          <h2 className="text-sm font-bold text-slate-900">Active Signal Directives</h2>
          <div className="flex items-center space-x-1.5">
            <span className="text-[11px] text-slate-500 uppercase font-bold tracking-wider mr-1">Filter:</span>
            {['ALL', 'LONG', 'SHORT', 'FLAT'].map((dir) => (
              <button
                key={dir}
                onClick={() => setFilterDirection(dir)}
                className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all ${
                  filterDirection === dir
                    ? 'bg-indigo-600 text-white shadow-xs'
                    : 'bg-white/80 text-slate-600 border border-slate-200/80 hover:bg-slate-100/80 hover:text-slate-900'
                }`}
              >
                {dir}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Direction</th>
                <th className="py-2.5 px-3">Confidence</th>
                <th className="py-2.5 px-3">Model Agreement</th>
                <th className="py-2.5 px-3">Regime</th>
                <th className="py-2.5 px-3">Horizon</th>
                <th className="py-2.5 px-3">Reason Codes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {filteredSignals.map((sig: SignalItem) => (
                <tr key={sig.symbol} className="hover:bg-indigo-50/40 transition-colors">
                  <td className="py-2 px-3 font-bold text-slate-900 font-sans">{sig.symbol}</td>
                  <td className="py-2 px-3"><StatusBadge status={sig.direction} size="sm" /></td>
                  <td className="py-2 px-3">
                    <div className="flex items-center space-x-2">
                      <div className="w-16 bg-slate-200/80 h-1.5 rounded-full overflow-hidden">
                        <div className="bg-indigo-600 h-full rounded-full" style={{ width: `${Math.round(sig.confidence * 100)}%` }} />
                      </div>
                      <span className="text-slate-800 font-semibold">{(sig.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="py-2 px-3">
                    <div className="flex items-center space-x-2">
                      <div className="w-16 bg-slate-200/80 h-1.5 rounded-full overflow-hidden">
                        <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${Math.round(sig.model_agreement * 100)}%` }} />
                      </div>
                      <span className="text-slate-800 font-semibold">{(sig.model_agreement * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="py-2 px-3 font-sans"><span className="text-slate-700">{sig.regime}</span></td>
                  <td className="py-2 px-3"><span className="text-slate-500">{sig.forecast_horizon || `${sig.forecast_horizon_days || 5}d`}</span></td>
                  <td className="py-2 px-3 text-slate-600 max-w-xs truncate font-sans text-[11px]" title={sig.reason_codes.join(', ')}>
                    {sig.reason_codes.join(', ') || 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default SignalsPage;
