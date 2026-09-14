import React, { useEffect, useState } from 'react';
import { Database, CheckCircle, Clock } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import StatusBadge from '../components/StatusBadge';
import type { MarketDataResponse } from '../types';
import { getMarketData } from '../services/api';

interface MarketDataPageProps {
  data?: MarketDataResponse | null;
}

export const MarketDataPage: React.FC<MarketDataPageProps> = ({ data: propData }) => {
  const [data, setData] = useState<MarketDataResponse | null>(propData || null);
  const [loading, setLoading] = useState(!propData);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!propData) {
      getMarketData()
        .then(setData)
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false));
    }
  }, [propData]);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading market data...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load market data: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No market data available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Market Data & Ingestion Feeds</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Normalized tick/bar ingestion, freshness validation, and historical parquet storage</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
        <MetricCard
          title="Market Status"
          value={data.market_status}
          subtitle="US Equities Market"
          badge="ACTIVE"
          icon={Clock}
        />
        <MetricCard
          title="Data Ingestion Provider"
          value={data.data_provider.split('/')[0]}
          subtitle="Local & API Provider"
          badge="READY"
          icon={Database}
        />
        <MetricCard
          title="Monitored Symbols"
          value={data.total_symbols_monitored}
          subtitle={`Missing: ${data.missing_symbols.length} | Stale: ${data.stale_symbols.length}`}
          badge="HEALTHY"
          icon={CheckCircle}
        />
      </div>

      <div className="glass-card p-5">
        <h3 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">
          Monitored Universe Feed & Bar Freshness
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Latest Close</th>
                <th className="py-2.5 px-3">24h Change</th>
                <th className="py-2.5 px-3">24h Volume</th>
                <th className="py-2.5 px-3">Bar Timestamp</th>
                <th className="py-2.5 px-3">Freshness Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {data.symbols.map((sym) => (
                <tr key={sym.symbol} className="hover:bg-indigo-50/40 transition-colors">
                  <td className="py-2 px-3 font-bold text-indigo-700 font-sans">{sym.symbol}</td>
                  <td className="py-2 px-3 font-semibold text-slate-900">${sym.latest_close.toFixed(2)}</td>
                  <td className={`py-2 px-3 font-bold ${sym.change_pct_24h >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                    {sym.change_pct_24h >= 0 ? '+' : ''}{sym.change_pct_24h.toFixed(2)}%
                  </td>
                  <td className="py-2 px-3 text-slate-600">{(sym.volume_24h / 1e6).toFixed(1)}M</td>
                  <td className="py-2 px-3 text-slate-500">
                    {sym.latest_timestamp ? new Date(sym.latest_timestamp).toLocaleDateString() : 'N/A'}
                  </td>
                  <td className="py-2 px-3"><StatusBadge status={sym.freshness_status} size="sm" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default MarketDataPage;
