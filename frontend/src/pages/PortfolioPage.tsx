import React, { useEffect, useState } from 'react';
import { getPortfolio } from '../services/api';
import type { PortfolioSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { DollarSign, PieChart, ShieldAlert, BarChart3 } from 'lucide-react';

const PortfolioPage: React.FC = () => {
  const [data, setData] = useState<PortfolioSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPortfolio()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Portfolio Engine data...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load portfolio: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No portfolio data available.</div>;

  const totalVal = data.total_value ?? data.equity;
  const positionsList = data.positions || [];
  const longExp = data.long_exposure ?? data.gross_exposure;
  const shortExp = data.short_exposure ?? 0.0;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Portfolio & Capital Allocation</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Current positions, capital distribution, and gross/net exposure telemetry</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard
          title="Total Portfolio Value"
          value={`$${totalVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          subtitle="Equity + Cash"
          icon={DollarSign}
        />
        <MetricCard
          title="Cash Available"
          value={`$${data.cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
          subtitle={`Invested: $${data.invested_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
          icon={PieChart}
        />
        <MetricCard
          title="Total Unrealized P&L"
          value={`${data.unrealized_pnl >= 0 ? '+' : ''}$${data.unrealized_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          subtitle="Mark-to-market"
          icon={BarChart3}
          trend={data.unrealized_pnl >= 0 ? 'up' : 'down'}
        />
        <MetricCard
          title="Active Positions"
          value={positionsList.length}
          subtitle={`Gross: ${(data.gross_exposure * 100).toFixed(1)}% | Net: ${(data.net_exposure * 100).toFixed(1)}%`}
          icon={ShieldAlert}
        />
      </div>

      {/* Allocation breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="glass-card p-5 lg:col-span-1">
          <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Exposure Summary</h2>
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
              <span className="text-emerald-600 font-bold">{(longExp * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Short Exposure:</span>
              <span className="text-rose-600 font-bold">{(shortExp * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500 font-sans font-medium">Leverage:</span>
              <span className="text-indigo-600 font-bold">{data.leverage.toFixed(2)}x</span>
            </div>
          </div>
        </div>

        <div className="glass-card p-5 lg:col-span-2">
          <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Active Holdings</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-700">
              <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
                <tr>
                  <th className="py-2.5 px-3">Symbol</th>
                  <th className="py-2.5 px-3">Side</th>
                  <th className="py-2.5 px-3">Quantity</th>
                  <th className="py-2.5 px-3">Entry Price</th>
                  <th className="py-2.5 px-3">Current Price</th>
                  <th className="py-2.5 px-3">Market Value</th>
                  <th className="py-2.5 px-3">Weight</th>
                  <th className="py-2.5 px-3">P&L ($ / %)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
                {positionsList.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-slate-400 font-sans">
                      No active positions held in portfolio.
                    </td>
                  </tr>
                ) : (
                  positionsList.map((pos) => {
                    const isPositive = pos.unrealized_pnl >= 0;
                    return (
                      <tr key={pos.symbol} className="hover:bg-indigo-50/40 transition-colors">
                        <td className="py-2 px-3 font-bold text-slate-900 font-sans">{pos.symbol}</td>
                        <td className="py-2 px-3"><StatusBadge status={pos.side} size="sm" /></td>
                        <td className="py-2 px-3">{pos.quantity.toLocaleString()}</td>
                        <td className="py-2 px-3">${(pos.average_entry_price ?? pos.average_entry ?? 0).toFixed(2)}</td>
                        <td className="py-2 px-3">${pos.current_price.toFixed(2)}</td>
                        <td className="py-2 px-3">${pos.market_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                        <td className="py-2 px-3">{((pos.weight_pct ?? (pos.weight * 100)) || 0).toFixed(2)}%</td>
                        <td className={`py-2 px-3 font-bold ${isPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
                          {isPositive ? '+' : ''}${pos.unrealized_pnl.toFixed(2)} ({isPositive ? '+' : ''}{(pos.unrealized_pnl_pct * 100).toFixed(2)}%)
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PortfolioPage;
