import React, { useEffect, useState } from 'react';
import { getBacktest } from '../services/api';
import type { BacktestSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { History, TrendingUp, DollarSign, Award, AlertCircle } from 'lucide-react';

const BacktestPage: React.FC = () => {
  const [data, setData] = useState<BacktestSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getBacktest()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Backtest Engine Archives...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load backtests: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No backtest data available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Research & Backtesting Analytics</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Historical simulation results, statistical diagnostics, and risk-adjusted metrics</p>
      </div>

      {/* Prominent Historical Disclaimer Alert */}
      <div className="p-3.5 bg-indigo-50/80 border border-indigo-200/80 rounded-xl flex items-start space-x-2.5 shadow-2xs">
        <AlertCircle className="w-4 h-4 text-indigo-600 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-indigo-950 leading-relaxed">
          <span className="font-bold uppercase tracking-wider text-indigo-800 mr-1.5">HISTORICAL RESEARCH SIMULATION DISCLAIMER:</span>
          Backtest metrics represent historical in-sample and out-of-sample simulation results. Past performance and theoretical simulations do NOT guarantee future live execution returns.
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard title="Completed Runs" value={data.total_runs} subtitle="Simulation configurations" icon={History} />
        <MetricCard title="Active Strategy" value="Multi-Model Ensemble" subtitle="Walk-forward validated" icon={Award} />
        <MetricCard title="Execution Universe" value="S&P 500 Equities" subtitle="Historical data feed" icon={DollarSign} />
        <MetricCard title="Risk Model" value="Hard Constraint Engine" subtitle="Phase 12 compliance" icon={TrendingUp} />
      </div>

      <div className="space-y-4">
        {data.runs.length === 0 ? (
          <div className="glass-card p-8 text-center text-slate-400 font-sans">
            No historical backtest runs recorded yet. Execute backtest pipeline to populate simulation archives.
          </div>
        ) : (
          data.runs.map((run) => (
            <div key={run.run_id} className="glass-card p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-3">
                <div>
                  <div className="flex items-center space-x-2.5">
                    <h2 className="text-base font-bold text-slate-900">{run.run_id}</h2>
                    <StatusBadge status="HISTORICAL" size="sm" />
                  </div>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Strategy: <span className="text-indigo-600 font-mono font-semibold">{run.strategy}</span> | Dataset: <span className="text-slate-700 font-mono">{run.dataset}</span> ({run.dataset_provenance})
                  </p>
                </div>
                <div className="text-left sm:text-right font-mono text-xs text-slate-500">
                  <div>Period: <span className="text-slate-900 font-semibold">{run.start_date}</span> to <span className="text-slate-900 font-semibold">{run.end_date}</span></div>
                  <div className="mt-0.5">Trades: <span className="text-slate-900 font-semibold">{run.total_trades}</span> | Turnover: <span className="text-slate-900 font-semibold">{(run.turnover * 100).toFixed(1)}%</span></div>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <div className="p-2.5 bg-white/80 rounded-lg border border-slate-200/80 font-mono shadow-2xs">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Initial Capital</span>
                  <div className="text-sm font-bold text-slate-900 mt-0.5">${run.initial_capital.toLocaleString()}</div>
                </div>
                <div className="p-2.5 bg-white/80 rounded-lg border border-slate-200/80 font-mono shadow-2xs">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Final Equity</span>
                  <div className="text-sm font-bold text-emerald-600 mt-0.5">${run.final_equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                </div>
                <div className="p-2.5 bg-white/80 rounded-lg border border-slate-200/80 font-mono shadow-2xs">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Total Return</span>
                  <div className="text-sm font-bold text-emerald-600 mt-0.5">+{(run.total_return * 100).toFixed(2)}%</div>
                </div>
                <div className="p-2.5 bg-white/80 rounded-lg border border-slate-200/80 font-mono shadow-2xs">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Sharpe Ratio</span>
                  <div className="text-sm font-bold text-indigo-600 mt-0.5">{run.sharpe_ratio.toFixed(2)}</div>
                </div>
                <div className="p-2.5 bg-white/80 rounded-lg border border-slate-200/80 font-mono shadow-2xs">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Max Drawdown</span>
                  <div className="text-sm font-bold text-rose-600 mt-0.5">-{(run.max_drawdown * 100).toFixed(2)}%</div>
                </div>
                <div className="p-2.5 bg-white/80 rounded-lg border border-slate-200/80 font-mono shadow-2xs">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Sortino Ratio</span>
                  <div className="text-sm font-bold text-indigo-600 mt-0.5">{run.sortino_ratio.toFixed(2)}</div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default BacktestPage;
