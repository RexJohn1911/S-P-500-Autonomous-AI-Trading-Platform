import React, { useEffect, useState } from 'react';
import { getAutonomousLoop } from '../services/api';
import type { AutonomousLoopSummary } from '../types';
import MetricCard from '../components/MetricCard';
import { RotateCcw, CheckCircle2, Clock, AlertOctagon, Check } from 'lucide-react';

const STAGES = [
  'Market Data',
  'Validation',
  'Features',
  'Models',
  'Regime',
  'Signals',
  'Portfolio',
  'Risk',
  'Orders',
  'Execution',
  'Reconciliation',
  'Monitoring',
  'Audit',
  'Checkpoint',
];

const AutonomousLoopPage: React.FC = () => {
  const [data, setData] = useState<AutonomousLoopSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAutonomousLoop()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Autonomous Loop Telemetry...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load autonomous loop telemetry: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No autonomous loop data available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Autonomous Trading Loop</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Deterministic state machine execution cycle and 14-stage pipeline telemetry</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard
          title="Loop State"
          value={data.state}
          subtitle={`Cycle ID: ${data.cycle_id || 'N/A'}`}
          icon={RotateCcw}
        />
        <MetricCard
          title="Cycles Completed"
          value={data.cycles_completed}
          subtitle={`Failed: ${data.failed_cycles}`}
          icon={CheckCircle2}
        />
        <MetricCard
          title="Last Cycle Duration"
          value={`${data.cycle_duration_sec.toFixed(2)}s`}
          subtitle={`Retry count: ${data.retry_count}`}
          icon={Clock}
        />
        <MetricCard
          title="Recovery State"
          value={data.recovery_state.toUpperCase()}
          subtitle={`Checkpoint: ${data.checkpoint_age_sec.toFixed(0)}s ago`}
          icon={AlertOctagon}
        />
      </div>

      {/* 14 Canonical Pipeline Stages */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Canonical 14-Stage Pipeline Status</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-2.5">
          {STAGES.map((stage, idx) => {
            const isCompleted = idx <= (STAGES.indexOf(data.last_successful_stage) >= 0 ? STAGES.indexOf(data.last_successful_stage) : STAGES.length - 1);
            return (
              <div
                key={stage}
                className={`p-2.5 rounded-lg border text-xs font-mono flex flex-col justify-between transition-all shadow-2xs ${
                  isCompleted
                    ? 'bg-emerald-50/70 border-emerald-200/80 text-emerald-900'
                    : 'bg-white/60 border-slate-200/80 text-slate-500'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] text-slate-400 font-bold">#{idx + 1}</span>
                  {isCompleted ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <div className="w-1.5 h-1.5 rounded-full bg-slate-300" />}
                </div>
                <div className="font-semibold text-slate-900 font-sans text-xs">{stage}</div>
                <div className="text-[10px] text-slate-500 mt-1">{isCompleted ? 'VERIFIED' : 'PENDING'}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent cycle events */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Recent Autonomous Cycle Events</h2>
        <div className="space-y-2">
          {data.recent_events.length === 0 ? (
            <p className="text-xs text-slate-400 font-sans">No recent cycle events logged.</p>
          ) : (
            data.recent_events.map((evt, idx) => (
              <div key={idx} className="p-2.5 bg-white/80 border border-slate-200/80 rounded-lg flex items-center justify-between text-xs font-mono shadow-2xs">
                <span className="text-slate-800">{evt.message || JSON.stringify(evt)}</span>
                <span className="text-slate-500">{evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ''}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default AutonomousLoopPage;
