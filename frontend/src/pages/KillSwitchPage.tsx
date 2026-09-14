import React, { useEffect, useState } from 'react';
import { getKillSwitch } from '../services/api';
import type { KillSwitchSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import KillSwitchModal from '../components/KillSwitchModal';
import { ShieldAlert, ShieldCheck, AlertOctagon, CheckCircle2, Lock } from 'lucide-react';

const KillSwitchPage: React.FC = () => {
  const [data, setData] = useState<KillSwitchSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalAction, setModalAction] = useState<'trigger' | 'reset'>('trigger');

  const fetchStatus = () => {
    getKillSwitch()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const openTriggerModal = () => {
    setModalAction('trigger');
    setModalOpen(true);
  };

  const openResetModal = () => {
    setModalAction('reset');
    setModalOpen(true);
  };

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Kill Switch Safety System...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load kill switch state: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No kill switch data available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Global Emergency Kill Switch & Safety Controls</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Fail-closed emergency halt authority, safety gate verification, and circuit breaker telemetry</p>
      </div>

      {/* Hero Banner */}
      <div
        className={`glass-panel p-5 rounded-2xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 transition-all shadow-sm ${
          data.is_triggered
            ? 'bg-rose-50/90 border-rose-200'
            : 'bg-emerald-50/80 border-emerald-200'
        }`}
      >
        <div className="flex items-center space-x-4">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center border shadow-xs ${
              data.is_triggered
                ? 'bg-rose-100 border-rose-300 text-rose-600 animate-pulse'
                : 'bg-emerald-100 border-emerald-300 text-emerald-600'
            }`}
          >
            {data.is_triggered ? <AlertOctagon className="w-6 h-6" /> : <ShieldCheck className="w-6 h-6" />}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">System Safety State:</span>
              <StatusBadge status={data.status} size="sm" />
            </div>
            <h2 className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">
              {data.is_triggered ? 'EMERGENCY HALT ACTIVE — TRADING BLOCKED' : 'SYSTEM ARMED & OPERATIONAL'}
            </h2>
            <p className="text-xs text-slate-600 mt-0.5">
              {data.is_triggered
                ? `Triggered by ${data.triggered_by || 'SYSTEM'} due to: "${data.trigger_reason || 'Safety Limit'}"`
                : 'Continuous health monitoring active. Automated safety thresholds armed.'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3 self-end md:self-auto">
          {data.is_triggered ? (
            <button
              onClick={openResetModal}
              disabled={!data.reset_permitted}
              className={`px-4 py-2 rounded-xl font-bold text-xs tracking-wide transition-all shadow-sm flex items-center space-x-1.5 ${
                data.reset_permitted
                  ? 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-600/20 cursor-pointer'
                  : 'bg-slate-200 text-slate-400 border border-slate-300 cursor-not-allowed'
              }`}
            >
              <CheckCircle2 className="w-4 h-4" />
              <span>REQUEST SAFE RESET</span>
            </button>
          ) : (
            <button
              onClick={openTriggerModal}
              className="px-4 py-2 bg-rose-600 hover:bg-rose-700 text-white font-bold text-xs tracking-wide rounded-xl transition-all shadow-sm shadow-rose-600/20 flex items-center space-x-1.5 cursor-pointer"
            >
              <AlertOctagon className="w-4 h-4" />
              <span>TRIGGER EMERGENCY HALT</span>
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard
          title="Kill Switch Status"
          value={data.status}
          subtitle={`Triggered: ${data.is_triggered ? 'YES' : 'NO'}`}
          icon={data.is_triggered ? AlertOctagon : ShieldCheck}
        />
        <MetricCard
          title="Reset Authority"
          value={data.reset_permitted ? 'PERMITTED' : 'LOCKED / BLOCKED'}
          subtitle="Verification required"
          icon={Lock}
        />
        <MetricCard
          title="Active Critical Incidents"
          value={data.active_critical_incidents}
          subtitle="Must be 0 for reset"
          icon={ShieldAlert}
          trend={data.active_critical_incidents > 0 ? 'down' : 'neutral'}
        />
        <MetricCard
          title="Current Safety State"
          value={data.current_safety_state.toUpperCase()}
          subtitle={`Affected: ${data.affected_systems.length} systems`}
          icon={ShieldCheck}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="glass-card p-5">
          <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Fail-Closed Safety Contract</h2>
          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Trigger Reason:</span>
              <span className="text-rose-600 font-bold">{data.trigger_reason || 'None (Normal State)'}</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Trigger Source / Operator:</span>
              <span className="text-slate-900 font-semibold">{data.triggered_by || 'N/A'}</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Trigger Timestamp:</span>
              <span className="text-slate-600">{data.triggered_at ? new Date(data.triggered_at).toLocaleString() : 'N/A'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500 font-sans font-medium">Reset Verification Status:</span>
              <span className={data.reset_permitted ? 'text-emerald-700 font-bold' : 'text-rose-600 font-bold'}>
                {data.reset_permitted ? 'ALL GATES PASS' : 'SYSTEM CONSTRAINTS FAIL'}
              </span>
            </div>
          </div>
        </div>

        <div className="glass-card p-5">
          <h2 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">Affected Subsystems (Circuit Isolation)</h2>
          <div className="flex flex-wrap gap-1.5">
            {data.affected_systems.length === 0 ? (
              <span className="text-xs text-slate-400 font-sans">No subsystems currently quarantined or halted.</span>
            ) : (
              data.affected_systems.map((sys) => (
                <span key={sys} className="px-2.5 py-1 bg-rose-50 text-rose-700 border border-rose-200 rounded-lg text-xs font-mono font-semibold shadow-2xs">
                  {sys} (HALTED)
                </span>
              ))
            )}
          </div>
        </div>
      </div>

      <KillSwitchModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        action={modalAction}
        onSuccess={() => {
          setModalOpen(false);
          fetchStatus();
        }}
      />
    </div>
  );
};

export default KillSwitchPage;
