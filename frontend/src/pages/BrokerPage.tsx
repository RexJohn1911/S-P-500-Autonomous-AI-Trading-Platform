import React, { useEffect, useState } from 'react';
import { getBroker } from '../services/api';
import type { BrokerSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { Server, ShieldCheck, Activity, KeyRound, DollarSign, Layers } from 'lucide-react';

const BrokerPage: React.FC = () => {
  const [data, setData] = useState<BrokerSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getBroker()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Connecting to Broker Telemetry...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load broker telemetry: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No broker telemetry available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Broker Connectivity & Integration</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Alpaca / Paper gateway telemetry, rate-limiting headroom, and account state</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard
          title="Broker Provider"
          value={data.provider.toUpperCase()}
          subtitle={`Mode: ${data.execution_mode}`}
          icon={Server}
        />
        <MetricCard
          title="Connection Status"
          value={data.connected ? 'CONNECTED' : 'OFFLINE'}
          subtitle={`Auth: ${data.authenticated ? 'VERIFIED' : 'FAILED'}`}
          icon={ShieldCheck}
        />
        <MetricCard
          title="API Latency"
          value={`${data.latency_ms.toFixed(1)} ms`}
          subtitle="Round-trip time"
          icon={Activity}
        />
        <MetricCard
          title="Rate Limit Headroom"
          value={`${data.rate_limit_remaining} / 200`}
          subtitle="Requests remaining"
          icon={Layers}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="glass-card p-5">
          <div className="flex items-center space-x-2.5 mb-3.5 pb-2 border-b border-slate-100">
            <KeyRound className="w-4 h-4 text-indigo-600" />
            <h2 className="text-sm font-bold text-slate-900">Security & Credential Masking</h2>
          </div>
          <div className="p-3 bg-white/80 border border-slate-200/80 rounded-lg text-xs space-y-2 shadow-2xs">
            <div className="flex justify-between items-center text-slate-600">
              <span className="font-sans">Broker Credential Status:</span>
              <span className="text-emerald-700 font-bold font-mono">SECURE (BACKEND ISOLATED)</span>
            </div>
            <div className="flex justify-between items-center text-slate-600">
              <span className="font-sans">API Key Exposure:</span>
              <span className="text-emerald-700 font-mono font-semibold">REDACTED / RESTRICTED</span>
            </div>
            <div className="flex justify-between items-center text-slate-600">
              <span className="font-sans">Direct Frontend Calls:</span>
              <span className="text-emerald-700 font-bold font-mono">BLOCKED (0 DIRECT CALLS)</span>
            </div>
          </div>

          <h3 className="text-xs font-bold text-slate-600 uppercase tracking-wider mt-5 mb-2.5">Supported Capabilities</h3>
          <div className="flex flex-wrap gap-1.5">
            {data.capabilities.map((cap) => (
              <span key={cap} className="px-2.5 py-1 bg-indigo-50/80 text-indigo-700 border border-indigo-200/80 rounded-md text-xs font-mono font-semibold shadow-2xs">
                {cap}
              </span>
            ))}
          </div>
        </div>

        <div className="glass-card p-5">
          <div className="flex items-center space-x-2.5 mb-3.5 pb-2 border-b border-slate-100">
            <DollarSign className="w-4 h-4 text-emerald-600" />
            <h2 className="text-sm font-bold text-slate-900">Broker Account Details</h2>
          </div>
          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Account Status:</span>
              <StatusBadge status={data.account_status} size="sm" />
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Equity Reported:</span>
              <span className="text-slate-900 font-bold">${data.equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Cash Balance:</span>
              <span className="text-slate-900 font-bold">${data.cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
            </div>
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-sans font-medium">Buying Power:</span>
              <span className="text-emerald-600 font-bold">${data.buying_power.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500 font-sans font-medium">Last Error:</span>
              <span className="text-slate-600 font-sans">{data.last_error || 'None'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BrokerPage;
