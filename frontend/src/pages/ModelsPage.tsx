import React, { useEffect, useState } from 'react';
import { BrainCircuit, Cpu, Sparkles } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import StatusBadge from '../components/StatusBadge';
import type { ModelsData } from '../types';
import { getModels } from '../services/api';

interface ModelsPageProps {
  data?: ModelsData | null;
}

export const ModelsPage: React.FC<ModelsPageProps> = ({ data: propData }) => {
  const [data, setData] = useState<ModelsData | null>(propData || null);
  const [loading, setLoading] = useState(!propData);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!propData) {
      getModels()
        .then(setData)
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false));
    }
  }, [propData]);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading AI models...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load models: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No models data available.</div>;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">AI & Machine Learning Ensemble</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Linear baselines, Gradient Boosted Trees, Deep Neural Nets, and Market Regime inference</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
        <MetricCard
          title="Ensemble Status"
          value={data.ensemble_status}
          subtitle="Multi-Model Quant Ensemble"
          badge="OPERATIONAL"
          icon={BrainCircuit}
        />
        <MetricCard
          title="Active AI Models"
          value={data.total_models_available}
          subtitle="Baseline ML & Deep Learning"
          badge="READY"
          icon={Cpu}
        />
        <MetricCard
          title="Market Regime"
          value={data.regime_model.current_regime}
          subtitle={`Confidence: ${(data.regime_model.confidence * 100).toFixed(0)}%`}
          badge="ACTIVE"
          icon={Sparkles}
        />
      </div>

      <div className="glass-card p-5">
        <h3 className="text-sm font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100">
          Ensemble Member Model Specifications & Latencies
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Model Name</th>
                <th className="py-2.5 px-3">Architecture Category</th>
                <th className="py-2.5 px-3">Version</th>
                <th className="py-2.5 px-3">Ensemble Weight</th>
                <th className="py-2.5 px-3">Inference Latency</th>
                <th className="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {data.active_models.map((m) => (
                <tr key={m.model_name} className="hover:bg-indigo-50/40 transition-colors">
                  <td className="py-2 px-3 font-bold text-slate-900 font-sans">{m.model_name}</td>
                  <td className="py-2 px-3 text-slate-600 font-sans">{m.model_type}</td>
                  <td className="py-2 px-3 text-slate-500">{m.version}</td>
                  <td className="py-2 px-3 font-bold text-indigo-600">{(m.weight * 100).toFixed(0)}%</td>
                  <td className="py-2 px-3 text-slate-700">{m.avg_inference_latency_ms.toFixed(1)} ms</td>
                  <td className="py-2 px-3"><StatusBadge status={m.status} size="sm" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ModelsPage;
