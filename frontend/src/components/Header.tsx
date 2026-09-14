import React from 'react';
import { AlertOctagon, RefreshCw, Cpu, Activity, ShieldAlert } from 'lucide-react';
import StatusBadge from './StatusBadge';

interface HeaderProps {
  title?: string;
  executionMode?: string;
  systemStatus?: string;
  killSwitchState?: string;
  killSwitchTriggered?: boolean;
  onRefresh?: () => void;
  isLoading?: boolean;
  onOpenKillSwitchModal?: () => void;
  onKillSwitchClick?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  executionMode = 'PAPER',
  systemStatus = 'HEALTHY',
  killSwitchState,
  killSwitchTriggered = false,
  onRefresh,
  isLoading = false,
  onOpenKillSwitchModal,
  onKillSwitchClick,
}) => {
  const isTriggered = killSwitchTriggered || killSwitchState === 'TRIGGERED';
  const effectiveKsState = killSwitchState || (isTriggered ? 'TRIGGERED' : 'ARMED');
  const handleKsClick = onKillSwitchClick || onOpenKillSwitchModal;

  return (
    <header className="glass-panel border-b border-white/75 px-5 py-1.5 flex items-center justify-between sticky top-0 z-30 select-none h-[42px]">
      {/* Brand & Subtitle */}
      <div className="flex items-center space-x-2.5">
        <div className="w-6 h-6 rounded-md bg-gradient-to-tr from-indigo-600 to-blue-500 flex items-center justify-center text-white shadow-xs">
          <Cpu className="w-3.5 h-3.5" />
        </div>
        <div className="flex items-baseline space-x-2">
          <h1 className="text-xs sm:text-sm font-bold text-slate-900 tracking-tight leading-none">
            {title || 'S&P 500 AI Trading System'}
          </h1>
          <span className="hidden md:inline text-[10.5px] text-slate-500 font-medium">
            Phase 19 Dashboard <span className="text-slate-300">•</span> Autonomous Engine v1.0.0
          </span>
        </div>
      </div>

      {/* Status Indicators & Quick Actions */}
      <div className="flex items-center space-x-2">
        {/* Mode Pill */}
        <div className="hidden sm:flex items-center space-x-1 px-2 py-0.5 rounded-full bg-slate-100/90 border border-slate-200/90 text-[10.5px] text-slate-600 font-mono shadow-2xs">
          <span className="text-slate-400 font-bold uppercase text-[9.5px]">Mode:</span>
          <StatusBadge status={executionMode} size="sm" />
        </div>

        {/* Health Pill */}
        <div className="hidden sm:flex items-center space-x-1 px-2 py-0.5 rounded-full bg-slate-100/90 border border-slate-200/90 text-[10.5px] text-slate-600 font-mono shadow-2xs">
          <Activity className="w-3 h-3 text-slate-400" />
          <span className="text-slate-400 font-bold uppercase text-[9.5px]">Health:</span>
          <StatusBadge status={systemStatus} size="sm" />
        </div>

        {/* Kill Switch Button */}
        {handleKsClick && (
          <button
            onClick={handleKsClick}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold border transition-all shadow-xs ${
              isTriggered
                ? 'bg-rose-50 border-rose-200 text-rose-700 animate-pulse hover:bg-rose-100'
                : 'bg-emerald-50/90 border-emerald-200 text-emerald-800 hover:bg-emerald-100'
            }`}
          >
            {isTriggered ? (
              <ShieldAlert className="w-3 h-3 text-rose-600" />
            ) : (
              <AlertOctagon className="w-3 h-3 text-emerald-600" />
            )}
            <span className="font-mono text-[10.5px] tracking-wide">KILL SWITCH: {effectiveKsState}</span>
          </button>
        )}

        {/* Refresh Button */}
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="w-7 h-7 rounded-full border border-slate-200 bg-white/90 text-slate-600 hover:text-slate-900 hover:bg-white flex items-center justify-center transition-all shadow-2xs disabled:opacity-50"
            title="Refresh State"
          >
            <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin text-indigo-600' : ''}`} />
          </button>
        )}
      </div>
    </header>
  );
};

export default Header;
