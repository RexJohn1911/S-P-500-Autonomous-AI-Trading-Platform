import React from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bot,
  BrainCircuit,
  Database,
  FileCheck2,
  FileSpreadsheet,
  FileText,
  History,
  LayoutDashboard,
  Network,
  Radio,
  Scale,
  Shield,
  ShieldAlert,
} from 'lucide-react';
import type { PageId } from '../types';

interface SidebarProps {
  activeTab?: string;
  setActiveTab?: (tab: string) => void;
  activePage?: PageId;
  onSelectPage?: (page: PageId) => void;
  alertsCount?: number;
  incidentsCount?: number;
  activeIncidentCount?: number;
  isKillSwitchTriggered?: boolean;
}

interface NavItem {
  id: PageId;
  label: string;
  icon: React.ReactNode;
  badge?: string | number;
  danger?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  activePage,
  onSelectPage,
  alertsCount = 0,
  incidentsCount = 0,
  activeIncidentCount = 0,
  isKillSwitchTriggered = false,
}) => {
  const currentTab = activeTab || activePage || 'overview';
  const handleSelect = (id: PageId) => {
    if (setActiveTab) setActiveTab(id);
    if (onSelectPage) onSelectPage(id);
  };

  const effectiveIncidentCount = incidentsCount || activeIncidentCount;

  const navItems: NavItem[] = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={13.5} /> },
    { id: 'health', label: 'System Health', icon: <Activity size={13.5} /> },
    { id: 'market-data', label: 'Market Data', icon: <Database size={13.5} /> },
    { id: 'models', label: 'AI Models', icon: <BrainCircuit size={13.5} /> },
    { id: 'signals', label: 'Signals', icon: <Radio size={13.5} /> },
    { id: 'portfolio', label: 'Portfolio', icon: <BarChart3 size={13.5} /> },
    { id: 'risk', label: 'Risk Engine', icon: <Shield size={13.5} /> },
    { id: 'orders', label: 'Orders', icon: <FileText size={13.5} /> },
    { id: 'executions', label: 'Executions', icon: <FileCheck2 size={13.5} /> },
    { id: 'broker', label: 'Broker Status', icon: <Network size={13.5} /> },
    { id: 'autonomous-loop', label: 'Autonomous Loop', icon: <Bot size={13.5} /> },
    {
      id: 'alerts',
      label: 'Safety Alerts',
      icon: <AlertTriangle size={13.5} />,
      badge: alertsCount > 0 ? alertsCount : undefined,
    },
    {
      id: 'incidents',
      label: 'Incidents',
      icon: <ShieldAlert size={13.5} />,
      badge: effectiveIncidentCount > 0 ? effectiveIncidentCount : undefined,
    },
    {
      id: 'kill-switch',
      label: 'Kill Switch',
      icon: <ShieldAlert size={13.5} />,
      badge: isKillSwitchTriggered ? 'TRIGGERED' : undefined,
      danger: isKillSwitchTriggered,
    },
    { id: 'reconciliation', label: 'Reconciliation', icon: <Scale size={13.5} /> },
    { id: 'audit', label: 'Audit Trail', icon: <FileSpreadsheet size={13.5} /> },
    { id: 'backtest', label: 'Research Backtest', icon: <History size={13.5} /> },
  ];

  return (
    <nav className="glass-panel border-b border-white/75 px-3 py-1 sticky top-[42px] z-20 select-none overflow-x-auto no-scrollbar shadow-2xs h-[39px] flex items-center">
      <div className="flex items-center space-x-1 min-w-max">
        {navItems.map((item) => {
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => handleSelect(item.id)}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-[11.5px] font-semibold transition-all ${
                isActive
                  ? 'bg-white/95 text-indigo-700 shadow-xs border border-white font-bold'
                  : item.danger
                  ? 'text-rose-600 hover:bg-rose-50'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-white/70'
              }`}
            >
              <span
                className={`transition-colors ${
                  isActive
                    ? 'text-indigo-600'
                    : item.danger
                    ? 'text-rose-500'
                    : 'text-slate-400 group-hover:text-slate-600'
                }`}
              >
                {item.icon}
              </span>
              <span>{item.label}</span>
              {item.badge !== undefined && (
                <span
                  className={`text-[9.5px] font-bold px-1.5 py-0.2 rounded-full font-mono ${
                    item.danger
                      ? 'bg-rose-500 text-white'
                      : isActive
                      ? 'bg-indigo-100 text-indigo-700'
                      : 'bg-slate-200/90 text-slate-700'
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
};

export default Sidebar;
