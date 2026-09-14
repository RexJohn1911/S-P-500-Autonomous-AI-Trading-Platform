import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import KillSwitchModal from './components/KillSwitchModal';
import { getOverview, getKillSwitch } from './services/api';
import type { SystemOverview, KillSwitchSummary } from './types';

// Pages
import OverviewPage from './pages/OverviewPage';
import HealthPage from './pages/HealthPage';
import MarketDataPage from './pages/MarketDataPage';
import ModelsPage from './pages/ModelsPage';
import SignalsPage from './pages/SignalsPage';
import PortfolioPage from './pages/PortfolioPage';
import RiskPage from './pages/RiskPage';
import OrdersPage from './pages/OrdersPage';
import ExecutionsPage from './pages/ExecutionsPage';
import BrokerPage from './pages/BrokerPage';
import AutonomousLoopPage from './pages/AutonomousLoopPage';
import AlertsPage from './pages/AlertsPage';
import IncidentsPage from './pages/IncidentsPage';
import KillSwitchPage from './pages/KillSwitchPage';
import ReconciliationPage from './pages/ReconciliationPage';
import AuditPage from './pages/AuditPage';
import BacktestPage from './pages/BacktestPage';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [overview, setOverview] = useState<SystemOverview | null>(null);
  const [killSwitch, setKillSwitch] = useState<KillSwitchSummary | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalAction, setModalAction] = useState<'trigger' | 'reset'>('trigger');

  const fetchGlobalState = async () => {
    try {
      const [ovData, ksData] = await Promise.all([getOverview(), getKillSwitch()]);
      setOverview(ovData);
      setKillSwitch(ksData);
    } catch {
      // Backend may be offline or starting up
    }
  };

  useEffect(() => {
    let isMounted = true;

    const syncState = async () => {
      try {
        const [ovData, ksData] = await Promise.all([getOverview(), getKillSwitch()]);
        if (isMounted) {
          setOverview(ovData);
          setKillSwitch(ksData);
        }
      } catch {
        // Backend may be offline or starting up
      }
    };

    void syncState();
    const interval = setInterval(() => {
      void syncState();
    }, 10000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleKillSwitchClick = () => {
    if (killSwitch?.is_triggered) {
      setModalAction('reset');
    } else {
      setModalAction('trigger');
    }
    setModalOpen(true);
  };

  const renderActivePage = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPage onNavigate={(tab: string) => setActiveTab(tab)} />;
      case 'health':
        return <HealthPage />;
      case 'market-data':
        return <MarketDataPage />;
      case 'models':
        return <ModelsPage />;
      case 'signals':
        return <SignalsPage />;
      case 'portfolio':
        return <PortfolioPage />;
      case 'risk':
        return <RiskPage />;
      case 'orders':
        return <OrdersPage />;
      case 'executions':
        return <ExecutionsPage />;
      case 'broker':
        return <BrokerPage />;
      case 'autonomous-loop':
        return <AutonomousLoopPage />;
      case 'alerts':
        return <AlertsPage />;
      case 'incidents':
        return <IncidentsPage />;
      case 'kill-switch':
        return <KillSwitchPage />;
      case 'reconciliation':
        return <ReconciliationPage />;
      case 'audit':
        return <AuditPage />;
      case 'backtest':
        return <BacktestPage />;
      default:
        return <OverviewPage onNavigate={(tab: string) => setActiveTab(tab)} />;
    }
  };

  return (
    <div className="relative min-h-screen flex flex-col bg-slate-50/60 text-slate-900 font-sans selection:bg-indigo-500 selection:text-white">
      {/* Liquid Ambient Gradient Blobs */}
      <div className="ambient-bg" aria-hidden="true">
        <div className="ambient-blob-1" />
        <div className="ambient-blob-2" />
        <div className="ambient-blob-3" />
      </div>

      {/* Main App Container */}
      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Top Header */}
        <Header
          executionMode={overview?.execution_mode || 'PAPER'}
          systemStatus={overview?.system_status || 'HEALTHY'}
          killSwitchState={killSwitch?.status}
          killSwitchTriggered={killSwitch?.is_triggered || false}
          onKillSwitchClick={handleKillSwitchClick}
          onRefresh={fetchGlobalState}
        />

        {/* Compact Horizontal Navigation Toolbar */}
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          alertsCount={overview?.recent_alerts.length || 0}
          incidentsCount={overview?.recent_incidents.length || 0}
          isKillSwitchTriggered={killSwitch?.is_triggered || false}
        />

        {/* Main Content Area */}
        <main className="flex-1 max-w-7xl w-full mx-auto p-3.5 sm:p-5 space-y-3.5">
          {renderActivePage()}
        </main>
      </div>

      {/* Global Kill Switch Confirmation Modal */}
      <KillSwitchModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        action={modalAction}
        onSuccess={() => {
          setModalOpen(false);
          void fetchGlobalState();
        }}
      />
    </div>
  );
};

export default App;
