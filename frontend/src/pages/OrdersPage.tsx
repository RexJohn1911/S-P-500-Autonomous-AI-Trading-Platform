import React, { useEffect, useState } from 'react';
import { getOrders } from '../services/api';
import type { OrderSummary } from '../types';
import StatusBadge from '../components/StatusBadge';
import MetricCard from '../components/MetricCard';
import { ShoppingCart, CheckCircle2, Clock, XCircle } from 'lucide-react';

const OrdersPage: React.FC = () => {
  const [data, setData] = useState<OrderSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('ALL');

  useEffect(() => {
    getOrders()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-slate-500 font-mono text-xs">Loading Orders Registry...</div>;
  if (error) return <div className="p-8 text-rose-600 font-mono text-xs">Failed to load orders: {error}</div>;
  if (!data) return <div className="p-8 text-slate-500 font-mono text-xs">No orders data available.</div>;

  const filledCount = data.orders.filter((o) => o.status === 'FILLED').length;
  const activeCount = data.orders.filter((o) => ['SUBMITTED', 'PENDING', 'ACCEPTED', 'PARTIALLY_FILLED'].includes(o.status)).length;
  const canceledCount = data.orders.filter((o) => ['CANCELED', 'REJECTED', 'EXPIRED'].includes(o.status)).length;

  const filteredOrders = data.orders.filter((o) => {
    if (statusFilter === 'ALL') return true;
    if (statusFilter === 'ACTIVE') return ['SUBMITTED', 'PENDING', 'ACCEPTED', 'PARTIALLY_FILLED'].includes(o.status);
    if (statusFilter === 'FILLED') return o.status === 'FILLED';
    if (statusFilter === 'CANCELED') return ['CANCELED', 'REJECTED', 'EXPIRED'].includes(o.status);
    return true;
  });

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">Orders Lifecycle Management</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Auditable record of all submitted, working, filled, and canceled orders</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <MetricCard title="Total Orders" value={data.total_orders} subtitle="Lifecycle submissions" icon={ShoppingCart} />
        <MetricCard title="Working / Active" value={activeCount} subtitle="In-flight orders" icon={Clock} />
        <MetricCard title="Filled" value={filledCount} subtitle="Successfully executed" icon={CheckCircle2} />
        <MetricCard title="Canceled / Rejected" value={canceledCount} subtitle="Terminal states" icon={XCircle} />
      </div>

      <div className="glass-card p-5">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3.5 pb-2 border-b border-slate-100">
          <h2 className="text-sm font-bold text-slate-900">Order History & Execution State</h2>
          <div className="flex items-center space-x-1.5">
            <span className="text-[11px] text-slate-500 uppercase font-bold tracking-wider mr-1">Filter:</span>
            {['ALL', 'ACTIVE', 'FILLED', 'CANCELED'].map((filter) => (
              <button
                key={filter}
                onClick={() => setStatusFilter(filter)}
                className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all ${
                  statusFilter === filter
                    ? 'bg-indigo-600 text-white shadow-xs'
                    : 'bg-white/80 text-slate-600 border border-slate-200/80 hover:bg-slate-100/80 hover:text-slate-900'
                }`}
              >
                {filter}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100/70 text-slate-600 uppercase border-b border-slate-200/80 font-bold text-[11px] tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Client Order ID</th>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Side</th>
                <th className="py-2.5 px-3">Type</th>
                <th className="py-2.5 px-3">Quantity</th>
                <th className="py-2.5 px-3">Filled Qty</th>
                <th className="py-2.5 px-3">Avg Price</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Mode</th>
                <th className="py-2.5 px-3">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11.5px]">
              {filteredOrders.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-slate-400 font-sans">
                    No orders matching filter criteria.
                  </td>
                </tr>
              ) : (
                filteredOrders.map((ord) => (
                  <tr key={ord.client_order_id} className="hover:bg-indigo-50/40 transition-colors">
                    <td className="py-2 px-3 text-slate-500 truncate max-w-xs" title={ord.client_order_id}>
                      {ord.client_order_id.slice(0, 16)}...
                    </td>
                    <td className="py-2 px-3 font-bold text-slate-900 font-sans">{ord.symbol}</td>
                    <td className="py-2 px-3"><StatusBadge status={ord.side} size="sm" /></td>
                    <td className="py-2 px-3"><span className="text-slate-700">{ord.order_type}</span></td>
                    <td className="py-2 px-3">{ord.quantity.toLocaleString()}</td>
                    <td className="py-2 px-3">{ord.filled_quantity.toLocaleString()}</td>
                    <td className="py-2 px-3 font-semibold text-slate-900">${ord.average_fill_price ? ord.average_fill_price.toFixed(2) : '0.00'}</td>
                    <td className="py-2 px-3"><StatusBadge status={ord.status} size="sm" /></td>
                    <td className="py-2 px-3"><span className="text-slate-500">{ord.execution_mode}</span></td>
                    <td className="py-2 px-3 text-slate-500">{ord.submitted_at ? new Date(ord.submitted_at).toLocaleTimeString() : 'N/A'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default OrdersPage;
