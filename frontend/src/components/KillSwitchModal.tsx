import React, { useState } from 'react';
import { AlertOctagon, AlertTriangle, CheckCircle2, X } from 'lucide-react';
import type { KillSwitchData } from '../types';
import { triggerKillSwitch, resetKillSwitch } from '../services/api';

interface KillSwitchModalProps {
  isOpen: boolean;
  onClose: () => void;
  action?: 'trigger' | 'reset';
  onSuccess?: () => void;
  data?: KillSwitchData | null;
  onTrigger?: (reason: string, operator: string) => Promise<void>;
  onReset?: (operator: string) => Promise<void>;
}

export const KillSwitchModal: React.FC<KillSwitchModalProps> = ({
  isOpen,
  onClose,
  action,
  onSuccess,
  data,
  onTrigger,
  onReset,
}) => {
  const [reason, setReason] = useState('');
  const [operator, setOperator] = useState('WEB_OPERATOR');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const isTriggered = action ? action === 'reset' : data?.kill_switch.is_triggered ?? false;

  const handleAction = async () => {
    setIsSubmitting(true);
    setError(null);
    try {
      if (isTriggered) {
        if (onReset) {
          await onReset(operator);
        } else {
          await resetKillSwitch(operator);
        }
      } else {
        if (!reason.trim()) {
          setError('A valid reason must be provided to activate the emergency kill switch.');
          setIsSubmitting(false);
          return;
        }
        if (onTrigger) {
          await onTrigger(reason, operator);
        } else {
          await triggerKillSwitch(reason, operator);
        }
      }
      if (onSuccess) onSuccess();
      onClose();
    } catch (e: any) {
      setError(e.message || 'Action failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
      <div
        className="glass-panel rounded-2xl max-w-md w-full p-6 shadow-2xl border border-white/80 bg-white/90"
      >
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center space-x-2.5">
            {isTriggered ? (
              <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600">
                <CheckCircle2 className="w-5 h-5" />
              </div>
            ) : (
              <div className="w-8 h-8 rounded-full bg-rose-100 flex items-center justify-center text-rose-600">
                <AlertOctagon className="w-5 h-5" />
              </div>
            )}
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              {isTriggered ? 'Request Safe Kill Switch Reset' : 'Activate Global Kill Switch'}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div
          className={`p-3 rounded-xl text-xs flex items-start space-x-2.5 mb-4 border ${
            isTriggered
              ? 'bg-indigo-50/80 border-indigo-200 text-indigo-900'
              : 'bg-rose-50/80 border-rose-200 text-rose-900'
          }`}
        >
          <AlertTriangle className={`w-4 h-4 flex-shrink-0 mt-0.5 ${isTriggered ? 'text-indigo-600' : 'text-rose-600'}`} />
          <div className="leading-relaxed">
            {isTriggered
              ? 'Resetting the kill switch will re-arm the system and allow the autonomous loop to resume trading once all safety gates pass.'
              : 'CRITICAL SAFETY ACTION: Activating the global kill switch immediately halts all autonomous cycles and cancels pending orders.'}
          </div>
        </div>

        {!isTriggered && (
          <div className="mb-4">
            <label className="block text-[11px] font-bold text-slate-600 uppercase tracking-wider mb-1">
              Reason for Emergency Activation (Required):
            </label>
            <input
              type="text"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g., Abnormal market volatility / Manual risk halt"
              className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-rose-500/20 focus:border-rose-500 font-mono shadow-xs"
            />
          </div>
        )}

        <div className="mb-5">
          <label className="block text-[11px] font-bold text-slate-600 uppercase tracking-wider mb-1">
            Operator Identifier:
          </label>
          <input
            type="text"
            value={operator}
            onChange={(e) => setOperator(e.target.value)}
            className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 font-mono shadow-xs"
          />
        </div>

        {error && (
          <div className="p-2.5 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-700 mb-4 font-mono">
            Error: {error}
          </div>
        )}

        <div className="flex justify-end space-x-2.5">
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-semibold transition-colors shadow-xs"
          >
            Cancel
          </button>
          <button
            onClick={handleAction}
            disabled={isSubmitting}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all shadow-sm ${
              isTriggered
                ? 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-600/20'
                : 'bg-rose-600 hover:bg-rose-700 text-white shadow-rose-600/20'
            }`}
          >
            {isSubmitting
              ? 'Executing...'
              : isTriggered
              ? 'Authorize Reset to ARMED'
              : 'CONFIRM EMERGENCY HALT'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default KillSwitchModal;
