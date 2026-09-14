import React from 'react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  const s = (status || 'UNKNOWN').toUpperCase();

  // Clean, high-contrast, premium liquid-glass pill styling
  let bg = 'rgba(241, 245, 249, 0.7)';
  let text = '#475569';
  let border = 'rgba(203, 213, 225, 0.6)';
  let dotColor = '#94a3b8';

  if (['HEALTHY', 'PASS', 'ACTIVE', 'ARMED', 'FILLED', 'BUY', 'LONG', 'READY', 'OPERATIONAL', 'FRESH', 'MATCHED', 'RESOLVED', 'CLEAN'].includes(s)) {
    bg = 'rgba(236, 253, 245, 0.85)';
    text = '#047857';
    border = 'rgba(167, 243, 208, 0.8)';
    dotColor = '#10b981';
  } else if (['DEGRADED', 'WARNING', 'WARNINGS', 'FLAT', 'PARTIALLY_FILLED', 'CLEARING', 'ACKNOWLEDGED', 'MITIGATING', 'HISTORICAL'].includes(s)) {
    bg = 'rgba(254, 252, 232, 0.85)';
    text = '#b45309';
    border = 'rgba(253, 230, 138, 0.8)';
    dotColor = '#f59e0b';
  } else if (['CRITICAL', 'VIOLATION', 'TRIGGERED', 'REJECTED', 'SELL', 'SHORT', 'MISMATCH', 'OPEN', 'FAILED', 'HIGH'].includes(s)) {
    bg = 'rgba(254, 242, 242, 0.85)';
    text = '#b91c1c';
    border = 'rgba(254, 202, 202, 0.8)';
    dotColor = '#ef4444';
  } else if (['PAPER', 'INFO'].includes(s)) {
    bg = 'rgba(239, 246, 255, 0.85)';
    text = '#1d4ed8';
    border = 'rgba(191, 219, 254, 0.8)';
    dotColor = '#3b82f6';
  } else if (['LIVE'].includes(s)) {
    bg = 'rgba(250, 245, 255, 0.85)';
    text = '#7e22ce';
    border = 'rgba(233, 213, 255, 0.8)';
    dotColor = '#a855f7';
  }

  const padding = size === 'sm' ? '0.15rem 0.5rem' : '0.2rem 0.65rem';
  const fontSize = size === 'sm' ? '0.6875rem' : '0.75rem';
  const dotSize = size === 'sm' ? '5px' : '6px';

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.35rem',
        padding,
        fontSize,
        fontWeight: 600,
        borderRadius: '9999px',
        backgroundColor: bg,
        color: text,
        border: `1px solid ${border}`,
        letterSpacing: '0.025em',
        whiteSpace: 'nowrap',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        boxShadow: '0 1px 2px rgba(15, 23, 42, 0.03)',
      }}
    >
      <span
        style={{
          width: dotSize,
          height: dotSize,
          borderRadius: '50%',
          backgroundColor: dotColor,
          display: 'inline-block',
          boxShadow: `0 0 4px ${dotColor}`,
        }}
      />
      {s}
    </span>
  );
};

export default StatusBadge;
