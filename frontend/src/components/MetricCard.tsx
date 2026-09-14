import React from 'react';
import StatusBadge from './StatusBadge';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  badge?: string;
  icon?: React.ReactNode | React.ComponentType<any>;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  badge,
  icon: Icon,
  trend,
  trendValue,
}) => {
  const renderIcon = () => {
    if (!Icon) return null;
    if (React.isValidElement(Icon)) return Icon;
    if (typeof Icon === 'function') {
      const Comp = Icon as React.ComponentType<{ className?: string }>;
      return <Comp className="w-3.5 h-3.5" />;
    }
    return null;
  };

  return (
    <div
      className="glass-card"
      style={{
        padding: '0.85rem 1rem',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        minHeight: '100px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          {Icon && (
            <div
              style={{
                width: '24px',
                height: '24px',
                borderRadius: '6px',
                backgroundColor: 'rgba(238, 242, 255, 0.85)',
                border: '1px solid rgba(224, 231, 255, 0.85)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#4f46e5',
              }}
            >
              {renderIcon()}
            </div>
          )}
          <span
            style={{
              fontSize: '0.7rem',
              color: '#64748b',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}
          >
            {title}
          </span>
        </div>
        {badge && <StatusBadge status={badge} size="sm" />}
      </div>

      <div
        style={{
          fontSize: '1.45rem',
          fontWeight: 700,
          color: '#0f172a',
          margin: '0.1rem 0',
          letterSpacing: '-0.025em',
          lineHeight: 1.15,
        }}
      >
        {value}
      </div>

      {(subtitle || trendValue) && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontSize: '0.725rem',
            marginTop: '0.15rem',
            color: '#64748b',
          }}
        >
          {trendValue && (
            <span
              style={{
                fontWeight: 600,
                color: trend === 'up' ? '#059669' : trend === 'down' ? '#dc2626' : '#64748b',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.15rem',
              }}
            >
              {trend === 'up' ? '▲' : trend === 'down' ? '▼' : '•'} {trendValue}
            </span>
          )}
          {subtitle && <span>{subtitle}</span>}
        </div>
      )}
    </div>
  );
};

export default MetricCard;
