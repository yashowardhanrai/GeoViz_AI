import { formatNumber } from '../../utils/colors';

export default function KPICard({ label, value, delta, prefix = '', suffix = '', color }) {
  const deltaNum = parseFloat(delta);
  const deltaClass = isNaN(deltaNum)
    ? 'neutral'
    : deltaNum > 0 ? 'positive' : deltaNum < 0 ? 'negative' : 'neutral';

  const arrow = deltaNum > 0 ? '▲' : deltaNum < 0 ? '▼' : '';

  return (
    <div className="kpi-card" style={color ? { '--accent': color } : undefined}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={color ? { color } : undefined}>
        {prefix}{typeof value === 'number' ? formatNumber(value) : value}{suffix}
      </div>
      {delta != null && !isNaN(deltaNum) && (
        <div className={`kpi-delta ${deltaClass}`}>
          <span>{arrow}</span>
          <span>{formatNumber(Math.abs(deltaNum))}</span>
        </div>
      )}
    </div>
  );
}
