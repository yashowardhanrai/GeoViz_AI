/**
 * Custom Recharts tooltip with glassmorphism styling.
 */
export default function CustomTooltip({ active, payload, label, formatter }) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="custom-tooltip">
      <div className="tooltip-label">{label}</div>
      {payload.map((entry, i) => (
        <div className="tooltip-item" key={i}>
          <span
            className="tooltip-dot"
            style={{ background: entry.color || entry.stroke }}
          />
          <span>{entry.name || entry.dataKey}</span>
          <span className="tooltip-value">
            {formatter
              ? formatter(entry.value, entry.name)
              : (typeof entry.value === 'number'
                  ? entry.value.toFixed(2)
                  : entry.value)
            }
          </span>
        </div>
      ))}
    </div>
  );
}
