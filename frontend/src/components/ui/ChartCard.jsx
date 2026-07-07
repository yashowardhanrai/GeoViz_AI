export default function ChartCard({ title, subtitle, children, style }) {
  return (
    <div className="chart-card" style={style}>
      {title && <h4 className="chart-title">{title}</h4>}
      {subtitle && <p className="chart-subtitle">{subtitle}</p>}
      {children}
    </div>
  );
}
