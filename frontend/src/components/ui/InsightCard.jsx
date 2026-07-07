export default function InsightCard({ icon, children }) {
  return (
    <div className="insight-card">
      <span className="insight-icon">{icon || 'ℹ️'}</span>
      <div>{children}</div>
    </div>
  );
}
