export default function AlertBanner({ type = 'info', icon, children }) {
  const typeClass = {
    danger: 'alert-danger',
    success: 'alert-success',
    warning: 'alert-warning',
    info: 'alert-info',
  }[type] || 'alert-info';

  const defaultIcon = {
    danger: '🚨',
    success: '✅',
    warning: '⚠️',
    info: 'ℹ️',
  }[type] || 'ℹ️';

  return (
    <div className={`alert-banner ${typeClass}`}>
      <span className="alert-icon">{icon || defaultIcon}</span>
      <div>{children}</div>
    </div>
  );
}
