import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/',                icon: '🌍', label: 'Overview' },
  { path: '/risk-monitor',    icon: '🚨', label: 'Risk Monitor' },
  { path: '/market-analytics',icon: '📈', label: 'Market Analytics' },
  { path: '/macro-markets',   icon: '🏦', label: 'Macro Markets' },
  { path: '/conflict',        icon: '⚔️', label: 'Conflict Analytics' },
  { path: '/media',           icon: '📰', label: 'Media Intelligence' },
  { path: '/correlation',     icon: '📊', label: 'Correlation Heatmap' },
  { path: '/forecasting',     icon: '🔮', label: 'Forecasting' },
  { path: '/explainability',  icon: '🧠', label: 'Explainability' },
  { path: '/ai-insights',     icon: '🤖', label: 'AI Insights' },
  { path: '/model-results',   icon: '📋', label: 'Model Results' },
];

export default function Sidebar({ isOpen, onClose }) {
  return (
    <>
      <div 
        className={`sidebar-overlay${isOpen ? ' active' : ''}`} 
        onClick={onClose} 
      />

      <aside className={`sidebar${isOpen ? ' open' : ''}`}>
        <button 
          className="sidebar-close-btn" 
          onClick={onClose} 
          aria-label="Close menu"
        >
          ✕
        </button>

        <div className="sidebar-logo">
          <span className="icon">🌍</span>
          <span className="gradient-text">GeoVizAI</span>
        </div>
        <div className="sidebar-subtitle">
          AI-powered geopolitical risk intelligence
          <br />
          Ukraine · Russia · Israel · Palestine · Iran
        </div>

        <div className="sidebar-divider" />

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(({ path, icon, label }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              className={({ isActive }) =>
                `sidebar-link${isActive ? ' active' : ''}`
              }
              onClick={onClose}
            >
              <span className="link-icon">{icon}</span>
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-divider" />
          <div className="sidebar-footer-text">
            XGBoost v6.3 · Classifier v3
            <br />
            1,075 rows · Feb 2022 – Jun 2025
            <br />
            AUC 0.815 [0.739, 0.878]
          </div>
        </div>
      </aside>
    </>
  );
}
