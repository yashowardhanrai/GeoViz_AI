import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/ui/Sidebar';
import ParticleField from './components/three/ParticleField';
import Overview from './pages/Overview';
import RiskMonitor from './pages/RiskMonitor';
import MarketAnalytics from './pages/MarketAnalytics';
import MacroMarkets from './pages/MacroMarkets';
import ConflictAnalytics from './pages/ConflictAnalytics';
import MediaIntelligence from './pages/MediaIntelligence';
import CorrelationHeatmap from './pages/CorrelationHeatmap';
import Forecasting from './pages/Forecasting';
import Explainability from './pages/Explainability';
import AIInsights from './pages/AIInsights';
import ModelResults from './pages/ModelResults';

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <BrowserRouter>
      <div className="app-layout">
        {/* Background effects */}
        <div className="bg-grid" />
        <div className="bg-radial-glow" />
        <div className="bg-radial-glow-2" />
        <ParticleField />

        {/* Mobile Header */}
        <header className="mobile-header">
          <div className="mobile-header-logo">
            <span className="icon">🌍</span>
            <span className="gradient-text">GeoVizAI</span>
          </div>
          <button 
            className="mobile-menu-btn" 
            onClick={() => setMenuOpen(true)}
            aria-label="Open menu"
          >
            ☰
          </button>
        </header>

        {/* Sidebar */}
        <Sidebar isOpen={menuOpen} onClose={() => setMenuOpen(false)} />

        {/* Main content */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/risk-monitor" element={<RiskMonitor />} />
            <Route path="/market-analytics" element={<MarketAnalytics />} />
            <Route path="/macro-markets" element={<MacroMarkets />} />
            <Route path="/conflict" element={<ConflictAnalytics />} />
            <Route path="/media" element={<MediaIntelligence />} />
            <Route path="/correlation" element={<CorrelationHeatmap />} />
            <Route path="/forecasting" element={<Forecasting />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/ai-insights" element={<AIInsights />} />
            <Route path="/model-results" element={<ModelResults />} />
          </Routes>
          <Footer />
        </main>
      </div>
    </BrowserRouter>
  );
}

function Footer() {
  return (
    <footer style={{
      marginTop: 'var(--space-2xl)',
      paddingTop: 'var(--space-md)',
      borderTop: '1px solid var(--border-subtle)',
      fontSize: '0.75rem',
      color: 'var(--text-muted)',
      fontFamily: 'Inter, sans-serif',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      flexWrap: 'wrap',
      gap: 'var(--space-md)'
    }}>

      <div>
        <strong>Contributors:</strong> Yashowardhan Rai, Amogh Gupta, Anmol Agarwal, Katyayani Tripathi, and Sanjay Kumar Sonbhadra
      </div>
    </footer>
  );
}
