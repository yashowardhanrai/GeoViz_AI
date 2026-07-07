import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import LoadingState from '../components/ui/LoadingState';
import CustomTooltip from '../components/ui/CustomTooltip';
import GlobeScene from '../components/three/GlobeScene';
import { riskColor, CHART_COLORS } from '../utils/colors';

export default function Overview() {
  const { data, loading, error } = useGeoData('data');

  if (loading) return <LoadingState message="Loading geopolitical data..." />;
  if (error) return <div className="alert-banner alert-danger">Error: {error.message}</div>;
  if (!data || data.length === 0) return null;

  const latest = data[data.length - 1];
  const prev = data.length > 1 ? data[data.length - 2] : latest;

  // Downsample for chart performance
  const step = Math.max(1, Math.floor(data.length / 500));
  const chartData = data.filter((_, i) => i % step === 0 || i === data.length - 1);

  return (
    <div>
      <div className="page-header">
        <h1>
          <span className="page-icon">🌍</span>
          <span className="gradient-text">GeoVizAI</span>
        </h1>
        <p className="page-subtitle">
          AI-Powered Geopolitical Risk Intelligence Platform
        </p>
      </div>

      {/* 3D Globe */}
      <ChartCard>
        <GlobeScene height={380} />
      </ChartCard>

      {/* KPI Row */}
      <div className="kpi-grid">
        <KPICard
          label="Risk Score"
          value={latest.risk_score}
          delta={latest.risk_score - prev.risk_score}
          color={riskColor(latest.risk_score)}
        />
        <KPICard
          label="Risk Level"
          value={latest.risk_level || latest.regime || '—'}
          color={riskColor(latest.risk_score)}
        />
        <KPICard
          label="Market Stress"
          value={latest.market_stress}
          delta={latest.market_stress - prev.market_stress}
          color="#7b61ff"
        />
        <KPICard
          label="Crisis Index"
          value={latest.crisis_index}
          delta={latest.crisis_index - prev.crisis_index}
          color="#ff9500"
        />
      </div>

      <div className="divider" />

      {/* Risk Score vs Crisis Index */}
      <ChartCard title="Risk Score vs Crisis Index">
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10 }}
              tickFormatter={(v) => v?.slice(5) || ''}
              stroke="rgba(255,255,255,0.15)"
            />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line
              type="monotone" dataKey="risk_score" name="Risk Score"
              stroke="#ff3366" strokeWidth={2} dot={false}
            />
            <Line
              type="monotone" dataKey="crisis_index" name="Crisis Index"
              stroke="#ff9500" strokeWidth={2} dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This chart plots the daily geopolitical Risk Score (representing immediate volatilities and risk levels) against the Crisis Index (representing systemic stress and actual kinetic escalations).
        </p>
      </ChartCard>

      <div className="divider" />

      {/* Pressure Components */}
      <ChartCard title="Pressure Components">
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10 }}
              tickFormatter={(v) => v?.slice(5) || ''}
              stroke="rgba(255,255,255,0.15)"
            />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="global_pressure"   name="Global Pressure"   stroke={CHART_COLORS[0]} strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="market_stress"      name="Market Stress"      stroke={CHART_COLORS[1]} strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="conflict_pressure"  name="Conflict Pressure"  stroke={CHART_COLORS[3]} strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="media_pressure"     name="Media Pressure"     stroke={CHART_COLORS[4]} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This line chart tracks the individual sub-components of overall geopolitical pressure, separating kinetic conflict pressure, news media volume pressure, financial market stress, and global aggregate pressure.
        </p>
      </ChartCard>
    </div>
  );
}
