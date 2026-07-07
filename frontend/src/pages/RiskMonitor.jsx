import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import LoadingState from '../components/ui/LoadingState';
import CustomTooltip from '../components/ui/CustomTooltip';
import RiskGauge from '../components/ui/RiskGauge';
import { riskColor } from '../utils/colors';

export default function RiskMonitor() {
  const { data, loading, error } = useGeoData('data');

  if (loading) return <LoadingState message="Loading risk data..." />;
  if (error) return <div className="alert-banner alert-danger">Error: {error.message}</div>;
  if (!data || data.length === 0) return null;

  const latest = data[data.length - 1];
  const prev = data.length > 1 ? data[data.length - 2] : latest;
  const step = Math.max(1, Math.floor(data.length / 500));
  const chartData = data.filter((_, i) => i % step === 0 || i === data.length - 1);

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">🚨</span> Risk Monitor</h1>
      </div>

      {/* Risk Gauge */}
      <ChartCard>
        <RiskGauge value={latest.risk_score} />
      </ChartCard>

      {/* KPI Row */}
      <div className="kpi-grid">
        <KPICard label="Risk Score" value={latest.risk_score} delta={latest.risk_score - prev.risk_score} color={riskColor(latest.risk_score)} />
        <KPICard label="Risk Level" value={latest.risk_level || latest.regime || '—'} color={riskColor(latest.risk_score)} />
        <KPICard label="30D Risk" value={latest.risk_30d} delta={latest.risk_30d - prev.risk_30d} color="#ff9500" />
        <KPICard label="90D Risk" value={latest.risk_90d} delta={latest.risk_90d - prev.risk_90d} color="#00b4d8" />
      </div>

      <div className="divider" />

      {/* Risk Trend */}
      <ChartCard title="Risk Trend">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="risk_score" name="Risk Score" stroke="#ff3366" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="risk_7d" name="7D" stroke="#00b4d8" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="risk_30d" name="30D" stroke="#ff9500" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="risk_90d" name="90D" stroke="#00ff88" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This line chart monitors the daily raw risk score alongside its 7-day, 30-day, and 90-day rolling moving averages to visualize short-term and medium-term geopolitical risk trends.
        </p>
      </ChartCard>

      <div className="divider" />

      {/* Momentum */}
      <h3 className="section-header">Risk Momentum</h3>
      <div className="grid-2">
        <ChartCard title="Momentum vs Acceleration">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
              <YAxis stroke="rgba(255,255,255,0.15)" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
              <Line type="monotone" dataKey="risk_momentum" name="Momentum" stroke="#00b4d8" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="risk_acceleration" name="Acceleration" stroke="#ff9500" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
            This chart tracks the rate of change (first derivative/momentum) and the speed of change (second derivative/acceleration) of geopolitical risk to detect early indicators of systemic transitions.
          </p>
        </ChartCard>

        <ChartCard title="Risk Regime (numeric)">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
              <YAxis stroke="rgba(255,255,255,0.15)" />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="risk_regime" name="Risk Regime" stroke="#00b4d8" fill="rgba(0,180,216,0.15)" strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
          <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
            This area chart displays the numerical regime level (Stable=0, Elevated=1, High=2, Crisis=3) computed based on thresholded historical risk score dynamics.
          </p>
        </ChartCard>
      </div>

      <div className="divider" />

      {/* Crisis Indicators */}
      <h3 className="section-header">Crisis Indicators</h3>
      <ChartCard title="Crisis Index vs Global Pressure">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="crisis_index" name="Crisis Index" stroke="#ff3366" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="global_pressure" name="Global Pressure" stroke="#ff9500" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This overlay chart compares the systemic Crisis Index (representing the density of high-fatality escalations) against Global Pressure to map long-term structural stress levels.
        </p>
      </ChartCard>
    </div>
  );
}
