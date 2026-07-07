import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import LoadingState from '../components/ui/LoadingState';
import CustomTooltip from '../components/ui/CustomTooltip';
import { CHART_COLORS } from '../utils/colors';

export default function ConflictAnalytics() {
  const { data, loading, error } = useGeoData('data');

  if (loading) return <LoadingState message="Loading conflict data..." />;
  if (error) return <div className="alert-banner alert-danger">Error: {error.message}</div>;
  if (!data || data.length === 0) return null;

  const latest = data[data.length - 1];
  const prev = data.length > 1 ? data[data.length - 2] : latest;
  const step = Math.max(1, Math.floor(data.length / 500));
  const chartData = data.filter((_, i) => i % step === 0 || i === data.length - 1);

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">⚔️</span> Conflict Analytics</h1>
      </div>

      <div className="kpi-grid">
        <KPICard label="Fatalities" value={Math.round(latest.fatalities)} delta={Math.round(latest.fatalities - prev.fatalities)} color="#ff3366" />
        <KPICard label="Event Count" value={Math.round(latest.event_count)} delta={Math.round(latest.event_count - prev.event_count)} color="#ff9500" />
        <KPICard label="Severity" value={latest.severity} delta={latest.severity - prev.severity} color="#7b61ff" />
        <KPICard label="Conflict Pressure" value={latest.conflict_pressure} delta={latest.conflict_pressure - prev.conflict_pressure} color="#00b4d8" />
      </div>

      <div className="divider" />

      <ChartCard title="Fatalities Trend">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="fatalities" name="Fatalities" stroke="#ff3366" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="fatalities_30d" name="30D Avg" stroke="#ff9500" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="fatalities_90d" name="90D Avg" stroke="#00ff88" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This line chart monitors daily kinetic fatalities alongside 30-day and 90-day moving averages to track conflict severity over time.
        </p>
      </ChartCard>

      <ChartCard title="Conflict Severity">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="severity" name="Severity" stroke="#7b61ff" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="severity_7d" name="7D" stroke="#00b4d8" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="severity_30d" name="30D" stroke="#ff9500" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This chart measures daily conflict severity (a composite metric based on weapon types and tactical engagements) alongside short-term moving averages.
        </p>
      </ChartCard>

      <ChartCard title="Conflict Intensity">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="intensity" name="Intensity" stroke={CHART_COLORS[0]} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="intensity_7d" name="7D" stroke={CHART_COLORS[1]} strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="intensity_30d" name="30D" stroke={CHART_COLORS[4]} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This chart represents conflict intensity (kinetic density and geographical reach of battles) along with short-term trends.
        </p>
      </ChartCard>

      <ChartCard title="Daily Event Count">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="event_count" name="Event Count" stroke="#ff9500" fill="rgba(255,149,0,0.12)" strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This area chart logs the total daily quantity of military, political, and civil conflict events reported across the monitored regions.
        </p>
      </ChartCard>

      <ChartCard title="Conflict Pressure">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="conflict_pressure" name="Conflict Pressure" stroke="#00b4d8" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This chart displays the aggregate Conflict Pressure Index, combining fatalities, event counts, severity, and intensity to estimate localized kinetic load.
        </p>
      </ChartCard>
    </div>
  );
}
