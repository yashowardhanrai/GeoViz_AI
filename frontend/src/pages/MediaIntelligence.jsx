import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import ChartCard from '../components/ui/ChartCard';
import LoadingState from '../components/ui/LoadingState';
import CustomTooltip from '../components/ui/CustomTooltip';
import { CHART_COLORS } from '../utils/colors';

export default function MediaIntelligence() {
  const { data, loading, error } = useGeoData('data');

  if (loading) return <LoadingState message="Loading media data..." />;
  if (error) return <div className="alert-banner alert-danger">Error: {error.message}</div>;
  if (!data || data.length === 0) return null;

  const step = Math.max(1, Math.floor(data.length / 500));
  const chartData = data.filter((_, i) => i % step === 0 || i === data.length - 1);

  const hasNewsChange = data.some(r => r.news_change != null);
  const hasToneChange = data.some(r => r.tone_change != null);
  const hasTone30d = data.some(r => r.tone_30d != null);

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">📰</span> Media Intelligence</h1>
      </div>

      <ChartCard title="Daily News Volume">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="article_count" name="Article Count" stroke={CHART_COLORS[0]} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This line chart monitors daily news volume (raw article counts) matching geopolitical conflict zones, showing global media focus.
        </p>
      </ChartCard>

      <ChartCard title="Normalised Media Attention">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="article_count_norm" name="Normalised Attention" stroke={CHART_COLORS[1]} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This chart represents normalized media attention, scaling raw article counts relative to baseline news traffic to isolate anomalies.
        </p>
      </ChartCard>

      <ChartCard title="News Sentiment (GDELT avg tone)">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="avg_tone" name="Avg Tone" stroke={CHART_COLORS[2]} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This chart tracks the average tone of GDELT news coverage; more negative values correspond to tense or hostile reporting.
        </p>
      </ChartCard>

      {hasNewsChange && (
        <ChartCard title="Daily Change in News Volume">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
              <YAxis stroke="rgba(255,255,255,0.15)" />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="news_change" name="News Change" stroke={CHART_COLORS[4]} strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
            This chart logs daily absolute changes in news volume, helping identify sudden escalations or news shocks.
          </p>
        </ChartCard>
      )}

      {hasToneChange && (
        <ChartCard title="Daily Change in News Tone">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
              <YAxis stroke="rgba(255,255,255,0.15)" />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="tone_change" name="Tone Change" stroke={CHART_COLORS[3]} strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
            This line chart monitors daily shifts in average sentiment tone to detect rapid changes in news coverage bias or narrative stress.
          </p>
        </ChartCard>
      )}

      {hasTone30d && (
        <ChartCard title="30-Day Rolling Average Tone">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
              <YAxis stroke="rgba(255,255,255,0.15)" />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="tone_30d" name="30D Tone" stroke={CHART_COLORS[5]} strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
            This chart tracks the 30-day moving average of news tone to smooth out daily volatility and view longer-term media sentiment.
          </p>
        </ChartCard>
      )}

      <ChartCard title="Media Pressure Index">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="media_pressure" name="Media Pressure" stroke="#ff3366" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This index measures total Media Pressure by integrating news volume, change velocity, tone, and attention spikes into a single pressure rating.
        </p>
      </ChartCard>
    </div>
  );
}
