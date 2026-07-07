import {
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ZAxis
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import LoadingState from '../components/ui/LoadingState';
import CustomTooltip from '../components/ui/CustomTooltip';
import { ASSET_COLORS, REGIME_COLORS } from '../utils/colors';

export default function MarketAnalytics() {
  const { data, loading, error } = useGeoData('data');

  if (loading) return <LoadingState message="Loading market data..." />;
  if (error) return <div className="alert-banner alert-danger">Error: {error.message}</div>;
  if (!data || data.length === 0) return null;

  const latest = data[data.length - 1];
  const prev = data.length > 1 ? data[data.length - 2] : latest;
  const step = Math.max(1, Math.floor(data.length / 500));
  const chartData = data.filter((_, i) => i % step === 0 || i === data.length - 1);

  // Split data by regime for scatter
  const regimeGroups = {};
  data.forEach(row => {
    const r = row.regime || row.risk_level || 'Unknown';
    if (!regimeGroups[r]) regimeGroups[r] = [];
    regimeGroups[r].push(row);
  });

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">📈</span> Market Analytics</h1>
      </div>

      <div className="kpi-grid">
        <KPICard label="Oil Price (Brent)" value={latest.Close} prefix="$" delta={latest.Close - prev.Close} color={ASSET_COLORS.oil} />
        <KPICard label="Daily Return" value={latest.daily_return} suffix="%" delta={latest.daily_return - prev.daily_return} />
        <KPICard label="Volatility (30d)" value={latest.volatility_30} delta={latest.volatility_30 - prev.volatility_30} color="#7b61ff" />
        <KPICard label="Market Stress" value={latest.market_stress} delta={latest.market_stress - prev.market_stress} color="#ff3366" />
      </div>

      <div className="divider" />

      {/* Oil Price */}
      <ChartCard title="Brent Crude Oil Price (USD)">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="Close" name="Brent Crude" stroke={ASSET_COLORS.oil} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This line chart shows the daily closing price of Brent Crude oil in USD, reflecting energy market response to geopolitical events.
        </p>
      </ChartCard>

      {/* Multi-Asset */}
      <ChartCard title="Multi-Asset Price Trends">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="Close"       name="Oil"   stroke={ASSET_COLORS.oil}   strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="gold_close"   name="Gold"  stroke={ASSET_COLORS.gold}  strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="sp500_close"  name="S&P500" stroke={ASSET_COLORS.sp500} strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="dxy_close"    name="DXY"   stroke={ASSET_COLORS.dxy}   strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This multi-line chart compares normalized price trends across different macro assets (Oil, Gold, S&P 500, DXY index) to identify cross-market impacts.
        </p>
      </ChartCard>

      {/* Volatility */}
      <ChartCard title="30-Day Volatility by Asset">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="volatility_30"     name="Oil Vol"   stroke={ASSET_COLORS.oil}   strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="gold_volatility"    name="Gold Vol"  stroke={ASSET_COLORS.gold}  strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="sp500_volatility"   name="S&P Vol"   stroke={ASSET_COLORS.sp500} strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="dxy_volatility"     name="DXY Vol"   stroke={ASSET_COLORS.dxy}   strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This chart plots the 30-day rolling volatility for Brent Crude, Gold, S&P 500, and the US Dollar Index (DXY).
        </p>
      </ChartCard>

      {/* Market Stress */}
      <ChartCard title="Market Stress Index">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="market_stress" name="Market Stress" stroke="#ff3366" fill="rgba(255,51,102,0.12)" strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This area chart displays the Market Stress Index, a composite measure of financial and commodity market volatility.
        </p>
      </ChartCard>

      {/* Oil vs Risk Scatter */}
      <ChartCard title="Oil Price vs Risk Score by Regime">
        <ResponsiveContainer width="100%" height={340}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis type="number" dataKey="Close" name="Oil Price" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
            <YAxis type="number" dataKey="risk_score" name="Risk Score" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
            <ZAxis range={[20, 20]} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            {Object.entries(regimeGroups).map(([regime, points]) => (
              <Scatter
                key={regime}
                name={regime}
                data={points}
                fill={REGIME_COLORS[regime] || '#666'}
                opacity={0.6}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This scatter plot maps daily Brent Crude oil price against the geopolitical Risk Score, color-coded by the active risk regime, highlighting pricing differences under stable vs crisis environments.
        </p>
      </ChartCard>
    </div>
  );
}
