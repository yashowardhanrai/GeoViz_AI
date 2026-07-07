import {
  LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ZAxis
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import DataTable from '../components/ui/DataTable';
import LoadingState from '../components/ui/LoadingState';
import CustomTooltip from '../components/ui/CustomTooltip';
import { ASSET_COLORS, REGIME_COLORS, formatNumber } from '../utils/colors';

export default function MacroMarkets() {
  const { data, loading, error } = useGeoData('data');

  if (loading) return <LoadingState message="Loading macro market data..." />;
  if (error) return <div className="alert-banner alert-danger">Error: {error.message}</div>;
  if (!data || data.length === 0) return null;

  const latest = data[data.length - 1];
  const prev = data.length > 1 ? data[data.length - 2] : latest;
  const step = Math.max(1, Math.floor(data.length / 500));
  const chartData = data.filter((_, i) => i % step === 0 || i === data.length - 1);

  // Correlation with risk_score
  const marketCols = ['Close', 'daily_return', 'volatility_30', 'gold_close', 'sp500_close', 'dxy_close', 'market_stress'];
  const corrRows = marketCols.map(col => {
    const vals = data.filter(r => r[col] != null && r.risk_score != null);
    if (vals.length < 10) return { feature: col, correlation: null };
    const n = vals.length;
    const sumX = vals.reduce((s, r) => s + r[col], 0);
    const sumY = vals.reduce((s, r) => s + r.risk_score, 0);
    const sumXY = vals.reduce((s, r) => s + r[col] * r.risk_score, 0);
    const sumX2 = vals.reduce((s, r) => s + r[col] ** 2, 0);
    const sumY2 = vals.reduce((s, r) => s + r.risk_score ** 2, 0);
    const num = n * sumXY - sumX * sumY;
    const den = Math.sqrt((n * sumX2 - sumX ** 2) * (n * sumY2 - sumY ** 2));
    return { feature: col, correlation: den === 0 ? 0 : num / den };
  }).sort((a, b) => (b.correlation || 0) - (a.correlation || 0));

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">🏦</span> Macro Markets</h1>
        <p className="page-subtitle">Cross-asset market indicators and their relationship to geopolitical risk.</p>
      </div>

      <div className="kpi-grid">
        <KPICard label="Oil Price (Brent)" value={latest.Close} prefix="$" delta={latest.Close - prev.Close} color={ASSET_COLORS.oil} />
        <KPICard label="Daily Return" value={latest.daily_return} suffix="%" delta={latest.daily_return - prev.daily_return} />
        <KPICard label="Volatility (30d)" value={latest.volatility_30} delta={latest.volatility_30 - prev.volatility_30} color="#7b61ff" />
        <KPICard label="Market Stress" value={latest.market_stress} delta={latest.market_stress - prev.market_stress} color="#ff3366" />
      </div>

      <div className="divider" />

      {/* Gold vs Oil vs Risk */}
      <ChartCard title="Gold vs Oil vs Risk Score" subtitle="Triple-axis overlay">
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis yAxisId="left" stroke="rgba(255,255,255,0.15)" />
            <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.1)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line yAxisId="left" type="monotone" dataKey="Close" name="Oil (Brent)" stroke={ASSET_COLORS.oil} strokeWidth={2} dot={false} />
            <Line yAxisId="left" type="monotone" dataKey="gold_close" name="Gold" stroke={ASSET_COLORS.gold} strokeWidth={2} dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="risk_score" name="Risk Score" stroke="#ff3366" strokeWidth={1.5} strokeDasharray="5 5" dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This triple-axis overlay chart displays the relationship between safe-haven Gold prices, Brent Crude oil prices, and the aggregate geopolitical Risk Score to identify risk-hedging behaviors.
        </p>
      </ChartCard>

      {/* S&P 500 and DXY */}
      <ChartCard title="S&P 500 and US Dollar Index">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis yAxisId="left" stroke="rgba(255,255,255,0.15)" />
            <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.1)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line yAxisId="left" type="monotone" dataKey="sp500_close" name="S&P 500" stroke={ASSET_COLORS.sp500} strokeWidth={1.5} dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="dxy_close" name="DXY" stroke={ASSET_COLORS.dxy} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This chart compares the equity index (S&P 500) and the US Dollar Index (DXY) to visualize shifts between risk-on asset classes and standard safe-haven currencies.
        </p>
      </ChartCard>

      {/* Market Stress vs Risk Scatter */}
      <ChartCard title="Market Stress vs Risk Score">
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis type="number" dataKey="market_stress" name="Market Stress" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
            <YAxis type="number" dataKey="risk_score" name="Risk Score" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
            <ZAxis range={[15, 15]} />
            <Tooltip content={<CustomTooltip />} />
            <Scatter name="All dates" data={data} fill="#00f0ff" opacity={0.4} />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This scatter plot visualizes the historical correlation between macro-market stress and the aggregate geopolitical Risk Score across all observed dates.
        </p>
      </ChartCard>

      <div className="divider" />

      {/* Correlation Table */}
      <DataTable
        title="Correlation with Risk Score"
        columns={[
          { key: 'feature', label: 'Feature' },
          { key: 'correlation', label: 'Correlation', format: (v) => formatNumber(v, 3) },
        ]}
        rows={corrRows}
      />
    </div>
  );
}
