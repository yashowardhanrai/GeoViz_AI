import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import DataTable from '../components/ui/DataTable';
import LoadingState from '../components/ui/LoadingState';
import CustomTooltip from '../components/ui/CustomTooltip';

export default function Forecasting() {
  const { data: mainData, loading: l1 } = useGeoData('data');
  const { data: preds, loading: l2 } = useGeoData('predictions');

  if (l1 || l2) return <LoadingState message="Loading forecast data..." />;
  if (!mainData) return null;

  const step = Math.max(1, Math.floor(mainData.length / 500));
  const chartData = mainData.filter((_, i) => i % step === 0 || i === mainData.length - 1);

  // Baseline comparison data
  const baselineData = [
    { Model: 'Naive',         '1d_R2': 0.778, '3d_R2': 0.816, '7d_R2': 0.682, '1d_MAE': 1.002, '3d_MAE': 0.971, '7d_MAE': 1.172 },
    { Model: 'AR(14)-Ridge',  '1d_R2': 0.789, '3d_R2': 0.829, '7d_R2': 0.721, '1d_MAE': 0.986, '3d_MAE': 0.930, '7d_MAE': 1.100 },
    { Model: 'ARIMA (2,1,1)', '1d_R2': 0.844, '3d_R2': 0.823, '7d_R2': 0.709, '1d_MAE': 0.834, '3d_MAE': 0.897, '7d_MAE': 1.067 },
    { Model: 'XGBoost v6.3',  '1d_R2': 0.730, '3d_R2': 0.750, '7d_R2': 0.554, '1d_MAE': 1.144, '3d_MAE': 1.151, '7d_MAE': 1.478 },
  ];

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">🔮</span> Risk Forecasting</h1>
      </div>

      {/* XGBoost predictions */}
      <h3 className="section-header">XGBoost v6.3 — Test Set Predictions (3-day horizon)</h3>
      <p className="text-muted mb-md" style={{ fontSize: '0.8rem' }}>
        Test R² = 0.750 · MAE = 1.151 · Naive R² = 0.816 · 209 test rows (last 20% of timeline)
      </p>

      {preds && preds.length > 0 && (
        <ChartCard title="3-Day Ahead Risk Score — Test Set">
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={preds}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
              <YAxis stroke="rgba(255,255,255,0.15)" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
              <Line type="monotone" dataKey="Actual" name="Actual" stroke="#ff3366" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Predicted" name="XGBoost" stroke="#00b4d8" strokeWidth={2} strokeDasharray="6 3" dot={false} />
              <Line type="monotone" dataKey="Naive_Baseline" name="Naive" stroke="rgba(255,255,255,0.3)" strokeWidth={1} strokeDasharray="3 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
            This chart compares actual out-of-sample risk scores against predictions from the XGBoost v6.3 model and the naive baseline for a 3-day forecasting horizon.
          </p>
        </ChartCard>
      )}

      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <KPICard label="1-Day Test R²" value="0.730" delta={-0.048} color="#00b4d8" />
        <KPICard label="3-Day Test R²" value="0.750" delta={-0.066} color="#ff9500" />
        <KPICard label="7-Day Test R²" value="0.554" delta={-0.128} color="#ff3366" />
      </div>

      <div className="divider" />

      {/* Risk Score History */}
      <ChartCard title="Risk Score with Rolling Averages">
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
            <YAxis stroke="rgba(255,255,255,0.15)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="risk_score" name="Risk Score" stroke="#ff3366" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="risk_7d" name="7D" stroke="#00b4d8" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="risk_30d" name="30D" stroke="#ff9500" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="risk_90d" name="90D" stroke="#00ff88" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This historical overview chart shows the long-term risk score fluctuations alongside its rolling moving averages (7-day, 30-day, and 90-day).
        </p>
      </ChartCard>

      <div className="divider" />

      {/* Baseline Comparison */}
      <h3 className="section-header">Baseline Comparison — All Horizons</h3>
      <p className="text-muted mb-md" style={{ fontSize: '0.8rem' }}>
        ARIMA and AR(14)-Ridge exploit autocorrelation directly. XGBoost adds conflict, oil, and news signal.
        None beat naive on regression — the classifier is where multi-feature XGBoost adds unique value.
      </p>

      <DataTable
        columns={[
          { key: 'Model', label: 'Model' },
          { key: '1d_R2', label: '1-Day R²' },
          { key: '3d_R2', label: '3-Day R²' },
          { key: '7d_R2', label: '7-Day R²' },
          { key: '1d_MAE', label: '1-Day MAE' },
          { key: '3d_MAE', label: '3-Day MAE' },
          { key: '7d_MAE', label: '7-Day MAE' },
        ]}
        rows={baselineData}
      />
    </div>
  );
}
