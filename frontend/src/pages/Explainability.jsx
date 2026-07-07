import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import DataTable from '../components/ui/DataTable';
import LoadingState from '../components/ui/LoadingState';
import Tabs from '../components/ui/Tabs';
import CustomTooltip from '../components/ui/CustomTooltip';

export default function Explainability() {
  const { data: shapReg, loading: l1 } = useGeoData('shapRegression');
  const { data: shapClf, loading: l2 } = useGeoData('shapClassifier');

  if (l1 || l2) return <LoadingState message="Loading SHAP data..." />;

  const regTop20 = shapReg ? [...shapReg].sort((a, b) => b.Importance - a.Importance).slice(0, 20) : [];
  const clfTop15 = shapClf ? [...shapClf].sort((a, b) => b.Importance - a.Importance).slice(0, 15) : [];

  // Reverse for horizontal bar (bottom → top)
  const regChart = [...regTop20].reverse();
  const clfChart = [...clfTop15].reverse();

  const tabs = [
    {
      label: 'Regression SHAP (3-day)',
      content: (
        <div>
          <h3 className="section-header">Feature Importance — 3-Day Risk Score Forecast</h3>
          <p className="text-muted mb-md" style={{ fontSize: '0.8rem' }}>
            Mean |SHAP| across 836 training rows. is_monday ranked #47/47 (0.027), confirming
            dayofweek captured oil market microstructure, not conflict seasonality.
          </p>

          {regChart.length > 0 && (
            <ChartCard title="Top 20 SHAP Features — Regression Model">
              <ResponsiveContainer width="100%" height={550}>
                <BarChart data={regChart} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis type="number" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="Feature" width={140} tick={{ fontSize: 10, fill: '#9ca3af' }} stroke="rgba(255,255,255,0.15)" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="Importance" fill="#00b4d8" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
                This horizontal bar chart ranks the top 20 features by mean absolute SHAP value for the 3-day ahead regression model, indicating which indicators drive the forecast.
              </p>
            </ChartCard>
          )}

          <DataTable
            columns={[
              { key: 'Feature', label: 'Feature' },
              { key: 'Importance', label: 'Importance', format: (v) => Number(v).toFixed(4) },
            ]}
            rows={regTop20}
          />
        </div>
      ),
    },
    {
      label: 'Classifier SHAP (7-day)',
      content: (
        <div>
          <h3 className="section-header">Feature Importance — 7-Day Regime Change Classifier</h3>
          <p className="text-muted mb-md" style={{ fontSize: '0.8rem' }}>
            Mean |SHAP| from best OOF fold (fold 4, val AUC=0.873).
            events_30d_lag1 is the dominant predictor of regime transitions.
          </p>

          {clfChart.length > 0 && (
            <ChartCard title="Top 15 SHAP Features — Classifier">
              <ResponsiveContainer width="100%" height={480}>
                <BarChart data={clfChart} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis type="number" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="Feature" width={150} tick={{ fontSize: 10, fill: '#9ca3af' }} stroke="rgba(255,255,255,0.15)" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="Importance" fill="#00ff88" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
                This chart ranks the top 15 features by mean absolute SHAP value for the 7-day regime change classifier, showing which predictors trigger high-risk alerts.
              </p>
            </ChartCard>
          )}

          <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <KPICard label="Classifier AUC" value="0.815" color="#00ff88" />
            <KPICard label="95% CI" value="[0.739, 0.878]" />
            <KPICard label="F1 @ threshold 0.15" value="0.620" color="#ff9500" />
            <KPICard label="Brier (calibrated)" value="0.060" color="#00b4d8" />
          </div>
        </div>
      ),
    },
    {
      label: 'Baseline Comparison',
      content: (
        <div>
          <h3 className="section-header">Model vs Baseline Comparison</h3>

          <h4 className="mb-md">Regression — Test R² by Horizon</h4>
          <DataTable
            columns={[
              { key: 'model', label: 'Model' },
              { key: 'r2', label: 'R²' },
              { key: 'mae', label: 'MAE' },
            ]}
            rows={[
              { model: 'Naive', r2: '0.778 / 0.816 / 0.682', mae: '1.002 / 0.971 / 1.172' },
              { model: 'AR(14)-Ridge', r2: '0.789 / 0.829 / 0.721', mae: '0.986 / 0.930 / 1.100' },
              { model: 'ARIMA (2,1,1)', r2: '0.844 / 0.823 / 0.709', mae: '0.834 / 0.897 / 1.067' },
              { model: 'XGBoost v6.3', r2: '0.730 / 0.750 / 0.554', mae: '1.144 / 1.151 / 1.478' },
            ]}
          />

          <p className="text-muted mt-md mb-lg" style={{ fontSize: '0.8rem' }}>
            <strong>Note:</strong> ARIMA/AR(14) exploit autocorrelation directly. XGBoost adds conflict, oil, and news signal.
            The regime classifier is where multi-feature capacity adds unique value.
          </p>
        </div>
      ),
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">🧠</span> Model Explainability</h1>
        <p className="page-subtitle">
          SHAP values from trained XGBoost v6.3 models. Loaded from saved artefacts — no retraining on this page.
        </p>
      </div>

      <Tabs tabs={tabs} />
    </div>
  );
}
