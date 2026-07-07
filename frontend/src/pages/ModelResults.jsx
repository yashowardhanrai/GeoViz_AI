import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, ReferenceLine, ZAxis
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import DataTable from '../components/ui/DataTable';
import LoadingState from '../components/ui/LoadingState';
import Tabs from '../components/ui/Tabs';
import CustomTooltip from '../components/ui/CustomTooltip';
import AlertBanner from '../components/ui/AlertBanner';

export default function ModelResults() {
  const { data: shapReg, loading: l1 } = useGeoData('shapRegression');
  const { data: shapClf, loading: l2 } = useGeoData('shapClassifier');
  const { data: preds, loading: l3 } = useGeoData('predictions');
  const { data: oofData, loading: l4 } = useGeoData('oof');
  const { data: mainData, loading: l5 } = useGeoData('data');

  if (l1 || l2 || l3 || l4 || l5) return <LoadingState message="Loading model results..." />;

  // Regression comparison grouped bar data
  const regressionComparison = [
    { horizon: '1-day', 'XGBoost v6.3': 0.730, 'ARIMA (2,1,1)': 0.844, 'AR(14)-Ridge': 0.789, 'Naive': 0.778 },
    { horizon: '3-day', 'XGBoost v6.3': 0.750, 'ARIMA (2,1,1)': 0.823, 'AR(14)-Ridge': 0.829, 'Naive': 0.816 },
    { horizon: '7-day', 'XGBoost v6.3': 0.554, 'ARIMA (2,1,1)': 0.709, 'AR(14)-Ridge': 0.721, 'Naive': 0.682 },
  ];

  // SHAP data
  const regTop15 = shapReg ? [...shapReg].sort((a, b) => b.Importance - a.Importance).slice(0, 15).reverse() : [];
  const clfTop15 = shapClf ? [...shapClf].sort((a, b) => b.Importance - a.Importance).slice(0, 15).reverse() : [];

  // Ablation study data
  const ablation = [
    { Stage: 'S0: Pre-audit (v5)', '7d_R2': -0.089 },
    { Stage: 'S1: +circular', '7d_R2': -0.562 },
    { Stage: 'S2: −circular', '7d_R2': -0.562 },
    { Stage: 'S3: clean rank', '7d_R2': -0.550 },
    { Stage: 'S4: +bfill +PCA', '7d_R2': 0.499 },
    { Stage: 'S5: per-horizon', '7d_R2': 0.527 },
    { Stage: 'v6.3 final', '7d_R2': 0.810 },
  ];

  const ablationTable = [
    { Stage: 'S0: Pre-audit (v5)', Fix: 'Baseline — all leakage present', '1d': 0.621, '3d': 0.670, '7d': -0.089 },
    { Stage: 'S1: +circular +leaky rank', Fix: 'Add circular features back', '1d': -0.233, '3d': -0.108, '7d': -0.562 },
    { Stage: 'S2: −circular +leaky rank', Fix: 'Remove oil_shock/oil_momentum', '1d': -0.233, '3d': -0.108, '7d': -0.562 },
    { Stage: 'S3: −circular, clean rank', Fix: 'Fix expanding rank (risk_percentile)', '1d': -0.250, '3d': -0.101, '7d': -0.550 },
    { Stage: 'S4: +bfill fix +PCA fix', Fix: 'Fix bfill→ffill, train-only PCA', '1d': 0.736, '3d': 0.755, '7d': 0.499 },
    { Stage: 'S5: per-horizon tuning', Fix: 'Separate Optuna per horizon', '1d': 0.741, '3d': 0.740, '7d': 0.527 },
    { Stage: 'v6.3 final', Fix: 'STRING_AGG + is_monday', '1d': 0.896, '3d': 0.875, '7d': 0.810 },
  ];

  // OOF split
  const alertPoints = oofData ? oofData.filter(r => r.pred_label === 1) : [];
  const noAlertPoints = oofData ? oofData.filter(r => r.pred_label === 0) : [];

  const tabs = [
    {
      label: 'Overview',
      content: (
        <div>
          <h3 className="section-header">Key Metrics</h3>
          <div className="grid-2">
            <div>
              <h4 className="mb-md" style={{ color: 'var(--accent-cyan)' }}>Regression Model</h4>
              <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                <KPICard label="1-Day Test R²" value="0.730" delta={-0.048} />
                <KPICard label="3-Day Test R²" value="0.750" delta={-0.066} />
                <KPICard label="7-Day Test R²" value="0.554" delta={-0.128} />
              </div>
              <p className="text-muted" style={{ fontSize: '0.75rem' }}>
                None of the three horizons beat the naive persistence baseline. XGBoost's multi-feature
                capacity adds value in the classifier, not the regression.
              </p>
            </div>
            <div>
              <h4 className="mb-md" style={{ color: 'var(--accent-green)' }}>Regime Change Classifier</h4>
              <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                <KPICard label="AUC" value="0.815" color="#00ff88" />
                <KPICard label="F1 @ 0.15" value="0.620" color="#ff9500" />
                <KPICard label="Brier cal" value="0.060" color="#00b4d8" />
              </div>
              <p className="text-muted" style={{ fontSize: '0.75rem' }}>
                OOF AUC 0.8106 beats the trend heuristic (AUC ~0.48) by 0.33 points.
              </p>
            </div>
          </div>

          <div className="divider" />

          <ChartCard title="Regression Baseline Comparison — Test R²">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={regressionComparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="horizon" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 11 }} />
                <YAxis domain={[0.4, 0.95]} stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                <Bar dataKey="XGBoost v6.3" fill="#00b4d8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="ARIMA (2,1,1)" fill="#00ff88" radius={[4, 4, 0, 0]} />
                <Bar dataKey="AR(14)-Ridge" fill="rgba(255,255,255,0.3)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Naive" fill="rgba(255,255,255,0.12)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
              This grouped bar chart compares the test R² performance of XGBoost v6.3 against ARIMA, AR(14)-Ridge, and Naive baselines across 1-day, 3-day, and 7-day horizons.
            </p>
          </ChartCard>

          <AlertBanner type="info" icon="📌">
            <strong>is_monday finding:</strong> Replacing the raw dayofweek ordinal with a binary is_monday flag
            caused its SHAP rank to drop from #1/47 to #47/47. This confirms dayofweek was capturing the
            oil market microstructure artifact, not conflict seasonality.
          </AlertBanner>
        </div>
      ),
    },
    {
      label: 'Regression',
      content: (
        <div>
          <h3 className="section-header">Test Set Predictions — 3-Day Horizon</h3>
          {preds && preds.length > 0 && (
            <ChartCard title="3-Day Ahead Risk Score — Test Set">
              <ResponsiveContainer width="100%" height={380}>
                <LineChart data={preds}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5) || ''} stroke="rgba(255,255,255,0.15)" />
                  <YAxis stroke="rgba(255,255,255,0.15)" />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                  <Line type="monotone" dataKey="Actual" stroke="#ff3366" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Predicted" stroke="#00b4d8" strokeWidth={2} strokeDasharray="6 3" dot={false} />
                  <Line type="monotone" dataKey="Naive_Baseline" stroke="rgba(255,255,255,0.3)" strokeWidth={1} strokeDasharray="3 3" dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
                This line chart plots actual risk scores against model predictions on the held-out test set for the 3-day forecasting horizon.
              </p>
            </ChartCard>
          )}
        </div>
      ),
    },
    {
      label: 'Classifier',
      content: (
        <div>
          <h3 className="section-header">Classifier Performance — OOF Evaluation</h3>
          <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
            <KPICard label="AUC" value="0.8106" color="#00ff88" />
            <KPICard label="95% CI" value="[0.74, 0.88]" />
            <KPICard label="AP" value="0.511" />
            <KPICard label="F1" value="0.620" color="#ff9500" />
            <KPICard label="Brier raw" value="0.0807" />
            <KPICard label="Brier cal" value="0.0633" color="#00b4d8" />
          </div>

          <div className="divider" />

          <div className="grid-2">
            <div>
              <h4 className="section-header">Confusion Matrix (OOF, calibrated)</h4>
              <DataTable
                columns={[
                  { key: 'actual', label: '' },
                  { key: 'pred_no', label: 'Pred: No Alert' },
                  { key: 'pred_alert', label: 'Pred: Alert' },
                ]}
                rows={[
                  { actual: 'Actual: No Change', pred_no: '459', pred_alert: '22' },
                  { actual: 'Actual: Change', pred_no: '21', pred_alert: '35' },
                ]}
              />
              <p className="text-muted mt-sm" style={{ fontSize: '0.75rem' }}>
                35 true alerts · 21 missed · 22 false alarms · 459 correct negatives
              </p>
            </div>

            <ChartCard title="Brier Score — Before & After Calibration">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={[
                  { label: 'Raw', value: 0.0807 },
                  { label: 'Calibrated', value: 0.0633 },
                ]}>
                  <XAxis dataKey="label" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 0.14]} stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {/* Colors handled inline */}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-muted mt-sm" style={{ fontSize: '0.75rem', lineHeight: '1.3' }}>
                This bar chart compares the Brier score of the regime change classifier before and after isotonic calibration, highlighting a 22% improvement.
              </p>
            </ChartCard>
          </div>

          <div className="divider" />

          {oofData && oofData.length > 0 && (
            <ChartCard title="Alert History — OOF Predictions">
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="date" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 9 }} />
                  <YAxis dataKey="pred_cal" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
                  <ZAxis range={[20, 20]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                  <ReferenceLine y={0.15} stroke="#ff3366" strokeDasharray="6 3" />
                  <Scatter name="Alert Fired" data={alertPoints} fill="#ff3366" opacity={0.7} />
                  <Scatter name="No Alert" data={noAlertPoints} fill="rgba(255,255,255,0.15)" opacity={0.5} />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
                This scatter plot shows the out-of-fold calibrated probabilities and corresponding alert triggers across the timeline.
              </p>
            </ChartCard>
          )}
        </div>
      ),
    },
    {
      label: 'Explainability',
      content: (
        <div className="grid-2">
          <div>
            <h4 className="section-header">Regression SHAP (3-day)</h4>
            {regTop15.length > 0 && (
              <ChartCard>
                <ResponsiveContainer width="100%" height={450}>
                  <BarChart data={regTop15} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis type="number" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
                    <YAxis type="category" dataKey="Feature" width={130} tick={{ fontSize: 9, fill: '#9ca3af' }} stroke="rgba(255,255,255,0.15)" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="Importance" fill="#00b4d8" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
                  This bar chart lists the top 15 features by mean absolute SHAP value for the 3-day ahead regression model.
                </p>
              </ChartCard>
            )}
          </div>
          <div>
            <h4 className="section-header">Classifier SHAP (7-day)</h4>
            {clfTop15.length > 0 && (
              <ChartCard>
                <ResponsiveContainer width="100%" height={450}>
                  <BarChart data={clfTop15} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis type="number" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
                    <YAxis type="category" dataKey="Feature" width={130} tick={{ fontSize: 9, fill: '#9ca3af' }} stroke="rgba(255,255,255,0.15)" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="Importance" fill="#00ff88" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
                  This bar chart lists the top 15 features by mean absolute SHAP value for the 7-day regime change classifier.
                </p>
              </ChartCard>
            )}
          </div>
        </div>
      ),
    },
    {
      label: 'Ablation Study',
      content: (
        <div>
          <h3 className="section-header">Leakage Audit — Ablation Table</h3>
          <p className="text-muted mb-md" style={{ fontSize: '0.8rem' }}>
            Each stage applies one fix to the pipeline. Fixed hyperparameters held constant across S1–S3.
          </p>

          <DataTable
            columns={[
              { key: 'Stage', label: 'Stage' },
              { key: 'Fix', label: 'Fix Applied' },
              { key: '1d', label: '1-Day R²' },
              { key: '3d', label: '3-Day R²' },
              { key: '7d', label: '7-Day R²' },
            ]}
            rows={ablationTable}
          />

          <div className="divider" />

          <ChartCard title="7-Day R² Progression">
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={ablation}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="Stage" tick={{ fontSize: 9, angle: -20 }} stroke="rgba(255,255,255,0.15)" height={60} />
                <YAxis stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0.886} stroke="rgba(255,255,255,0.3)" strokeDasharray="6 3" label={{ value: 'Naive (0.886)', fill: '#9ca3af', fontSize: 10 }} />
                <ReferenceLine y={0} stroke="#ff3366" strokeDasharray="3 3" label={{ value: 'R²=0', fill: '#ff3366', fontSize: 10 }} />
                <Line type="monotone" dataKey="7d_R2" name="7-Day R²" stroke="#00b4d8" strokeWidth={2} dot={{ r: 5, fill: '#00b4d8' }} />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
              This progression line chart tracks the test R² improvement of the models across successive ablation stages and pipeline audits.
            </p>
          </ChartCard>
        </div>
      ),
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">📋</span> Model Results</h1>
        <p className="page-subtitle">
          Full research results for the GeoVizAI paper. XGBoost v6.3 · Regime Classifier v3 · leakage audit · bootstrap CIs.
        </p>
      </div>

      <Tabs tabs={tabs} />
    </div>
  );
}
