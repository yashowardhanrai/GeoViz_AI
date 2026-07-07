import { useGeoData } from '../hooks/useGeoData';
import ChartCard from '../components/ui/ChartCard';
import DataTable from '../components/ui/DataTable';
import LoadingState from '../components/ui/LoadingState';
import { correlationColor, formatNumber } from '../utils/colors';

export default function CorrelationHeatmap() {
  const { data, loading, error } = useGeoData('correlation');

  if (loading) return <LoadingState message="Computing correlation matrix..." />;
  if (error) return <div className="alert-banner alert-danger">Error: {error.message}</div>;
  if (!data) return null;

  const { columns, matrix } = data;

  // Sorted correlation pairs with risk_score
  const riskIdx = columns.indexOf('risk_score');
  const corrPairs = columns
    .map((col, i) => ({
      feature: col,
      correlation: riskIdx >= 0 ? matrix[i][riskIdx] : null,
    }))
    .filter(p => p.feature !== 'risk_score')
    .sort((a, b) => (b.correlation || 0) - (a.correlation || 0));

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">📊</span> Feature Correlation Heatmap</h1>
      </div>

      <ChartCard title="Correlation Matrix — Key Features">
        <div className="heatmap-container">
          {/* Column labels */}
          <div style={{ display: 'flex', marginLeft: 100, marginBottom: 4 }}>
            {columns.map((col, j) => (
              <div key={j} className="heatmap-label-h" style={{ width: 64, minWidth: 64 }}>
                {col.replace(/_/g, ' ').replace('close', '').trim()}
              </div>
            ))}
          </div>

          {/* Rows */}
          {matrix.map((row, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
              {/* Row label */}
              <div style={{
                width: 100, minWidth: 100, textAlign: 'right', paddingRight: 8,
                fontSize: '0.7rem', color: 'var(--text-muted)', whiteSpace: 'nowrap',
                overflow: 'hidden', textOverflow: 'ellipsis',
              }}>
                {columns[i].replace(/_/g, ' ')}
              </div>

              {/* Cells */}
              {row.map((val, j) => (
                <div
                  key={j}
                  className="heatmap-cell"
                  style={{
                    backgroundColor: correlationColor(val),
                    color: Math.abs(val) > 0.5 ? '#111' : 'var(--text-primary)',
                    width: 64, minWidth: 64, height: 42,
                  }}
                  title={`${columns[i]} × ${columns[j]}: ${formatNumber(val, 3)}`}
                >
                  {formatNumber(val, 2)}
                </div>
              ))}
            </div>
          ))}
        </div>
        <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
          This heatmap displays the Pearson correlation coefficients between key geopolitical risk, macro asset, kinetic conflict, and news media variables. Blue indicates negative correlation, and red indicates positive correlation.
        </p>
      </ChartCard>

      <div className="divider" />

      <DataTable
        title="Strongest Correlations with Risk Score"
        columns={[
          { key: 'feature', label: 'Feature' },
          { key: 'correlation', label: 'Correlation', format: (v) => formatNumber(v, 3) },
        ]}
        rows={corrPairs}
      />
    </div>
  );
}
