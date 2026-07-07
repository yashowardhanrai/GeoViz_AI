import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { useGeoData } from '../hooks/useGeoData';
import KPICard from '../components/ui/KPICard';
import ChartCard from '../components/ui/ChartCard';
import AlertBanner from '../components/ui/AlertBanner';
import InsightCard from '../components/ui/InsightCard';
import LoadingState from '../components/ui/LoadingState';
import CustomTooltip from '../components/ui/CustomTooltip';

const AlertTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isAlert = data.pred_label === 1;
    return (
      <div style={{
        background: 'rgba(9, 15, 30, 0.95)',
        border: `1px solid ${isAlert ? '#FF3040' : '#64748B'}`,
        padding: '8px 12px',
        borderRadius: '6px',
        boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
        fontSize: '11px',
        color: '#E2E8F0',
        fontFamily: 'JetBrains Mono, monospace'
      }}>
        <div style={{ fontWeight: 'bold', color: '#FFFFFF', marginBottom: '4px' }}>
          DATE: {data.date}
        </div>
        <div>PROBABILITY: <span style={{ fontWeight: 'bold', color: isAlert ? '#FF3040' : '#E2E8F0' }}>{data.pred_cal?.toFixed(4) || '0.0000'}</span></div>
        <div style={{ marginTop: '2px' }}>
          STATUS: <span style={{
            color: isAlert ? '#FF3040' : '#64748B',
            fontWeight: 'bold'
          }}>{isAlert ? 'REGIME ALERT FIRED' : 'CLEAR'}</span>
        </div>
      </div>
    );
  }
  return null;
};

export default function AIInsights() {
  const { data: mainData, loading: l1 } = useGeoData('data');
  const { data: oofData, loading: l2 } = useGeoData('oof');

  if (l1 || l2) return <LoadingState message="Loading AI insights..." />;
  if (!mainData || mainData.length === 0) return null;

  const latest = mainData[mainData.length - 1];
  const riskScore = latest.risk_score;
  const regime = latest.regime || latest.risk_level || 'Unknown';

  // Classifier alert
  let prob = 0;
  let alertFired = false;
  const threshold = 0.15;

  if (oofData && oofData.length > 0) {
    const latestClf = oofData[oofData.length - 1];
    prob = latestClf.pred_cal || 0;
    alertFired = latestClf.pred_label === 1;
  }

  // Compute mean values for insight comparisons
  const mean = (key) => {
    const vals = mainData.filter(r => r[key] != null).map(r => r[key]);
    return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  };

  // Generate narrative insights
  const insights = [];
  if (latest.market_stress > mean('market_stress'))
    insights.push({ icon: '📈', text: 'Market stress is above historical average, indicating elevated financial volatility.' });
  if (Math.abs(latest.daily_return) > 2)
    insights.push({ icon: '🛢️', text: 'Large oil price movements are contributing to geopolitical uncertainty.' });
  if (latest.conflict_pressure > mean('conflict_pressure'))
    insights.push({ icon: '⚔️', text: 'Conflict pressure remains elevated due to persistent violence and intensity.' });
  if (latest.fatalities > mean('fatalities'))
    insights.push({ icon: '💀', text: 'Fatalities are above average levels, increasing systemic geopolitical risk.' });
  if (latest.media_pressure > mean('media_pressure'))
    insights.push({ icon: '📰', text: 'Media attention is unusually high, suggesting increasing global focus on conflict events.' });
  if (Math.abs(latest.avg_tone) > 3)
    insights.push({ icon: '📉', text: 'News sentiment is highly negative, reinforcing market anxiety.' });
  if (latest.shock_score > 50)
    insights.push({ icon: '⚡', text: 'Shock indicators suggest abnormal changes in conflict and news dynamics.' });
  if (latest.risk_momentum > 0)
    insights.push({ icon: '📊', text: 'Short-term risk momentum is increasing.' });
  else
    insights.push({ icon: '📉', text: 'Short-term risk momentum is easing.' });

  const regimeMsgs = {
    Stable:   { icon: '✅', text: 'Current environment remains relatively stable.' },
    Elevated: { icon: '🟡', text: 'Risk conditions are elevated and should be monitored.' },
    High:     { icon: '🟠', text: 'Geopolitical conditions are stressed and require close attention.' },
    Crisis:   { icon: '🔴', text: 'Crisis conditions detected. Extreme uncertainty persists.' },
  };
  if (regimeMsgs[regime]) insights.push(regimeMsgs[regime]);

  // Alert scatter — separate by pred_label
  const alertPoints = oofData ? oofData.filter(r => r.pred_label === 1) : [];
  const noAlertPoints = oofData ? oofData.filter(r => r.pred_label === 0) : [];

  return (
    <div>
      <div className="page-header">
        <h1><span className="page-icon">🤖</span> AI Insights</h1>
      </div>

      {/* Classifier Alert */}
      <h3 className="section-header">7-Day Regime Change Alert</h3>

      {alertFired ? (
        <AlertBanner type="danger" icon="🚨">
          <strong>Regime change alert active</strong> — classifier probability ({prob.toFixed(3)}) exceeds threshold ({threshold}).
          Escalation to a higher risk regime likely within 7 days.
        </AlertBanner>
      ) : (
        <AlertBanner type="success" icon="✅">
          No regime change alert — probability ({prob.toFixed(3)}) below threshold ({threshold}).
        </AlertBanner>
      )}

      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <KPICard label="Escalation Probability" value={prob.toFixed(3)} color={alertFired ? '#ff3366' : '#00ff88'} />
        <KPICard label="Alert Threshold" value={threshold.toString()} />
        <KPICard label="Alert Status" value={alertFired ? 'ACTIVE' : 'Clear'} color={alertFired ? '#ff3366' : '#00ff88'} />
      </div>

      <p className="text-muted mb-md" style={{ fontSize: '0.75rem' }}>
        XGBoost Regime Classifier v3 · OOF AUC 0.8106 [0.7403, 0.8780] · isotonic calibration · threshold optimised for F1
      </p>

      {/* Alert History Scatter */}
      {oofData && oofData.length > 0 && (() => {
        // Sort chronologically
        const sortedOofData = [...oofData].sort((a, b) => new Date(a.date) - new Date(b.date));

        // Display exactly 5 quarterly target ticks
        const targets = ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01', '2025-01-01'];
        const xTicks = targets.map(t => {
          let closest = sortedOofData[0].date;
          let minDiff = Infinity;
          const targetTime = new Date(t).getTime();
          sortedOofData.forEach(r => {
            const diff = Math.abs(new Date(r.date).getTime() - targetTime);
            if (diff < minDiff) {
              minDiff = diff;
              closest = r.date;
            }
          });
          return closest;
        });

        const alertPoints = sortedOofData.filter(r => r.pred_label === 1);
        const noAlertPoints = sortedOofData.filter(r => r.pred_label === 0);

        return (
          <ChartCard>
            <div style={{ position: 'relative', width: '100%' }}>
              {/* Header Title & Subtitle */}
              <div style={{ marginBottom: '20px', paddingRight: '220px' }}>
                <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 'bold', color: '#FFFFFF', letterSpacing: '0.02em' }}>
                  Classifier Alert History (Out-of-Fold Probabilities)
                </h3>
                <p style={{ margin: '3px 0 0 0', fontSize: '11px', color: '#64748B' }}>
                  Calibrated probabilities and alert triggers over time.
                </p>
              </div>

              {/* Minimalist Legend Row */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '16px',
                fontSize: '11px',
                color: '#64748B',
                fontFamily: 'JetBrains Mono, monospace',
                marginTop: '-10px',
                marginBottom: '15px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <span style={{ color: '#FF3040', fontSize: '12px' }}>●</span> Alert
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <span style={{ color: '#64748B', fontSize: '12px', opacity: 0.6 }}>●</span> No Alert
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <span style={{ color: '#FFC107', fontWeight: 'bold' }}>— —</span> Threshold
                </div>
              </div>

              {/* Statistics Panel (Floating Glassmorphic Panel in Top-Right) */}
              <div style={{
                position: 'absolute',
                top: '-4px',
                right: '4px',
                background: 'rgba(15, 23, 42, 0.55)',
                backdropFilter: 'blur(8px)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '8px',
                padding: '8px 12px',
                color: '#E2E8F0',
                fontSize: '10px',
                fontFamily: 'JetBrains Mono, monospace',
                lineHeight: '1.45',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.45)',
                zIndex: 2,
                pointerEvents: 'none'
              }}>
                <div style={{ color: '#64748B', fontWeight: '700', fontSize: '8.5px', marginBottom: '4px', letterSpacing: '0.08em' }}>CLASSIFIER METRICS</div>
                <div>OOF ROC-AUC: <span style={{ color: '#00D26A', fontWeight: 'bold' }}>0.8106</span></div>
                <div>95% CI: <span style={{ color: '#94A3B8' }}>[0.7403, 0.8780]</span></div>
                <div>Threshold: <span style={{ color: '#FFC107', fontWeight: 'bold' }}>0.15</span></div>
              </div>

              {/* Responsive Container */}
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 10, bottom: 5, left: -22 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                  
                  <XAxis 
                    dataKey="date" 
                    ticks={xTicks}
                    stroke="rgba(255,255,255,0.12)" 
                    tickLine={false}
                    tick={{ fontSize: 9.5, fill: '#64748B', fontFamily: 'JetBrains Mono, monospace' }} 
                    tickFormatter={(v) => {
                      const d = new Date(v);
                      const m = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getMonth()];
                      return `${m} ${d.getFullYear()}`;
                    }}
                  />
                  
                  <YAxis 
                    dataKey="pred_cal"
                    domain={[0.0, 0.6]} 
                    ticks={[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
                    stroke="rgba(255,255,255,0.12)" 
                    tickLine={false}
                    tick={{ fontSize: 9.5, fill: '#64748B', fontFamily: 'JetBrains Mono, monospace' }} 
                    tickFormatter={(v) => v.toFixed(1)}
                  />
                  
                  <Tooltip content={<AlertTooltip />} />

                  {/* Threshold Line (Amber, Dashed) */}
                  <ReferenceLine 
                    y={threshold} 
                    stroke="#FFC107" 
                    strokeDasharray="5 5" 
                    strokeWidth={1.5} 
                    label={{ 
                      value: `Threshold = ${threshold}`, 
                      fill: '#FFC107', 
                      position: 'top', 
                      fontSize: 9.5, 
                      fontWeight: '700',
                      fontFamily: 'JetBrains Mono, monospace'
                    }} 
                  />

                  {/* Standard No-Alert Scatter Points (Flat Gray) */}
                  <Scatter 
                    name="Clear" 
                    data={noAlertPoints} 
                    isAnimationActive={false}
                    shape={(props) => {
                      const { cx, cy } = props;
                      return <circle cx={cx} cy={cy} r="2.5" fill="#64748B" opacity="0.4" />;
                    }}
                  />

                  {/* Highlighted Alert Scatter Points (Flat Solid Red, No Glow) */}
                  <Scatter 
                    name="Regime Alert" 
                    data={alertPoints} 
                    isAnimationActive={false}
                    shape={(props) => {
                      const { cx, cy } = props;
                      return <circle cx={cx} cy={cy} r="3" fill="#FF3040" />;
                    }}
                  />

                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <p className="text-muted mt-sm" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
              This scatter plot displays the out-of-fold calibrated probabilities and corresponding alert triggers across the timeline.
            </p>
          </ChartCard>
        );
      })()}

      <div className="divider" />

      {/* Current Assessment */}
      <h3 className="section-header">Current Assessment</h3>
      <div className="kpi-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <KPICard label="Risk Score" value={riskScore} color={riskScore >= 70 ? '#ff3366' : riskScore >= 50 ? '#ff9500' : riskScore >= 25 ? '#00b4d8' : '#00ff88'} />
        <KPICard label="Risk Regime" value={regime} color={riskScore >= 70 ? '#ff3366' : riskScore >= 50 ? '#ff9500' : riskScore >= 25 ? '#00b4d8' : '#00ff88'} />
      </div>

      <div className="divider" />

      {/* Narrative Insights */}
      <h3 className="section-header">Narrative Insights</h3>
      {insights.map((item, i) => (
        <InsightCard key={i} icon={item.icon}>{item.text}</InsightCard>
      ))}

      <div className="divider" />

      {/* Overall Outlook */}
      <h3 className="section-header">Overall Outlook</h3>
      {riskScore < 25 && <AlertBanner type="success">Low geopolitical risk environment.</AlertBanner>}
      {riskScore >= 25 && riskScore < 50 && <AlertBanner type="warning">Moderate geopolitical risk environment.</AlertBanner>}
      {riskScore >= 50 && riskScore < 75 && <AlertBanner type="danger">High geopolitical risk environment.</AlertBanner>}
      {riskScore >= 75 && <AlertBanner type="danger" icon="🔴">Critical geopolitical environment.</AlertBanner>}
    </div>
  );
}
