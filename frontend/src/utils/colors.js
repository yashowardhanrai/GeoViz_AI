// Color utilities and palette for GeoVizAI

export const REGIME_COLORS = {
  Stable:   '#00ff88',
  Elevated: '#00b4d8',
  High:     '#ff9500',
  Crisis:   '#ff3366',
};

export const REGIME_BG = {
  Stable:   'rgba(0, 255, 136, 0.08)',
  Elevated: 'rgba(0, 180, 216, 0.08)',
  High:     'rgba(255, 149, 0, 0.08)',
  Crisis:   'rgba(255, 51, 102, 0.08)',
};

export const CHART_COLORS = [
  '#00f0ff', // cyan
  '#7b61ff', // purple
  '#00ff88', // green
  '#ff3366', // red
  '#ff9500', // orange
  '#00b4d8', // blue
  '#ffd700', // gold
  '#e879f9', // pink
];

export const ASSET_COLORS = {
  oil:   '#ff9500',
  gold:  '#ffd700',
  sp500: '#00b4d8',
  dxy:   '#00ff88',
};

export function riskColor(score) {
  if (score >= 70) return '#ff3366';
  if (score >= 50) return '#ff9500';
  if (score >= 25) return '#00b4d8';
  return '#00ff88';
}

export function riskLabel(score) {
  if (score >= 70) return 'Crisis';
  if (score >= 50) return 'High';
  if (score >= 25) return 'Elevated';
  return 'Stable';
}

/** Interpolate between blue (-1) and red (+1) for correlation heatmap */
export function correlationColor(value) {
  if (value == null) return 'rgba(255,255,255,0.03)';
  const clamped = Math.max(-1, Math.min(1, value));

  if (clamped >= 0) {
    // 0 → white-ish, 1 → red
    const t = clamped;
    const r = Math.round(255);
    const g = Math.round(255 * (1 - t * 0.7));
    const b = Math.round(255 * (1 - t * 0.85));
    return `rgba(${r},${g},${b},0.85)`;
  } else {
    // -1 → blue, 0 → white-ish
    const t = -clamped;
    const r = Math.round(255 * (1 - t * 0.85));
    const g = Math.round(255 * (1 - t * 0.5));
    const b = Math.round(255);
    return `rgba(${r},${g},${b},0.85)`;
  }
}

export function formatNumber(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

export function formatPercent(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—';
  return `${Number(value).toFixed(decimals)}%`;
}
