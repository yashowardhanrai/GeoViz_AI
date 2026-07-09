// API configuration and fetch helpers for GeoVizAI frontend

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000/api';

async function fetchJSON(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  getData:           () => fetchJSON('/data'),
  getLatest:         () => fetchJSON('/data/latest'),
  getSummary:        () => fetchJSON('/data/summary'),
  getPredictions:    () => fetchJSON('/predictions'),
  getShapRegression: () => fetchJSON('/shap/regression'),
  getShapClassifier: () => fetchJSON('/shap/classifier'),
  getOOF:            () => fetchJSON('/oof'),
  getBaselines:      () => fetchJSON('/baselines'),
  getCorrelation:    () => fetchJSON('/correlation'),
};
