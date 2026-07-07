"""
GeoVizAI — Flask API Server
Serves the CSV dataset and model artefacts as JSON for the React frontend.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Go up one more to reach GeoVizAI root
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "geoviz_risk_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Columns to keep (only columns needed by the dashboard, skipping huge text fields)
KEEP_COLS = [
    "date", "risk_score", "risk_level", "risk_7d", "risk_30d", "risk_90d",
    "crisis_index", "global_pressure", "market_stress", "conflict_pressure",
    "media_pressure", "risk_momentum", "risk_acceleration", "risk_regime",
    "regime", "Close", "daily_return", "volatility_30", "volatility_90",
    "return_30d", "price_change", "gold_close", "gold_return",
    "gold_volatility", "gold_return_30d", "sp500_close", "sp500_return",
    "sp500_volatility", "sp500_return_30d", "dxy_close", "dxy_return",
    "dxy_volatility", "dxy_return_30d", "fatalities", "event_count",
    "severity", "fatalities_7d", "fatalities_30d", "fatalities_90d",
    "severity_7d", "severity_30d", "intensity", "intensity_7d",
    "intensity_30d", "article_count", "article_count_norm", "avg_tone",
    "news_change", "tone_change", "tone_30d", "shock_score",
    "risk_change", "risk_percentile", "risk_zscore",
]

# ─── cache ───────────────────────────────────────────────────────────
_cache = {}


def _nan_to_none(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _df_to_records(df):
    """Convert DataFrame to JSON-safe list of dicts."""
    records = df.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            row[k] = _nan_to_none(v)
    return records


def get_main_data():
    if "main" not in _cache:
        print(f"[API] Loading dataset from {DATA_PATH} ...")
        # Load only necessary columns to speed up loading (from 1.9GB file)
        df = pd.read_csv(DATA_PATH, usecols=lambda col: col in KEEP_COLS, low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        # Convert date to ISO string for JSON
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        _cache["main"] = df
        print(f"[API] Loaded {len(df)} rows, {len(df.columns)} columns")
    return _cache["main"]


# ─── routes ──────────────────────────────────────────────────────────

@app.route("/api/data")
def api_data():
    """Return the full dataset (all rows, stripped of text columns)."""
    df = get_main_data()
    return jsonify(_df_to_records(df))


@app.route("/api/data/latest")
def api_latest():
    """Return the last row of the dataset."""
    df = get_main_data()
    latest = df.iloc[-1].to_dict()
    for k, v in latest.items():
        latest[k] = _nan_to_none(v)
    return jsonify(latest)


@app.route("/api/data/summary")
def api_summary():
    """Return aggregate statistics for the dataset."""
    df = get_main_data()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[num_cols].describe().to_dict()
    # Clean NaN
    for col in stats:
        for k in stats[col]:
            stats[col][k] = _nan_to_none(stats[col][k])
    return jsonify({
        "rows": len(df),
        "columns": len(df.columns),
        "date_range": {
            "start": df["date"].iloc[0],
            "end": df["date"].iloc[-1],
        },
        "stats": stats,
    })


@app.route("/api/predictions")
def api_predictions():
    """Return XGBoost test-set predictions (predictions_v6_3d.csv)."""
    path = os.path.join(MODELS_DIR, "predictions_v6_3d.csv")
    df = pd.read_csv(path)
    # Add date index from main dataset
    main = get_main_data()
    n = len(df)
    df["date"] = main["date"].iloc[-n:].values
    return jsonify(_df_to_records(df))


@app.route("/api/shap/regression")
def api_shap_regression():
    """Return SHAP importance for the regression model."""
    path = os.path.join(MODELS_DIR, "shap_importance_v6.csv")
    df = pd.read_csv(path)
    return jsonify(_df_to_records(df))


@app.route("/api/shap/classifier")
def api_shap_classifier():
    """Return SHAP importance for the regime-change classifier."""
    path = os.path.join(MODELS_DIR, "shap_importance_regime_classifier_v3.csv")
    df = pd.read_csv(path)
    return jsonify(_df_to_records(df))


@app.route("/api/oof")
def api_oof():
    """Return OOF predictions from the regime classifier."""
    path = os.path.join(MODELS_DIR, "oof_predictions_regime_classifier_7d_v3.csv")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return jsonify(_df_to_records(df))


@app.route("/api/baselines")
def api_baselines():
    """Return baseline model comparison data."""
    path = os.path.join(MODELS_DIR, "paper_baselines.csv")
    df = pd.read_csv(path)
    return jsonify(_df_to_records(df))


@app.route("/api/correlation")
def api_correlation():
    """Return pre-computed correlation matrix for key features."""
    df = get_main_data()
    cols = [
        "risk_score", "Close", "daily_return", "volatility_30",
        "gold_close", "sp500_close", "dxy_close",
        "fatalities", "intensity", "article_count", "avg_tone",
        "market_stress", "conflict_pressure", "media_pressure",
    ]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr().round(3)
    return jsonify({
        "columns": available,
        "matrix": corr.values.tolist(),
    })


if __name__ == "__main__":
    # Pre-load data on startup
    get_main_data()
    app.run(host="0.0.0.0", port=5000, debug=True)
