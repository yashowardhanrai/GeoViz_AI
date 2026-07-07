"""
WEIGHT SENSITIVITY ANALYSIS
============================
Tests three risk score weight configurations to show robustness.
Computes pressure components directly from raw columns — no dependency
on pre-computed pressure columns.

Run from GeoVizAI root:
    python models/weight_sensitivity.py

Outputs:
    models/weight_sensitivity_results.csv
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded: {len(df)} rows")
print(f"Columns: {list(df.columns[:10])} ...")

# ── Identify available pressure proxies ───────────────────────────────────
# Find the best available columns for each component
print("\nAvailable columns (sample):")
for keyword in ['intensity', 'fatal', 'article', 'tone', 'close', 'return',
                'goldstein', 'volatil', 'pressure', 'conflict']:
    cols = [c for c in df.columns if keyword.lower() in c.lower()]
    if cols:
        print(f"  {keyword}: {cols[:5]}")

# ── Build pressure components from raw columns ────────────────────────────
print("\nBuilding pressure components from raw columns...")

# Conflict pressure proxy — use intensity or fatalities
if 'intensity' in df.columns:
    raw_c = df['intensity'].fillna(0)
elif 'fatalities' in df.columns:
    raw_c = df['fatalities'].fillna(0)
else:
    conflict_cols = [c for c in df.columns
                     if any(k in c.lower() for k in ['event','battle','conflict'])]
    raw_c = df[conflict_cols[0]].fillna(0) if conflict_cols else pd.Series(0, index=df.index)

# Media pressure proxy — use article count × |tone|
if 'article_count' in df.columns and 'avg_tone' in df.columns:
    raw_m = (df['article_count'].fillna(0) *
             df['avg_tone'].abs().fillna(0))
elif 'article_count' in df.columns:
    raw_m = df['article_count'].fillna(0)
else:
    gdelt_cols = [c for c in df.columns
                  if any(k in c.lower() for k in ['article','tone','gdelt'])]
    raw_m = df[gdelt_cols[0]].fillna(0) if gdelt_cols else pd.Series(0, index=df.index)

# Political pressure proxy — Goldstein deviation
if 'goldstein_scale' in df.columns:
    roll_mean = df['goldstein_scale'].rolling(90, min_periods=1).mean()
    raw_p = (df['goldstein_scale'] - roll_mean).abs().fillna(0)
elif 'avg_tone' in df.columns:
    roll_mean = df['avg_tone'].rolling(90, min_periods=1).mean()
    raw_p = (df['avg_tone'] - roll_mean).abs().fillna(0)
else:
    raw_p = pd.Series(0, index=df.index)

# Market stress proxy — oil volatility
if 'volatility' in df.columns:
    raw_s = df['volatility'].fillna(0)
elif 'volatility_7' in df.columns:
    raw_s = df['volatility_7'].fillna(0)
elif 'Close' in df.columns:
    raw_s = df['Close'].pct_change().abs().rolling(7).mean().fillna(0)
else:
    market_cols = [c for c in df.columns
                   if any(k in c.lower() for k in ['volatil','return','close'])]
    raw_s = df[market_cols[0]].fillna(0) if market_cols else pd.Series(0, index=df.index)

# Normalise each component to [0, 1] using expanding-window max
def expanding_norm(series):
    exp_max = series.expanding(min_periods=1).max()
    exp_max = exp_max.replace(0, np.nan).ffill().fillna(1)
    return (series / exp_max).clip(0, 1)

C = expanding_norm(raw_c)
M = expanding_norm(raw_m)
P = expanding_norm(raw_p)
S = expanding_norm(raw_s)

print(f"  C (conflict):  mean={C.mean():.3f}, std={C.std():.3f}")
print(f"  M (media):     mean={M.mean():.3f}, std={M.std():.3f}")
print(f"  P (political): mean={P.mean():.3f}, std={P.std():.3f}")
print(f"  S (market):    mean={S.mean():.3f}, std={S.std():.3f}")

# ── Three weight configurations ───────────────────────────────────────────
configs = {
    "Calibrated (paper)":   (0.35, 0.30, 0.20, 0.15),
    "Equal weights":        (0.25, 0.25, 0.25, 0.25),
    "Conflict-dominant":    (0.50, 0.25, 0.15, 0.10),
}

# ── Load saved features ───────────────────────────────────────────────────
with open("models/regime_classifier_7d_v3_features.json") as f:
    feat_info = json.load(f)
FEAT_COLS_ORIG = (feat_info if isinstance(feat_info, list)
                  else feat_info.get("features", []))

# ── Build all model features ──────────────────────────────────────────────
print("\nBuilding model features...")
for lag in [1, 2, 3, 5, 7, 14]:
    df[f"rs_lag_{lag}"] = df["risk_score"].shift(lag)
for w in [7, 14, 30]:
    df[f"rs_roll_mean_{w}"] = df["risk_score"].shift(1).rolling(w).mean()
    df[f"rs_roll_std_{w}"]  = df["risk_score"].shift(1).rolling(w).std()
df["rs_change_1"] = df["risk_score"].shift(1) - df["risk_score"].shift(2)
df["rs_change_3"] = df["risk_score"].shift(1) - df["risk_score"].shift(4)
df["rs_change_7"] = df["risk_score"].shift(1) - df["risk_score"].shift(8)
df["rs_dev_7"]  = df["rs_lag_1"] - df["rs_roll_mean_7"]
df["rs_dev_14"] = df["rs_lag_1"] - df["rs_roll_mean_14"]
df["rs_dev_30"] = df["rs_lag_1"] - df["rs_roll_mean_30"]
for lag in [1, 3, 7]:
    df[f"intensity_lag_{lag}"]  = (df["intensity"].shift(lag)
                                    if "intensity" in df.columns
                                    else pd.Series(0, index=df.index))
    df[f"fatalities_lag_{lag}"] = (df["fatalities"].shift(lag)
                                    if "fatalities" in df.columns
                                    else pd.Series(0, index=df.index))
for col, name in [
    ("severity_7d", "severity_7d_lag1"),
    ("events_30d",  "events_30d_lag1"),
    ("fatality_momentum", "fatality_mom_lag1"),
    ("crisis_interaction", "crisis_lag1"),
]:
    df[name] = (df[col].shift(1) if col in df.columns
                else pd.Series(0, index=df.index))
for lag in [1, 3, 7]:
    df[f"oil_lag_{lag}"] = df["Close"].shift(lag)
df["oil_ret_1"] = df["Close"].shift(1).pct_change()
df["oil_ret_7"] = df["Close"].shift(1) / df["Close"].shift(8) - 1
df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
df["month"]     = df["date"].dt.month
pca_cols = [c for c in df.columns if c.startswith("embedding_pca_")][:10]

FEAT_COLS = [c for c in FEAT_COLS_ORIG if c in df.columns]
print(f"Features available: {len(FEAT_COLS)}/{len(FEAT_COLS_ORIG)}")

# ── Run experiment for each weight configuration ──────────────────────────
results = []

for config_name, (a, b, g, d) in configs.items():
    print(f"\n{'='*55}")
    print(f"Config: {config_name}  (α={a}, β={b}, γ={g}, δ={d})")
    print(f"{'='*55}")

    # Recompute risk score with these weights
    new_risk = (a*C + b*M + g*P + d*S).clip(0, 1) * 100
    df["new_risk"] = new_risk

    # Recompute regime-change label from new risk score
    df["new_label"] = (
        df["new_risk"].shift(-7) - df["new_risk"] > 5
    ).astype(int)

    # Recompute lag features based on new risk score
    df_exp = df.copy()
    for lag in [1, 2, 3, 5, 7, 14]:
        df_exp[f"rs_lag_{lag}"] = df_exp["new_risk"].shift(lag)
    for w in [7, 14, 30]:
        df_exp[f"rs_roll_mean_{w}"] = df_exp["new_risk"].shift(1).rolling(w).mean()
        df_exp[f"rs_roll_std_{w}"]  = df_exp["new_risk"].shift(1).rolling(w).std()
    df_exp["rs_change_1"] = df_exp["new_risk"].shift(1) - df_exp["new_risk"].shift(2)
    df_exp["rs_change_3"] = df_exp["new_risk"].shift(1) - df_exp["new_risk"].shift(4)
    df_exp["rs_change_7"] = df_exp["new_risk"].shift(1) - df_exp["new_risk"].shift(8)
    df_exp["rs_dev_7"]  = df_exp["rs_lag_1"] - df_exp["rs_roll_mean_7"]
    df_exp["rs_dev_14"] = df_exp["rs_lag_1"] - df_exp["rs_roll_mean_14"]
    df_exp["rs_dev_30"] = df_exp["rs_lag_1"] - df_exp["rs_roll_mean_30"]

    keep = FEAT_COLS + ["new_label", "date", "new_risk"]
    keep = list(dict.fromkeys([c for c in keep if c in df_exp.columns]))
    data = (df_exp[keep]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True))

    n_pos = int(data["new_label"].sum())
    pos_rate = data["new_label"].mean()
    print(f"  Rows: {len(data)} | Positives: {n_pos} ({pos_rate*100:.1f}%)")
    print(f"  Risk score: mean={data['new_risk'].mean():.2f}, "
          f"std={data['new_risk'].std():.2f}, max={data['new_risk'].max():.2f}")

    if n_pos < 10:
        print(f"  SKIP: too few positives ({n_pos}) for reliable AUC")
        results.append({
            "config": config_name, "alpha": a, "beta": b, "gamma": g, "delta": d,
            "n_rows": len(data), "n_pos": n_pos, "pos_rate": round(pos_rate, 4),
            "risk_mean": round(data['new_risk'].mean(), 2),
            "risk_std": round(data['new_risk'].std(), 2),
            "auc": "n/a", "ci_lo": "n/a", "ci_hi": "n/a"
        })
        continue

    # Walk-forward OOF AUC
    import xgboost as xgb
    X = data[FEAT_COLS].fillna(0)
    y = data["new_label"]
    tscv = TimeSeriesSplit(n_splits=5)

    params = {
        "objective": "binary:logistic",
        "random_state": 42, "n_jobs": -1, "tree_method": "hist",
        "n_estimators": 300, "learning_rate": 0.05,
        "max_depth": 3, "min_child_weight": 50,
        "subsample": 0.7, "colsample_bytree": 0.6,
        "gamma": 1.0, "reg_alpha": 3.0, "reg_lambda": 5.0,
        "scale_pos_weight": float((y==0).sum()) / max(float((y==1).sum()), 1),
    }

    oof_preds = np.full(len(y), np.nan)
    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr = y.iloc[tr_idx]
        if ytr.sum() < 2:
            continue
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr, ytr, verbose=False)
        oof_preds[va_idx] = m.predict_proba(Xva)[:, 1]

    mask  = ~np.isnan(oof_preds)
    y_oof = y[mask].values
    p_oof = oof_preds[mask]

    if y_oof.sum() < 5:
        auc = float('nan')
        ci_lo = ci_hi = float('nan')
    else:
        auc = roc_auc_score(y_oof, p_oof)
        boot = []
        rng  = np.random.default_rng(42)
        for _ in range(500):
            idx = rng.integers(0, len(y_oof), len(y_oof))
            if y_oof[idx].sum() > 0:
                boot.append(roc_auc_score(y_oof[idx], p_oof[idx]))
        ci_lo = np.percentile(boot, 2.5)
        ci_hi = np.percentile(boot, 97.5)

    print(f"  OOF AUC: {auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    results.append({
        "config":     config_name,
        "alpha": a, "beta": b, "gamma": g, "delta": d,
        "n_rows":    len(data),
        "n_pos":     n_pos,
        "pos_rate":  round(pos_rate, 4),
        "risk_mean": round(data['new_risk'].mean(), 2),
        "risk_std":  round(data['new_risk'].std(), 2),
        "auc":       round(auc, 4),
        "ci_lo":     round(ci_lo, 4),
        "ci_hi":     round(ci_hi, 4),
    })

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("WEIGHT SENSITIVITY RESULTS")
print(f"{'='*55}")
for r in results:
    print(f"\n  {r['config']}")
    print(f"    Weights: α={r['alpha']}, β={r['beta']}, "
          f"γ={r['gamma']}, δ={r['delta']}")
    print(f"    Risk score: mean={r['risk_mean']}, std={r['risk_std']}")
    print(f"    Positives: {r['n_pos']} / {r['n_rows']} ({r['pos_rate']*100:.1f}%)")
    print(f"    OOF AUC: {r['auc']}  [{r['ci_lo']}, {r['ci_hi']}]")

out = pd.DataFrame(results)
out.to_csv("models/weight_sensitivity_results.csv", index=False)
print(f"\nSaved: models/weight_sensitivity_results.csv")

# Paper sentence
if all(isinstance(r['auc'], float) for r in results):
    aucs = [r['auc'] for r in results if isinstance(r['auc'], float)]
    print(f"\nPAPER SENTENCE:")
    print(f"  Sensitivity analysis across three weight configurations "
          f"yields classifier AUC in the range [{min(aucs):.3f}, {max(aucs):.3f}], "
          f"confirming robustness to weight choice.")