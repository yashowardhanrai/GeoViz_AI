"""
PAPER BASELINES
===============
Compares XGBoost models against standard baselines required for paper review:

REGRESSION BASELINES (vs xgboost_model.py v6.2):
  1. Naive persistence  — predict risk_score[t] for all horizons
  2. ARIMA(p,d,q)       — auto-selected via AIC on train, one-step-ahead walk-forward
  3. Linear AR(14)      — Ridge regression on 14 lags of risk_score

CLASSIFICATION BASELINES (vs regime_classifier_v3.py):
  1. Always-0           — majority class, standard imbalance baseline
  2. Trend heuristic    — rs_change_3 > 0 → predict regime change
  3. Logistic regression on same pruned feature set

All baselines use the same 80/20 temporal split as the main models so
numbers are directly comparable. Outputs a single results table and
saves to models/paper_baselines.csv.

Requires: statsmodels  (pip install statsmodels)
          scikit-learn (already installed)
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
HORIZONS     = [1, 3, 7]

# =====================================================
# LOAD DATASET  (same CSV used by all models)
# =====================================================

df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# =====================================================
# SHARED FEATURE ENGINEERING
# (mirror xgboost_model.py exactly so split indices match)
# =====================================================

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
    df[f"intensity_lag_{lag}"]  = df["intensity"].shift(lag)
    df[f"fatalities_lag_{lag}"] = df["fatalities"].shift(lag)

for col, name in [
    ("severity_7d",        "severity_7d_lag1"),
    ("events_30d",         "events_30d_lag1"),
    ("fatality_momentum",  "fatality_mom_lag1"),
    ("crisis_interaction", "crisis_lag1"),
]:
    if col in df.columns:
        df[name] = df[col].shift(1)

for lag in [1, 3, 7]:
    df[f"oil_lag_{lag}"] = df["Close"].shift(lag)
df["oil_ret_1"] = df["Close"].shift(1).pct_change()
df["oil_ret_7"] = df["Close"].shift(1) / df["Close"].shift(8) - 1

df["dayofweek"] = df["date"].dt.dayofweek
df["month"]     = df["date"].dt.month
df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)

pca_cols = [c for c in df.columns if c.startswith("embedding_pca_")][:10]

FEATURE_COLS = (
    [c for c in df.columns if c.startswith("rs_")]
    + [c for c in df.columns if c.startswith("intensity_lag_")]
    + [c for c in df.columns if c.startswith("fatalities_lag_")]
    + ["severity_7d_lag1", "events_30d_lag1", "fatality_mom_lag1", "crisis_lag1"]
    + [c for c in df.columns if c.startswith("oil_")]
    + ["dayofweek", "month", "is_monday"]
    + pca_cols
)
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))
CIRCULAR = {"oil_shock", "oil_momentum"}
FEATURE_COLS = [c for c in FEATURE_COLS if c not in CIRCULAR]

# Build targets and drop NaNs exactly as in main model
for h in HORIZONS:
    df[f"target_raw_{h}"]   = df["risk_score"].shift(-h)
    df[f"target_delta_{h}"] = df[f"target_raw_{h}"] - df["risk_score"]

df["target_change"] = (
    df["risk_regime"].shift(-7) != df["risk_regime"]
).astype(int)
df.loc[df["risk_regime"].shift(-7).isna(), "target_change"] = np.nan

keep = FEATURE_COLS \
     + [f"target_raw_{h}"   for h in HORIZONS] \
     + [f"target_delta_{h}" for h in HORIZONS] \
     + ["target_change", "risk_score"]
data = df[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()

X = data[FEATURE_COLS].fillna(0).reset_index(drop=True)
y_clf = data["target_change"].astype(int).reset_index(drop=True)

split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

risk_series = data["risk_score"].reset_index(drop=True)

print(f"Dataset: {len(X)} rows | Train: {split_idx} | Test: {len(X) - split_idx}")
print(f"Classification positives — train: {y_clf.iloc[:split_idx].sum()}  "
      f"test: {y_clf.iloc[split_idx:].sum()}")

# =====================================================
# ── REGRESSION BASELINES ─────────────────────────────
# =====================================================

reg_results = {}   # {horizon: {model_name: {r2, mae}}}

# ── 1. Naive persistence ──────────────────────────────

print("\n── Naive persistence baseline ──")
for h in HORIZONS:
    yr_test  = data[f"target_raw_{h}"].iloc[split_idx:].values
    naive    = X_test["rs_lag_1"].values
    r2  = r2_score(yr_test, naive)
    mae = mean_absolute_error(yr_test, naive)
    reg_results.setdefault(h, {})["Naive"] = {"r2": r2, "mae": mae}
    print(f"  {h}-day  R²={r2:.4f}  MAE={mae:.3f}")

# ── 2. AR(14) with Ridge ──────────────────────────────

print("\n── AR(14) Ridge baseline ──")
ar_cols = [f"rs_lag_{lag}" for lag in [1, 2, 3, 5, 7, 14]
           if f"rs_lag_{lag}" in X.columns]

scaler = StandardScaler()
X_ar_train = scaler.fit_transform(X_train[ar_cols])
X_ar_test  = scaler.transform(X_test[ar_cols])

for h in HORIZONS:
    yd_train = data[f"target_delta_{h}"].iloc[:split_idx].values
    yr_test  = data[f"target_raw_{h}"].iloc[split_idx:].values

    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_ar_train, yd_train)

    delta_pred = ridge.predict(X_ar_test)
    raw_pred   = X_test["rs_lag_1"].values + delta_pred

    r2  = r2_score(yr_test, raw_pred)
    mae = mean_absolute_error(yr_test, raw_pred)
    reg_results[h]["AR(14)-Ridge"] = {"r2": r2, "mae": mae}
    print(f"  {h}-day  R²={r2:.4f}  MAE={mae:.3f}")

# ── 3. ARIMA walk-forward ─────────────────────────────
# Auto-selects (p,d,q) from a small grid on the training series,
# then does a genuine walk-forward forecast: re-fits each step.
# We limit to 1-day only (walk-forward ARIMA for 3d/7d would need
# multi-step forecasting which adds another modelling decision;
# AR-Ridge already covers the multi-step regression baseline).

print("\n── ARIMA walk-forward (1-day) ──")

train_series = risk_series.iloc[:split_idx].values
test_series  = risk_series.iloc[split_idx:].values
n_test       = len(test_series)

# Grid search on AIC over training data
best_aic, best_order = np.inf, (1, 0, 0)
for p, d, q in product(range(0, 4), [0, 1], range(0, 3)):
    try:
        aic = ARIMA(train_series, order=(p, d, q)).fit().aic
        if aic < best_aic:
            best_aic, best_order = aic, (p, d, q)
    except Exception:
        continue
print(f"  Best ARIMA order (AIC): {best_order}  AIC={best_aic:.1f}")

# Walk-forward: for each test point, re-fit on all data up to t-1
arima_preds = []
history = list(train_series)
for i in range(n_test):
    try:
        m = ARIMA(history, order=best_order).fit()
        arima_preds.append(m.forecast(steps=1)[0])
    except Exception:
        arima_preds.append(history[-1])   # fall back to persistence
    history.append(test_series[i])

arima_preds = np.array(arima_preds)
yr_test_1d  = data["target_raw_1"].iloc[split_idx:].values

r2_arima  = r2_score(yr_test_1d, arima_preds)
mae_arima = mean_absolute_error(yr_test_1d, arima_preds)
reg_results[1]["ARIMA"] = {"r2": r2_arima, "mae": mae_arima}
print(f"  1-day  R²={r2_arima:.4f}  MAE={mae_arima:.3f}")

# ARIMA single-step extrapolation for 3d/7d (fit on train, forecast h steps ahead)
for h in [3, 7]:
    try:
        m = ARIMA(train_series, order=best_order).fit()
        preds_h = []
        hist = list(train_series)
        for i in range(n_test):
            fc = m.forecast(steps=h)
            preds_h.append(fc[-1])
            hist.append(test_series[i])
            m = m.append([test_series[i]], refit=False)
        yr_test_h = data[f"target_raw_{h}"].iloc[split_idx:].values
        r2_h  = r2_score(yr_test_h, preds_h)
        mae_h = mean_absolute_error(yr_test_h, preds_h)
    except Exception:
        r2_h, mae_h = np.nan, np.nan
    reg_results[h]["ARIMA"] = {"r2": r2_h, "mae": mae_h}
    print(f"  {h}-day  R²={r2_h:.4f}  MAE={mae_h:.3f}")

# =====================================================
# ── CLASSIFICATION BASELINES ──────────────────────────
# =====================================================

clf_results = {}

y_tr = y_clf.iloc[:split_idx]
y_te = y_clf.iloc[split_idx:]

# ── 1. Always-0 ───────────────────────────────────────

pred_0 = np.zeros(len(y_te), dtype=int)
clf_results["Always-0"] = {
    "auc":  np.nan,
    "ap":   np.nan,
    "f1":   f1_score(y_te, pred_0, zero_division=0),
    "prec": precision_score(y_te, pred_0, zero_division=0),
    "rec":  recall_score(y_te, pred_0, zero_division=0),
    "brier": brier_score_loss(y_te, pred_0),
}

# ── 2. Trend heuristic ────────────────────────────────

trend_score = X_test["rs_change_3"].values
trend_pred  = (trend_score > 0).astype(int)
try:
    trend_auc = roc_auc_score(y_te, trend_score)
    trend_ap  = average_precision_score(y_te, trend_score)
except Exception:
    trend_auc = trend_ap = np.nan

clf_results["Trend heuristic"] = {
    "auc":  trend_auc,
    "ap":   trend_ap,
    "f1":   f1_score(y_te, trend_pred, zero_division=0),
    "prec": precision_score(y_te, trend_pred, zero_division=0),
    "rec":  recall_score(y_te, trend_pred, zero_division=0),
    "brier": brier_score_loss(y_te, (trend_score - trend_score.min()) /
                               (np.ptp(trend_score) + 1e-9)),
}

# ── 3. Logistic Regression (same feature set as v3 classifier) ───

# Load the saved feature list from the regime classifier
try:
    import json
    with open("models/regime_classifier_7d_v3_features.json") as fh:
        meta = json.load(fh)
    clf_features = [f for f in meta["features"] if f in X.columns]
    print(f"\n── Logistic Regression baseline — {len(clf_features)} features ──")
except FileNotFoundError:
    clf_features = [c for c in X.columns if c.startswith("rs_lag_") or
                    c in ["events_30d_lag1", "severity_7d_lag1", "month"]]
    print(f"\n── Logistic Regression baseline (features.json not found) "
          f"— {len(clf_features)} features ──")

scaler_clf = StandardScaler()
X_clf_train = scaler_clf.fit_transform(X_train[clf_features].fillna(0))
X_clf_test  = scaler_clf.transform(X_test[clf_features].fillna(0))

spw = max((y_tr == 0).sum() / max(y_tr.sum(), 1), 1.0)

lr = LogisticRegression(
    C=0.1,
    class_weight={0: 1.0, 1: spw},
    max_iter=1000,
    random_state=RANDOM_STATE,
    solver="lbfgs",
)
lr.fit(X_clf_train, y_tr)
lr_probs = lr.predict_proba(X_clf_test)[:, 1]
lr_pred  = (lr_probs >= 0.5).astype(int)

try:
    lr_auc = roc_auc_score(y_te, lr_probs)
    lr_ap  = average_precision_score(y_te, lr_probs)
except Exception:
    lr_auc = lr_ap = np.nan

clf_results["Logistic Regression"] = {
    "auc":  lr_auc,
    "ap":   lr_ap,
    "f1":   f1_score(y_te, lr_pred, zero_division=0),
    "prec": precision_score(y_te, lr_pred, zero_division=0),
    "rec":  recall_score(y_te, lr_pred, zero_division=0),
    "brier": brier_score_loss(y_te, lr_probs),
}
print(f"  AUC={lr_auc:.4f}  AP={lr_ap:.4f}  F1={clf_results['Logistic Regression']['f1']:.4f}")

# ── 4. Load XGBoost regime classifier results (from saved OOF CSV) ──

try:
    oof = pd.read_csv("models/oof_predictions_regime_classifier_7d_v3.csv")
    xgb_auc = roc_auc_score(oof["actual"], oof["pred_raw"])
    xgb_ap  = average_precision_score(oof["actual"], oof["pred_raw"])
    xgb_f1  = f1_score(oof["actual"], oof["pred_label"], zero_division=0)
    xgb_pr  = precision_score(oof["actual"], oof["pred_label"], zero_division=0)
    xgb_re  = recall_score(oof["actual"], oof["pred_label"], zero_division=0)
    xgb_bs  = brier_score_loss(oof["actual"], oof["pred_cal"])
    clf_results["XGBoost (OOF)"] = {
        "auc": xgb_auc, "ap": xgb_ap,
        "f1": xgb_f1, "prec": xgb_pr, "rec": xgb_re, "brier": xgb_bs,
    }
except FileNotFoundError:
    print("  OOF predictions not found — run regime_classifier.py first.")

# =====================================================
# ── PRINT RESULTS ─────────────────────────────────────
# =====================================================

print("\n" + "=" * 72)
print("REGRESSION COMPARISON TABLE")
print("=" * 72)
models_reg = ["Naive", "AR(14)-Ridge", "ARIMA"]

# Header
header = f"{'Model':<22}"
for h in HORIZONS:
    header += f"  {'R²':>7} {'MAE':>6}  "
print(header)
print(f"{'':22}" + "".join([f"  {'─'*7} {'─'*6}  " for _ in HORIZONS]))

for mname in models_reg:
    row = f"{mname:<22}"
    for h in HORIZONS:
        d = reg_results[h].get(mname, {})
        r2  = d.get("r2",  np.nan)
        mae = d.get("mae", np.nan)
        r2_str  = f"{r2:.4f}"  if not np.isnan(r2)  else "   n/a"
        mae_str = f"{mae:.3f}" if not np.isnan(mae) else "  n/a"
        row += f"  {r2_str:>7} {mae_str:>6}  "
    print(row)

print()
print("  Note: XGBoost v6.2 results (from xgboost_model.py output):")
print("  1-day R²=0.7409  3-day R²=0.7400  7-day R²=0.5270")

print("\n" + "=" * 72)
print("CLASSIFICATION COMPARISON TABLE  (7-day regime change)")
print("=" * 72)
print(f"{'Model':<25} {'AUC':>7} {'AP':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Brier':>7}")
print(f"{'':25} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

for mname, d in clf_results.items():
    def fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "   n/a"
    print(f"{mname:<25} {fmt(d['auc']):>7} {fmt(d['ap']):>7} "
          f"{fmt(d['f1']):>7} {fmt(d['prec']):>7} {fmt(d['rec']):>7} "
          f"{fmt(d['brier']):>7}")

print("=" * 72)

# =====================================================
# ── SAVE CSV ──────────────────────────────────────────
# =====================================================

rows = []
for h in HORIZONS:
    for mname in models_reg:
        d = reg_results[h].get(mname, {})
        rows.append({
            "task": "regression", "horizon": h, "model": mname,
            "r2": d.get("r2", np.nan), "mae": d.get("mae", np.nan),
        })
rows.append({"task": "regression", "horizon": "1,3,7", "model": "XGBoost v6.2",
             "r2": "0.7409 / 0.7400 / 0.5270", "mae": "1.127 / 1.236 / 1.548"})

for mname, d in clf_results.items():
    rows.append({
        "task": "classification", "horizon": 7, "model": mname,
        "auc": d["auc"], "ap": d["ap"], "f1": d["f1"],
        "precision": d["prec"], "recall": d["rec"], "brier": d["brier"],
    })

pd.DataFrame(rows).to_csv("models/paper_baselines.csv", index=False)
print("\nSaved: models/paper_baselines.csv")