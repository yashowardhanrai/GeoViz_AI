"""
SMOTE + TEMPORAL BUFFER CLASSIFIER — Gap 3 Fix (v2)
=====================================================
Reads: data/processed/geoviz_risk_dataset.csv (read only)
Reads: models/regime_classifier_7d_v3_features.json (read only)
Reads: models/regime_classifier_7d_v3.pkl (read only)
Writes: models/smote_oof_predictions.csv (new file)

No pipeline rerun. No dataset modification. Safe to run.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("SMOTE available")
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available — using class weights only")

RANDOM_STATE = 42
BUFFER_DAYS  = 30
THRESHOLD    = 0.10

# ─────────────────────────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded: {len(df)} rows | Columns: {len(df.columns)}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Build ALL features (always — unconditional)
# ─────────────────────────────────────────────────────────────
print("Building features...")

for lag in [1, 2, 3, 5, 7, 14]:
    df[f"rs_lag_{lag}"] = df["risk_score"].shift(lag)

for w in [7, 14, 30]:
    df[f"rs_roll_mean_{w}"] = df["risk_score"].shift(1).rolling(w).mean()
    df[f"rs_roll_std_{w}"]  = df["risk_score"].shift(1).rolling(w).std()

df["rs_change_1"] = df["risk_score"].shift(1) - df["risk_score"].shift(2)
df["rs_change_3"] = df["risk_score"].shift(1) - df["risk_score"].shift(4)
df["rs_change_7"] = df["risk_score"].shift(1) - df["risk_score"].shift(8)
df["rs_dev_7"]    = df["rs_lag_1"] - df["rs_roll_mean_7"]
df["rs_dev_14"]   = df["rs_lag_1"] - df["rs_roll_mean_14"]
df["rs_dev_30"]   = df["rs_lag_1"] - df["rs_roll_mean_30"]

for lag in [1, 3, 7]:
    df[f"intensity_lag_{lag}"] = (
        df["intensity"].shift(lag) if "intensity" in df.columns
        else pd.Series(0, index=df.index)
    )
    df[f"fatalities_lag_{lag}"] = (
        df["fatalities"].shift(lag) if "fatalities" in df.columns
        else pd.Series(0, index=df.index)
    )

for col, name in [
    ("severity_7d",        "severity_7d_lag1"),
    ("events_30d",         "events_30d_lag1"),
    ("fatality_momentum",  "fatality_mom_lag1"),
    ("crisis_interaction", "crisis_lag1"),
]:
    df[name] = df[col].shift(1) if col in df.columns else pd.Series(0, index=df.index)

for lag in [1, 3, 7]:
    df[f"oil_lag_{lag}"] = df["Close"].shift(lag)

df["oil_ret_1"] = df["Close"].shift(1).pct_change()
df["oil_ret_7"] = df["Close"].shift(1) / df["Close"].shift(8) - 1
df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
df["month"]     = df["date"].dt.month

print(f"Columns after feature engineering: {len(df.columns)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: Load feature list from JSON
# ─────────────────────────────────────────────────────────────
with open("models/regime_classifier_7d_v3_features.json") as f:
    feat_info = json.load(f)
FEATURE_COLS = feat_info if isinstance(feat_info, list) else feat_info.get("features", [])

# Keep only features that exist in df after engineering
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"Features available: {len(FEATURE_COLS)}/{len(feat_info)}")

# Check for any missing
missing = [c for c in (feat_info if isinstance(feat_info, list) else feat_info.get("features",[]))
           if c not in df.columns]
if missing:
    print(f"  Missing (will skip): {missing[:5]}{'...' if len(missing)>5 else ''}")

# ─────────────────────────────────────────────────────────────
# STEP 4: Build target and prepare data
# ─────────────────────────────────────────────────────────────
df["target"] = (
    df["risk_score"].shift(-7) - df["risk_score"] > 5
).astype(int)

keep = FEATURE_COLS + ["target", "date", "risk_score"]
keep = list(dict.fromkeys(keep))

# Verify all keep columns exist
missing_keep = [c for c in keep if c not in df.columns]
if missing_keep:
    print(f"Still missing: {missing_keep}")
    keep = [c for c in keep if c in df.columns]

data = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
print(f"\nAfter dropna: {len(data)} rows | "
      f"Positives: {data.target.sum()} ({data.target.mean()*100:.1f}%)")

X = data[FEATURE_COLS].fillna(0)
y = data["target"]

# ─────────────────────────────────────────────────────────────
# STEP 5: Walk-forward OOF with temporal buffer + SMOTE
# ─────────────────────────────────────────────────────────────
print(f"\nWalk-forward OOF ({BUFFER_DAYS}-day temporal buffer)...")
tscv = TimeSeriesSplit(n_splits=5)

params = {
    "objective":        "binary:logistic",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "tree_method":      "hist",
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        3,
    "min_child_weight": 50,
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "gamma":            1.0,
    "reg_alpha":        3.0,
    "reg_lambda":       5.0,
    "scale_pos_weight": float((y == 0).sum()) / max(float((y == 1).sum()), 1),
    "eval_metric":      "auc",
}

oof_preds = np.full(len(y), np.nan)
fold_aucs = []

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X)):
    # Apply temporal buffer
    if len(tr_idx) > BUFFER_DAYS:
        tr_idx = tr_idx[:-BUFFER_DAYS]

    Xtr = X.iloc[tr_idx]
    ytr = y.iloc[tr_idx]
    Xva = X.iloc[va_idx]
    yva = y.iloc[va_idx]

    n_pos_tr = int(ytr.sum())
    n_pos_va = int(yva.sum())

    if n_pos_tr < 2:
        print(f"  Fold {fold_idx+1}: Skip — {n_pos_tr} train positives")
        continue

    # SMOTE within training fold only
    if SMOTE_AVAILABLE and n_pos_tr >= 3:
        try:
            k = min(3, n_pos_tr - 1)
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            Xtr_res, ytr_res = sm.fit_resample(Xtr, ytr)
            smote_note = f"SMOTE {len(Xtr)}→{len(Xtr_res)}"
        except Exception as e:
            Xtr_res, ytr_res = Xtr, ytr
            smote_note = f"SMOTE failed: {e}"
    else:
        Xtr_res, ytr_res = Xtr, ytr
        smote_note = "no SMOTE"

    model = xgb.XGBClassifier(**params)
    model.fit(Xtr_res, ytr_res, verbose=False)
    proba = model.predict_proba(Xva)[:, 1]
    oof_preds[va_idx] = proba

    if n_pos_va >= 2:
        fold_auc = roc_auc_score(yva, proba)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold_idx+1}: {smote_note} | "
              f"Val AUC={fold_auc:.4f} ({n_pos_va} val pos)")
    else:
        print(f"  Fold {fold_idx+1}: {smote_note} | "
              f"Val AUC=n/a ({n_pos_va} val pos)")

# ─────────────────────────────────────────────────────────────
# STEP 6: Pooled OOF metrics
# ─────────────────────────────────────────────────────────────
mask    = ~np.isnan(oof_preds)
y_oof   = y[mask].values
p_oof   = oof_preds[mask]
n_oof   = mask.sum()
pos_oof = int(y_oof.sum())

oof_auc = roc_auc_score(y_oof, p_oof)
oof_ap  = average_precision_score(y_oof, p_oof)
pred_lb = (p_oof >= THRESHOLD).astype(int)
oof_f1  = f1_score(y_oof, pred_lb, zero_division=0)

print(f"\n{'='*60}")
print(f"SMOTE + BUFFER RESULTS")
print(f"{'='*60}")
print(f"OOF rows: {n_oof} | Positives: {pos_oof}")
print(f"ROC-AUC:  {oof_auc:.4f}")
print(f"Avg Prec: {oof_ap:.4f}")
print(f"F1@{THRESHOLD}: {oof_f1:.4f}")
print(f"\nvs original regime_classifier.py:")
print(f"  AUC: 0.8106 → {oof_auc:.4f}  "
      f"({'IMPROVED' if oof_auc > 0.8106 else 'no improvement'})")
print(f"  AP:  0.3148 → {oof_ap:.4f}  "
      f"({'IMPROVED' if oof_ap  > 0.3148 else 'no improvement'})")
print(f"  F1:  0.4472 → {oof_f1:.4f}  "
      f"({'IMPROVED' if oof_f1  > 0.4472 else 'no improvement'})")

# Bootstrap CI
boot_aucs = []
for _ in range(1000):
    idx = np.random.choice(len(y_oof), len(y_oof), replace=True)
    if y_oof[idx].sum() > 0:
        boot_aucs.append(roc_auc_score(y_oof[idx], p_oof[idx]))
ci_lo = np.percentile(boot_aucs, 2.5)
ci_hi = np.percentile(boot_aucs, 97.5)
print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

# Save
pd.DataFrame({
    "date":   data.loc[mask, "date"].values,
    "actual": y_oof,
    "pred":   p_oof,
    "label":  pred_lb,
}).to_csv("models/smote_oof_predictions.csv", index=False)
print("\nSaved: models/smote_oof_predictions.csv")