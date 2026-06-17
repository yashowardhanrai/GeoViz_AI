"""
GEOGRAPHIC / TEMPORAL OOS VALIDATION — Gap 1 Fix (v2)
=======================================================
Tests your existing saved model on a structurally different
conflict period — Israel/Gaza (Oct 2023 onwards) — using only
your existing dataset and saved model files.

Reads:  data/processed/geoviz_risk_dataset.csv  (read only)
Reads:  models/regime_classifier_7d_v3.pkl       (read only)
Reads:  models/regime_classifier_7d_v3_features.json (read only)
Writes: models/geographic_oos_results.csv         (new file)

No pipeline rerun. No dataset modification. Safe to run.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

THRESHOLD = 0.10

# ─────────────────────────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded: {len(df)} rows ({df.date.min().date()} to {df.date.max().date()})")

# ─────────────────────────────────────────────────────────────
# STEP 2: Build ALL features unconditionally
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
    df[name] = (
        df[col].shift(1) if col in df.columns
        else pd.Series(0, index=df.index)
    )

for lag in [1, 3, 7]:
    df[f"oil_lag_{lag}"] = df["Close"].shift(lag)

df["oil_ret_1"] = df["Close"].shift(1).pct_change()
df["oil_ret_7"] = df["Close"].shift(1) / df["Close"].shift(8) - 1
df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
df["month"]     = df["date"].dt.month

print(f"Columns after engineering: {len(df.columns)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: Load feature list and keep only available columns
# ─────────────────────────────────────────────────────────────
with open("models/regime_classifier_7d_v3_features.json") as f:
    feat_info = json.load(f)

raw_features = feat_info if isinstance(feat_info, list) else feat_info.get("features", [])
FEATURE_COLS = [c for c in raw_features if c in df.columns]

missing = [c for c in raw_features if c not in df.columns]
print(f"Features loaded: {len(FEATURE_COLS)}/{len(raw_features)}")
if missing:
    print(f"  Skipping missing: {missing}")

# ─────────────────────────────────────────────────────────────
# STEP 4: Build regime-change target
# ─────────────────────────────────────────────────────────────
df["target"] = (
    df["risk_score"].shift(-7) - df["risk_score"] > 5
).astype(int)

keep = FEATURE_COLS + ["target", "date", "risk_score"]
keep = list(dict.fromkeys(keep))
data = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
print(f"After dropna: {len(data)} rows | "
      f"Positives: {data.target.sum()} ({data.target.mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 5: Temporal-geographic split
# ─────────────────────────────────────────────────────────────
print("\nTemporal-geographic split:")
print("  Train: 2022-03-01 to 2023-09-30 (Ukraine/Russia dominant)")
print("  Test:  2023-10-01 to 2025-06-13 (Israel/Gaza conflict starts)")

mask_tr = data["date"] <= "2023-09-30"
mask_te = data["date"] >= "2023-10-01"
data_tr = data[mask_tr].reset_index(drop=True)
data_te = data[mask_te].reset_index(drop=True)

X_te = data_te[FEATURE_COLS].fillna(0)
y_te = data_te["target"]

print(f"  Train: {len(data_tr)} rows, {data_tr.target.sum()} pos")
print(f"  Test:  {len(data_te)} rows, {y_te.sum()} pos "
      f"({y_te.mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 6: Load existing model and predict on test split
# ─────────────────────────────────────────────────────────────
print("\nLoading model: models/regime_classifier_7d_v3.pkl")
clf_pkg = joblib.load("models/regime_classifier_7d_v3.pkl")

if isinstance(clf_pkg, dict):
    clf   = clf_pkg.get("model")
    calib = clf_pkg.get("calibrator")
    print(f"  Loaded dict with keys: {list(clf_pkg.keys())}")
else:
    clf, calib = clf_pkg, None
    print("  Loaded model directly")

# Get predictions
proba_raw = clf.predict_proba(X_te)[:, 1]
if calib is not None:
    try:
        proba_cal = calib.transform(proba_raw)
        print("  Applied isotonic calibration")
    except Exception:
        proba_cal = proba_raw
        print("  Calibration failed — using raw probabilities")
else:
    proba_cal = proba_raw

# ─────────────────────────────────────────────────────────────
# STEP 7: Metrics
# ─────────────────────────────────────────────────────────────
pred_label = (proba_cal >= THRESHOLD).astype(int)

geo_auc = roc_auc_score(y_te, proba_cal)
geo_ap  = average_precision_score(y_te, proba_cal)
geo_f1  = f1_score(y_te, pred_label, zero_division=0)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_te, pred_label)
tn, fp, fn, tp = cm.ravel()

# Bootstrap CI
boot_aucs = []
rng = np.random.default_rng(42)
for _ in range(1000):
    idx = rng.integers(0, len(y_te), len(y_te))
    if y_te.iloc[idx].sum() > 0:
        boot_aucs.append(roc_auc_score(y_te.iloc[idx], proba_cal[idx]))
ci_lo = np.percentile(boot_aucs, 2.5)
ci_hi = np.percentile(boot_aucs, 97.5)

print(f"\n{'='*65}")
print(f"GEOGRAPHIC / TEMPORAL OOS VALIDATION")
print(f"{'='*65}")
print(f"Test period:   2023-10-01 to 2025-06-13")
print(f"Test rows:     {len(y_te)} | Positives: {int(y_te.sum())} "
      f"({y_te.mean()*100:.1f}%)")
print(f"ROC-AUC:       {geo_auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Avg Precision: {geo_ap:.4f}")
print(f"F1@{THRESHOLD}:      {geo_f1:.4f}")
print(f"Confusion:     TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"Recall:        {tp/(tp+fn) if (tp+fn)>0 else 0:.3f}")
print()
print(f"Comparison vs OOF (same-period) AUC=0.8106:")
delta = geo_auc - 0.8106
if geo_auc >= 0.75:
    verdict = "GOOD — model generalises to new conflict type"
elif geo_auc >= 0.65:
    verdict = "MODERATE — partial generalisation"
elif geo_auc >= 0.55:
    verdict = "WEAK — limited generalisation"
else:
    verdict = "POOR — model does not generalise (single-cycle limitation)"
print(f"  Delta: {delta:+.4f} | Verdict: {verdict}")

# ─────────────────────────────────────────────────────────────
# STEP 8: Save results
# ─────────────────────────────────────────────────────────────
pd.DataFrame({
    "date":     data_te["date"].values,
    "actual":   y_te.values,
    "pred_cal": proba_cal,
    "label":    pred_label,
}).to_csv("models/geographic_oos_results.csv", index=False)
print("\nSaved: models/geographic_oos_results.csv")

print(f"\nPAPER TEXT:")
print(f"  To test temporal generalisation, the trained model was evaluated")
print(f"  on the post-October-2023 period ({len(data_te)} rows), which coincides")
print(f"  with the onset of the Israel-Gaza conflict — a structurally different")
print(f"  escalation dynamic from the Ukraine/Russia period used in training.")
print(f"  The model achieved AUC={geo_auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}],")
print(f"  {verdict.lower()}.")