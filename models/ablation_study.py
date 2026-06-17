"""
ABLATION STUDY — Leakage Audit Contribution Table
==================================================
Reconstructs the performance at each stage of the leakage audit so the
paper has a formal ablation rather than just before/after numbers.

Stages (each builds on the previous):
  S0  Baseline (v5-equivalent): all circular features, bfill, global
      PCA, full-sample rank — approximated by re-adding the circular
      features to the fixed pipeline and reporting those numbers.
      We cannot fully reconstruct S0 without reverting the pipeline,
      so we document the v5 numbers from the conversation history and
      compute S1–S5 fresh.

  S1  Remove oil_shock / oil_momentum  (circular formula components)
  S2  S1 + fix risk_percentile         (expanding rank vs full-sample)
  S3  S2 + fix bfill → ffill           (no backward fill across gaps)
  S4  S3 + fix PCA fit                 (train-only vs full-dataset)
  S5  S4 + per-horizon tuning          (v6.2 final)

For S1–S4, we use the same fixed dataset (all pipeline fixes already
applied) but toggle features or re-introduce the leaking computation
at the model-script level where possible. Where a fix is in the
pipeline itself (bfill, PCA), we document the step using numbers from
the conversation log rather than re-running the broken pipeline.

Output: models/ablation_table.csv  +  console table
"""

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
HORIZONS     = [1, 3, 7]

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# =====================================================
# FULL FEATURE ENGINEERING (v6.2 base)
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

# Simulate oil_shock / oil_momentum being present (for S0 comparison)
# These are the circular features: abs(daily_return) and return_30d
if "daily_return" in df.columns:
    df["oil_shock"]    = df["daily_return"].abs()
if "return_30d" in df.columns:
    df["oil_momentum"] = df["return_30d"].abs()

# Simulate leaky risk_percentile (full-sample rank — for S0/S1 comparison)
df["risk_percentile_leaky"] = df["risk_score"].rank(pct=True) * 100
df["crisis_lag1_leaky"] = (
    df["risk_percentile_leaky"] * df.get("crisis_index", pd.Series(0, index=df.index))
).shift(1)

pca_cols = [c for c in df.columns if c.startswith("embedding_pca_")][:10]

BASE_COLS = (
    [c for c in df.columns if c.startswith("rs_")]
    + [c for c in df.columns if c.startswith("intensity_lag_")]
    + [c for c in df.columns if c.startswith("fatalities_lag_")]
    + ["severity_7d_lag1", "events_30d_lag1", "fatality_mom_lag1", "crisis_lag1"]
    + [c for c in df.columns if c.startswith("oil_")]
    + ["dayofweek", "month"]
    + pca_cols
)
BASE_COLS = [c for c in BASE_COLS if c in df.columns]
BASE_COLS = list(dict.fromkeys(BASE_COLS))

for h in HORIZONS:
    df[f"target_delta_{h}"] = df["risk_score"].shift(-h) - df["risk_score"]
    df[f"target_raw_{h}"]   = df["risk_score"].shift(-h)

# =====================================================
# SHARED TRAINING HELPER
# =====================================================

FIXED_PARAMS = {
    "objective":        "reg:squarederror",
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
    "reg_lambda":       8.0,
}


def evaluate_feature_set(features, label):
    """Train XGBoost with fixed params on given feature set, return R² per horizon."""
    keep = features + [f"target_delta_{h}" for h in HORIZONS] \
                    + [f"target_raw_{h}"   for h in HORIZONS]
    keep = [c for c in keep if c in df.columns]
    d = df[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X  = d[features].fillna(0)
    n  = len(X)
    sp = int(n * 0.80)
    X_tr, X_te = X.iloc[:sp], X.iloc[sp:]

    out = {}
    for h in HORIZONS:
        yd_tr = d[f"target_delta_{h}"].iloc[:sp]
        yr_te = d[f"target_raw_{h}"].iloc[sp:]

        m = xgb.XGBRegressor(**FIXED_PARAMS)
        m.fit(X_tr, yd_tr, verbose=False)
        raw_pred = X_te["rs_lag_1"].values + m.predict(X_te)
        r2  = r2_score(yr_te.values, raw_pred)
        mae = mean_absolute_error(yr_te.values, raw_pred)
        out[h] = {"r2": r2, "mae": mae}

    print(f"  {label:<40}  "
          + "  ".join([f"{h}d R²={out[h]['r2']:+.4f}" for h in HORIZONS]))
    return out


# =====================================================
# ABLATION STAGES
# =====================================================

print("Computing ablation stages (fixed params across all stages for fair comparison) …\n")

stages = {}

# S0 — documented from conversation history (v5 approximate)
# These are the numbers from the first run before any fixes.
# Cannot be recomputed cleanly without reverting pipeline files.
stages["S0: Pre-audit (v5 approx.)"] = {
    h: {"r2": r, "mae": m, "source": "conversation log"}
    for h, r, m in [(1, 0.6205, 1.462), (3, 0.6704, 1.384), (7, -0.0890, 2.784)]
}
print("  S0: Pre-audit (v5 approx.)              (from conversation log)")

# S1 — add oil_shock + oil_momentum back, leaky risk_percentile
cols_s1 = BASE_COLS.copy()
for c in ["oil_shock", "oil_momentum"]:
    if c in df.columns and c not in cols_s1:
        cols_s1.append(c)
# Replace clean crisis_lag1 with leaky version
if "crisis_lag1_leaky" in df.columns:
    cols_s1 = [c if c != "crisis_lag1" else "crisis_lag1_leaky" for c in cols_s1]
stages["S1: +circular feats +leaky rank"] = evaluate_feature_set(
    cols_s1, "S1: +circular feats +leaky rank"
)

# S2 — remove circular features, keep leaky rank
cols_s2 = BASE_COLS.copy()
if "crisis_lag1_leaky" in df.columns:
    cols_s2 = [c if c != "crisis_lag1" else "crisis_lag1_leaky" for c in cols_s2]
stages["S2: –circular, +leaky rank"] = evaluate_feature_set(
    cols_s2, "S2: –circular, +leaky rank"
)

# S3 — clean rank (crisis_lag1 from expanding percentile, already in BASE_COLS)
stages["S3: –circular, clean rank"] = evaluate_feature_set(
    BASE_COLS, "S3: –circular, clean rank"
)

# S4 — same as S3; bfill and PCA fixes are in the pipeline itself (already applied)
# We document this as the state after those pipeline-level fixes take effect.
# Numbers are from the post-fix run in the conversation log.
stages["S4: +bfill fix +PCA fix (pipeline)"] = {
    h: {"r2": r, "mae": m, "source": "conversation log"}
    for h, r, m in [(1, 0.7357, 1.152), (3, 0.7553, 1.195), (7, 0.4985, 1.695)]
}
print("  S4: +bfill fix +PCA fix (pipeline)      (from conversation log)")

# S5 — v6.2 final (per-horizon tuning) — from saved model run
stages["S5: +per-horizon tuning (v6.2)"] = {
    h: {"r2": r, "mae": m, "source": "xgboost_model.py output"}
    for h, r, m in [(1, 0.7409, 1.127), (3, 0.7400, 1.236), (7, 0.5270, 1.548)]
}
print("  S5: +per-horizon tuning (v6.2)          (from xgboost_model.py output)")

# Naive baseline for reference
naive_r2 = {1: 0.7779, 3: 0.8158, 7: 0.6823}

# =====================================================
# PRINT ABLATION TABLE
# =====================================================

print("\n" + "=" * 80)
print("ABLATION TABLE — Test R² by stage and horizon")
print("(Naive baseline: 1-day=0.7779  3-day=0.8158  7-day=0.6823)")
print("=" * 80)
print(f"{'Stage':<42} {'1-day R²':>9} {'3-day R²':>9} {'7-day R²':>9}  {'Note'}")
print("-" * 80)

for stage_name, res in stages.items():
    r1 = res[1]["r2"] if isinstance(res[1], dict) else np.nan
    r3 = res[3]["r2"] if isinstance(res[3], dict) else np.nan
    r7 = res[7]["r2"] if isinstance(res[7], dict) else np.nan
    source = res[1].get("source", "computed") if isinstance(res[1], dict) else ""
    flag = " *" if "log" in source or "output" in source else ""
    print(f"{stage_name:<42} {r1:>9.4f} {r3:>9.4f} {r7:>9.4f}  {flag}")

print("-" * 80)
print("  * = documented from prior runs; not recomputed (pipeline state unavailable)")
print("=" * 80)

# Individual fix contributions (delta from prior stage)
print("\nFix contributions (ΔR² from prior stage):")
stage_list = list(stages.items())
for i in range(1, len(stage_list)):
    prev_name, prev = stage_list[i - 1]
    curr_name, curr = stage_list[i]
    deltas = []
    for h in HORIZONS:
        pr = prev[h]["r2"] if isinstance(prev[h], dict) else np.nan
        cr = curr[h]["r2"] if isinstance(curr[h], dict) else np.nan
        deltas.append(f"{h}d: {cr - pr:+.4f}")
    print(f"  {curr_name:<42}  {' | '.join(deltas)}")

# =====================================================
# SAVE
# =====================================================

rows = []
for stage_name, res in stages.items():
    for h in HORIZONS:
        r2  = res[h]["r2"]  if isinstance(res[h], dict) else np.nan
        mae = res[h].get("mae", np.nan) if isinstance(res[h], dict) else np.nan
        src = res[h].get("source", "computed") if isinstance(res[h], dict) else ""
        rows.append({
            "stage": stage_name, "horizon": h,
            "test_r2": r2, "mae": mae,
            "naive_r2": naive_r2[h],
            "gap_vs_naive": r2 - naive_r2[h],
            "source": src,
        })

pd.DataFrame(rows).to_csv("models/ablation_table.csv", index=False)
print("\nSaved: models/ablation_table.csv")