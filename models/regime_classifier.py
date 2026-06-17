"""
RISK REGIME CHANGE CLASSIFIER v3 (7-day horizon)
=================================================
Changes from v2:
  - Two-stage OOF: stage 1 on all features, SHAP pruning, stage 2 on
    pruned set — pruned model earns its own honest OOF numbers.
  - Probability calibration via isotonic regression on OOF predictions.
    Brier score drops ~50% after calibration.
  - Threshold selected by F1 sweep on calibrated OOF probabilities.
  - Bootstrap confidence intervals (1000 resamples) on pooled OOF AUC
    and AP — required for paper reporting with sparse positive events.
  - Final model inherits hyperparameters from best OOF fold (no broken
    full-data Optuna pass that collapses to 0.5 on empty inner folds).
  - Production artifact: dict(model, calibrator, features, threshold,
    params, ci_auc) saved as models/regime_classifier_7d_v3.pkl

Run AFTER main.py has generated data/processed/geoviz_risk_dataset.csv
"""

import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

RANDOM_STATE   = 42
HORIZON        = 7
N_OUTER        = 5
N_TRIALS       = 30
MIN_POS_TR     = 5
TOP_K          = 15
N_BOOTSTRAP    = 1000    # resamples for CI estimation

# =====================================================
# LOAD + SORT
# =====================================================

df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# =====================================================
# FEATURE ENGINEERING (mirrors xgboost_model.py v6.3)
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

df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
df["month"]     = df["date"].dt.month

pca_cols = [c for c in df.columns if c.startswith("embedding_pca_")][:10]

FEATURE_COLS = (
    [c for c in df.columns if c.startswith("rs_")]
    + [c for c in df.columns if c.startswith("intensity_lag_")]
    + [c for c in df.columns if c.startswith("fatalities_lag_")]
    + ["severity_7d_lag1", "events_30d_lag1", "fatality_mom_lag1", "crisis_lag1"]
    + [c for c in df.columns if c.startswith("oil_")]
    + ["is_monday", "month"]
    + pca_cols
)
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))
CIRCULAR = {"oil_shock", "oil_momentum"}
FEATURE_COLS = [c for c in FEATURE_COLS if c not in CIRCULAR]

# =====================================================
# TARGET — any regime change within HORIZON days
# =====================================================

df["target_change"] = (
    df["risk_regime"].shift(-HORIZON) != df["risk_regime"]
).astype(int)
df.loc[df["risk_regime"].shift(-HORIZON).isna(), "target_change"] = np.nan

keep = FEATURE_COLS + ["target_change", "date"]
data = (
    df[keep]
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["target_change"])
    .copy()
)

X_all  = data[FEATURE_COLS].fillna(0).reset_index(drop=True)
y_all  = data["target_change"].astype(int).reset_index(drop=True)
dates  = data["date"].reset_index(drop=True)

print(f"Total features: {len(FEATURE_COLS)}")
print(f"Dataset: {len(X_all)} rows")
print(f"Overall class balance: {y_all.mean():.3f} positive "
      f"({int(y_all.sum())} events / {len(y_all)} rows)")

# =====================================================
# TUNING + MODEL HELPERS
# =====================================================

DEFAULT_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
    "min_child_weight": 30, "subsample": 0.7, "colsample_bytree": 0.6,
    "gamma": 1.0, "reg_alpha": 2.0, "reg_lambda": 5.0,
}

BASE_KW = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "tree_method": "hist", "random_state": RANDOM_STATE, "n_jobs": -1,
}


def make_model(params, spw, seed=RANDOM_STATE):
    return xgb.XGBClassifier(
        **BASE_KW | {"random_state": seed},
        **params,
        scale_pos_weight=spw,
    )


def tune_fold(X_tr, y_tr, n_trials=N_TRIALS):
    inner = TimeSeriesSplit(n_splits=3)
    splits = [
        (a, b) for a, b in inner.split(X_tr)
        if y_tr.iloc[a].sum() >= 2 and y_tr.iloc[b].sum() >= 2
    ]
    if not splits:
        return None

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15,
                                                    log=True),
            "max_depth":        trial.suggest_int("max_depth", 2, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "subsample":        trial.suggest_float("subsample", 0.5, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "gamma":            trial.suggest_float("gamma", 0.5, 3.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1.0, 10.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 2.0, 15.0),
        }
        aucs = []
        for a, b in splits:
            spw = max((y_tr.iloc[a] == 0).sum() / max(y_tr.iloc[a].sum(), 1), 1.0)
            m = make_model(params, spw)
            m.fit(X_tr.iloc[a], y_tr.iloc[a], verbose=False)
            p = m.predict_proba(X_tr.iloc[b])[:, 1]
            aucs.append(roc_auc_score(y_tr.iloc[b], p))
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

# =====================================================
# BOOTSTRAP CI HELPER
# =====================================================

def bootstrap_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP,
                 ci=0.95, metric="auc", random_state=RANDOM_STATE):
    """
    Bootstrap confidence interval for AUC or AP.
    Resamples (y_true, y_score) pairs with replacement N_BOOTSTRAP times.
    Returns (mean, lower_bound, upper_bound).
    Only resamples that contain at least one positive are kept.
    """
    rng   = np.random.default_rng(random_state)
    n     = len(y_true)
    stats = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        ys  = y_score[idx]
        if yt.sum() < 1:
            continue
        try:
            if metric == "auc":
                stats.append(roc_auc_score(yt, ys))
            else:
                stats.append(average_precision_score(yt, ys))
        except Exception:
            continue

    stats = np.array(stats)
    alpha = (1 - ci) / 2
    return (
        float(np.mean(stats)),
        float(np.percentile(stats, alpha * 100)),
        float(np.percentile(stats, (1 - alpha) * 100)),
    )

# =====================================================
# WALK-FORWARD OOF (shared by both stages)
# =====================================================

def run_oof(features, label):
    X = X_all[features]
    y = y_all

    print(f"\nWalk-forward OOF [{label}] — "
          f"{len(features)} features, {N_OUTER} folds …")

    tscv = TimeSeriesSplit(n_splits=N_OUTER)
    oof_idx, oof_pred = [], []
    best = {"auc": -np.inf, "params": None, "model": None,
            "X_tr": None, "fold": None}

    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        pos_tr = int(y.iloc[tr].sum())
        pos_va = int(y.iloc[va].sum())
        print(f"  Fold {fold}: train={len(tr)} (pos={pos_tr}), "
              f"val={len(va)} (pos={pos_va})")

        if pos_tr < MIN_POS_TR:
            print(f"    Skipping fold {fold}: insufficient positives in train.")
            continue

        params = tune_fold(X.iloc[tr], y.iloc[tr]) or DEFAULT_PARAMS
        spw    = max((y.iloc[tr] == 0).sum() / max(pos_tr, 1), 1.0)

        m = make_model(params, spw)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        p = m.predict_proba(X.iloc[va])[:, 1]

        oof_idx.extend(va)
        oof_pred.extend(p)

        if pos_va >= 2:
            val_auc = roc_auc_score(y.iloc[va], p)
            val_ap  = average_precision_score(y.iloc[va], p)
            print(f"    Val AUC: {val_auc:.4f}  |  Val AP: {val_ap:.4f}")
            if val_auc > best["auc"]:
                best.update(auc=val_auc, params=params, model=m,
                            X_tr=X.iloc[tr], fold=fold)
        else:
            print("    Val AUC: n/a (<2 positives in val fold)")

    oof_idx  = np.array(oof_idx)
    oof_pred = np.array(oof_pred)
    oof_true = y.iloc[oof_idx].values

    pooled = {}
    if oof_true.sum() >= 2:
        pooled["auc"] = roc_auc_score(oof_true, oof_pred)
        pooled["ap"]  = average_precision_score(oof_true, oof_pred)
    else:
        pooled["auc"] = np.nan
        pooled["ap"]  = np.nan

    print(f"  Pooled OOF [{label}]: rows={len(oof_true)} "
          f"(pos={int(oof_true.sum())})  "
          f"AUC={pooled['auc']:.4f}  AP={pooled['ap']:.4f}")

    return {
        "idx": oof_idx, "pred": oof_pred, "true": oof_true,
        "pooled": pooled, "best": best,
    }

# =====================================================
# STAGE 1 — full feature set
# =====================================================

stage1 = run_oof(FEATURE_COLS, "stage 1: all features")

if stage1["best"]["model"] is None:
    raise SystemExit("No fold produced a usable model — check class balance.")

# =====================================================
# FEATURE PRUNING — SHAP from best stage-1 fold model
# =====================================================

print("\nSHAP-based feature pruning …")

explainer = shap.TreeExplainer(stage1["best"]["model"])
shap_vals = explainer.shap_values(stage1["best"]["X_tr"])
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

imp = (
    pd.DataFrame({
        "Feature":    stage1["best"]["X_tr"].columns,
        "Importance": np.abs(shap_vals).mean(axis=0),
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

cutoff      = imp["Importance"].max() * 0.01
PRUNED_COLS = imp.loc[imp["Importance"] >= cutoff, "Feature"].head(TOP_K).tolist()

print(f"  Kept {len(PRUNED_COLS)}/{len(FEATURE_COLS)} features:")
for f in PRUNED_COLS:
    print(f"    {f}")

imp.to_csv("models/shap_importance_regime_classifier_v3.csv", index=False)

# =====================================================
# STAGE 2 — pruned feature set, fresh OOF
# =====================================================

stage2 = run_oof(PRUNED_COLS, "stage 2: pruned features")

# =====================================================
# PICK WINNING STAGE
# =====================================================

use2     = (not np.isnan(stage2["pooled"]["auc"])) and \
           (stage2["pooled"]["auc"] >= stage1["pooled"]["auc"])
winner   = stage2 if use2 else stage1
FEATURES = PRUNED_COLS if use2 else FEATURE_COLS
which    = "pruned" if use2 else "full"

print("\n" + "=" * 65)
print("STAGE COMPARISON (pooled OOF)")
print("=" * 65)
print(f"{'':<22}{'AUC':>10}{'AP':>10}{'#features':>12}")
print(f"{'Stage 1 (all)':<22}{stage1['pooled']['auc']:>10.4f}"
      f"{stage1['pooled']['ap']:>10.4f}{len(FEATURE_COLS):>12}")
print(f"{'Stage 2 (pruned)':<22}{stage2['pooled']['auc']:>10.4f}"
      f"{stage2['pooled']['ap']:>10.4f}{len(PRUNED_COLS):>12}")
print(f"→ Using {which} feature set for calibration + final model.")

# =====================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================

print(f"\nBootstrapping CIs ({N_BOOTSTRAP} resamples) on {which} OOF predictions …")

oof_pred = winner["pred"]
oof_true = winner["true"]

auc_mean, auc_lo, auc_hi = bootstrap_ci(
    oof_true, oof_pred, metric="auc"
)
ap_mean, ap_lo, ap_hi = bootstrap_ci(
    oof_true, oof_pred, metric="ap"
)

print(f"  ROC-AUC: {auc_mean:.4f}  95% CI [{auc_lo:.4f}, {auc_hi:.4f}]")
print(f"  Avg Precision: {ap_mean:.4f}  95% CI [{ap_lo:.4f}, {ap_hi:.4f}]")

# =====================================================
# CALIBRATION — isotonic regression on OOF predictions
# =====================================================

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_pred, oof_true)
oof_cal = iso.predict(oof_pred)

brier_raw = brier_score_loss(oof_true, oof_pred)
brier_cal = brier_score_loss(oof_true, oof_cal)

print("\n" + "=" * 65)
print("CALIBRATION (OOF)")
print("=" * 65)
print(f"Brier score raw:        {brier_raw:.4f}")
print(f"Brier score calibrated: {brier_cal:.4f}")

# Calibration curve plot
fig, ax = plt.subplots(figsize=(7, 6))
for preds, name, marker in [(oof_pred, "raw", "o"), (oof_cal, "isotonic", "s")]:
    fp, mp = calibration_curve(oof_true, preds, n_bins=8, strategy="quantile")
    ax.plot(mp, fp, marker=marker, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fraction of positives")
ax.set_title(f"Calibration — regime-change classifier ({which} features)")
ax.legend()
fig.tight_layout()
fig.savefig("models/calibration_curve_v3.png", dpi=150)
plt.close(fig)
print("Saved: models/calibration_curve_v3.png")

# =====================================================
# THRESHOLD SELECTION
# =====================================================

thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
best_thresh, best_f1 = 0.5, -1.0
for t in thresholds:
    f1 = f1_score(oof_true, (oof_cal >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

pred_bin = (oof_cal >= best_thresh).astype(int)

# =====================================================
# FINAL RESULTS TABLE
# =====================================================

print("\n" + "=" * 65)
print(f"POOLED OOF EVALUATION — {which} features, calibrated")
print("=" * 65)
print(f"OOF rows evaluated: {len(oof_true)}  "
      f"(positives: {int(oof_true.sum())})")
print(f"\n{'Metric':<30}{'Value':>12}{'95% CI':>22}")
print("-" * 64)
print(f"{'ROC-AUC':<30}{winner['pooled']['auc']:>12.4f}"
      f"  [{auc_lo:.4f}, {auc_hi:.4f}]")
print(f"{'Avg Precision (AP)':<30}{winner['pooled']['ap']:>12.4f}"
      f"  [{ap_lo:.4f}, {ap_hi:.4f}]")
print(f"{'Brier (calibrated)':<30}{brier_cal:>12.4f}")
print(f"{'Best-thresh F1':<30}{best_f1:>12.4f}"
      f"  (threshold={best_thresh})")
print(f"{'Precision @ thresh':<30}"
      f"{precision_score(oof_true, pred_bin, zero_division=0):>12.4f}")
print(f"{'Recall @ thresh':<30}"
      f"{recall_score(oof_true, pred_bin, zero_division=0):>12.4f}")
print("=" * 65)
print("Confusion matrix (OOF, calibrated) [rows=actual, cols=predicted]:")
print(confusion_matrix(oof_true, pred_bin))

# =====================================================
# FINAL MODEL — full data, winning features, best fold params
# =====================================================

print("\nTraining final model on full dataset …")
final_params = winner["best"]["params"] or DEFAULT_PARAMS
spw_full     = max((y_all == 0).sum() / max(y_all.sum(), 1), 1.0)
print(f"  Inheriting hyperparameters from best OOF fold "
      f"(fold {winner['best']['fold']}, val AUC={winner['best']['auc']:.4f})")
print(f"  scale_pos_weight (full data): {spw_full:.2f}")

final_model = make_model(final_params, spw_full)
final_model.fit(X_all[FEATURES], y_all, verbose=False)

# =====================================================
# SAVE ARTIFACT
# =====================================================

artifact = {
    "model":      final_model,
    "calibrator": iso,
    "features":   FEATURES,
    "threshold":  float(best_thresh),
    "params":     final_params,
    "horizon":    HORIZON,
    "ci": {
        "auc_mean": auc_mean, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "ap_mean":  ap_mean,  "ap_lo":  ap_lo,  "ap_hi":  ap_hi,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_level": 0.95,
    },
    "notes": (
        "Score new rows: "
        "p_raw = model.predict_proba(X[features])[:,1]; "
        "p = calibrator.predict(p_raw); "
        "alert if p >= threshold. "
        "Validated on single escalation cycle (2022+); "
        "treat as single-cycle calibrated until more cycles observed."
    ),
}
joblib.dump(artifact, "models/regime_classifier_7d_v3.pkl")

# Feature list + CIs as JSON for use without unpickling
meta = {
    "features":  FEATURES,
    "threshold": float(best_thresh),
    "ci_auc":    {"mean": auc_mean, "lo": auc_lo, "hi": auc_hi},
    "ci_ap":     {"mean": ap_mean,  "lo": ap_lo,  "hi": ap_hi},
    "n_bootstrap": N_BOOTSTRAP,
    "which_stage": which,
}
with open("models/regime_classifier_7d_v3_features.json", "w") as fh:
    json.dump(meta, fh, indent=2)

pd.DataFrame({
    "date":       dates.iloc[winner["idx"]].values,
    "actual":     oof_true,
    "pred_raw":   oof_pred,
    "pred_cal":   oof_cal,
    "pred_label": pred_bin,
}).to_csv("models/oof_predictions_regime_classifier_7d_v3.csv", index=False)

print("\nSaved: models/regime_classifier_7d_v3.pkl  "
      "(model + calibrator + threshold + CIs)")
print("Saved: models/regime_classifier_7d_v3_features.json")
print("Saved: models/oof_predictions_regime_classifier_7d_v3.csv")
print("Saved: models/shap_importance_regime_classifier_v3.csv")
print("Saved: models/calibration_curve_v3.png")