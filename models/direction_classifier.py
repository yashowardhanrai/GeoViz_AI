"""
DIRECTION CLASSIFIER — Gap 2 Fix
==================================
Converts the regression problem into a 3-class direction problem:
  Decrease  : 7-day change < -2 points
  Stable    : 7-day change in [-2, +2]
  Increase  : 7-day change > +2 points

Naive persistence ALWAYS predicts "Stable" → F1=0 on Decrease/Increase
XGBoost can genuinely beat this.

Run from GeoVizAI root:
    python models/direction_classifier.py

Outputs:
    models/direction_classifier_results.csv
    models/direction_confusion_matrix.png
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
HORIZON      = 7        # 7-day direction prediction
CHANGE_THRES = 2.0      # ±2 points = "stable" zone

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded: {len(df)} rows ({df.date.min().date()} to {df.date.max().date()})")

# ── Feature engineering (mirror xgboost_model.py) ────────────────────────────
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
    df[f"intensity_lag_{lag}"]  = df["intensity"].shift(lag)  if "intensity"  in df.columns else 0
    df[f"fatalities_lag_{lag}"] = df["fatalities"].shift(lag) if "fatalities" in df.columns else 0
for col, name in [
    ("severity_7d","severity_7d_lag1"),("events_30d","events_30d_lag1"),
    ("fatality_momentum","fatality_mom_lag1"),("crisis_interaction","crisis_lag1"),
]:
    if col in df.columns: df[name] = df[col].shift(1)
for lag in [1, 3, 7]:
    df[f"oil_lag_{lag}"] = df["Close"].shift(lag)
df["oil_ret_1"]  = df["Close"].shift(1).pct_change()
df["oil_ret_7"]  = df["Close"].shift(1) / df["Close"].shift(8) - 1
df["is_monday"]  = (df["date"].dt.dayofweek == 0).astype(int)
df["month"]      = df["date"].dt.month
pca_cols = [c for c in df.columns if c.startswith("embedding_pca_")][:10]

FEATURE_COLS = (
    [c for c in df.columns if c.startswith("rs_")]
    + [c for c in df.columns if c.startswith("intensity_lag_")]
    + [c for c in df.columns if c.startswith("fatalities_lag_")]
    + ["severity_7d_lag1","events_30d_lag1","fatality_mom_lag1","crisis_lag1"]
    + [c for c in df.columns if c.startswith("oil_")]
    + ["is_monday","month"] + pca_cols
)
FEATURE_COLS = [c for c in dict.fromkeys(FEATURE_COLS) if c in df.columns]
CIRCULAR     = {"oil_shock","oil_momentum"}
FEATURE_COLS = [c for c in FEATURE_COLS if c not in CIRCULAR]

# ── Build direction target ────────────────────────────────────────────────────
df["future_change"] = df["risk_score"].shift(-HORIZON) - df["risk_score"]
df["direction"] = pd.cut(
    df["future_change"],
    bins=[-np.inf, -CHANGE_THRES, CHANGE_THRES, np.inf],
    labels=["Decrease", "Stable", "Increase"]
)

keep = FEATURE_COLS + ["direction", "future_change", "risk_score", "date"]
keep = list(dict.fromkeys(keep))
data = df[keep].replace([np.inf,-np.inf], np.nan).dropna().copy()
data = data.reset_index(drop=True)
print(f"After dropna: {len(data)} rows")

# ── Class distribution ────────────────────────────────────────────────────────
print("\nDirection class distribution:")
counts = data["direction"].value_counts()
for cls, cnt in counts.items():
    print(f"  {cls}: {cnt} ({cnt/len(data)*100:.1f}%)")

# ── Date-based split (same as xgboost_model.py v6.5) ─────────────────────────
mask_train = data["date"] <= "2023-12-31"
mask_test  = data["date"] >= "2024-01-01"
data_train = data[mask_train].reset_index(drop=True)
data_test  = data[mask_test].reset_index(drop=True)

X_train = data_train[FEATURE_COLS].fillna(0)
y_train = data_train["direction"].astype(str)
X_test  = data_test[FEATURE_COLS].fillna(0)
y_test  = data_test["direction"].astype(str)

print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"Test class distribution:")
for cls, cnt in pd.Series(y_test).value_counts().items():
    print(f"  {cls}: {cnt} ({cnt/len(y_test)*100:.1f}%)")

# ── Naive baseline: always predict "Stable" ───────────────────────────────────
print("\n--- NAIVE BASELINE (always predict Stable) ---")
naive_pred = ["Stable"] * len(y_test)
naive_f1   = f1_score(y_test, naive_pred, average="macro", zero_division=0)
naive_acc  = accuracy_score(y_test, naive_pred)
print(f"  Macro F1: {naive_f1:.4f}")
print(f"  Accuracy: {naive_acc:.4f}")
print(classification_report(y_test, naive_pred, zero_division=0))

# ── Optuna tuning ─────────────────────────────────────────────────────────────
print("\nRunning Optuna (50 trials) for direction classifier...")
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        "objective":        "multi:softprob",
        "num_class":        3,
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "tree_method":      "hist",
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "gamma":            trial.suggest_float("gamma", 0.1, 3.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.5, 8.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0),
        "scale_pos_weight": 1,
        "eval_metric":      "mlogloss",
        "early_stopping_rounds": 15,
    }
    scores = []
    for tr_idx, va_idx in tscv.split(X_train):
        Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        label_map = {"Decrease": 0, "Stable": 1, "Increase": 2}
        ytr_enc = ytr.map(label_map)
        yva_enc = yva.map(label_map)
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr, ytr_enc, eval_set=[(Xva, yva_enc)], verbose=False)
        pred = m.predict(Xva)
        pred_labels = [list(label_map.keys())[p] for p in pred]
        scores.append(f1_score(yva, pred_labels, average="macro", zero_division=0))
    return float(np.mean(scores))

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f"Best CV macro F1: {study.best_value:.4f}")

# ── Train final model ─────────────────────────────────────────────────────────
label_map = {"Decrease": 0, "Stable": 1, "Increase": 2}
inv_map   = {0: "Decrease", 1: "Stable", 2: "Increase"}

best_params = {
    **study.best_params,
    "objective":    "multi:softprob",
    "num_class":    3,
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
    "tree_method":  "hist",
}

y_train_enc = y_train.map(label_map)
y_test_enc  = y_test.map(label_map)

model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train_enc)

# ── Evaluate ──────────────────────────────────────────────────────────────────
proba      = model.predict_proba(X_test)
pred_enc   = model.predict(X_test)
pred_label = [inv_map[p] for p in pred_enc]

macro_f1 = f1_score(y_test, pred_label, average="macro", zero_division=0)
acc      = accuracy_score(y_test, pred_label)

# Per-class F1
report = classification_report(y_test, pred_label,
                                target_names=["Decrease","Stable","Increase"],
                                output_dict=True, zero_division=0)

# Multi-class AUC (OvR)
y_bin = label_binarize(y_test_enc, classes=[0,1,2])
auc_macro = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")

print("\n" + "="*65)
print("DIRECTION CLASSIFIER RESULTS")
print("="*65)
print(f"Macro F1:        {macro_f1:.4f}  (naive: {naive_f1:.4f})")
print(f"Accuracy:        {acc:.4f}     (naive: {naive_acc:.4f})")
print(f"AUC (macro OvR): {auc_macro:.4f}")
print()
print("Per-class results:")
for cls in ["Decrease", "Stable", "Increase"]:
    r = report.get(cls, {})
    print(f"  {cls:<10}: F1={r.get('f1-score',0):.3f}  "
          f"Prec={r.get('precision',0):.3f}  "
          f"Rec={r.get('recall',0):.3f}  "
          f"n={int(r.get('support',0))}")
print()
print("Confusion matrix (rows=actual, cols=predicted):")
print("              Decrease  Stable  Increase")
cm = confusion_matrix(y_test, pred_label,
                      labels=["Decrease","Stable","Increase"])
for i, row_lbl in enumerate(["Decrease","Stable","Increase"]):
    print(f"  {row_lbl:<10}  {cm[i][0]:>6}  {cm[i][1]:>6}  {cm[i][2]:>8}")

print(f"\nBeats naive: {'YES' if macro_f1 > naive_f1 else 'NO'}")
print(f"Improvement: {macro_f1 - naive_f1:+.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame({
    "date":      data_test["date"].values,
    "actual":    y_test.values,
    "predicted": pred_label,
    "prob_dec":  proba[:,0],
    "prob_sta":  proba[:,1],
    "prob_inc":  proba[:,2],
}).to_csv("models/direction_classifier_results.csv", index=False)
print("Saved: models/direction_classifier_results.csv")

joblib.dump(model, "models/direction_classifier_7d.pkl")
print("Saved: models/direction_classifier_7d.pkl")