"""
SHAP INTERACTION VALUES
========================
Computes SHAP interaction values to reveal which feature PAIRS
jointly drive the risk score predictions.

Unlike mean |SHAP| (which shows individual feature importance),
SHAP interactions show: "when feature A is high AND feature B
is high, how much more does the prediction increase compared
to their individual effects?"

For paper: top interaction pairs reveal joint conflict-market
dynamics (e.g., high fatalities + oil spike = disproportionate
risk increase).

Run from GeoVizAI root:
    python models/shap_interactions.py

Outputs:
    models/shap_interactions_top20.csv
    models/shap_interaction_matrix.csv

Requires: shap, xgboost, joblib, pandas, numpy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import joblib

# =====================================================
# LOAD DATA + MODEL
# =====================================================

print("Loading data and model...")

df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Load the 3-day model (best for interaction analysis —
# captures medium-term dynamics between conflict and markets)
try:
    model = joblib.load("models/xgboost_v6_3d.pkl")
    print("  Loaded: xgboost_v6_3d.pkl")
except FileNotFoundError:
    try:
        model = joblib.load("models/xgboost_v6_1d.pkl")
        print("  Loaded: xgboost_v6_1d.pkl")
    except FileNotFoundError:
        print("ERROR: No saved model found. Run xgboost_model.py first.")
        exit(1)

# =====================================================
# RECONSTRUCT FEATURES
# (mirror xgboost_model.py feature engineering)
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
    df[f"intensity_lag_{lag}"]  = df["intensity"].shift(lag)  \
        if "intensity" in df.columns else 0
    df[f"fatalities_lag_{lag}"] = df["fatalities"].shift(lag) \
        if "fatalities" in df.columns else 0

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

# Get feature columns from model
feature_names = model.get_booster().feature_names
if feature_names is None:
    print("ERROR: model has no feature names. Re-run xgboost_model.py.")
    exit(1)

# Build X using model's exact feature list
X = df[[c for c in feature_names if c in df.columns]].fillna(0)
X = X.reindex(columns=feature_names, fill_value=0)
X = X.dropna()

# Use training window only (2022-03-01 to 2023-12-31)
# to match xgboost_model.py v6.5
dates_avail = df.loc[X.index, "date"] if "date" in df.columns else None
if dates_avail is not None:
    train_mask = dates_avail <= "2023-12-31"
    X_train = X[train_mask.values[:len(X)]]
else:
    X_train = X.iloc[:int(len(X) * 0.65)]

# Use a sample for interaction computation (expensive)
N_SAMPLE = min(200, len(X_train))
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), N_SAMPLE, replace=False)
X_sample = X_train.iloc[sample_idx].reset_index(drop=True)

print(f"  Features: {len(feature_names)}")
print(f"  Sample for interactions: {N_SAMPLE} rows")

# =====================================================
# SHAP INTERACTION VALUES
# =====================================================

print("\nComputing SHAP interaction values (this takes 2-5 minutes)...")
explainer  = shap.TreeExplainer(model)
shap_inter = explainer.shap_interaction_values(X_sample)
# shap_inter shape: (n_samples, n_features, n_features)

print(f"  Interaction matrix shape: {shap_inter.shape}")

# =====================================================
# AGGREGATE INTERACTIONS
# =====================================================

# Mean absolute interaction value per feature pair
mean_inter = np.abs(shap_inter).mean(axis=0)

# Diagonal = main effects (same as standard SHAP)
# Off-diagonal = true interaction effects between pairs
n_feat = len(feature_names)
inter_rows = []

for i in range(n_feat):
    for j in range(i + 1, n_feat):   # upper triangle only
        inter_val = mean_inter[i, j]
        inter_rows.append({
            "feature_1":    feature_names[i],
            "feature_2":    feature_names[j],
            "interaction":  round(float(inter_val), 6),
        })

inter_df = pd.DataFrame(inter_rows).sort_values(
    "interaction", ascending=False
).reset_index(drop=True)

# =====================================================
# MAIN EFFECTS (diagonal)
# =====================================================

main_effects = pd.DataFrame({
    "feature":     feature_names,
    "main_effect": [mean_inter[i, i] for i in range(n_feat)],
}).sort_values("main_effect", ascending=False).reset_index(drop=True)

# =====================================================
# PRINT RESULTS
# =====================================================

print("\n" + "=" * 65)
print("TOP 10 MAIN EFFECTS (diagonal SHAP interaction)")
print("=" * 65)
print(main_effects.head(10).to_string(index=False))

print("\n" + "=" * 65)
print("TOP 20 FEATURE INTERACTIONS (off-diagonal)")
print("=" * 65)
print(f"{'Feature 1':<25} {'Feature 2':<25} {'Interaction':>12}")
print("-" * 65)
for _, row in inter_df.head(20).iterrows():
    print(f"  {row['feature_1']:<23} {row['feature_2']:<23} "
          f"{row['interaction']:>12.4f}")

# =====================================================
# KEY INSIGHT DETECTION
# =====================================================

print("\n--- KEY INTERACTION FINDINGS ---")

# Look for conflict × market interactions
conflict_feats = [f for f in feature_names if any(
    k in f for k in ["fatalities", "intensity", "severity", "events"])]
market_feats   = [f for f in feature_names if any(
    k in f for k in ["oil", "Close", "return", "volatility"])]
news_feats     = [f for f in feature_names if any(
    k in f for k in ["article", "tone", "embedding", "pca"])]

top_cross = inter_df[
    (inter_df["feature_1"].isin(conflict_feats) &
     inter_df["feature_2"].isin(market_feats)) |
    (inter_df["feature_1"].isin(market_feats) &
     inter_df["feature_2"].isin(conflict_feats))
].head(5)

if len(top_cross):
    print("\nStrongest conflict × market interactions:")
    for _, row in top_cross.iterrows():
        print(f"  {row['feature_1']} × {row['feature_2']}: "
              f"{row['interaction']:.4f}")
else:
    print("  No strong conflict × market interactions found")

top_news_conf = inter_df[
    (inter_df["feature_1"].isin(news_feats) &
     inter_df["feature_2"].isin(conflict_feats)) |
    (inter_df["feature_1"].isin(conflict_feats) &
     inter_df["feature_2"].isin(news_feats))
].head(5)

if len(top_news_conf):
    print("\nStrongest news × conflict interactions:")
    for _, row in top_news_conf.iterrows():
        print(f"  {row['feature_1']} × {row['feature_2']}: "
              f"{row['interaction']:.4f}")

# =====================================================
# SAVE
# =====================================================

inter_df.head(20).to_csv("models/shap_interactions_top20.csv", index=False)
print(f"\nSaved: models/shap_interactions_top20.csv")

# Full interaction matrix as CSV
inter_matrix = pd.DataFrame(
    mean_inter,
    index=feature_names,
    columns=feature_names
)
inter_matrix.to_csv("models/shap_interaction_matrix.csv")
print(f"Saved: models/shap_interaction_matrix.csv")

print("\n" + "=" * 65)
print("PAPER TEXT TEMPLATE")
print("=" * 65)
top3 = inter_df.head(3)
print(f"""
SHAP interaction analysis reveals that the strongest feature-pair
interaction is between {top3.iloc[0]['feature_1']} and
{top3.iloc[0]['feature_2']} (mean |SHAP interaction| =
{top3.iloc[0]['interaction']:.4f}), followed by
{top3.iloc[1]['feature_1']} × {top3.iloc[1]['feature_2']}
({top3.iloc[1]['interaction']:.4f}) and
{top3.iloc[2]['feature_1']} × {top3.iloc[2]['feature_2']}
({top3.iloc[2]['interaction']:.4f}). These interaction values
quantify the joint nonlinear contribution of feature pairs beyond
their individual additive effects, providing deeper interpretability
than standard SHAP importance rankings alone.
""")