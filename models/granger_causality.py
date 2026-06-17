"""
GRANGER CAUSALITY ANALYSIS
===========================
Tests whether news volume, conflict intensity, and oil returns
Granger-cause the geopolitical risk score and each other.

H0: X does NOT Granger-cause Y (X adds no predictive power
    for Y beyond Y's own lags).

Uses statsmodels grangercausalitytests with lag orders 1, 3, 7.
Reports F-statistic and p-value for each direction.

Run from GeoVizAI root:
    python models/granger_causality.py

Outputs:
    models/granger_causality_results.csv
    models/granger_causality_summary.txt
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

RANDOM_STATE = 42
LAGS = [1, 3, 7]    # test at 1-day, 3-day, 7-day lag orders

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 65)
print("GRANGER CAUSALITY ANALYSIS")
print(f"Dataset: {len(df)} rows "
      f"({df.date.min().date()} to {df.date.max().date()})")
print("=" * 65)

# =====================================================
# SELECT SERIES
# =====================================================

# Core variables for causality testing
variables = {
    "risk_score":    df["risk_score"],
    "conflict_int":  df["intensity"]       if "intensity"      in df.columns
                     else df.get("conflict_pressure", pd.Series(dtype=float)),
    "article_count": df["article_count"]   if "article_count"  in df.columns
                     else pd.Series(dtype=float),
    "avg_tone":      df["avg_tone"]        if "avg_tone"       in df.columns
                     else pd.Series(dtype=float),
    "oil_return":    df["daily_return"]    if "daily_return"   in df.columns
                     else pd.Series(dtype=float),
    "fatalities":    df["fatalities"]      if "fatalities"     in df.columns
                     else pd.Series(dtype=float),
}

# Remove empty series
variables = {k: v for k, v in variables.items()
             if v is not None and len(v.dropna()) > 50}

print(f"\nVariables tested: {list(variables.keys())}")

# =====================================================
# STATIONARITY CHECK (ADF test)
# =====================================================

print("\n--- Augmented Dickey-Fuller stationarity test ---")
stationarity = {}
for name, series in variables.items():
    s = series.dropna()
    try:
        adf_stat, pval, _, _, _, _ = adfuller(s, autolag="AIC")
        is_stationary = pval < 0.05
        stationarity[name] = is_stationary
        print(f"  {name:<18}: ADF={adf_stat:+.3f}  p={pval:.4f}  "
              f"{'✓ stationary' if is_stationary else '✗ non-stationary → differencing'}")
    except Exception as e:
        stationarity[name] = False
        print(f"  {name:<18}: ERROR — {e}")

# First-difference non-stationary series
variables_proc = {}
for name, series in variables.items():
    if stationarity.get(name, False):
        variables_proc[name] = series.values
    else:
        variables_proc[name] = series.diff().fillna(0).values
        if name != "risk_score":   # keep risk_score level for interpretability
            print(f"  → {name}: using first differences")

# =====================================================
# GRANGER CAUSALITY TESTS
# =====================================================

print("\n--- Granger causality tests ---")
print(f"Lag orders tested: {LAGS}\n")

# Define test pairs: (cause → effect) pairs of interest
test_pairs = [
    # Does conflict cause risk score?
    ("conflict_int",  "risk_score"),
    # Does news volume cause risk score?
    ("article_count", "risk_score"),
    # Does news tone cause risk score?
    ("avg_tone",      "risk_score"),
    # Does oil return cause risk score?
    ("oil_return",    "risk_score"),
    # Does fatalities cause risk score?
    ("fatalities",    "risk_score"),
    # Reverse: does risk score cause conflict?
    ("risk_score",    "conflict_int"),
    # Does risk score cause news volume?
    ("risk_score",    "article_count"),
    # Does conflict cause news volume (media response)?
    ("conflict_int",  "article_count"),
    # Does news cause conflict (escalation via media)?
    ("article_count", "conflict_int"),
    # Does oil cause conflict?
    ("oil_return",    "conflict_int"),
]

# Keep only pairs where both variables exist
test_pairs = [(x, y) for x, y in test_pairs
              if x in variables_proc and y in variables_proc]

rows = []

for cause, effect in test_pairs:
    x = variables_proc[cause]
    y = variables_proc[effect]

    # Align lengths
    min_len = min(len(x), len(y))
    data_pair = np.column_stack([
        y[:min_len],   # grangercausalitytests expects [effect, cause]
        x[:min_len]
    ])

    print(f"  {cause} → {effect}:")

    best_lag = None
    best_pval = 1.0

    for lag in LAGS:
        try:
            results = grangercausalitytests(
                data_pair, maxlag=lag, verbose=False
            )
            # Get result for this specific lag
            res = results[lag]
            f_stat = res[0]["ssr_ftest"][0]
            p_val  = res[0]["ssr_ftest"][1]
            sig    = "***" if p_val < 0.001 else \
                     "**"  if p_val < 0.01  else \
                     "*"   if p_val < 0.05  else "n.s."

            print(f"    lag={lag}: F={f_stat:.3f}  p={p_val:.4f}  {sig}")

            rows.append({
                "cause":       cause,
                "effect":      effect,
                "lag":         lag,
                "f_stat":      round(f_stat, 4),
                "p_value":     round(p_val, 6),
                "significant": p_val < 0.05,
                "stars":       sig,
                "direction":   f"{cause} → {effect}",
            })

            if p_val < best_pval:
                best_pval = p_val
                best_lag  = lag

        except Exception as e:
            print(f"    lag={lag}: ERROR — {e}")
            rows.append({
                "cause": cause, "effect": effect,
                "lag": lag, "f_stat": np.nan,
                "p_value": np.nan, "significant": False,
                "stars": "err",
                "direction": f"{cause} → {effect}",
            })

    # Summary for this pair
    if best_lag is not None and best_pval < 0.05:
        print(f"    → SIGNIFICANT at lag {best_lag} (p={best_pval:.4f})")
    else:
        print(f"    → No significant Granger causality found")
    print()

# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(rows)
results_df.to_csv("models/granger_causality_results.csv", index=False)
print(f"Saved: models/granger_causality_results.csv")

# =====================================================
# SUMMARY TABLE — for paper
# =====================================================

print("\n" + "=" * 65)
print("GRANGER CAUSALITY SUMMARY — For Paper")
print("=" * 65)
print(f"{'Causal Direction':<35} {'Lag 1':>8} {'Lag 3':>8} {'Lag 7':>8}")
print("-" * 65)

for cause, effect in test_pairs:
    direction = f"{cause} → {effect}"
    row_str = f"{direction:<35}"
    for lag in LAGS:
        match = results_df[
            (results_df["cause"] == cause) &
            (results_df["effect"] == effect) &
            (results_df["lag"] == lag)
        ]
        if len(match) and not pd.isna(match.iloc[0]["p_value"]):
            p = match.iloc[0]["p_value"]
            s = match.iloc[0]["stars"]
            row_str += f"  {p:.4f}{s:>3}"
        else:
            row_str += f"{'n/a':>11}"
    print(row_str)

print("=" * 65)
print("*** p<0.001  ** p<0.01  * p<0.05  n.s. = not significant")

# =====================================================
# KEY FINDINGS for paper text
# =====================================================

sig_results = results_df[results_df["significant"]].copy()

print("\n--- KEY SIGNIFICANT FINDINGS ---")
if len(sig_results) == 0:
    print("  No significant Granger causality found at any lag.")
    print("  Interpretation: variables evolve independently — ")
    print("  each source adds orthogonal signal to the risk score.")
else:
    for _, row in sig_results.drop_duplicates(
            subset=["cause","effect"]).iterrows():
        print(f"  {row['direction']}: "
              f"significant at lag {row['lag']} "
              f"(p={row['p_value']:.4f} {row['stars']})")

# Save summary text
summary_lines = [
    "GRANGER CAUSALITY ANALYSIS SUMMARY",
    f"Dataset: {len(df)} rows",
    f"Lag orders: {LAGS}",
    "",
    "Significant causal directions (p < 0.05):",
]
for _, row in sig_results.drop_duplicates(
        subset=["cause","effect"]).iterrows():
    summary_lines.append(
        f"  {row['direction']}: lag={row['lag']}, "
        f"F={row['f_stat']:.3f}, p={row['p_value']:.4f} {row['stars']}"
    )
if len(sig_results) == 0:
    summary_lines.append("  None — variables evolve independently")

with open("models/granger_causality_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print("\nSaved: models/granger_causality_summary.txt")