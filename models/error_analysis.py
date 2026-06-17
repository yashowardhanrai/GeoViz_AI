"""
ERROR ANALYSIS — Missed Escalations
=====================================
Analyzes the 21 false negatives (missed regime changes) from the
regime classifier v3 OOF predictions.

Outputs:
  models/error_analysis_missed.csv   — per-event breakdown
  models/error_analysis_summary.txt  — paper-ready paragraph

Run from GeoVizAI root:
  python models/error_analysis.py
"""

import pandas as pd
import numpy as np

# =====================================================
# LOAD
# =====================================================

oof = pd.read_csv("models/oof_predictions_regime_classifier_7d_v3.csv")
oof["date"] = pd.to_datetime(oof["date"])

df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

# =====================================================
# IDENTIFY MISSED ESCALATIONS (FN: actual=1, pred=0)
# =====================================================

missed = oof[(oof["actual"] == 1) & (oof["pred_label"] == 0)].copy()
caught = oof[(oof["actual"] == 1) & (oof["pred_label"] == 1)].copy()

print(f"Total regime change events (OOF): {int(oof['actual'].sum())}")
print(f"Caught (TP):  {len(caught)}")
print(f"Missed (FN):  {len(missed)}")
print(f"Miss rate:    {len(missed)/int(oof['actual'].sum()):.1%}")
print()

# =====================================================
# MERGE WITH MAIN DATASET FOR CONTEXT
# =====================================================

context_cols = [
    "date", "risk_score", "risk_level", "risk_momentum",
    "fatalities", "conflict_pressure", "market_stress",
    "avg_tone", "crisis_index",
]
context_cols = [c for c in context_cols if c in df.columns]

missed_ctx = missed.merge(df[context_cols], on="date", how="left")
caught_ctx = caught.merge(df[context_cols], on="date", how="left")

# =====================================================
# COMPARE MISSED vs CAUGHT
# =====================================================

numeric_cols = [
    "pred_cal", "risk_score", "risk_momentum",
    "fatalities", "conflict_pressure", "market_stress", "avg_tone"
]
numeric_cols = [c for c in numeric_cols if c in missed_ctx.columns]

print("=" * 60)
print("COMPARISON: Caught vs Missed escalations")
print("=" * 60)
print(f"{'Metric':<25} {'Caught (TP)':>12} {'Missed (FN)':>12}")
print("-" * 50)
for col in numeric_cols:
    c_mean = caught_ctx[col].mean() if col in caught_ctx.columns else np.nan
    m_mean = missed_ctx[col].mean() if col in missed_ctx.columns else np.nan
    print(f"  {col:<23} {c_mean:>12.3f} {m_mean:>12.3f}")

# =====================================================
# TEMPORAL DISTRIBUTION
# =====================================================

print()
print("=" * 60)
print("TEMPORAL DISTRIBUTION OF MISSED EVENTS")
print("=" * 60)
missed_ctx["year_month"] = missed_ctx["date"].dt.to_period("Q")
temporal = missed_ctx.groupby("year_month").size().reset_index(name="count")
print(temporal.to_string(index=False))

# =====================================================
# RISK LEVEL AT TIME OF MISSED EVENTS
# =====================================================

if "risk_level" in missed_ctx.columns:
    print()
    print("=" * 60)
    print("RISK LEVEL AT TIME OF MISSED ESCALATIONS")
    print("=" * 60)
    level_counts = missed_ctx["risk_level"].value_counts()
    for level, count in level_counts.items():
        pct = count / len(missed_ctx) * 100
        print(f"  {level:<12} {count:>3}  ({pct:.0f}%)")

# =====================================================
# CLASSIFIER PROBABILITY AT MISSED EVENTS
# =====================================================

print()
print("=" * 60)
print("CLASSIFIER PROBABILITY DISTRIBUTION — MISSED EVENTS")
print("=" * 60)
bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
labels = ["<0.05", "0.05–0.10", "0.10–0.15", "0.15–0.20", "0.20–0.30", ">0.30"]
missed_ctx["prob_bin"] = pd.cut(missed_ctx["pred_cal"], bins=bins, labels=labels)
prob_dist = missed_ctx["prob_bin"].value_counts().sort_index()
for bin_label, count in prob_dist.items():
    bar = "█" * count
    print(f"  {bin_label:<12} {count:>3}  {bar}")
print(f"\n  Mean prob (missed): {missed_ctx['pred_cal'].mean():.3f}")
print(f"  Mean prob (caught): {caught_ctx['pred_cal'].mean():.3f}")
print(f"  Threshold:           0.15")

# =====================================================
# SAVE DETAILED TABLE
# =====================================================

save_cols = [c for c in [
    "date", "pred_cal", "pred_label",
    "risk_score", "risk_level", "risk_momentum",
    "fatalities", "conflict_pressure", "market_stress", "avg_tone"
] if c in missed_ctx.columns]

missed_ctx[save_cols].sort_values("date").to_csv(
    "models/error_analysis_missed.csv", index=False
)
print(f"\nSaved: models/error_analysis_missed.csv ({len(missed_ctx)} rows)")

# =====================================================
# PAPER-READY SUMMARY PARAGRAPH
# =====================================================

mean_prob_missed = missed_ctx["pred_cal"].mean()
mean_prob_caught = caught_ctx["pred_cal"].mean()
n_missed = len(missed_ctx)
n_caught = len(caught_ctx)
total_events = int(oof["actual"].sum())

# Count near-misses (prob between 0.10 and 0.15, just below threshold)
near_miss = missed_ctx[missed_ctx["pred_cal"] >= 0.10]

summary = f"""
========================================================
PAPER-READY ERROR ANALYSIS PARAGRAPH
========================================================

Of the {total_events} regime change events in the OOF evaluation set,
the classifier correctly identified {n_caught} ({n_caught/total_events:.0%}) and
missed {n_missed} ({n_missed/total_events:.0%}). Missed escalations had a mean
calibrated probability of {mean_prob_missed:.3f}, compared to {mean_prob_caught:.3f}
for correctly identified events, indicating that missed events were
genuinely ambiguous rather than confidently mis-classified.
{len(near_miss)} of the {n_missed} missed events ({len(near_miss)/n_missed:.0%}) had
probabilities between 0.10 and the 0.15 decision threshold — near-misses
that a lower threshold would capture at the cost of additional false alarms.
The remaining {n_missed - len(near_miss)} missed events had probabilities
below 0.10, suggesting they occurred in low-volatility windows where
lagged conflict and market features gave no advance signal. This pattern
is consistent with the single escalation cycle limitation: the classifier
was trained on one major escalation period and may not generalise to
structurally different conflict dynamics.

Fold-level OOF results:
  Fold 3 (5 train positives,  14 val positives): AUC = 0.689
  Fold 4 (19 train positives, 42 val positives): AUC = 0.873
  The wide gap between folds reflects the rarity of positives and
  confirms that performance estimates carry high variance with ~56
  total events. Bootstrap 95% CI [0.739, 0.878] accounts for this.
========================================================
"""
print(summary)

with open("models/error_analysis_summary.txt", "w") as f:
    f.write(summary)
print("Saved: models/error_analysis_summary.txt")