import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 55)
print("DATASET QUALITY CHECK")
print("=" * 55)
print(f"Rows:           {len(df)}")
print(f"Date range:     {df.date.min().date()} to {df.date.max().date()}")
print(f"Risk score mean:{df.risk_score.mean():.2f}")
print(f"Risk score std: {df.risk_score.std():.2f}")
print(f"Autocorr(1):    {df.risk_score.autocorr(1):.4f}")
print(f"Autocorr(7):    {df.risk_score.autocorr(7):.4f}")

# 80/20 split preview (after ~14 lag rows dropped)
n_model  = len(df) - 14
split    = int(n_model * 0.80)
test_df  = df.iloc[split + 14:]
print(f"\nExpected test set: {len(test_df)} rows")
print(f"Test date range:  {test_df.date.min().date()} to {test_df.date.max().date()}")
print(f"Test std:         {test_df.risk_score.std():.2f}")

naive_r2 = r2_score(
    test_df.risk_score.values[1:],
    test_df.risk_score.values[:-1]
)
print(f"Naive 1d R²:      {naive_r2:.4f}")

print("\nRisk score by year:")
for yr in [2022, 2023, 2024, 2025]:
    sub = df[df.date.dt.year == yr]["risk_score"]
    if len(sub):
        print(f"  {yr}: n={len(sub):3d}  "
              f"mean={sub.mean():.1f}  "
              f"std={sub.std():.2f}  "
              f"max={sub.max():.1f}")

print("\n" + "=" * 55)
if df.risk_score.std() > 6 and df.risk_score.autocorr(1) > 0.90 and len(df) > 1000:
    print("✓ DATASET IS STRONG — proceed with all model runs")
else:
    print("⚠ Dataset may still be weak — check std and autocorr")
print("=" * 55)