"""
FIX LAST ROW ZEROS
===================
Forward-fills derived conflict/pressure columns that show zero
on the last row due to incomplete rolling window calculations.

Run from GeoVizAI root:
  python models/fix_last_row.py
"""

import pandas as pd
import numpy as np

PATH = "data/processed/geoviz_risk_dataset.csv"

df = pd.read_csv(PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Rows: {len(df)}")
print(f"Last date: {df['date'].iloc[-1].date()}")
print()

# Columns that should never be zero if fatalities > 0
# These are rolling/derived features that go to zero on the last row
ffill_cols = [
    "event_count", "severity", "conflict_pressure",
    "intensity", "global_pressure", "crisis_index",
    "media_pressure", "market_stress",
    "severity_7d", "severity_30d",
    "fatalities_30d", "fatalities_90d",
    "intensity_7d", "intensity_30d",
]
ffill_cols = [c for c in ffill_cols if c in df.columns]

print("Checking for zero-tail values:")
for col in ffill_cols:
    last_val = df[col].iloc[-1]
    prev_val = df[col].iloc[-2]
    if last_val == 0 and prev_val != 0:
        print(f"  {col}: last={last_val:.3f}, prev={prev_val:.3f} → FIXING")
    else:
        print(f"  {col}: last={last_val:.3f} ✓")

print()

# Forward-fill: replace trailing zeros with last non-zero value
for col in ffill_cols:
    # Only fix if last value is 0 and there are non-zero values before it
    mask = (df[col] == 0) & (df[col].shift(1) != 0)
    if mask.any():
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].ffill()
        print(f"Fixed: {col}")

print()
print("After fix — last row values:")
for col in ffill_cols:
    print(f"  {col}: {df[col].iloc[-1]:.3f}")

df.to_csv(PATH, index=False)
print(f"\nSaved: {PATH}")
print("Restart dashboard to see the fix.")