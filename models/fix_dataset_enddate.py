"""
FIX DATASET END DATE
=====================
Caps geoviz_risk_dataset.csv at 2025-06-30 to remove empty 2026 rows
that cause crisis_index and global_pressure to drop to zero.

Run once from GeoVizAI root:
  python models/fix_dataset_enddate.py

This does NOT require re-running main.py — it filters the existing CSV.
After running, restart the Streamlit dashboard to see the fix.
"""

import pandas as pd

END_DATE = "2025-06-30"
PATH = "data/processed/geoviz_risk_dataset.csv"

df = pd.read_csv(PATH)
df["date"] = pd.to_datetime(df["date"])

before = len(df)
df = df[df["date"] <= END_DATE].copy()
after = len(df)

print(f"Rows before: {before}")
print(f"Rows after:  {after}  (removed {before - after} rows after {END_DATE})")
print(f"Date range:  {df['date'].min().date()} → {df['date'].max().date()}")

# Verify crisis_index is no longer zero-tailed
last_30 = df.tail(30)
print(f"\nLast 30 rows — crisis_index stats:")
print(f"  mean: {last_30['crisis_index'].mean():.2f}")
print(f"  min:  {last_30['crisis_index'].min():.2f}")
print(f"  zeros: {(last_30['crisis_index'] == 0).sum()}")

df.to_csv(PATH, index=False)
print(f"\nSaved: {PATH}")
print("Restart dashboard to see the fix.")