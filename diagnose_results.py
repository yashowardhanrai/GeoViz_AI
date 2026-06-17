"""
CHECK COUNTRIES AND MAIN.PY FULL CONFIG
========================================
"""
import pandas as pd
import os

# ── Dataset country check ──────────────────────────────────────────────────────
df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

print("=" * 60)
print("COUNTRIES IN CURRENT DATASET")
print("=" * 60)
country_cols = [c for c in df.columns if "country" in c.lower()]
print(f"Country columns found: {country_cols}")

# Check if there's a country column
if "country" in df.columns:
    print(df["country"].value_counts())
else:
    print("No 'country' column — dataset may be single-country or merged")
    print(f"Total columns: {len(df.columns)}")
    print("First 30 columns:", df.columns[:30].tolist())

# ── Show full main.py ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FULL MAIN.PY CONTENT")
print("=" * 60)
if os.path.exists("main.py"):
    with open("main.py") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"{i+1:4d}: {line}", end="")
else:
    print("main.py not found")

# ── Risk score volatility by quarter ──────────────────────────────────────────
print("\n" + "=" * 60)
print("RISK SCORE VOLATILITY BY QUARTER")
print("=" * 60)
df["quarter"] = df["date"].dt.to_period("Q")
qstats = df.groupby("quarter")["risk_score"].agg(["mean", "std", "min", "max", "count"])
print(qstats.to_string())