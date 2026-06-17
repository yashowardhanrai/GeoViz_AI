import pandas as pd
import os

# ── Dataset check ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/geoviz_risk_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

print("=" * 55)
print("CURRENT DATASET")
print("=" * 55)
print(f"  Rows:          {len(df)}")
print(f"  Date range:    {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Calendar days: {(df['date'].max() - df['date'].min()).days}")
print(f"  Missing vs 1075: {1075 - len(df)} rows")
print(f"  Risk score mean: {df['risk_score'].mean():.2f}")
print(f"  Risk score std:  {df['risk_score'].std():.2f}")

# Check for gaps in dates
df_sorted = df.sort_values("date").reset_index(drop=True)
df_sorted["gap"] = df_sorted["date"].diff().dt.days
big_gaps = df_sorted[df_sorted["gap"] > 3][["date", "gap"]]
print(f"\n  Date gaps > 3 days: {len(big_gaps)}")
if len(big_gaps) > 0:
    print(big_gaps.to_string(index=False))

# ── main.py config check ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("MAIN.PY DATE CONFIGURATION")
print("=" * 55)

if os.path.exists("main.py"):
    with open("main.py") as f:
        lines = f.readlines()
    keywords = ["start", "end", "date", "DATE", "2022", "2023",
                "2024", "2025", "2026", "today", "country", "COUNTRY"]
    for i, line in enumerate(lines):
        if any(k in line for k in keywords) and line.strip():
            print(f"  Line {i+1:3d}: {line.rstrip()}")
else:
    print("  main.py not found in current directory")

# ── Check connectors for date config ──────────────────────────────────────────
print("\n" + "=" * 55)
print("CONNECTOR DATE CONFIGS")
print("=" * 55)

connector_files = [
    "src/connectors/acled_connector.py",
    "src/connectors/gdelt_connector.py",
    "src/connectors/yfinance_connector.py",
]
for fpath in connector_files:
    if os.path.exists(fpath):
        with open(fpath) as f:
            content = f.read()
        lines = content.split("\n")
        print(f"\n  {fpath}:")
        for i, line in enumerate(lines):
            if any(k in line for k in ["start", "end", "date", "2022", "2025", "today"]):
                if line.strip() and not line.strip().startswith("#"):
                    print(f"    Line {i+1}: {line.rstrip()}")
    else:
        print(f"  {fpath}: NOT FOUND")