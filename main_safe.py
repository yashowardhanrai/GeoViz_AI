"""
SAFE MAIN.PY WRAPPER
=====================
Runs the full GeoVizAI pipeline with:
  1. end_date capped at 2025-06-30 (ACLED data boundary)
  2. Validation before overwriting the dataset
  3. Backup of existing dataset before any write
  4. Clear error messages at each stage

Run from GeoVizAI root:
    python main_safe.py

This replaces running main.py directly.
"""

from datetime import date
import time
import os
import shutil
import traceback
import pandas as pd

# ── Safe end date (ACLED boundary) ────────────────────────────────────────────
START_DATE = "2022-02-24"
END_DATE   = "2025-06-30"   # Hard cap — ACLED data ends here
                             # Do NOT use date.today() — fetches beyond ACLED range

DATASET_PATH = "data/processed/geoviz_risk_dataset.csv"
BACKUP_PATH  = "data/processed/geoviz_risk_dataset_backup.csv"

def backup_existing():
    if os.path.exists(DATASET_PATH):
        shutil.copy(DATASET_PATH, BACKUP_PATH)
        df = pd.read_csv(DATASET_PATH)
        print(f"  Backed up existing dataset: {len(df)} rows → {BACKUP_PATH}")
    else:
        print("  No existing dataset to back up")

def validate_dataset(df):
    """Check the new dataset is valid before saving."""
    errors = []

    # Must have minimum rows
    if len(df) < 500:
        errors.append(f"Too few rows: {len(df)} (expected 800+)")

    # Must have risk_score
    if "risk_score" not in df.columns:
        errors.append("Missing 'risk_score' column")

    # Must cover the full date range
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].min() > pd.Timestamp("2022-03-01"):
            errors.append(f"Start date too late: {df['date'].min().date()}")
        if df["date"].max() < pd.Timestamp("2025-01-01"):
            errors.append(f"End date too early: {df['date'].max().date()}")

    # Risk score must be reasonable
    if "risk_score" in df.columns:
        mean_rs = df["risk_score"].mean()
        if mean_rs < 5 or mean_rs > 90:
            errors.append(f"Suspicious risk_score mean: {mean_rs:.2f}")

    # Must not have excessive NaNs
    nan_pct = df.isnull().mean().mean() * 100
    if nan_pct > 20:
        errors.append(f"Too many NaNs: {nan_pct:.1f}%")

    return errors

def restore_backup():
    if os.path.exists(BACKUP_PATH):
        shutil.copy(BACKUP_PATH, DATASET_PATH)
        print(f"  Restored backup: {BACKUP_PATH}")
    else:
        print("  No backup to restore")

def main():
    start_time = time.time()

    print("=" * 60)
    print("GEOVIZAI SAFE PIPELINE RUN")
    print(f"Start: {START_DATE}  End: {END_DATE}")
    print("=" * 60)

    # Step 0: Backup existing dataset
    print("\n[0/6] Backing up existing dataset...")
    backup_existing()

    try:
        # ── IMPORTS ───────────────────────────────────────────────────────────
        from src.connectors.yfinance_connector import YahooFinanceConnector
        from src.connectors.acled_connector import ACLEDConnector
        from src.connectors.gdelt_connector import GDELTConnector
        from src.features.market_features import add_market_features
        from src.features.conflict_features import conflict_intensity
        from src.features.risk_score import classify_risk
        from src.pipelines.dataset_builder import build_dataset
        from src.storage.csv_store import save_dataset

        # ── STEP 1: MARKET DATA ───────────────────────────────────────────────
        print("\n[1/6] Loading market data (yFinance)...")
        try:
            market_connector = YahooFinanceConnector()
            market_data = market_connector.pull(START_DATE, END_DATE)
            oil_df   = add_market_features(market_data[0])
            gold_df  = add_market_features(market_data[1])
            sp500_df = add_market_features(market_data[2])
            dxy_df   = add_market_features(market_data[3])
            print(f"  Oil records:   {len(oil_df)}")
            print(f"  Gold records:  {len(gold_df)}")
            print(f"  SP500 records: {len(sp500_df)}")
            print(f"  DXY records:   {len(dxy_df)}")
        except Exception as e:
            print(f"  ERROR in market data: {e}")
            traceback.print_exc()
            restore_backup()
            return

        # ── STEP 2: ACLED DATA ────────────────────────────────────────────────
        print("\n[2/6] Loading conflict data (ACLED)...")
        try:
            acled_connector = ACLEDConnector()
            raw_records = acled_connector.pull(START_DATE, END_DATE)
            print(f"  Raw ACLED events: {len(raw_records)}")
            if len(raw_records) < 1000:
                print(f"  WARNING: Very few ACLED events ({len(raw_records)}) "
                      f"— API may have rate-limited or returned partial data")
        except Exception as e:
            print(f"  ERROR in ACLED: {e}")
            traceback.print_exc()
            restore_backup()
            return

        # ── STEP 3: CONFLICT FEATURES ─────────────────────────────────────────
        print("\n[3/6] Building conflict features...")
        try:
            conflict_df = conflict_intensity(raw_records)
            print(f"  Conflict days: {len(conflict_df)}")
            if len(conflict_df) < 500:
                print(f"  WARNING: Only {len(conflict_df)} conflict days "
                      f"— expected 800+")
        except Exception as e:
            print(f"  ERROR in conflict features: {e}")
            traceback.print_exc()
            restore_backup()
            return

        # ── STEP 4: GDELT DATA ────────────────────────────────────────────────
        print("\n[4/6] Loading GDELT data...")
        try:
            gdelt_connector = GDELTConnector()
            gdelt_df = gdelt_connector.pull(START_DATE, END_DATE)
            print(f"  GDELT records: {len(gdelt_df)}")
        except Exception as e:
            print(f"  ERROR in GDELT: {e}")
            print("  Continuing without GDELT (will use zero-fills)...")
            gdelt_df = pd.DataFrame()   # empty — pipeline handles this

        # ── STEP 5: BUILD DATASET ─────────────────────────────────────────────
        print("\n[5/6] Building merged dataset...")
        try:
            dataset = build_dataset(
                market_df=oil_df,
                conflict_df=conflict_df,
                gdelt_df=gdelt_df
            )
            print(f"  Dataset after merge: {dataset.shape}")

            # Add gold, SP500, DXY
            for df_asset, cols, renames in [
                (gold_df,  ["Close","daily_return","volatility_30","return_30d"],
                 {"Close":"gold_close","daily_return":"gold_return",
                  "volatility_30":"gold_volatility","return_30d":"gold_return_30d"}),
                (sp500_df, ["Close","daily_return","volatility_30","return_30d"],
                 {"Close":"sp500_close","daily_return":"sp500_return",
                  "volatility_30":"sp500_volatility","return_30d":"sp500_return_30d"}),
                (dxy_df,   ["Close","daily_return","volatility_30","return_30d"],
                 {"Close":"dxy_close","daily_return":"dxy_return",
                  "volatility_30":"dxy_volatility","return_30d":"dxy_return_30d"}),
            ]:
                df_asset = df_asset.reset_index()
                df_asset["date"] = pd.to_datetime(df_asset["Date"])
                dataset = pd.merge(
                    dataset,
                    df_asset[["date"] + cols].rename(columns=renames),
                    on="date", how="left"
                )

            # Forward fill only (no backward fill — leakage prevention)
            numeric_cols = dataset.select_dtypes(include="number").columns
            dataset[numeric_cols] = dataset[numeric_cols].ffill().fillna(0)
            dataset = dataset.fillna("")

            # Risk level
            dataset["risk_level"] = dataset["risk_score"].apply(classify_risk)

            print(f"  Final dataset: {dataset.shape}")

        except Exception as e:
            print(f"  ERROR building dataset: {e}")
            traceback.print_exc()
            restore_backup()
            return

        # ── STEP 6: VALIDATE + SAVE ───────────────────────────────────────────
        print("\n[6/6] Validating and saving...")
        errors = validate_dataset(dataset)

        if errors:
            print(f"  VALIDATION FAILED — NOT saving:")
            for err in errors:
                print(f"    ✗ {err}")
            print(f"  Existing dataset preserved.")
            restore_backup()
            return
        else:
            print(f"  Validation passed:")
            print(f"    ✓ Rows:       {len(dataset)}")
            if "date" in dataset.columns:
                dataset["date"] = pd.to_datetime(dataset["date"])
                print(f"    ✓ Date range: {dataset['date'].min().date()} "
                      f"to {dataset['date'].max().date()}")
            print(f"    ✓ Risk score mean: {dataset['risk_score'].mean():.2f}")

            save_dataset(dataset, "geoviz_risk_dataset.csv")
            print(f"\n  Dataset saved: {DATASET_PATH}")

        # ── SUMMARY ───────────────────────────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)
        print("\n" + "=" * 60)
        print(f"Pipeline complete in {elapsed}s")
        print(f"Dataset: {len(dataset)} rows")
        if "date" in dataset.columns:
            print(f"Range:   {dataset['date'].min().date()} "
                  f"to {dataset['date'].max().date()}")
        print(f"Backup:  {BACKUP_PATH}")
        print("=" * 60)

        # Compare with backup
        if os.path.exists(BACKUP_PATH):
            old_df = pd.read_csv(BACKUP_PATH)
            diff = len(dataset) - len(old_df)
            print(f"\nRow change: {len(old_df)} → {len(dataset)} "
                  f"({'+'if diff>=0 else ''}{diff} rows)")
            if diff < -50:
                print("  WARNING: Dataset shrank by >50 rows. "
                      "Consider restoring backup.")
                print(f"  To restore: copy {BACKUP_PATH} to {DATASET_PATH}")

    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        traceback.print_exc()
        print("\nRestoring backup...")
        restore_backup()


if __name__ == "__main__":
    main()