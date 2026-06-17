import pandas as pd


# ==========================================================
# BUILD GDELT FEATURES
# ==========================================================
# Lightweight wrapper used when GDELT data needs to be
# pre-processed before passing to build_dataset().
# The main GDELT feature engineering (rolling article counts,
# tone windows, etc.) happens inside dataset_builder.py
# after the merge with market and conflict data.
#
# Note: if this function is not being called in your pipeline
# (i.e. gdelt_df goes directly into build_dataset without
# passing through here), it is safe to leave as-is — it only
# performs renaming and null-filling that build_dataset also
# handles independently.
# ==========================================================

def build_gdelt_features(df):

    df = df.copy()

    df.rename(
        columns={
            "event_date": "date"
        },
        inplace=True
    )

    df["date"] = pd.to_datetime(df["date"])

    df["article_count"] = (
        df["article_count"]
        .fillna(0)
    )

    df["avg_tone"] = (
        df["avg_tone"]
        .fillna(0)
    )

    return df