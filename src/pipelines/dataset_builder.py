import pandas as pd

from src.features.risk_score import (
    calculate_risk_score,
    add_risk_trend
)

from src.features.sentiment_features import (
    add_sentiment_features
)

from src.features.event_features import (
    add_event_features
)

from src.features.embedding_features import (
    add_embedding_features
)

from src.features.news_sequence_features import (
    add_news_sequence_features
)

from src.features.macro_features import (
    add_macro_features
)


def build_dataset(

    market_df,

    conflict_df,

    gdelt_df

):

    market_df = market_df.copy()

    conflict_df = conflict_df.copy()

    gdelt_df = gdelt_df.copy()

    # =====================================================
    # DATE CONVERSION
    # =====================================================

    market_df = market_df.reset_index()

    market_df["date"] = (

        pd.to_datetime(

            market_df["Date"]

        )

        .dt.normalize()

    )

    conflict_df["date"] = (

        pd.to_datetime(

            conflict_df["date"]

        )

        .dt.normalize()

    )

    gdelt_df["date"] = (

        pd.to_datetime(

            gdelt_df["event_date"]

        )

        .dt.normalize()

    )

    # =====================================================
    # CONFLICT + MARKET MERGE
    # FIX: conflict_df is now the LEFT (spine) table.
    # This gives calendar days (~1045 rows) instead of
    # trading days only (840 rows). Market data gaps on
    # weekends/holidays are forward-filled below.
    # =====================================================

    dataset = pd.merge(

        conflict_df,    # LEFT spine — calendar days

        market_df,      # RIGHT — trading days, NaN on weekends

        on="date",

        how="left"

    )

    # Forward-fill market columns over weekend/holiday gaps
    # (last known oil price persists until next trading day)
    market_cols = [
        "Close", "High", "Low", "Open", "Volume",
        "daily_return", "volatility_7", "volatility_30",
        "volatility_90", "volatility", "moving_average_7",
        "moving_average_30", "moving_average_90",
        "moving_average", "return_7d", "return_30d",
        "return_90d", "price_change", "price_acceleration",
        "abs_daily_return", "oil_shock", "volatility_change",
        "oil_momentum", "trend_7_30", "trend_30_90",
        "momentum_7", "momentum_30", "momentum_90",
    ]
    for col in market_cols:
        if col in dataset.columns:
            dataset[col] = dataset[col].ffill()

    # =====================================================
    # GDELT MERGE
    # =====================================================

    dataset = pd.merge(

        dataset,

        gdelt_df[
            [

                "date",

                "article_count",

                "avg_tone",

                "source",

                "url",

                "themes",

                "locations",

                "persons",

                "organizations",

                "counts"

            ]

        ],

        on="date",

        how="left"

    )

    # =====================================================
    # HANDLE CONFLICT MISSING VALUES
    # =====================================================

    dataset["fatalities"] = (

        dataset["fatalities"]

        .ffill()

        .fillna(0)

    )

    dataset["intensity"] = (

        dataset["intensity"]

        .ffill()

        .fillna(0)

    )

    # =====================================================
    # HANDLE GDELT MISSING VALUES
    # (no bfill: only fill forward from past values, so a
    #  future article_count/avg_tone reading can't leak
    #  backward into an earlier row's features)
    # =====================================================

    dataset["article_count"] = (

        dataset["article_count"]

        .replace(

            0,

            pd.NA

        )

        .ffill()

    )

    dataset["avg_tone"] = (

        dataset["avg_tone"]

        .replace(

            0,

            pd.NA

        )

        .ffill()

    )

    dataset["article_count"] = (

        dataset["article_count"]

        .fillna(0)

        .clip(

            lower=0

        )

    )

    dataset["avg_tone"] = (

        dataset["avg_tone"]

        .fillna(0)

        .clip(

            lower=-10,

            upper=10

        )

    )

    # =====================================================
    # SORT DATASET
    # =====================================================

    dataset = (

        dataset

        .sort_values(

            "date"

        )

        .reset_index(

            drop=True

        )

    )

    # =====================================================
    # CONFLICT FEATURES
    # =====================================================

    dataset["fatalities_7d"] = (

        dataset["fatalities"]

        .rolling(

            7,

            min_periods=1

        )

        .sum()

    )

    dataset["fatalities_30d"] = (

        dataset["fatalities"]

        .rolling(

            30,

            min_periods=1

        )

        .sum()

    )

    dataset["fatalities_90d"] = (

        dataset["fatalities"]

        .rolling(

            90,

            min_periods=1

        )

        .sum()

    )

    dataset["intensity_7d"] = (

        dataset["intensity"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    dataset["intensity_30d"] = (

        dataset["intensity"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    dataset["intensity_90d"] = (

        dataset["intensity"]

        .rolling(

            90,

            min_periods=1

        )

        .mean()

    )

    # =====================================================
    # NEWS FEATURES
    # =====================================================

    dataset["article_count_7d"] = (

        dataset["article_count"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    dataset["article_count_30d"] = (

        dataset["article_count"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    dataset["article_count_90d"] = (

        dataset["article_count"]

        .rolling(

            90,

            min_periods=1

        )

        .mean()

    )

    dataset["article_volatility_7d"] = (

        dataset["article_count"]

        .rolling(

            7,

            min_periods=1

        )

        .std()

    )

    dataset["article_volatility_30d"] = (

        dataset["article_count"]

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )

    # =====================================================
    # TONE FEATURES
    # =====================================================

    dataset["tone_7d"] = (

        dataset["avg_tone"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    dataset["tone_30d"] = (

        dataset["avg_tone"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    dataset["tone_90d"] = (

        dataset["avg_tone"]

        .rolling(

            90,

            min_periods=1

        )

        .mean()

    )

    dataset["tone_std_7d"] = (

        dataset["avg_tone"]

        .rolling(

            7,

            min_periods=1

        )

        .std()

    )

    dataset["tone_std_30d"] = (

        dataset["avg_tone"]

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )

    # =====================================================
    # MEDIA ATTENTION
    # =====================================================

    dataset["article_count_norm"] = (

        dataset["article_count_30d"]

        /

        dataset["article_count_30d"].max()

    ) * 100

    # =====================================================
    # MARKET STRESS
    # =====================================================

    dataset["market_stress"] = (

        dataset["volatility_30"]

        +

        abs(

            dataset["daily_return"]

        )

        +

        abs(

            dataset["return_30d"]

        ) / 10

    )

    # =====================================================
    # CONFLICT PRESSURE
    # =====================================================

    if "severity" in dataset.columns:

        dataset["conflict_pressure"] = (

            dataset["intensity"]

            +

            dataset["severity"]

        )

    else:

        dataset["conflict_pressure"] = (

            dataset["intensity"]

        )

    # =====================================================
    # MEDIA PRESSURE
    # =====================================================

    dataset["media_pressure"] = (

        dataset["article_count_norm"]

        +

        abs(

            dataset["avg_tone"]

        )

    )

    # =====================================================
    # GLOBAL PRESSURE
    # =====================================================

    dataset["global_pressure"] = (

        dataset["market_stress"]

        +

        dataset["conflict_pressure"]

        +

        dataset["media_pressure"]

    )

    # =====================================================
    # SHOCK FEATURES
    # =====================================================

    dataset["conflict_acceleration"] = (

        dataset["intensity"]

        .diff()

    )

    dataset["fatality_change"] = (

        dataset["fatalities"]

        .diff()

    )

    dataset["tone_change"] = (

        dataset["avg_tone"]

        .diff()

    )

    dataset["news_change"] = (

        dataset["article_count"]

        .diff()

    )

    dataset["news_change"] = (

        dataset["news_change"]

        .clip(

            lower=-100,

            upper=100

        )

    )

    # =====================================================
    # VOLATILITY REGIME
    # =====================================================

    if "volatility_90" in dataset.columns:

        dataset["volatility_regime"] = (

            dataset["volatility_30"]

            >

            dataset["volatility_90"]

        ).astype(int)

    else:

        dataset["volatility_regime"] = 0

    # =====================================================
    # CALCULATE RISK SCORE
    # =====================================================

    dataset["risk_score"] = (

        dataset

        .apply(

            calculate_risk_score,

            axis=1

        )

    )

    # =====================================================
    # LAYER 1
    # RISK FEATURES
    # =====================================================

    dataset = add_risk_trend(

        dataset

    )

    # =====================================================
    # LAYER 2A
    # SENTIMENT FEATURES
    # =====================================================

    dataset = add_sentiment_features(

        dataset

    )

    # =====================================================
    # LAYER 2B
    # EVENT FEATURES
    # =====================================================

    dataset = add_event_features(

        dataset

    )

    # =====================================================
    # LAYER 2C
    # EMBEDDING FEATURES
    # =====================================================

    dataset = add_embedding_features(

        dataset

    )

    # =====================================================
    # LAYER 2D
    # NEWS SEQUENCE FEATURES
    # =====================================================

    dataset = add_news_sequence_features(

        dataset

    )

    # =====================================================
    # LAYER 3
    # MACRO FEATURES
    # =====================================================

    dataset = add_macro_features(

        dataset

    )

    # =====================================================
    # CLEAN NUMERIC NaNs
    # (forward-fill only, then zero-fill any remaining
    #  leading NaNs — no bfill, so future values can't
    #  leak backward into earlier rows)
    # =====================================================

    numeric_cols = (

        dataset

        .select_dtypes(

            include="number"

        )

        .columns

    )

    dataset[numeric_cols] = (

        dataset[numeric_cols]

        .ffill()

        .fillna(0)

    )

    # =====================================================
    # CLEAN STRING COLUMNS
    # =====================================================

    object_cols = (

        dataset

        .select_dtypes(

            include="object"

        )

        .columns

    )

    dataset[object_cols] = (

        dataset[object_cols]

        .fillna("")

    )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    dataset = (

        dataset

        .drop_duplicates()

        .reset_index(

            drop=True

        )

    )

    # =====================================================
    # SORT FINAL DATASET
    # =====================================================

    dataset = (

        dataset

        .sort_values(

            "date"

        )

        .reset_index(

            drop=True

        )

    )

    # =====================================================
    # FINAL SHAPE
    # =====================================================

    print(

        "\nFinal Dataset Shape:",

        dataset.shape

    )

    print(

        "Total Features:",

        len(

            dataset.columns

        )

    )

    # =====================================================
    # RETURN DATASET
    # =====================================================

    return dataset