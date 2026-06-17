import pandas as pd


# ==========================================================
# MAIN FUNCTION
# ==========================================================
#
# IMPORTANT — same-day feature assumption:
# All momentum and sequence features here are computed from
# same-day sentiment/escalation/fear/uncertainty scores
# (i.e. today's FinBERT output on today's GDELT text).
# These are valid for forecasting risk_score[t+h] ONLY IF
# same-day sentiment is genuinely available at prediction
# time — which holds when the pipeline runs end-of-day on
# current GDELT data.  For intraday or pre-market use cases
# these features would need an additional .shift(1) lag.
#
# Rolling means within this file (news_pressure_7/30,
# sequence_risk_7d/30d, etc.) are computed WITHOUT .shift(1)
# on the underlying scores, meaning today's value is included
# in today's rolling average.  This is consistent with the
# same-day-available assumption above — each row's rolling
# window ends at time t, not t-1.
#
# None of these features appear in the model FEATURE_COLS
# (xgboost_model.py) or the regime classifier — they exist
# in the dataset CSV as intermediate/exploratory columns.
# If you add them to a model, add .shift(1) to all of them
# first to maintain the forecasting-safe boundary.
# ==========================================================

def add_news_sequence_features(df):

    # ====================================================
    # SENTIMENT MOMENTUM
    # ====================================================

    df["sentiment_momentum_7"] = (

        df["sentiment_score"]

        -

        df["sentiment_score"].shift(7)

    )

    df["sentiment_momentum_30"] = (

        df["sentiment_score"]

        -

        df["sentiment_score"].shift(30)

    )

    # ====================================================
    # FEAR MOMENTUM
    # ====================================================

    df["fear_momentum_7"] = (

        df["fear_score"]

        -

        df["fear_score"].shift(7)

    )

    df["fear_momentum_30"] = (

        df["fear_score"]

        -

        df["fear_score"].shift(30)

    )

    # ====================================================
    # NEWS PRESSURE
    # ====================================================

    df["news_pressure_7"] = (

        df["article_count_norm"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    df["news_pressure_30"] = (

        df["article_count_norm"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    # ====================================================
    # ESCALATION MOMENTUM
    # ====================================================

    df["escalation_momentum_7"] = (

        df["escalation_score"]

        -

        df["escalation_score"].shift(7)

    )

    df["escalation_momentum_30"] = (

        df["escalation_score"]

        -

        df["escalation_score"].shift(30)

    )

    # ====================================================
    # ARTICLE VELOCITY
    # ====================================================

    df["article_velocity"] = (

        df["article_count_norm"]

        -

        df["article_count_norm"]

        .shift(1)

    )

    # ====================================================
    # ARTICLE ACCELERATION
    # ====================================================

    df["article_acceleration"] = (

        df["article_velocity"]

        .diff()

    )

    # ====================================================
    # ATTENTION SCORE
    # ====================================================

    df["attention_score"] = (

        df["news_pressure_30"]

        *

        (

            1

            +

            df["escalation_score"]

        )

    )

    # ====================================================
    # SEQUENCE RISK
    # ====================================================

    df["sequence_risk"] = (

        0.40

        * df["fear_score"]

        +

        0.30

        * df["escalation_score"]

        +

        0.20

        * df["uncertainty_score"]

        +

        0.10

        * df["article_count_norm"]

    )

    # ====================================================
    # MOVING AVERAGES
    # ====================================================

    df["sequence_risk_7d"] = (

        df["sequence_risk"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    df["sequence_risk_30d"] = (

        df["sequence_risk"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    # ====================================================
    # SEQUENCE MOMENTUM
    # ====================================================

    df["sequence_momentum"] = (

        df["sequence_risk_7d"]

        -

        df["sequence_risk_30d"]

    )

    # ====================================================
    # ESCALATION PERSISTENCE
    # ====================================================

    df["escalation_persistence"] = (

        df["escalation_7d"]

        -

        df["escalation_30d"]

    )

    # ====================================================
    # SEQUENCE VOLATILITY
    # ====================================================

    df["sequence_volatility_30"] = (

        df["sequence_risk"]

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )

    # ====================================================
    # ATTENTION MOMENTUM
    # ====================================================

    df["attention_momentum"] = (

        df["attention_score"]

        -

        df["attention_score"]

        .shift(7)

    )

    # ====================================================
    # CLEAN NUMERIC NaNs
    # ====================================================

    numeric_cols = (

        df

        .select_dtypes(

            include="number"

        )

        .columns

    )

    df[numeric_cols] = (

        df[numeric_cols]

        .fillna(0)

    )

    return df