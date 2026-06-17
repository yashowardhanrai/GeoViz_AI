import pandas as pd
import numpy as np
from scipy.stats import entropy


def calculate_risk_score(row):

    # =====================================================
    # MARKET VARIABLES
    # =====================================================

    daily_return = row.get(
        "daily_return",
        0
    )

    volatility = row.get(
        "volatility_30",
        row.get(
            "volatility",
            0
        )
    )

    oil_price = row.get(
        "Close",
        0
    )

    return_30d = row.get(
        "return_30d",
        0
    )

    # =====================================================
    # CONFLICT VARIABLES
    # =====================================================

    intensity = row.get(
        "intensity",
        0
    )

    intensity_30d = row.get(
        "intensity_30d",
        0
    )

    fatalities = row.get(
        "fatalities",
        0
    )

    fatalities_30d = row.get(
        "fatalities_30d",
        0
    )

    severity = row.get(
        "severity",
        0
    )

    severity_7d = row.get(
        "severity_7d",
        0
    )

    event_count = row.get(
        "event_count",
        0
    )

    conflict_acceleration = row.get(
        "conflict_acceleration",
        0
    )

    fatality_change = row.get(
        "fatality_change",
        0
    )

    # =====================================================
    # NEWS VARIABLES
    # =====================================================

    article_count_norm = row.get(
        "article_count_norm",
        0
    )

    avg_tone = row.get(
        "avg_tone",
        0
    )

    tone_30d = row.get(
        "tone_30d",
        0
    )

    news_change = row.get(
        "news_change",
        0
    )

    tone_change = row.get(
        "tone_change",
        0
    )

    # =====================================================
    # HANDLE NaNs
    # =====================================================

    values = [

        daily_return,
        volatility,
        oil_price,
        return_30d,

        intensity,
        intensity_30d,

        fatalities,
        fatalities_30d,

        severity,
        severity_7d,

        event_count,

        conflict_acceleration,
        fatality_change,

        article_count_norm,
        avg_tone,
        tone_30d,
        news_change,
        tone_change

    ]

    values = [

        0 if pd.isna(v)

        else v

        for v in values

    ]

    (

        daily_return,
        volatility,
        oil_price,
        return_30d,

        intensity,
        intensity_30d,

        fatalities,
        fatalities_30d,

        severity,
        severity_7d,

        event_count,

        conflict_acceleration,
        fatality_change,

        article_count_norm,
        avg_tone,
        tone_30d,
        news_change,
        tone_change

    ) = values

    # =====================================================
    # MARKET SCORES
    # =====================================================

    oil_score = min(

        abs(daily_return) * 5,

        100

    )

    volatility_score = min(

        volatility * 10,

        100

    )

    trend_score = min(

        abs(return_30d),

        100

    )

    # =====================================================
    # CONFLICT SCORES
    # =====================================================

    intensity_score = min(

        intensity * 3,

        100

    )

    intensity_memory_score = min(

        intensity_30d * 2,

        100

    )

    fatality_score = min(

        fatalities / 10,

        100

    )

    conflict_memory_score = min(

        fatalities_30d / 50,

        100

    )

    severity_score = min(

        severity,

        100

    )

    severity_memory_score = min(

        severity_7d,

        100

    )

    event_score = min(

        event_count,

        100

    )

    acceleration_score = min(

        abs(conflict_acceleration) * 10,

        100

    )

    fatality_change_score = min(

        abs(fatality_change),

        100

    )

    # =====================================================
    # NEWS SCORES
    # =====================================================

    article_score = min(

        article_count_norm,

        100

    )

    tone_score = min(

        abs(avg_tone) * 10,

        100

    )

    tone_memory_score = min(

        abs(tone_30d) * 10,

        100

    )

    news_change_score = min(

        abs(news_change),

        100

    )

    tone_change_score = min(

        abs(tone_change) * 20,

        100

    )

    # =====================================================
    # OIL PREMIUM
    # =====================================================

    if oil_price >= 120:

        price_score = 100

    elif oil_price >= 100:

        price_score = 75

    elif oil_price >= 80:

        price_score = 50

    else:

        price_score = 20

    # =====================================================
    # FINAL SCORE
    # Weights sum to 1.02 (slight over-unity by design).
    # The min(score, 100) cap at the end ensures the output
    # is always in [0, 100] regardless, so the 0.02 excess
    # has no practical effect on the score range.
    # Do not change weights without re-validating regime
    # thresholds (25/50/75) against historical data.
    # =====================================================

    score = (

        0.10 * oil_score +

        0.10 * volatility_score +

        0.05 * trend_score +

        0.15 * intensity_score +

        0.10 * intensity_memory_score +

        0.10 * fatality_score +

        0.10 * conflict_memory_score +

        0.10 * severity_score +

        0.05 * severity_memory_score +

        0.03 * event_score +

        0.03 * acceleration_score +

        0.02 * fatality_change_score +

        0.03 * article_score +

        0.02 * tone_score +

        0.01 * tone_memory_score +

        0.005 * news_change_score +

        0.005 * tone_change_score +

        0.02 * price_score

    )

    return round(

        min(score, 100),

        2

    )


def classify_risk(score):

    if score < 20:

        return "Low"

    elif score < 40:

        return "Moderate"

    elif score < 70:

        return "High"

    else:

        return "Critical"


def add_risk_trend(df):

    # ====================================================
    # RISK MOVING AVERAGES
    # ====================================================

    df["risk_7d"] = (

        df["risk_score"]

        .rolling(
            7,
            min_periods=1
        )

        .mean()

    )

    df["risk_30d"] = (

        df["risk_score"]

        .rolling(
            30,
            min_periods=1
        )

        .mean()

    )

    df["risk_90d"] = (

        df["risk_score"]

        .rolling(
            90,
            min_periods=1
        )

        .mean()

    )

    # ====================================================
    # SHOCK SCORE
    # ====================================================

    df["shock_score"] = (

        abs(
            df["conflict_acceleration"]
        ) * 10

        +

        abs(
            df["fatality_change"]
        ) * 0.20

        +

        abs(
            df["tone_change"]
        ) * 5

        +

        abs(
            df["news_change"]
        ) * 0.10

    )

    df["shock_score"] = (

        df["shock_score"]

        .clip(
            upper=100
        )

    )

    # ====================================================
    # RISK CHANGE
    # ====================================================

    df["risk_change"] = (

        df["risk_score"]

        .diff()

    )

    # ====================================================
    # RISK ACCELERATION
    # ====================================================

    df["risk_acceleration"] = (

        df["risk_change"]

        .diff()

    )

    # ====================================================
    # RISK MOMENTUM
    # ====================================================

    df["risk_momentum"] = (

        df["risk_7d"]

        -

        df["risk_30d"]

    )

    # ====================================================
    # RISK PERCENTILE
    # Expanding-window rank: each row ranked only against
    # rows up to and including itself — no future leakage.
    # ====================================================

    df["risk_percentile"] = (

        df["risk_score"]

        .expanding(
            min_periods=1
        )

        .rank(
            pct=True
        )

        * 100

    )

    # ====================================================
    # RISK Z-SCORE
    # ====================================================

    df["risk_zscore"] = (

        df["risk_score"]

        -

        df["risk_score"]
        .rolling(
            90,
            min_periods=1
        )
        .mean()

    ) / (

        df["risk_score"]
        .rolling(
            90,
            min_periods=1
        )
        .std()

    )

    # ====================================================
    # RISK REGIME
    # Absolute thresholds against risk_score (no future data).
    # ====================================================

    df["risk_regime"] = 0

    df.loc[
        df["risk_score"] >= 25,
        "risk_regime"
    ] = 1

    df.loc[
        df["risk_score"] >= 50,
        "risk_regime"
    ] = 2

    df.loc[
        df["risk_score"] >= 75,
        "risk_regime"
    ] = 3

    # ====================================================
    # REGIME LABELS
    # ====================================================

    df["regime"] = "Stable"

    df.loc[
        df["risk_regime"] == 1,
        "regime"
    ] = "Elevated"

    df.loc[
        df["risk_regime"] == 2,
        "regime"
    ] = "High"

    df.loc[
        df["risk_regime"] == 3,
        "regime"
    ] = "Crisis"

    # ====================================================
    # GLOBAL PRESSURE
    # ====================================================

    df["global_pressure"] = (

        df["market_stress"]

        +

        df["conflict_pressure"]

        +

        df["media_pressure"]

    )

    # ====================================================
    # CRISIS INDEX
    # ====================================================

    df["crisis_index"] = (

        0.40 * df["market_stress"]

        +

        0.40 * df["conflict_pressure"]

        +

        0.20 * df["media_pressure"]

    )

    # ====================================================
    # ROLLING MEANS
    # ====================================================

    df["risk_mean_60"] = (
        df["risk_score"]
        .rolling(60, min_periods=1)
        .mean()
    )

    df["risk_mean_90"] = (
        df["risk_score"]
        .rolling(90, min_periods=1)
        .mean()
    )

    df["intensity_mean_60"] = (
        df["intensity"]
        .rolling(60, min_periods=1)
        .mean()
    )

    df["intensity_mean_90"] = (
        df["intensity"]
        .rolling(90, min_periods=1)
        .mean()
    )

    # ====================================================
    # ROLLING STANDARD DEVIATIONS
    # ====================================================

    df["risk_std_30"] = (
        df["risk_score"]
        .rolling(30, min_periods=1)
        .std()
    )

    df["risk_std_60"] = (
        df["risk_score"]
        .rolling(60, min_periods=1)
        .std()
    )

    df["intensity_std_30"] = (
        df["intensity"]
        .rolling(30, min_periods=1)
        .std()
    )

    df["intensity_std_60"] = (
        df["intensity"]
        .rolling(60, min_periods=1)
        .std()
    )

    # ====================================================
    # MOMENTUM FEATURES
    # ====================================================

    df["momentum_7"] = (
        df["risk_score"]
        - df["risk_score"].shift(7)
    )

    df["momentum_30"] = (
        df["risk_score"]
        - df["risk_score"].shift(30)
    )

    df["intensity_momentum_7"] = (
        df["intensity"]
        - df["intensity"].shift(7)
    )

    df["intensity_momentum_30"] = (
        df["intensity"]
        - df["intensity"].shift(30)
    )

    # ====================================================
    # EMA FEATURES
    # ====================================================

    df["risk_ema_14"] = (
        df["risk_score"]
        .ewm(span=14, adjust=False)
        .mean()
    )

    df["risk_ema_30"] = (
        df["risk_score"]
        .ewm(span=30, adjust=False)
        .mean()
    )

    df["intensity_ema_14"] = (
        df["intensity"]
        .ewm(span=14, adjust=False)
        .mean()
    )

    df["intensity_ema_30"] = (
        df["intensity"]
        .ewm(span=30, adjust=False)
        .mean()
    )

    # ====================================================
    # TREND STRENGTH
    # ====================================================

    df["trend_strength"] = (
        df["risk_ema_14"]
        - df["risk_ema_30"]
    )

    df["intensity_trend_strength"] = (
        df["intensity_ema_14"]
        - df["intensity_ema_30"]
    )

    # ====================================================
    # LONG MEMORY
    # ====================================================

    df["risk_lag_60"]      = df["risk_score"].shift(60)
    df["risk_lag_90"]      = df["risk_score"].shift(90)
    df["intensity_lag_60"] = df["intensity"].shift(60)
    df["intensity_lag_90"] = df["intensity"].shift(90)

    # ====================================================
    # LONG TERM AVERAGES
    # ====================================================

    df["risk_mean_180"] = (
        df["risk_score"]
        .rolling(180, min_periods=1)
        .mean()
    )

    df["intensity_mean_180"] = (
        df["intensity"]
        .rolling(180, min_periods=1)
        .mean()
    )

    # ====================================================
    # LONG MOMENTUM
    # ====================================================

    df["momentum_60"] = (
        df["risk_score"]
        - df["risk_score"].shift(60)
    )

    df["intensity_momentum_60"] = (
        df["intensity"]
        - df["intensity"].shift(60)
    )

    # ====================================================
    # EMA 60
    # ====================================================

    df["risk_ema_60"] = (
        df["risk_score"]
        .ewm(span=60, adjust=False)
        .mean()
    )

    df["intensity_ema_60"] = (
        df["intensity"]
        .ewm(span=60, adjust=False)
        .mean()
    )

    # ====================================================
    # INTERACTION FEATURES
    # ====================================================

    df["risk_x_intensity"] = (
        df["risk_30d"]
        * df["intensity_30d"]
    )

    df["shock_x_risk"] = (
        df["shock_score"]
        * df["risk_30d"]
    )

    df["fatality_intensity_ratio"] = (
        df["fatalities_30d"]
        / (df["intensity_30d"] + 1)
    )

    df["risk_volatility_interaction"] = (
        df["risk_std_30"]
        * df["risk_30d"]
    )

    df["momentum_shock"] = (
        df["momentum_30"]
        * df["shock_score"]
    )

    df["crisis_interaction"] = (
        df["crisis_index"]
        * df["risk_percentile"]
    )

    # ====================================================
    # TIME FEATURES
    # Fixed: use lowercase "date" column (consistent with
    # all other files in the pipeline). Falls back to "Date"
    # if "date" is absent, to avoid silent KeyErrors.
    # ====================================================

    date_col = "date" if "date" in df.columns else "Date"

    df["month"]     = df[date_col].dt.month
    df["quarter"]   = df[date_col].dt.quarter
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["year"]      = df[date_col].dt.year

    # is_monday: binary flag used by model scripts instead of
    # raw dayofweek. dayofweek is kept for cyclical encoding only
    # (dayofweek_sin/cos below). Do NOT add dayofweek to
    # FEATURE_COLS — use is_monday instead.
    df["is_monday"] = (df[date_col].dt.dayofweek == 0).astype(int)

    # ====================================================
    # CYCLICAL FEATURES
    # These encode periodicity correctly for tree models
    # (sin/cos pair encodes circular distance).
    # ====================================================

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["dayofweek_sin"] = np.sin(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["dayofweek_cos"] = np.cos(
        2 * np.pi * df["dayofweek"] / 7
    )

    # ====================================================
    # REGIME PERSISTENCE
    # ====================================================

    df["regime_change"] = (
        df["risk_regime"]
        .diff()
    )

    df["regime_duration"] = (

        df["risk_regime"]

        .groupby(

            (

                df["risk_regime"]

                !=

                df["risk_regime"]

                .shift()

            ).cumsum()

        )

        .cumcount()

    )

    # ====================================================
    # RISK VOLATILITY
    # ====================================================

    df["risk_volatility_30"] = (
        df["risk_score"]
        .rolling(30, min_periods=1)
        .std()
    )

    df["risk_volatility_90"] = (
        df["risk_score"]
        .rolling(90, min_periods=1)
        .std()
    )

    # ====================================================
    # SKEWNESS
    # ====================================================

    df["risk_skew_30"] = (
        df["risk_score"]
        .rolling(30, min_periods=1)
        .skew()
    )

    # ====================================================
    # KURTOSIS
    # ====================================================

    df["risk_kurtosis_30"] = (
        df["risk_score"]
        .rolling(30, min_periods=1)
        .kurt()
    )

    # ====================================================
    # ENTROPY
    # ====================================================

    df["risk_entropy_30"] = (

        df["risk_score"]

        .rolling(

            30,

            min_periods=5

        )

        .apply(

            lambda x:

            entropy(

                np.histogram(

                    x,

                    bins=10

                )[0]

                + 1e-8

            ),

            raw=False

        )

    )

    # ====================================================
    # CLEAN NaNs
    # ====================================================

    df = df.fillna(0)

    return df