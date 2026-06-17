import pandas as pd


# =====================================================
# DAILY RETURNS
# =====================================================

def add_daily_return(df):

    df = df.copy()

    df["daily_return"] = (

        df["Close"]

        .pct_change()

        * 100

    )

    return df


# =====================================================
# VOLATILITY FEATURES
# =====================================================

def add_volatility_features(df):

    df = df.copy()

    returns = df["Close"].pct_change()

    df["volatility_7"] = (

        returns

        .rolling(

            7,

            min_periods=1

        )

        .std()

        * 100

    )

    df["volatility_30"] = (

        returns

        .rolling(

            30,

            min_periods=1

        )

        .std()

        * 100

    )

    df["volatility_90"] = (

        returns

        .rolling(

            90,

            min_periods=1

        )

        .std()

        * 100

    )

    # compatibility

    df["volatility"] = df["volatility_30"]

    return df


# =====================================================
# MOVING AVERAGES
# =====================================================

def add_moving_average_features(df):

    df = df.copy()

    df["moving_average_7"] = (

        df["Close"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    df["moving_average_30"] = (

        df["Close"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    df["moving_average_90"] = (

        df["Close"]

        .rolling(

            90,

            min_periods=1

        )

        .mean()

    )

    df["moving_average"] = df["moving_average_30"]

    return df


# =====================================================
# PRICE RETURNS
# =====================================================

def add_price_change_features(df):

    df = df.copy()

    df["return_7d"] = (

        df["Close"]

        .pct_change(

            periods=7

        )

        * 100

    )

    df["return_30d"] = (

        df["Close"]

        .pct_change(

            periods=30

        )

        * 100

    )

    df["return_90d"] = (

        df["Close"]

        .pct_change(

            periods=90

        )

        * 100

    )

    return df


# =====================================================
# MARKET SHOCK FEATURES
# =====================================================

def add_market_shock_features(df):

    df = df.copy()

    df["price_change"] = (

        df["Close"]

        .diff()

    )

    df["price_acceleration"] = (

        df["price_change"]

        .diff()

    )

    df["abs_daily_return"] = (

        df["daily_return"]

        .abs()

    )

    df["oil_shock"] = (

        df["abs_daily_return"]

    )

    df["volatility_change"] = (

        df["volatility_30"]

        .diff()

    )

    df["oil_momentum"] = (

        df["return_30d"]

    )

    return df


# =====================================================
# TREND FEATURES
# =====================================================

def add_trend_features(df):

    df = df.copy()

    df["trend_7_30"] = (

        df["moving_average_7"]

        -

        df["moving_average_30"]

    )

    df["trend_30_90"] = (

        df["moving_average_30"]

        -

        df["moving_average_90"]

    )

    return df


# =====================================================
# MOMENTUM FEATURES
# =====================================================

def add_momentum_features(df):

    df = df.copy()

    df["momentum_7"] = (

        df["Close"]

        -

        df["moving_average_7"]

    )

    df["momentum_30"] = (

        df["Close"]

        -

        df["moving_average_30"]

    )

    df["momentum_90"] = (

        df["Close"]

        -

        df["moving_average_90"]

    )

    return df


# =====================================================
# Z-SCORE FEATURES
# =====================================================

def add_zscore_features(df):

    df = df.copy()

    rolling_mean = (

        df["Close"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    rolling_std = (

        df["Close"]

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )

    df["price_zscore"] = (

        df["Close"]

        -

        rolling_mean

    ) / rolling_std

    return df


# =====================================================
# EMA FEATURES
# =====================================================

def add_ema_features(df):

    df = df.copy()

    df["ema_14"] = (

        df["Close"]

        .ewm(

            span=14,

            adjust=False

        )

        .mean()

    )

    df["ema_30"] = (

        df["Close"]

        .ewm(

            span=30,

            adjust=False

        )

        .mean()

    )

    df["ema_60"] = (

        df["Close"]

        .ewm(

            span=60,

            adjust=False

        )

        .mean()

    )

    return df


# =====================================================
# BOLLINGER FEATURES
# =====================================================

def add_bollinger_features(df):

    df = df.copy()

    rolling_mean = (

        df["Close"]

        .rolling(

            20,

            min_periods=1

        )

        .mean()

    )

    rolling_std = (

        df["Close"]

        .rolling(

            20,

            min_periods=1

        )

        .std()

    )

    df["bollinger_upper"] = (

        rolling_mean

        +

        2 * rolling_std

    )

    df["bollinger_lower"] = (

        rolling_mean

        -

        2 * rolling_std

    )

    df["bollinger_width"] = (

        df["bollinger_upper"]

        -

        df["bollinger_lower"]

    )

    return df


# =====================================================
# VOLATILITY MOMENTUM
# =====================================================

def add_volatility_momentum(df):

    df = df.copy()

    df["volatility_momentum"] = (

        df["volatility_7"]

        -

        df["volatility_30"]

    )

    return df


# =====================================================
# MARKET REGIME
# =====================================================

def add_market_regime(df):

    df = df.copy()

    df["market_regime"] = (

        df["volatility_30"]

        >

        df["volatility_90"]

    ).astype(int)

    return df


# =====================================================
# MASTER FUNCTION
# =====================================================

def add_market_features(df):

    df = add_daily_return(df)

    df = add_volatility_features(df)

    df = add_moving_average_features(df)

    df = add_price_change_features(df)

    df = add_market_shock_features(df)

    df = add_trend_features(df)

    df = add_momentum_features(df)

    df = add_zscore_features(df)

    df = add_ema_features(df)

    df = add_bollinger_features(df)

    df = add_volatility_momentum(df)

    df = add_market_regime(df)

    df = df.fillna(0)

    return df