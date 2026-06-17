import pandas as pd


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def add_macro_features(df):

    # ====================================================
    # MARKET STRESS INDEX
    # ====================================================

    if all(
        col in df.columns
        for col in [
            "volatility_30",
            "daily_return",
            "return_30d"
        ]
    ):

        df["market_stress_index"] = (

            df["volatility_30"]

            +

            abs(
                df["daily_return"]
            )

            +

            abs(
                df["return_30d"]
            ) / 10

        )

    else:

        df["market_stress_index"] = 0


    # ====================================================
    # CONFLICT-MARKET INTERACTION
    # ====================================================

    if "risk_score" in df.columns:

        df["macro_conflict_risk"] = (

            df["market_stress_index"]

            *

            df["risk_score"]

        )

    else:

        df["macro_conflict_risk"] = 0


    # ====================================================
    # GLOBAL PRESSURE MOMENTUM
    # ====================================================

    if "global_pressure" in df.columns:

        df["global_pressure_7d"] = (

            df["global_pressure"]

            .rolling(

                7,

                min_periods=1

            )

            .mean()

        )

        df["global_pressure_30d"] = (

            df["global_pressure"]

            .rolling(

                30,

                min_periods=1

            )

            .mean()

        )

        df["global_pressure_momentum"] = (

            df["global_pressure_7d"]

            -

            df["global_pressure_30d"]

        )

    else:

        df["global_pressure_7d"] = 0

        df["global_pressure_30d"] = 0

        df["global_pressure_momentum"] = 0


    # ====================================================
    # RISK REGIME INTERACTION
    # ====================================================

    if "risk_regime" in df.columns:

        df["regime_pressure"] = (

            df["risk_regime"]

            *

            df["market_stress_index"]

        )

    else:

        df["regime_pressure"] = 0


    # ====================================================
    # CRISIS INTERACTION
    # ====================================================

    if "crisis_index" in df.columns:

        df["macro_crisis_interaction"] = (

            df["crisis_index"]

            *

            df["market_stress_index"]

        )

    else:

        df["macro_crisis_interaction"] = 0


    # ====================================================
    # LONG-TERM MACRO RISK
    # ====================================================

    df["macro_risk_30d"] = (

        df["market_stress_index"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    df["macro_risk_90d"] = (

        df["market_stress_index"]

        .rolling(

            90,

            min_periods=1

        )

        .mean()

    )


    # ====================================================
    # MACRO VOLATILITY
    # ====================================================

    df["macro_risk_std_30"] = (

        df["market_stress_index"]

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )


    # ====================================================
    # MACRO MOMENTUM
    # ====================================================

    df["macro_momentum_30"] = (

        df["market_stress_index"]

        -

        df["market_stress_index"]

        .shift(30)

    )
    # ====================================================
    # MACRO TREND
    # ====================================================

    df["macro_trend"] = (

        df["macro_risk_30d"]

        -

        df["macro_risk_90d"]

    )

    # ====================================================
    # VOLATILITY RATIO
    # ====================================================

    df["macro_volatility_ratio"] = (

        df["macro_risk_30d"]

        /

        (

            df["macro_risk_90d"]

            +

            1e-8

        )

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