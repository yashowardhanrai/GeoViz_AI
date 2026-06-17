import pandas as pd


# =====================================================
# EVENT WEIGHTS
# =====================================================

EVENT_WEIGHTS = {

    "Battles": 5,

    "Explosions/Remote violence": 4,

    "Violence against civilians": 6,

    "Strategic developments": 3,

    "Protests": 1,

    "Riots": 2

}


# =====================================================
# BUILD DAILY CONFLICT DATA
# =====================================================

def build_daily_conflict(records):

    if not records:

        return pd.DataFrame(

            columns=[

                "date",

                "fatalities",

                "event_count",

                "severity"

            ]

        )

    rows = []

    for record in records:

        try:

            event_type = record.get(

                "event_type",

                ""

            )

            weight = EVENT_WEIGHTS.get(

                event_type,

                1

            )

            rows.append(

                {

                    "date":

                        record.get(

                            "event_date"

                        ),

                    "fatalities":

                        float(

                            record.get(

                                "fatalities",

                                0

                            )

                        ),

                    "event_weight":

                        weight,

                    "event_count":

                        1

                }

            )

        except:

            continue

    df = pd.DataFrame(

        rows

    )

    df["date"] = pd.to_datetime(

        df["date"],

        errors="coerce"

    )

    df = df.dropna(

        subset=[

            "date"

        ]

    )

    daily = (

        df

        .groupby(

            "date",

            as_index=False

        )

        .agg(

            {

                "fatalities": "sum",

                "event_count": "sum",

                "event_weight": "sum"

            }

        )

    )

    daily.rename(

        columns={

            "event_weight":

                "severity"

        },

        inplace=True

    )

    daily = (

        daily

        .sort_values(

            "date"

        )

    )

    return daily


# =====================================================
# CONFLICT INTENSITY
# =====================================================

def conflict_intensity(

    records,

    window=7

):

    daily = build_daily_conflict(

        records

    )

    if daily.empty:

        return pd.DataFrame(

            columns=[

                "date",

                "fatalities",

                "event_count",

                "severity",

                "intensity"

            ]

        )

    # =================================================
    # 7 DAY FEATURES
    # =================================================

    daily["fatalities_7d"] = (

        daily["fatalities"]

        .rolling(

            7,

            min_periods=1

        )

        .sum()

    )

    daily["events_7d"] = (

        daily["event_count"]

        .rolling(

            7,

            min_periods=1

        )

        .sum()

    )

    daily["severity_7d"] = (

        daily["severity"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    # =================================================
    # 30 DAY FEATURES
    # =================================================

    daily["fatalities_30d"] = (

        daily["fatalities"]

        .rolling(

            30,

            min_periods=1

        )

        .sum()

    )

    daily["events_30d"] = (

        daily["event_count"]

        .rolling(

            30,

            min_periods=1

        )

        .sum()

    )

    daily["severity_30d"] = (

        daily["severity"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    # =================================================
    # 90 DAY FEATURES
    # =================================================

    daily["fatalities_90d"] = (

        daily["fatalities"]

        .rolling(

            90,

            min_periods=1

        )

        .sum()

    )

    daily["events_90d"] = (

        daily["event_count"]

        .rolling(

            90,

            min_periods=1

        )

        .sum()

    )

    daily["severity_90d"] = (

        daily["severity"]

        .rolling(

            90,

            min_periods=1

        )

        .mean()

    )

    # =================================================
    # INTENSITY
    # =================================================

    daily["intensity"] = (

        0.50 * daily["severity_7d"]

        +

        0.30 * daily["events_7d"]

        +

        0.20 * daily["fatalities_7d"]

    ) / 100

    daily["intensity"] = (

        daily["intensity"]

        .round(

            2

        )

    )

    # =================================================
    # MOMENTUM FEATURES
    # =================================================

    daily["severity_momentum"] = (

        daily["severity_7d"]

        -

        daily["severity_30d"]

    )

    daily["event_momentum"] = (

        daily["events_7d"]

        -

        daily["events_30d"]

    )

    daily["fatality_momentum"] = (

        daily["fatalities_7d"]

        -

        daily["fatalities_30d"]

    )

    daily["intensity_momentum"] = (

        daily["intensity"]

        -

        daily["intensity"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    # =================================================
    # CHANGE FEATURES
    # =================================================

    daily["intensity_change"] = (

        daily["intensity"]

        .diff()

    )

    daily["intensity_acceleration"] = (

        daily["intensity_change"]

        .diff()

    )

    # =================================================
    # CONFLICT REGIME
    # Fixed: rolling mean now uses .shift(1) so today's
    # intensity is compared against the mean of past days
    # only — not against a window that includes today itself.
    # =================================================

    daily["conflict_regime"] = (

        daily["intensity"]

        >

        daily["intensity"]

        .shift(1)

        .rolling(

            90,

            min_periods=1

        )

        .mean()

    ).astype(int)

    # =================================================
    # VOLATILITY
    # =================================================

    daily["intensity_std_30"] = (

        daily["intensity"]

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )

    daily["severity_std_30"] = (

        daily["severity"]

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )

    # =================================================
    # EVENT DENSITY
    # =================================================

    daily["event_density"] = (

        daily["events_30d"]

        /

        (

            daily["fatalities_30d"]

            +

            1

        )

    )

    # =================================================
    # CONFLICT PRESSURE
    # =================================================

    daily["conflict_pressure"] = (

        daily["severity_30d"]

        *

        daily["intensity"]

    )

    # =================================================
    # LONG TERM TREND
    # =================================================

    daily["intensity_trend"] = (

        daily["intensity"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

        -

        daily["intensity"]

        .rolling(

            90,

            min_periods=1

        )

        .mean()

    )

    # =================================================
    # PERSISTENCE
    # =================================================

    daily["conflict_persistence"] = (

        daily["intensity"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

        -

        daily["intensity"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    # =================================================
    # CLEAN NaNs
    # =================================================

    numeric_cols = (

        daily

        .select_dtypes(

            include="number"

        )

        .columns

    )

    daily[numeric_cols] = (

        daily[numeric_cols]

        .fillna(0)

    )

    return daily