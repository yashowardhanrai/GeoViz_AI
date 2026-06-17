import pandas as pd


# ==========================================================
# EVENT KEYWORDS
# ==========================================================

WAR_WORDS = [

    "war",
    "battle",
    "offensive",
    "invasion",
    "missile",
    "shelling"

]

TERROR_WORDS = [

    "terror",
    "terrorism",
    "bomb",
    "attack",
    "hostage"

]

SANCTION_WORDS = [

    "sanction",
    "embargo",
    "restriction"

]

MILITARY_WORDS = [

    "military",
    "troops",
    "army",
    "navy",
    "air force"

]

CEASEFIRE_WORDS = [

    "ceasefire",
    "truce",
    "peace agreement"

]

# Maximum possible raw keyword counts per category
# (used to normalise scores to [0, 1] so escalation_score
# is stable regardless of how many keywords are in each list)
_WAR_MAX       = len(WAR_WORDS)       # 6
_TERROR_MAX    = len(TERROR_WORDS)    # 5
_SANCTION_MAX  = len(SANCTION_WORDS)  # 3
_MILITARY_MAX  = len(MILITARY_WORDS)  # 5
_CEASEFIRE_MAX = len(CEASEFIRE_WORDS) # 3


# ==========================================================
# EVENT PROBABILITIES
# ==========================================================

def extract_event_probabilities(text):

    if pd.isna(text):

        return {

            "war_probability": 0.0,

            "terrorism_probability": 0.0,

            "sanction_probability": 0.0,

            "military_probability": 0.0,

            "ceasefire_probability": 0.0

        }

    text = str(text).lower()

    # Normalise each count to [0, 1] so that a category with more
    # keywords doesn't dominate escalation_score by construction.
    return {

        "war_probability":

            min(
                sum(
                    word in text
                    for word in WAR_WORDS
                ) / _WAR_MAX,
                1.0
            ),

        "terrorism_probability":

            min(
                sum(
                    word in text
                    for word in TERROR_WORDS
                ) / _TERROR_MAX,
                1.0
            ),

        "sanction_probability":

            min(
                sum(
                    word in text
                    for word in SANCTION_WORDS
                ) / _SANCTION_MAX,
                1.0
            ),

        "military_probability":

            min(
                sum(
                    word in text
                    for word in MILITARY_WORDS
                ) / _MILITARY_MAX,
                1.0
            ),

        "ceasefire_probability":

            min(
                sum(
                    word in text
                    for word in CEASEFIRE_WORDS
                ) / _CEASEFIRE_MAX,
                1.0
            )

    }


# ==========================================================
# ESCALATION SCORE
# ==========================================================

def compute_escalation_score(row):

    # Weights sum to 1.0 (0.40+0.20+0.20+0.20-0.10 net = 0.90
    # in the deescalation-penalised case, 1.00 without ceasefire).
    # With normalised inputs each component is in [0,1], so
    # escalation_score is bounded to roughly [-0.10, 1.00].

    score = (

        0.40

        * row["war_probability"]

        +

        0.20

        * row["terrorism_probability"]

        +

        0.20

        * row["sanction_probability"]

        +

        0.20

        * row["military_probability"]

        -

        0.10

        * row["ceasefire_probability"]

    )

    return score


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def add_event_features(df):

    # -----------------------------------------
    # Event probabilities (normalised to [0,1])
    # -----------------------------------------

    event_df = (

        df["semantic_text"]

        .fillna("")

        .apply(

            extract_event_probabilities

        )

        .apply(

            pd.Series

        )

    )

    df = pd.concat(

        [

            df,

            event_df

        ],

        axis=1

    )

    # -----------------------------------------
    # Escalation score
    # -----------------------------------------

    df["escalation_score"] = (

        df.apply(

            compute_escalation_score,

            axis=1

        )

    )

    # -----------------------------------------
    # Rolling averages
    # Same-day features: rolling windows end at
    # time t (today's score included), consistent
    # with the end-of-day pipeline assumption.
    # Do NOT use in model FEATURE_COLS without
    # adding .shift(1) first.
    # -----------------------------------------

    df["escalation_7d"] = (

        df["escalation_score"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    df["escalation_30d"] = (

        df["escalation_score"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    # -----------------------------------------
    # Momentum
    # -----------------------------------------

    df["escalation_change"] = (

        df["escalation_score"]

        .diff()

    )

    # -----------------------------------------
    # Persistence
    # -----------------------------------------

    df["event_persistence"] = (

        df["escalation_7d"]

        -

        df["escalation_30d"]

    )

    # -----------------------------------------
    # Escalation volatility
    # -----------------------------------------

    df["escalation_std_30"] = (

        df["escalation_score"]

        .rolling(

            30,

            min_periods=1

        )

        .std()

    )

    # -----------------------------------------
    # Event regime
    # Fixed: compare today's escalation_score
    # against the rolling mean of PAST days only
    # (.shift(1) on the rolling mean), so today's
    # value is not included in its own threshold.
    # -----------------------------------------

    df["event_regime"] = (

        df["escalation_score"]

        >

        df["escalation_score"]

        .shift(1)

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    ).astype(int)

    # -----------------------------------------
    # Clean numeric NaNs
    # -----------------------------------------

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

    df = df.copy()

    return df