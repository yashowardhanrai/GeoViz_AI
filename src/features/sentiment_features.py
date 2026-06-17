
import pandas as pd
from transformers import pipeline


# ==========================================================
# LOAD FINBERT
# ==========================================================

sentiment_model = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None
)


# ==========================================================
# UNCERTAINTY WORDS
# ==========================================================

UNCERTAINTY_WORDS = [

    "uncertain",
    "risk",
    "threat",
    "war",
    "conflict",
    "attack",
    "escalation",
    "sanctions",
    "crisis",
    "military",
    "terrorism"

]


# ==========================================================
# BUILD SEMANTIC TEXT
# ==========================================================

def build_semantic_text(df):

    df["semantic_text"] = (

        df["themes"]

        .fillna("")

        .astype(str)

        + " "

        +

        df["persons"]

        .fillna("")

        .astype(str)

        + " "

        +

        df["organizations"]

        .fillna("")

        .astype(str)

        + " "

        +

        df["locations"]

        .fillna("")

        .astype(str)

    )

    return df


# ==========================================================
# EXTRACT SENTIMENT
# ==========================================================

def extract_sentiment(text):

    if pd.isna(text):

        return {

            "positive_prob": 0,

            "negative_prob": 0,

            "neutral_prob": 0,

            "sentiment_score": 0

        }

    text = str(text)[:512]

    try:

        results = sentiment_model(text)[0]

        scores = {

            item["label"].lower(): item["score"]

            for item in results

        }

        positive = scores.get("positive", 0)

        negative = scores.get("negative", 0)

        neutral = scores.get("neutral", 0)

        sentiment_score = positive - negative

        return {

            "positive_prob": positive,

            "negative_prob": negative,

            "neutral_prob": neutral,

            "sentiment_score": sentiment_score

        }

    except Exception:

        return {

            "positive_prob": 0,

            "negative_prob": 0,

            "neutral_prob": 0,

            "sentiment_score": 0

        }


# ==========================================================
# UNCERTAINTY SCORE
# ==========================================================

def compute_uncertainty(text):

    if pd.isna(text):

        return 0

    text = str(text).lower()

    score = sum(

        word in text

        for word in UNCERTAINTY_WORDS

    )

    return score


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def add_sentiment_features(df):

    # ------------------------------------------
    # Build semantic text
    # ------------------------------------------

    df = build_semantic_text(df)

    # ------------------------------------------
    # FinBERT sentiment
    # ------------------------------------------

    sentiment_df = (

        df["semantic_text"]

        .fillna("")

        .apply(extract_sentiment)

        .apply(pd.Series)

    )

    df = pd.concat(

        [

            df,

            sentiment_df

        ],

        axis=1

    )

    # ------------------------------------------
    # Fear score
    # ------------------------------------------

    df["fear_score"] = (

        df["negative_prob"]

        -

        df["positive_prob"]

    )

    # ------------------------------------------
    # Uncertainty score
    # ------------------------------------------

    df["uncertainty_score"] = (

        df["semantic_text"]

        .apply(compute_uncertainty)

    )

    # ------------------------------------------
    # Rolling sentiment features
    # ------------------------------------------

    df["sentiment_7d"] = (

        df["sentiment_score"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    df["sentiment_30d"] = (

        df["sentiment_score"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    # ------------------------------------------
    # Fear moving averages
    # ------------------------------------------

    df["fear_7d"] = (

        df["fear_score"]

        .rolling(

            7,

            min_periods=1

        )

        .mean()

    )

    df["fear_30d"] = (

        df["fear_score"]

        .rolling(

            30,

            min_periods=1

        )

        .mean()

    )

    # ------------------------------------------
    # Momentum
    # ------------------------------------------

    df["sentiment_change"] = (

        df["sentiment_score"]

        .diff()

    )

    df["fear_change"] = (

        df["fear_score"]

        .diff()

    )

        # ------------------------------------------
        # CLEAN NUMERIC NaNs
        # ------------------------------------------

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

    # ------------------------------------------
    # DEFRAGMENT DATAFRAME
    # ------------------------------------------

    df = df.copy()

    return df