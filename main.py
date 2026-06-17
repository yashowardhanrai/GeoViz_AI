from datetime import date
import time
import pandas as pd

from src.connectors.yfinance_connector import (
    YahooFinanceConnector
)

from src.connectors.acled_connector import (
    ACLEDConnector
)

from src.connectors.gdelt_connector import (
    GDELTConnector
)

from src.features.market_features import (
    add_market_features
)

from src.features.conflict_features import (
    conflict_intensity
)

from src.features.risk_score import (
    classify_risk
)

from src.pipelines.dataset_builder import (
    build_dataset
)

from src.storage.csv_store import (
    save_dataset
)


def main():

    start_time = time.time()

    start_date = "2022-02-24"

    end_date = str(

        date.today()

    )

    # ===================================================
    # MARKET DATA
    # ===================================================

    print(

        "\nLoading market data..."

    )

    market_connector = YahooFinanceConnector()

    market_data = market_connector.pull(

        start_date,

        end_date

    )

    oil_df = add_market_features(

        market_data[0]

    )

    gold_df = add_market_features(

        market_data[1]

    )

    sp500_df = add_market_features(

        market_data[2]

    )

    dxy_df = add_market_features(

        market_data[3]

    )

    print(

        f"Oil records loaded: {len(oil_df)}"

    )

    # ===================================================
    # ACLED DATA
    # ===================================================

    print(

        "\nLoading conflict data..."

    )

    acled_connector = ACLEDConnector()

    raw_records = acled_connector.pull(

        start_date,

        end_date

    )

    print(

        f"Raw ACLED events loaded: {len(raw_records)}"

    )

    # ===================================================
    # CONFLICT FEATURES
    # ===================================================

    print(

        "\nBuilding conflict features..."

    )

    conflict_df = conflict_intensity(

        raw_records

    )

    print(

        f"Conflict days generated: {len(conflict_df)}"

    )

    # ===================================================
    # GDELT DATA
    # ===================================================

    print(

        "\nLoading GDELT data..."

    )

    gdelt_connector = GDELTConnector()

    gdelt_df = gdelt_connector.pull(

        start_date,

        end_date

    )

    print(

        f"GDELT records loaded: {len(gdelt_df)}"

    )

    # ===================================================
    # BUILD DATASET
    # ===================================================

    print(

        "\nBuilding dataset..."

    )

    dataset = build_dataset(

        market_df=oil_df,

        conflict_df=conflict_df,

        gdelt_df=gdelt_df

    )

    print(

        f"Dataset shape: {dataset.shape}"

    )

    # ===================================================
    # GOLD FEATURES
    # ===================================================

    gold_df = gold_df.reset_index()

    gold_df["date"] = pd.to_datetime(

        gold_df["Date"]

    )

    dataset = pd.merge(

        dataset,

        gold_df[
            [

                "date",

                "Close",

                "daily_return",

                "volatility_30",

                "return_30d"

            ]

        ].rename(

            columns={

                "Close":

                "gold_close",

                "daily_return":

                "gold_return",

                "volatility_30":

                "gold_volatility",

                "return_30d":

                "gold_return_30d"

            }

        ),

        on="date",

        how="left"

    )

    # ===================================================
    # SP500 FEATURES
    # ===================================================

    sp500_df = sp500_df.reset_index()

    sp500_df["date"] = pd.to_datetime(

        sp500_df["Date"]

    )

    dataset = pd.merge(

        dataset,

        sp500_df[
            [

                "date",

                "Close",

                "daily_return",

                "volatility_30",

                "return_30d"

            ]

        ].rename(

            columns={

                "Close":

                "sp500_close",

                "daily_return":

                "sp500_return",

                "volatility_30":

                "sp500_volatility",

                "return_30d":

                "sp500_return_30d"

            }

        ),

        on="date",

        how="left"

    )

    # ===================================================
    # DXY FEATURES
    # ===================================================

    dxy_df = dxy_df.reset_index()

    dxy_df["date"] = pd.to_datetime(

        dxy_df["Date"]

    )

    dataset = pd.merge(

        dataset,

        dxy_df[
            [

                "date",

                "Close",

                "daily_return",

                "volatility_30",

                "return_30d"

            ]

        ].rename(

            columns={

                "Close":

                "dxy_close",

                "daily_return":

                "dxy_return",

                "volatility_30":

                "dxy_volatility",

                "return_30d":

                "dxy_return_30d"

            }

        ),

        on="date",

        how="left"

    )

    # ===================================================
    # HANDLE MISSING VALUES
    # (no bfill on the merged dataset — gold/SP500/DXY
    #  calendar gaps must not pull a future price/return
    #  backward into an earlier date's row)
    # ===================================================

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

    dataset = dataset.fillna("")

    # ===================================================
    # RISK LEVEL
    # ===================================================

    dataset["risk_level"] = (

        dataset["risk_score"]

        .apply(

            classify_risk

        )

    )

    # ===================================================
    # SAVE DATASET
    # ===================================================

    save_dataset(

        dataset,

        "geoviz_risk_dataset.csv"

    )

    # ===================================================
    # SUMMARY
    # ===================================================

    latest_row = dataset.iloc[-1]

    print(

        "\n" + "=" * 70

    )

    print(

        "DATASET SUMMARY"

    )

    print(

        "=" * 70

    )

    print(

        f"Dataset Shape: {dataset.shape}"

    )

    print(

        f"Date Range: "

        f"{dataset['date'].min().date()} "

        f"to "

        f"{dataset['date'].max().date()}"

    )

    print(

        f"\nAverage Risk Score: "

        f"{round(dataset['risk_score'].mean(),2)}"

    )

    print(

        f"Maximum Risk Score: "

        f"{round(dataset['risk_score'].max(),2)}"

    )

    print(

        f"Current Risk Score: "

        f"{round(latest_row['risk_score'],2)}"

    )

    print(

        f"Current Risk Level: "

        f"{latest_row['risk_level']}"

    )

    print(

        f"30-Day Risk Trend: "

        f"{round(latest_row['risk_30d'],2)}"

    )

    print(

        "\nLatest Records:\n"

    )

    print(

        dataset[

            [

                "date",

                "Close",

                "daily_return",

                "volatility_30",

                "fatalities",

                "fatalities_30d",

                "intensity",

                "intensity_30d",

                "article_count",

                "tone_30d",

                "gold_close",

                "sp500_close",

                "dxy_close",

                "risk_score",

                "risk_level"

            ]

        ]

        .tail(10)

    )

    print(

        "\nTotal Runtime:",

        round(

            time.time() - start_time,

            2

        ),

        "seconds"

    )

    print(

        "\nDataset successfully saved!"

    )


if __name__ == "__main__":

    main()