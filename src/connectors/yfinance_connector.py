from datetime import datetime

import yfinance as yf

from src.connectors.base_connector import BaseConnector
from src.schemas.canonical_record import CanonicalRecord


class YahooFinanceConnector(BaseConnector):

    source_id = "yfinance"

    cadence = "daily"

    TICKERS = {

    "oil": "BZ=F",

    "gold": "GC=F",

    "sp500": "SPY",

    "dxy": "DX-Y.NYB"

}

    def __init__(self):
        pass

    def pull(self, start, end):

        records = []

        for asset_name, ticker in self.TICKERS.items():

            print(
                f"\nDownloading {asset_name.upper()}..."
            )

            try:

                df = yf.download(

                    ticker,

                    start=start,

                    end=end,

                    progress=False,

                    auto_adjust=True

                )

            except Exception as e:

                print(
                    f"Error downloading {asset_name}: {e}"
                )

                continue

            if df.empty:

                print(
                    f"{asset_name.upper()} returned empty dataframe."
                )

                continue

            # -------------------------------------
            # Flatten MultiIndex columns
            # -------------------------------------

            if (
                hasattr(df.columns, "nlevels")
                and df.columns.nlevels > 1
            ):

                df.columns = (
                    df.columns
                    .get_level_values(0)
                )

            # -------------------------------------
            # Remove invalid rows
            # -------------------------------------

            df = df.dropna(

                subset=[

                    "Close",

                    "Open",

                    "High",

                    "Low"

                ]

            )

            # -------------------------------------
            # Remove duplicate dates
            # -------------------------------------

            df = (
                df[~df.index.duplicated()]
            )

            # -------------------------------------
            # Sort chronologically
            # -------------------------------------

            df = (
                df
                .sort_index()
            )

            # -------------------------------------
            # Add asset column
            # -------------------------------------

            df["asset"] = asset_name

            print(
                f"{asset_name.upper()} records loaded: "
                f"{len(df)}"
            )

            print(
                "Date range:",
                df.index.min().date(),
                "to",
                df.index.max().date()
            )

            records.append(df)

        return records

    def normalize(self, row, asset_name):

        close_price = row["Close"]

        open_price = row["Open"]

        high_price = row["High"]

        low_price = row["Low"]

        volume = row["Volume"]

        return CanonicalRecord(

            timestamp=str(

                row.name.date()

            ),

            geo_id="GLOBAL",

            geo_sub=None,

            source="yfinance",

            entity_type=asset_name,

            value=float(

                close_price

            ),

            unit="price",

            data_status="observed",

            pull_ts=datetime.utcnow().isoformat(),

            metadata={

                "open":
                    float(open_price),

                "high":
                    float(high_price),

                "low":
                    float(low_price),

                "volume":
                    float(volume)

            }

        )