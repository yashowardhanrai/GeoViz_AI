from google.cloud import bigquery

from src.connectors.base_connector import BaseConnector


class GDELTConnector(BaseConnector):

    source_id = "gdelt"

    cadence = "daily"

    def __init__(self):

        self.client = bigquery.Client(

            project="geoviz-498707"

        )

    # =====================================================
    # PULL DATA
    # =====================================================

    def pull(

        self,

        start,

        end

    ):

        print(

            "\nLoading GDELT data..."

        )

        # -------------------------------------------------------
        # TEXT COLUMNS: STRING_AGG instead of ANY_VALUE.
        # ANY_VALUE returns an arbitrary single row's value per day
        # (BigQuery makes no guarantee which row is chosen), so
        # semantic_text built from themes/persons/organizations would
        # vary across runs on the same date range, breaking
        # reproducibility and degrading embedding quality.
        # STRING_AGG concatenates all values for the day (space-
        # separated, truncated to first 1000 chars per column to
        # stay within BigQuery limits), giving a representative
        # day-level summary for downstream NLP/embedding steps.
        # -------------------------------------------------------

        query = f"""

        SELECT

            DATE(

                PARSE_TIMESTAMP(

                    '%Y%m%d%H%M%S',

                    CAST(DATE AS STRING)

                )

            ) AS event_date,

            COUNT(*) AS article_count,

            AVG(

                SAFE_CAST(

                    SPLIT(

                        V2Tone,

                        ","

                    )[OFFSET(0)]

                    AS FLOAT64

                )

            ) AS avg_tone,

            ANY_VALUE(

                SourceCommonName

            ) AS source,

            ANY_VALUE(

                DocumentIdentifier

            ) AS url,

            STRING_AGG(

                DISTINCT SUBSTR(

                    V2Themes,

                    1,

                    500

                ),

                ' '

                LIMIT 1000

            ) AS themes,

            STRING_AGG(

                DISTINCT SUBSTR(

                    V2Locations,

                    1,

                    300

                ),

                ' '

                LIMIT 1000

            ) AS locations,

            STRING_AGG(

                DISTINCT SUBSTR(

                    V2Persons,

                    1,

                    300

                ),

                ' '

                LIMIT 1000

            ) AS persons,

            STRING_AGG(

                DISTINCT SUBSTR(

                    V2Organizations,

                    1,

                    300

                ),

                ' '

                LIMIT 1000

            ) AS organizations,

            ANY_VALUE(

                V2Counts

            ) AS counts

        FROM

            `gdelt-bq.gdeltv2.gkg`

        WHERE

            DATE(

                PARSE_TIMESTAMP(

                    '%Y%m%d%H%M%S',

                    CAST(DATE AS STRING)

                )

            )

            BETWEEN '{start}'

            AND '{end}'

        GROUP BY

            event_date

        ORDER BY

            event_date

        """

        df = (

            self.client

            .query(

                query

            )

            .to_dataframe()

        )

        print(

            f"GDELT records loaded: {len(df)}"

        )

        print(

            "Date range:",

            df.event_date.min(),

            "to",

            df.event_date.max()

        )

        print(

            "Columns:"

        )

        print(

            df.columns.tolist()

        )

        return df

    # =====================================================
    # NORMALIZE
    # =====================================================

    def normalize(

        self,

        raw

    ):

        raw = raw.copy()

        raw["event_date"] = (

            raw["event_date"]

            .astype(

                "datetime64[ns]"

            )

        )

        raw["article_count"] = (

            raw["article_count"]

            .fillna(

                0

            )

        )

        raw["avg_tone"] = (

            raw["avg_tone"]

            .fillna(

                0

            )

        )

        text_cols = [

            "source",

            "url",

            "themes",

            "locations",

            "persons",

            "organizations",

            "counts"

        ]

        for col in text_cols:

            if col in raw.columns:

                raw[col] = (

                    raw[col]

                    .fillna(

                        ""

                    )

                    .astype(

                        str

                    )

                )

        return raw