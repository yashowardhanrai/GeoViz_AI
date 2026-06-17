
from datetime import datetime
import os

import pandas as pd
import requests

from dotenv import load_dotenv

from src.connectors.base_connector import BaseConnector
from src.schemas.canonical_record import CanonicalRecord
from src.utils.iso_lookup import get_iso3

load_dotenv()


class ACLEDConnector(BaseConnector):

    source_id = "acled"

    cadence = "daily"

    MAX_EVENTS = 5000

    def __init__(self):

        self.username = os.getenv(
            "ACLED_USERNAME"
        )

        self.password = os.getenv(
            "ACLED_PASSWORD"
        )

        self.token = self._get_token()

    def _get_token(self):

        response = requests.post(

            "https://acleddata.com/oauth/token",

            data={

                "username":
                    self.username,

                "password":
                    self.password,

                "grant_type":
                    "password",

                "client_id":
                    "acled",

                "scope":
                    "authenticated"

            }

        )

        response.raise_for_status()

        return response.json()[

            "access_token"

        ]
    
    def _fetch_period(

        self,

        country,

        start_date,

        end_date

    ):

        response = requests.get(

            "https://acleddata.com/api/acled/read",

            params={

                "country":
                    country,

                "event_date":
                    start_date,

                "event_date_where":
                    ">=",

                "limit":
                    self.MAX_EVENTS

            },

            headers={

                "Authorization":

                f"Bearer {self.token}"

            },

            timeout=60

        )

        response.raise_for_status()

        data = response.json()

        events = data.get(

            "data",

            []

        )

        filtered_events = []

        for event in events:

            event_date = event.get(

                "event_date"

            )

            if not event_date:

                continue

            if (

                start_date

                <=

                event_date

                <=

                end_date

            ):

                filtered_events.append(

                    event

                )

        return filtered_events
    
    def _split_month_into_weeks(

        self,

        start_date,

        end_date

    ):

        dates = pd.date_range(

            start=start_date,

            end=end_date,

            freq="7D"

        )

        ranges = []

        for i in range(

            len(dates)

        ):

            week_start = (

                dates[i]

                .strftime(

                    "%Y-%m-%d"

                )

            )

            if i < (

                len(dates)

                - 1

            ):

                week_end = (

                    dates[i + 1]

                    -

                    pd.Timedelta(

                        days=1

                    )

                ).strftime(

                    "%Y-%m-%d"

                )

            else:

                week_end = end_date

            ranges.append(

                (

                    week_start,

                    week_end

                )

            )

        return ranges
    
    def _split_week_into_days(

        self,

        start_date,

        end_date

    ):

        dates = pd.date_range(

            start=start_date,

            end=end_date,

            freq="D"

        )

        ranges = []

        for d in dates:

            day = d.strftime(

                "%Y-%m-%d"

            )

            ranges.append(

                (

                    day,

                    day

                )

            )

        return ranges
    
    def _adaptive_fetch(

        self,

        country,

        start_date,

        end_date

    ):

        events = self._fetch_period(

            country,

            start_date,

            end_date

        )

        # ----------------------------------
        # Safe case
        # ----------------------------------

        if len(events) < self.MAX_EVENTS:

            return events

        print(

            f"{country}: "

            f"{start_date} -> {end_date} "

            f"hit 5000 limit."

        )

        all_events = []

        # ----------------------------------
        # Split month into weeks
        # ----------------------------------

        week_ranges = (

            self._split_month_into_weeks(

                start_date,

                end_date

            )

        )

        for (

            week_start,

            week_end

        ) in week_ranges:

            week_events = (

                self._fetch_period(

                    country,

                    week_start,

                    week_end

                )

            )

            # ----------------------------------
            # Weekly overflow
            # ----------------------------------

            if (

                len(

                    week_events

                )

                >=

                self.MAX_EVENTS

            ):

                print(

                    f"Weekly overflow "

                    f"{week_start}"

                )

                day_ranges = (

                    self

                    ._split_week_into_days(

                        week_start,

                        week_end

                    )

                )

                for (

                    day_start,

                    day_end

                ) in day_ranges:

                    day_events = (

                        self

                        ._fetch_period(

                            country,

                            day_start,

                            day_end

                        )

                    )

                    all_events.extend(

                        day_events

                    )

            else:

                all_events.extend(

                    week_events

                )

        return all_events
    
    def pull(

        self,

        start,

        end

    ):

        countries = [

            "Ukraine",

            "Russia",

            "Israel",

            "Palestine",

            "Iran"

        ]

        # ==========================================
        # Monthly periods
        # ==========================================

        month_starts = pd.date_range(

            start=start,

            end=end,

            freq="MS"

        )

        period_ranges = []

        for i in range(

            len(

                month_starts

            )

        ):

            period_start = (

                month_starts[i]

                .strftime(

                    "%Y-%m-%d"

                )

            )

            if i < (

                len(

                    month_starts

                )

                - 1

            ):

                period_end = (

                    month_starts[

                        i + 1

                    ]

                    -

                    pd.Timedelta(

                        days=1

                    )

                ).strftime(

                    "%Y-%m-%d"

                )

            else:

                period_end = end

            period_ranges.append(

                (

                    period_start,

                    period_end

                )

            )

        all_events = []

        # ==========================================
        # Loop countries
        # ==========================================

        for country in countries:

            print(

                f"\n========== "

                f"{country}"

                f" =========="

            )

            country_total = 0

            for (

                period_start,

                period_end

            ) in period_ranges:

                try:

                    print(

                        f"Loading "

                        f"{period_start}"

                        f" -> "

                        f"{period_end}"

                    )

                    events = (

                        self

                        ._adaptive_fetch(

                            country,

                            period_start,

                            period_end

                        )

                    )

                    print(

                        f"Loaded "

                        f"{len(events)} "

                        f"events"

                    )

                    country_total += len(

                        events

                    )

                    all_events.extend(

                        events

                    )

                except Exception as e:

                    print(

                        f"Error loading "

                        f"{country} "

                        f"{period_start}: "

                        f"{e}"

                    )

            print(

                f"Total "

                f"{country}: "

                f"{country_total}"

            )

        # ==========================================
        # Remove duplicates
        # ==========================================

        unique_events = {}

        for event in all_events:

            event_id = event.get(

                "event_id_cnty"

            )

            if event_id:

                unique_events[

                    event_id

                ] = event

        all_events = list(

            unique_events.values()

        )

        # ==========================================
        # Summary
        # ==========================================

        print(

            "\nTOTAL EVENTS:",

            len(

                all_events

            )

        )

        if all_events:

            dates = [

                e["event_date"]

                for e in all_events

                if e.get(

                    "event_date"

                )

            ]

            print(

                "Earliest Date:",

                min(

                    dates

                )

            )

            print(

                "Latest Date:",

                max(

                    dates

                )

            )

        return all_events
    
    def normalize(

        self,

        raw

    ):

        fatalities = raw.get(

            "fatalities",

            0

        )

        try:

            fatalities = float(

                fatalities

            )

        except:

            fatalities = 0.0

        latitude = raw.get(

            "latitude"

        )

        longitude = raw.get(

            "longitude"

        )

        try:

            latitude = float(

                latitude

            )

        except:

            latitude = None

        try:

            longitude = float(

                longitude

            )

        except:

            longitude = None

        return CanonicalRecord(

            timestamp=raw.get(

                "event_date"

            ),

            geo_id=get_iso3(

                raw.get(

                    "country",

                    ""

                )

            ),

            geo_sub=raw.get(

                "admin1"

            ),

            source="acled",

            entity_type="conflict_event",

            value=fatalities,

            unit="fatalities",

            data_status="observed",

            pull_ts=datetime.utcnow().isoformat(),

            metadata={

                # ==================================
                # IDs
                # ==================================

                "event_id":

                    raw.get(

                        "event_id_cnty"

                    ),

                # ==================================
                # Event Classification
                # ==================================

                "event_type":

                    raw.get(

                        "event_type"

                    ),

                "sub_event_type":

                    raw.get(

                        "sub_event_type"

                    ),

                # ==================================
                # Actors
                # ==================================

                "actor1":

                    raw.get(

                        "actor1"

                    ),

                "actor2":

                    raw.get(

                        "actor2"

                    ),

                "assoc_actor_1":

                    raw.get(

                        "assoc_actor_1"

                    ),

                "assoc_actor_2":

                    raw.get(

                        "assoc_actor_2"

                    ),

                # ==================================
                # Geography
                # ==================================

                "country":

                    raw.get(

                        "country"

                    ),

                "admin1":

                    raw.get(

                        "admin1"

                    ),

                "admin2":

                    raw.get(

                        "admin2"

                    ),

                "location":

                    raw.get(

                        "location"

                    ),

                "latitude":

                    latitude,

                "longitude":

                    longitude,

                # ==================================
                # Time
                # ==================================

                "year":

                    raw.get(

                        "year"

                    ),

                "time_precision":

                    raw.get(

                        "time_precision"

                    ),

                # ==================================
                # Sources
                # ==================================

                "source":

                    raw.get(

                        "source"

                    ),

                "source_scale":

                    raw.get(

                        "source_scale"

                    ),

                # ==================================
                # Notes
                # ==================================

                "notes":

                    raw.get(

                        "notes"

                    ),

                # ==================================
                # Fatalities
                # ==================================

                "fatalities":

                    fatalities

            }

        )












    


