import pandas as pd
from datetime import datetime
from pathlib import Path


def register_freshness(
    source,
    latest_data_date,
    record_count,
    status="OK"
):

    row = {
        "source": source,
        "last_pull_ts": datetime.utcnow(),
        "latest_data_date": latest_data_date,
        "record_count": record_count,
        "status": status
    }

    file_path = Path("data/freshness_matrix.csv")

    df = pd.DataFrame([row])

    if file_path.exists():

        df.to_csv(
            file_path,
            mode="a",
            header=False,
            index=False
        )

    else:

        df.to_csv(
            file_path,
            index=False
        )

    print(f"Freshness updated for {source}")