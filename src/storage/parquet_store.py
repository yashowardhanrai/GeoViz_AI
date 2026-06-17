from pathlib import Path
import pandas as pd


def save_records(records, source):

    df = pd.DataFrame(records)

    output_dir = Path(
        f"data/raw/{source}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        output_dir /
        "records.parquet"
    )

    df.to_parquet(
        file_path,
        index=False
    )

    print(
        f"Saved {len(df)} records"
    )