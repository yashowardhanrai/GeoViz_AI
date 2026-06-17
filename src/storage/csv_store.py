import os


def save_dataset(df, filename):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    path = f"data/processed/{filename}"

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Dataset saved: {path}"
    )