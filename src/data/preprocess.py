import pandas as pd
from pathlib import Path


def preprocess(input_path: str, output_path: str) -> None:
    """Load raw taxi data, clean it, and save processed version."""
    df = pd.read_parquet(input_path)

    # compute trip duration in minutes
    df['duration'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds() / 60

    # filter unrealistic durations
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    # filter non-positive distances
    df = df[df['trip_distance'] > 0]

    # extract time features
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['tpep_pickup_datetime'].dt.month

    # keep relevant columns
    columns_to_keep = [
        'PULocationID', 'DOLocationID', 'trip_distance',
        'passenger_count', 'pickup_hour', 'pickup_dayofweek',
        'pickup_month', 'duration'
    ]
    df = df[columns_to_keep].dropna()

    # save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    import sys
    preprocess(sys.argv[1], sys.argv[2])