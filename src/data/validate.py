import pandas as pd


def validate_dataset(path: str) -> None:
    """Run basic data quality checks."""
    df = pd.read_parquet(path)

    # schema checks
    expected_columns = {
        'PULocationID', 'DOLocationID', 'trip_distance',
        'passenger_count', 'pickup_hour', 'pickup_dayofweek',
        'pickup_month', 'duration'
    }
    assert set(df.columns) == expected_columns, \
        f"Unexpected columns: {set(df.columns) ^ expected_columns}"

    # range checks
    assert df['duration'].between(1, 60).all(), "Duration out of range"
    assert df['trip_distance'].gt(0).all(), "Non-positive distance found"
    assert df['pickup_hour'].between(0, 23).all(), "Invalid hour"

    # no missing values
    assert df.isnull().sum().sum() == 0, "Unexpected nulls"

    # row count sanity
    assert len(df) > 1000, f"Dataset too small: {len(df)} rows"

    print(f"✓ {path}: {len(df)} rows, all checks passed")


if __name__ == "__main__":
    validate_dataset("data/processed/train.parquet")
    validate_dataset("data/processed/test.parquet")