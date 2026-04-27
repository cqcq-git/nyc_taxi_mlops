from pathlib import Path

import pandas as pd

from src.data.preprocess import preprocess


def test_preprocess_creates_duration_column(tmp_path: Path):
    test_data = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2026-01-01 10:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2026-01-01 10:15:00"]),
            "PULocationID": [100],
            "DOLocationID": [200],
            "trip_distance": [2.5],
            "passenger_count": [1],
        }
    )

    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    test_data.to_parquet(input_path)

    preprocess(str(input_path), str(output_path))

    result = pd.read_parquet(output_path)

    assert "duration" in result.columns
    assert result["duration"].iloc[0] == 15.0


def test_preprocess_filters_invalid_durations(tmp_path: Path):
    test_data = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 12:00:00",
                ]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                [
                    "2026-01-01 10:00:30",
                    "2026-01-01 11:15:00",
                    "2026-01-01 13:30:00",
                ]
            ),
            "PULocationID": [100, 100, 100],
            "DOLocationID": [200, 200, 200],
            "trip_distance": [0.1, 2.5, 30.0],
            "passenger_count": [1, 1, 1],
        }
    )

    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    test_data.to_parquet(input_path)

    preprocess(str(input_path), str(output_path))

    result = pd.read_parquet(output_path)

    assert len(result) == 1
    assert result["duration"].iloc[0] == 15.0
