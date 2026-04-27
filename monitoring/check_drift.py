from pathlib import Path

import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


def check_data_drift(reference_path: str, current_path: str, output_path: str):
    print(f"Loading reference data from {reference_path}")
    reference = pd.read_parquet(reference_path)

    print(f"Loading current data from {current_path}")
    current = pd.read_parquet(current_path)

    print(f"Reference: {len(reference)} rows | Current: {len(current)} rows")

    report = Report([
        DataDriftPreset(),
        DataSummaryPreset(),
    ])

    evaluation = report.run(reference_data=reference, current_data=current)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    evaluation.save_html(output_path)

    print(f"Data drift report saved to {output_path}")


def check_prediction_drift(
    reference_path: str,
    current_path: str,
    model_path: str,
    output_path: str,
):
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    print("Loading datasets")
    reference = pd.read_parquet(reference_path).copy()
    current = pd.read_parquet(current_path).copy()

    ref_features = reference.drop("duration", axis=1)
    cur_features = current.drop("duration", axis=1)

    reference["prediction"] = model.predict(ref_features)
    current["prediction"] = model.predict(cur_features)

    print(
        f"Reference predictions: mean={reference['prediction'].mean():.2f}, "
        f"std={reference['prediction'].std():.2f}"
    )
    print(
        f"Current predictions: mean={current['prediction'].mean():.2f}, "
        f"std={current['prediction'].std():.2f}"
    )

    report = Report([
        DataDriftPreset(columns=["prediction"]),
    ])

    evaluation = report.run(reference_data=reference, current_data=current)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    evaluation.save_html(output_path)

    print(f"Prediction drift report saved to {output_path}")


if __name__ == "__main__":
    check_data_drift(
        reference_path="data/processed/train.parquet",
        current_path="data/processed/test.parquet",
        output_path="monitoring/reports/data_drift.html",
    )

    check_prediction_drift(
        reference_path="data/processed/train.parquet",
        current_path="data/processed/test.parquet",
        model_path="model/production_model.pkl",
        output_path="monitoring/reports/prediction_drift.html",
    )

    print("All drift reports generated.")