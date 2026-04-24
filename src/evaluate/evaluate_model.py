# src/evaluate/evaluate_model.py

import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def load_models():
    return {
        "xgboost_baseline": joblib.load("model/xgboost_model.pkl"),
        "xgboost_tuned": joblib.load("model/xgboost_tuned_model.pkl"),
        "xgboost_tuned_cv": joblib.load("model/xgboost_tuned_cv_model.pkl"),
    }


def compute_metrics(y_true, y_pred):
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true.clip(lower=1))) * 100,
        "pct_within_5min": np.mean(np.abs(y_true - y_pred) <= 5) * 100,
        "pct_within_10min": np.mean(np.abs(y_true - y_pred) <= 10) * 100,
    }


def evaluate_on_dataset(models, X, y, dataset_name):
    rows = []

    for model_name, model in models.items():
        preds = model.predict(X)
        metrics = compute_metrics(y, preds)

        row = {
            "model": model_name,
            "dataset": dataset_name,
            **metrics,
        }
        rows.append(row)

        print(f"\n--- {model_name} on {dataset_name} ---")
        print(f"  RMSE:             {metrics['rmse']:.3f} min")
        print(f"  MAE:              {metrics['mae']:.3f} min")
        print(f"  Median AE:        {metrics['median_ae']:.3f} min")
        print(f"  R²:               {metrics['r2']:.4f}")
        print(f"  MAPE:             {metrics['mape']:.1f}%")
        print(f"  Within 5 min:     {metrics['pct_within_5min']:.1f}%")
        print(f"  Within 10 min:    {metrics['pct_within_10min']:.1f}%")

    return pd.DataFrame(rows)


def overfitting_check(models, X_train, y_train, X_test, y_test):
    rows = []

    print(f"\n{'=' * 70}")
    print("OVERFITTING CHECK: Train vs Test RMSE")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<25} {'Train RMSE':>12} {'Test RMSE':>12} {'Gap':>12} {'Status':<20}")
    print(f"  {'-' * 80}")

    for model_name, model in models.items():
        train_rmse = root_mean_squared_error(y_train, model.predict(X_train))
        test_rmse = root_mean_squared_error(y_test, model.predict(X_test))
        gap = test_rmse - train_rmse

        if gap < 0.5:
            status = "Healthy"
        elif gap < 1.0:
            status = "Moderate"
        else:
            status = "Overfitting"

        rows.append(
            {
                "model": model_name,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "gap": gap,
                "status": status,
            }
        )

        print(f"  {model_name:<25} {train_rmse:>12.3f} {test_rmse:>12.3f} {gap:>+12.3f} {status:<20}")

    return pd.DataFrame(rows)


def slice_analysis(models, X, y, feature_name, feature_values, slice_labels=None):
    rows = []

    if feature_name not in X.columns:
        print(f"\nSkipping slice analysis for missing feature: {feature_name}")
        return pd.DataFrame(rows)

    if slice_labels is None:
        slice_labels = {v: str(v) for v in feature_values}

    print(f"\n{'=' * 70}")
    print(f"SLICE ANALYSIS: {feature_name}")
    print(f"{'=' * 70}")

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        print(f"  {'Slice':<20} {'RMSE':>8} {'MAE':>8} {'Count':>8}")
        print(f"  {'-' * 44}")

        for value in feature_values:
            mask = X[feature_name] == value
            count = int(mask.sum())

            if count == 0:
                continue

            preds = model.predict(X[mask])
            rmse = root_mean_squared_error(y[mask], preds)
            mae = mean_absolute_error(y[mask], preds)
            label = slice_labels.get(value, str(value))

            rows.append(
                {
                    "model": model_name,
                    "feature": feature_name,
                    "slice_value": value,
                    "slice_label": label,
                    "count": count,
                    "rmse": rmse,
                    "mae": mae,
                }
            )

            print(f"  {label:<20} {rmse:>8.3f} {mae:>8.3f} {count:>8}")

    return pd.DataFrame(rows)


def error_analysis(models, X, y, top_n=10):
    rows = []

    print(f"\n{'=' * 70}")
    print(f"ERROR ANALYSIS: Top {top_n} Worst Predictions")
    print(f"{'=' * 70}")

    has_pu = "PULocationID" in X.columns
    has_do = "DOLocationID" in X.columns
    has_distance = "trip_distance" in X.columns

    for model_name, model in models.items():
        preds = model.predict(X)
        errors = np.abs(y.values - preds)
        worst_indices = np.argsort(errors)[-top_n:][::-1]

        print(f"\n--- {model_name} ---")
        print(f"  {'Actual':>10} {'Predicted':>10} {'Error':>10} {'PU Zone':>10} {'DO Zone':>10} {'Distance':>10}")
        print(f"  {'-' * 70}")

        for rank, idx in enumerate(worst_indices, start=1):
            row = {
                "model": model_name,
                "rank": rank,
                "actual": float(y.iloc[idx]),
                "predicted": float(preds[idx]),
                "absolute_error": float(errors[idx]),
                "PULocationID": float(X.iloc[idx]["PULocationID"]) if has_pu else np.nan,
                "DOLocationID": float(X.iloc[idx]["DOLocationID"]) if has_do else np.nan,
                "trip_distance": float(X.iloc[idx]["trip_distance"]) if has_distance else np.nan,
            }
            rows.append(row)

            print(
                f"  {row['actual']:>10.1f} "
                f"{row['predicted']:>10.1f} "
                f"{row['absolute_error']:>10.1f} "
                f"{row['PULocationID']:>10.0f} "
                f"{row['DOLocationID']:>10.0f} "
                f"{row['trip_distance']:>10.1f}"
            )

    return pd.DataFrame(rows)


def model_comparison_summary(test_metrics_df):
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Within 5min':>12} {'Within 10min':>13}")
    print(f"  {'-' * 80}")

    best_row = test_metrics_df.sort_values("rmse").iloc[0]
    best_model = best_row["model"]

    for _, row in test_metrics_df.iterrows():
        print(
            f"  {row['model']:<25} "
            f"{row['rmse']:>10.3f} "
            f"{row['mae']:>10.3f} "
            f"{row['r2']:>10.4f} "
            f"{row['pct_within_5min']:>11.1f}% "
            f"{row['pct_within_10min']:>12.1f}%"
        )

    print(f"\n{'=' * 70}")
    print("RECOMMENDATION")
    print(f"{'=' * 70}")
    print(f"\n  Best model: {best_model}")
    print(f"  Test RMSE: {best_row['rmse']:.3f}")
    print(f"  Test MAE: {best_row['mae']:.3f}")
    print(f"  Test R²: {best_row['r2']:.4f}")

    return best_model


def log_dataframe(df, artifact_file):
    if df.empty:
        return

    output_path = Path("reports") / artifact_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        df.to_csv(output_path, index=False)

    mlflow.log_artifact(str(output_path), artifact_path=str(output_path.parent))

def evaluate_all():
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train = train_df.drop("duration", axis=1)
    y_train = train_df["duration"]
    X_test = test_df.drop("duration", axis=1)
    y_test = test_df["duration"]

    models = load_models()

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment("nyc_taxi_evaluation")

    with mlflow.start_run(run_name="model_comparison"):
        print(f"\n{'=' * 70}")
        print("TEST SET RESULTS")
        print(f"{'=' * 70}")
        test_metrics_df = evaluate_on_dataset(models, X_test, y_test, "test")

        overfitting_df = overfitting_check(models, X_train, y_train, X_test, y_test)

        day_labels = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        day_slice_df = slice_analysis(
            models,
            X_test,
            y_test,
            "pickup_dayofweek",
            range(7),
            day_labels,
        )

        hour_labels = {h: f"{h}:00" for h in range(24)}
        hour_slice_df = slice_analysis(
            models,
            X_test,
            y_test,
            "pickup_hour",
            range(24),
            hour_labels,
        )

        error_df = error_analysis(models, X_test, y_test, top_n=10)

        best_model = model_comparison_summary(test_metrics_df)

        # Log scalar metrics
        for _, row in test_metrics_df.iterrows():
            model_name = row["model"]
            for metric_name in [
                "rmse",
                "mae",
                "median_ae",
                "r2",
                "mape",
                "pct_within_5min",
                "pct_within_10min",
            ]:
                mlflow.log_metric(
                    f"{model_name}_test_{metric_name}",
                    float(row[metric_name]),
                )

        # Log structured analysis artifacts
        log_dataframe(test_metrics_df, "tables/test_metrics.json")
        log_dataframe(overfitting_df, "tables/overfitting_check.json")
        log_dataframe(day_slice_df, "tables/slice_analysis_dayofweek.json")
        log_dataframe(hour_slice_df, "tables/slice_analysis_hour.json")
        log_dataframe(error_df, "tables/error_analysis_top10.json")

        mlflow.log_param("best_model", best_model)
        mlflow.log_param("num_models_compared", len(models))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        print("\n✓ All metrics and analysis tables logged to MLflow experiment 'nyc_taxi_evaluation'")

    return best_model


if __name__ == "__main__":
    best = evaluate_all()
    print(f"\n{'=' * 70}")
    print(f"DONE — Deploy '{best}' as the production model")
    print(f"{'=' * 70}")