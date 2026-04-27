import os
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_model():
    # load preprocessed data
    df = pd.read_parquet("data/processed/train.parquet")

    X = df.drop("duration", axis=1)
    y = df["duration"]

    # split into train and validation
    val_size = 0.3
    random_state = 42
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    # set up MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("nyc_taxi_duration")

    with mlflow.start_run():
        # model configuration
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": random_state,
        }

        # log split info and model params
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        # log model params
        for key, value in params.items():
            mlflow.log_param(key, value)

        # train
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # evaluate on both sets
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        train_rmse = root_mean_squared_error(y_train, train_preds)
        val_rmse = root_mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)

        # log and save the model
        mlflow.xgboost.log_model(model, "xgboost_model")
        Path("model").mkdir(exist_ok=True)
        joblib.dump(model, "model/xgboost_model.pkl")

        print(f"Train rows: {len(X_train)} | Val rows: {len(X_val)}")
        print(f"Train RMSE: {train_rmse:.3f} | Val RMSE: {val_rmse:.3f} | Val MAE: {val_mae:.3f}")


if __name__ == "__main__":
    train_model()
