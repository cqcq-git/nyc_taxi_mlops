import os
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

RANDOM_STATE = 42
VAL_SIZE = 0.3
N_TRIALS = 20


def objective(trial, X_train, X_val, y_train, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": RANDOM_STATE,
    }

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)

        mlflow.log_metric("val_rmse", rmse)
        return rmse


if __name__ == "__main__":
    # load preprocessed data once
    df = pd.read_parquet("data/processed/train.parquet")

    X = df.drop("duration", axis=1)
    y = df["duration"]

    # split once for all trials
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("nyc_taxi_tuning")

    with mlflow.start_run(run_name="optuna_single_split"):
        mlflow.log_param("val_size", VAL_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("n_trials", N_TRIALS)

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X_train, X_val, y_train, y_val),
            n_trials=N_TRIALS,
        )

        best_params = study.best_params | {"random_state": RANDOM_STATE}

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_rmse", study.best_value)

        # retrain final model on the full dataset with the best params
        final_model = XGBRegressor(**best_params)
        final_model.fit(X, y)

        full_preds = final_model.predict(X)
        full_train_rmse = root_mean_squared_error(y, full_preds)
        mlflow.log_metric("final_train_rmse_full_data", full_train_rmse)
        mlflow.log_param("final_training_rows", len(X))

        mlflow.xgboost.log_model(final_model, "best_xgboost_model")
        Path("model").mkdir(exist_ok=True)
        joblib.dump(final_model, "model/xgboost_tuned_model.pkl")

        print(f"Best params: {study.best_params}")
        print(f"Best validation RMSE: {study.best_value:.3f}")
        print(f"Final model retrained on {len(X)} rows")
        print(f"Final full-data train RMSE: {full_train_rmse:.3f}")
