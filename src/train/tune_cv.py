import os
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

N_SPLITS = 5
RANDOM_STATE = 42
N_TRIALS = 20


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": RANDOM_STATE,
    }

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rmses = []

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_param("cv_folds", N_SPLITS)
        mlflow.log_param("cv_shuffle", True)
        mlflow.log_param("cv_random_state", RANDOM_STATE)
        mlflow.log_param("dataset_rows", len(X))
        mlflow.log_params(params)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            fold_rmse = root_mean_squared_error(y_val, preds)
            fold_rmses.append(fold_rmse)

            mlflow.log_metric(f"fold_{fold_idx}_rmse", fold_rmse)

        cv_rmse_mean = float(np.mean(fold_rmses))
        cv_rmse_std = float(np.std(fold_rmses))

        mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
        mlflow.log_metric("cv_rmse_std", cv_rmse_std)

        trial.set_user_attr("fold_rmses", fold_rmses)
        return cv_rmse_mean


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/train.parquet")
    X = df.drop("duration", axis=1)
    y = df["duration"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("nyc_taxi_tuning")

    with mlflow.start_run(run_name="optuna_5fold_cv"):
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", N_SPLITS)
        mlflow.log_param("cv_strategy", "KFold")
        mlflow.log_param("dataset_rows", len(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS)

        best_params = study.best_params | {"random_state": RANDOM_STATE}

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_rmse", study.best_value)

        # retrain final model on the full dataset with the best params
        final_model = XGBRegressor(**best_params)
        final_model.fit(X, y)

        full_preds = final_model.predict(X)
        full_train_rmse = root_mean_squared_error(y, full_preds)
        mlflow.log_metric("final_train_rmse_full_data", full_train_rmse)
        mlflow.log_param("final_training_rows", len(X))

        mlflow.xgboost.log_model(final_model, "best_xgboost_model_cv")
        Path("model").mkdir(exist_ok=True)
        joblib.dump(final_model, "model/xgboost_tuned_cv_model.pkl")

        print(f"Best params: {study.best_params}")
        print(f"Best 5-fold CV RMSE: {study.best_value:.3f}")
        print(f"Final model retrained on {len(X)} rows")
        print(f"Final full-data train RMSE: {full_train_rmse:.3f}")
