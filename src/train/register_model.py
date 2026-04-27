import os
import shutil
from pathlib import Path
import joblib
import mlflow
import mlflow.xgboost

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = "nyc_taxi_duration_predictor"
PRODUCTION_MODEL_PATH = Path("model/production_model.pkl")

MODEL_PATHS = {
    "xgboost_baseline": "model/xgboost_model.pkl",
    "xgboost_tuned": "model/xgboost_tuned_model.pkl",
    "xgboost_tuned_cv": "model/xgboost_tuned_cv_model.pkl",
}

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# find latest evaluation run
experiment = client.get_experiment_by_name("nyc_taxi_evaluation")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

eval_run = runs[0]
best_model = eval_run.data.params["best_model"]
model_path = MODEL_PATHS[best_model]

print(f"Best model from evaluation: {best_model}")
print(f"Loading model from: {model_path}")

# copy best model to stable production path
PRODUCTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(model_path, PRODUCTION_MODEL_PATH)
print(f"Copied best model to: {PRODUCTION_MODEL_PATH}")


model = joblib.load(model_path)

# log and register model
mlflow.set_experiment("nyc_taxi_model_registry")

with mlflow.start_run(run_name=f"register_{best_model}") as run:
    mlflow.log_param("selected_model", best_model)
    mlflow.log_param("source_evaluation_run_id", eval_run.info.run_id)
    mlflow.log_param("local_model_path", model_path)
    mlflow.log_param("production_model_path", str(PRODUCTION_MODEL_PATH))
    mlflow.xgboost.log_model(model, artifact_path="model")

    model_uri = f"runs:/{run.info.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print(f"Registered model '{REGISTERED_MODEL_NAME}' version {registered_model.version}")
