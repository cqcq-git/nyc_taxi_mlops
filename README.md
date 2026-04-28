````markdown
# NYC Taxi MLOps

A MLOps project for building, tuning, evaluating, serving, monitoring, and registering an NYC taxi trip-duration prediction model.

The project uses:

- **uv** for Python environment and dependency management
- **DVC** for reproducible pipeline orchestration
- **MLflow** for experiment tracking, evaluation logging, and model registry
- **Optuna** for hyperparameter tuning
- **XGBoost** for regression modeling
- **FastAPI + Uvicorn** for model serving
- **Docker + Docker Compose** for containerization and orchestration
- **Prometheus + Grafana** for monitoring
- **Evidently** for drift detection
- **Pytest + Ruff + GitHub Actions** for testing and CI

The goal of this project is to build the MLOps workflow step by step, not to hide everything behind a black-box code generator.

---

## Project Structure

```text
nyc_taxi_mlops/
├── scripts/
│   ├── setup.sh
│   └── run_pipeline.sh
├── src/
│   ├── data/
│   ├── features/
│   ├── train/
│   │   ├── train_model.py
│   │   ├── tune.py
│   │   └── tune_cv.py
│   ├── evaluate/
│   │   └── evaluate_model.py
│   ├── register/
│   │   └── register_model.py
│   └── serve/
│       └── api.py
├── monitoring/
│   ├── check_drift.py
│   ├── prometheus.yml
│   ├── RUNBOOK.md
│   └── reports/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_api.py
│   └── test_preprocess.py
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── reports/
│   └── tables/
├── .github/workflows/
│   └── ci.yml
├── dvc.yaml
├── dvc.lock
├── pyproject.toml
├── uv.lock
└── README.md
````

---

## Quick Start

This project is designed so the main workflow can be run with two scripts.

### 1. Set up the project

```bash
./scripts/setup.sh
```

This script:

* initializes Git if needed
* installs project dependencies with `uv`
* creates the project directory structure
* downloads NYC taxi parquet data into `data/raw/`

### 2. Run the full ML pipeline

```bash
./scripts/run_pipeline.sh
```

This script:

* starts a local MLflow tracking server
* runs the full DVC pipeline with `dvc repro`
* stops the local MLflow process at the end

This is the preferred local one-command workflow.

---

## Pipeline Overview

The DVC pipeline includes the following stages:

```text
raw data
  ↓
preprocess_train
  ↓
preprocess_test
  ↓
train_model
  ↓
tune_model
  ↓
tune_model_cv
  ↓
evaluate_model
  ↓
register_model
```

### Preprocessing

The preprocessing stages transform raw NYC taxi data into processed train/test parquet files.

Outputs:

```text
data/processed/train.parquet
data/processed/test.parquet
```

### Training

`src/train/train_model.py` trains a baseline XGBoost model.

Output:

```text
model/xgboost_model.pkl
```

### Hyperparameter Tuning

`src/train/tune.py` uses Optuna with a single train/validation split.

Output:

```text
model/xgboost_tuned_model.pkl
```

### Cross-Validation Tuning

`src/train/tune_cv.py` uses Optuna with 5-fold cross-validation.

Output:

```text
model/xgboost_tuned_cv_model.pkl
```

### Evaluation

`src/evaluate/evaluate_model.py` compares all trained models on the test set.

It logs:

* RMSE
* MAE
* Median absolute error
* R²
* MAPE
* percentage of predictions within 5 minutes
* percentage of predictions within 10 minutes
* overfitting checks
* slice analysis
* top prediction errors

Outputs:

```text
reports/tables/test_metrics.json
reports/tables/overfitting_check.json
reports/tables/slice_analysis_dayofweek.json
reports/tables/slice_analysis_hour.json
reports/tables/error_analysis_top10.json
```

The evaluation step also logs the selected best model name to MLflow as:

```text
best_model
```

### Model Registration

`src/register/register_model.py` reads the latest evaluation result from MLflow, finds the selected `best_model`, loads the corresponding local model artifact, logs it as an MLflow model, and registers it as:

```text
nyc_taxi_duration_predictor
```

It also creates the stable production model artifact:

```text
model/production_model.pkl
```

This file is automatically tracked through the DVC pipeline stage output.

---

## Inference API

The project includes a FastAPI inference service for online predictions.

Main file:

```text
src/serve/api.py
```

The API loads:

```text
model/production_model.pkl
```

which is automatically created by the `register_model` DVC stage.

### Run Locally

```bash
uv run uvicorn src.serve.api:app --reload --port 8000
```

### API Endpoints

| Endpoint   | Purpose              |
| ---------- | -------------------- |
| `/health`  | service health check |
| `/predict` | prediction endpoint  |
| `/metrics` | Prometheus metrics   |
| `/docs`    | Swagger UI           |

### Example Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PULocationID": 100,
    "DOLocationID": 200,
    "trip_distance": 2.5,
    "passenger_count": 1,
    "pickup_hour": 14,
    "pickup_dayofweek": 2,
    "pickup_month": 1
  }'
```

---

## Docker and Docker Compose

### Build API Image

```bash
docker build --no-cache -t nyc-taxi-api -f docker/Dockerfile .
```

### Run API Container

```bash
docker run --rm -p 8000:8000 nyc-taxi-api
```

### Run Full Monitoring Stack

```bash
docker compose -f docker/docker-compose.yml up --build
```

Services:

| Service      | URL                                                      |
| ------------ | -------------------------------------------------------- |
| FastAPI      | [http://localhost:8000](http://localhost:8000)           |
| Swagger Docs | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Prometheus   | [http://localhost:9090](http://localhost:9090)           |
| Grafana      | [http://localhost:3000](http://localhost:3000)           |

Grafana default credentials:

```text
admin / admin
```

---

## Monitoring and Drift Detection

### Prometheus Metrics

The API exposes Prometheus metrics at:

```text
/metrics
```

Tracked metrics include:

* prediction request count
* prediction latency
* prediction value distribution
* prediction errors

### Drift Detection

Generate Evidently drift reports:

```bash
uv run python monitoring/check_drift.py
```

Reports are generated in:

```text
monitoring/reports/
```

Open reports:

```bash
open monitoring/reports/data_drift.html
open monitoring/reports/prediction_drift.html
```

---

## MLflow Setup

For local development, the pipeline uses MLflow with:

* SQLite backend store
* local artifact store

The intended local MLflow command is:

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

However, you usually do not need to run this manually because `scripts/run_pipeline.sh` starts MLflow for you.

### Backend Store vs Artifact Store

MLflow uses two storage layers:

```text
backend store = metadata database
artifact store = files
```

The backend store contains:

* experiments
* runs
* params
* metrics
* tags
* model registry metadata

In this project:

```text
mlflow.db
```

The artifact store contains:

* logged models
* JSON reports
* plots
* other generated files

In this project:

```text
mlartifacts/
```

---

## Running Individual Steps Manually

You can still run each script manually.

First start MLflow:

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Then run scripts:

```bash
uv run python src/train/train_model.py
uv run python src/train/tune.py
uv run python src/train/tune_cv.py
uv run python src/evaluate/evaluate_model.py
uv run python src/register/register_model.py
```

Or use DVC directly:

```bash
dvc repro
```

---

## DVC Usage

Reproduce the full pipeline:

```bash
dvc repro
```

Push DVC-tracked data and artifacts to the configured remote:

```bash
dvc push
```

DVC tracks data/model/report artifacts. Git tracks source code and pipeline definitions.

Typical commit flow:

```bash
dvc repro
dvc push

git add dvc.yaml dvc.lock src scripts pyproject.toml uv.lock .gitignore README.md
git commit -m "Update MLOps pipeline"
git push
```

---

## Testing and CI

### Run Tests

```bash
uv run pytest tests/ -v
```

### Run Linter

```bash
uv run ruff check src/ tests/ monitoring/
```

### GitHub Actions

The project includes a basic GitHub Actions CI workflow:

```text
.github/workflows/ci.yml
```

Current CI checks:

* Ruff linting
* preprocessing tests

The workflow intentionally excludes API/model tests in CI for now because model artifacts are DVC-managed and not committed directly to GitHub.

---

## Git Ignore Recommendations

Generated artifacts should not be committed directly to Git.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/

data/raw/
data/processed/
model/
reports/
monitoring/reports/

mlruns/
mlartifacts/
mlflow.db
mlflow_server.log
```

DVC should handle large data and model artifacts.

---

## Notes on Local Development

This is a local project, so the workflow intentionally favors readability over production complexity.

Current local workflow:

```bash
# initial setup
./scripts/setup.sh

# run full ML pipeline
./scripts/run_pipeline.sh

# start monitoring stack
docker compose -f docker/docker-compose.yml up --build

# generate drift reports
uv run python monitoring/check_drift.py
```

```
```
