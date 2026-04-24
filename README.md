# NYC Taxi MLOps

A MLOps project for building, tuning, evaluating, and registering an NYC taxi trip-duration prediction model.

The project uses:

- **uv** for Python environment and dependency management
- **DVC** for reproducible pipeline orchestration
- **MLflow** for experiment tracking, evaluation logging, and model registry
- **Optuna** for hyperparameter tuning
- **XGBoost** for regression modeling

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
│   └── register/
│       └── register_model.py
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── reports/
│   └── tables/
├── dvc.yaml
├── dvc.lock
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Quick Start

This project is designed so the main workflow can be run with two scripts.

### 1. Set up the project

```bash
./scripts/setup.sh
```

This script:

- initializes Git if needed
- installs project dependencies with `uv`
- creates the project directory structure
- downloads NYC taxi parquet data into `data/raw/`

### 2. Run the full ML pipeline

```bash
./scripts/run_pipeline.sh
```

This script:

- starts a local MLflow tracking server
- runs the full DVC pipeline with `dvc repro`
- stops the local MLflow process at the end

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

- RMSE
- MAE
- Median absolute error
- R²
- MAPE
- percentage of predictions within 5 minutes
- percentage of predictions within 10 minutes
- overfitting checks
- slice analysis
- top prediction errors

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

---

## MLflow Setup

For local development, the pipeline uses MLflow with:

- SQLite backend store
- local artifact store

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

- experiments
- runs
- params
- metrics
- tags
- model registry metadata

In this project:

```text
mlflow.db
```

The artifact store contains:

- logged models
- JSON reports
- plots
- other generated files

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
git commit -m "Add full ML pipeline"
git push
```

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
mlruns/
mlartifacts/
mlflow.db
mlflow_server.log
```

DVC should handle large data and model artifacts.

---

## Notes on Local Development

This is a local learning project, so the workflow intentionally favors readability over production complexity.

Current local workflow:

```bash
./scripts/setup.sh
./scripts/run_pipeline.sh
```




