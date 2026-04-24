#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# initialize Git only if not already initialized
if [ ! -d ".git" ]; then
  git init
fi

# add dependencies
uv add pandas numpy scikit-learn xgboost jupyter matplotlib seaborn pyarrow mlflow optuna dvc joblib

# create directory structure
mkdir -p \
  data/raw \
  data/processed \
  model \
  reports/tables \
  src/data \
  src/features \
  src/train \
  src/evaluate \
  src/register \
  src/serve \
  tests \
  notebooks \
  airflow/dags \
  monitoring \
  docker

# download taxi data only if missing
if [ ! -f data/raw/yellow_tripdata_2026-01.parquet ]; then
  curl -L -o data/raw/yellow_tripdata_2026-01.parquet \
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet
fi

if [ ! -f data/raw/yellow_tripdata_2026-02.parquet ]; then
  curl -L -o data/raw/yellow_tripdata_2026-02.parquet \
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-02.parquet
fi

echo "Setup complete!"