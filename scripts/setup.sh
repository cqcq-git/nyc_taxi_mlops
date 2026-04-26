#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# initialize Git only if not already initialized
if [ ! -d ".git" ]; then
  git init
fi

# add dependencies
# production deps: only what the inference container needs, keep it minimal
uv add \
  fastapi \
  uvicorn \
  pydantic \
  xgboost \
  pandas \
  numpy \
  scikit-learn \
  joblib \
  pyarrow
# dev deps: training, experimentation, exploration tools
uv add --dev \
  mlflow \
  optuna \
  great-expectations \
  evidently \
  dvc \
  jupyter \
  matplotlib \
  seaborn


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

# --- .dockerignore ---
# create if missing, ensures small Docker images from day one
if [ ! -f ".dockerignore" ]; then
  cat > .dockerignore << 'EOF'
.venv/
__pycache__/
*.pyc
.git/
.dvc/
.dvc/cache/
data/
notebooks/
mlruns/
mlflow.db
.pytest_cache/
.ruff_cache/
.DS_Store
.env
*.ipynb
tests/
EOF
fi


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
echo "Next steps:"
echo "1. Run the pipeline: ./scripts/run_pipeline.sh"
