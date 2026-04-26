#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Start the MLflow server
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  > mlflow_server.log 2>&1 &

sleep 5

# run the pipeline
dvc repro

# Cleanup: stop MLflow server
lsof -t -i :5000 | xargs kill 2>/dev/null || true