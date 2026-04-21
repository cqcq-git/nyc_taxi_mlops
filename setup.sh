#!/bin/bash
set -e  # exit immediately if any command fails

# create project and initialize Git + UV, uv step already done since we run inside project directory
# uv init nyc_taxi_mlops
# cd nyc_taxi_mlops
git init

# add dependencies
uv add pandas numpy scikit-learn xgboost jupyter matplotlib seaborn pyarrow

# create directory structure
mkdir -p data/raw data/processed model \
    src/data src/features src/train src/evaluate src/serve \
    tests notebooks airflow/dags monitoring docker

# download taxi data
curl -o data/raw/yellow_tripdata_2026-01.parquet \
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet
curl -o data/raw/yellow_tripdata_2026-02.parquet \
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-02.parquet
echo "Setup complete!"
