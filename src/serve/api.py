from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app
import time

MODEL_PATH = Path("model/production_model.pkl")

FEATURE_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
]


app = FastAPI(
    title="NYC Taxi Trip Duration Predictor",
    description="Predicts NYC taxi trip duration in minutes.",
    version="1.0.0",
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["status"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent on prediction in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTION_VALUE = Histogram(
    "prediction_value_minutes",
    "Distribution of predicted trip durations",
    buckets=(1, 5, 10, 15, 20, 30, 45, 60),
)

app.mount("/metrics", make_asgi_app())


class TripFeatures(BaseModel):
    PULocationID: int = Field(..., example=100)
    DOLocationID: int = Field(..., example=200)
    trip_distance: float = Field(..., gt=0, example=2.5)
    passenger_count: int = Field(..., ge=1, example=1)
    pickup_hour: int = Field(..., ge=0, le=23, example=14)
    pickup_dayofweek: int = Field(..., ge=0, le=6, example=2)
    pickup_month: int = Field(..., ge=1, le=12, example=1)


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float
    input: TripFeatures


@app.get("/")
def root():
    return {
        "service": "NYC Taxi Trip Duration Predictor",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "metrics": "/metrics",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TripFeatures):
    start_time = time.time()

    try:
        row = data.model_dump()
        features = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        prediction = float(model.predict(features)[0])

        PREDICTION_COUNT.labels(status="success").inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_VALUE.observe(prediction)

        return PredictionResponse(
            predicted_duration_minutes=prediction,
            input=data,
        )

    except Exception as e:
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )
