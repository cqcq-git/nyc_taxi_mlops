from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


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
    try:
        row = data.model_dump()
        features = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        prediction = float(model.predict(features)[0])

        return PredictionResponse(
            predicted_duration_minutes=prediction,
            input=data,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )