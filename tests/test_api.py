from fastapi.testclient import TestClient

from src.serve.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert "service" in body
    assert "endpoints" in body


def test_predict_valid_input():
    payload = {
        "PULocationID": 100,
        "DOLocationID": 200,
        "trip_distance": 2.5,
        "passenger_count": 1,
        "pickup_hour": 14,
        "pickup_dayofweek": 2,
        "pickup_month": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "predicted_duration_minutes" in body
    assert body["predicted_duration_minutes"] > 0


def test_predict_invalid_input():
    payload = {
        "PULocationID": "bad",
        "DOLocationID": 200,
        "trip_distance": 2.5,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    