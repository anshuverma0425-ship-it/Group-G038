from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_INPUT = {
    "battery_start_pct": 40,
    "session_minutes": 120,
    "charger_kw": 7.2,
    "temp_c": 30,
    "vehicle_age_yr": 3,
    "queue_before": 2,
    "vehicle": "car",
    "station_type": "Highway"
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_prediction():
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))


def test_rejects_queue_above_20():
    data = {**VALID_INPUT, "queue_before": 21}
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_rejects_charger_below_1_5():
    data = {**VALID_INPUT, "charger_kw": 1.0}
    response = client.post("/predict", json=data)
    assert response.status_code == 422
