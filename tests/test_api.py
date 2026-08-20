from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Predictive Maintenance API is running"

def test_predict():
    payload = {
        "machine_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 42.0,
        "tool_wear": 100.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in (0, 1)
    assert 0 <= data["failure_probability"] <= 1
    assert data["message"] in (
        "Machine failure predicted",
        "No machine failure predicted",
    )

def test_predict_invalid_input():
    payload = {
        "machine_type": "M",
        "air_temperature": "invalid",
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 42.0,
        "tool_wear": 100.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422