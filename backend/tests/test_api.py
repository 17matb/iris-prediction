from app.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_predict_valid_payload():
    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)


def test_predict_missing_field():
    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_type():
    payload = {
        "sepal_length_cm": "not a number",
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
