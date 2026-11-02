"""API tests to run without starting a server, using FastAPI's TestClient."""
import os

os.environ["API_KEY"] = "test-key"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import MODEL_PATH, app  # noqa: E402

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}
PASSENGER = {"Pclass": 1, "Name": "Cumings, Mrs. John Bradley", "Sex": "female", "Age": 38,
             "SibSp": 1, "Parch": 0, "Fare": 71.28, "Cabin": "C85", "Embarked": "C"}

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_rejects_missing_api_key():
    assert client.post("/predict", json=PASSENGER).status_code == 401


def test_rejects_wrong_api_key():
    r = client.post("/predict", json=PASSENGER, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


@needs_model
def test_predict_valid_passenger():
    r = client.post("/predict", json=PASSENGER, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["survived"] in (0, 1)
    assert 0 <= body["survival_probability"] <= 1


@needs_model
def test_predict_with_missing_optional_fields():
    r = client.post("/predict", json={"Pclass": 3, "Sex": "male"}, headers=HEADERS)
    assert r.status_code == 200


def test_rejects_invalid_input():
    bad = {**PASSENGER, "Pclass": 5, "Sex": "robot", "Age": -3}
    assert client.post("/predict", json=bad, headers=HEADERS).status_code == 422


def test_rejects_unknown_fields():
    assert client.post("/predict", json={**PASSENGER, "hack": 1}, headers=HEADERS).status_code == 422


@needs_model
def test_batch_predict():
    r = client.post("/predict/batch", json=[PASSENGER, {**PASSENGER, "Sex": "male", "Pclass": 3}],
                    headers=HEADERS)
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_batch_limit():
    r = client.post("/predict/batch", json=[PASSENGER] * 101, headers=HEADERS)
    assert r.status_code == 413