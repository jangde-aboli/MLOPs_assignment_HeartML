import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from heartml.data import clean, split_xy
from heartml.features import build_pipeline


@pytest.fixture
def client(raw_frame, tmp_path, monkeypatch):
    """API backed by a tiny model trained on synthetic data, so the tests
    don't depend on the real training run."""
    X, y = split_xy(clean(raw_frame))
    model_path = tmp_path / "heart_model.joblib"
    joblib.dump(build_pipeline(LogisticRegression(max_iter=500)).fit(X, y), model_path)
    (tmp_path / "model_meta.json").write_text('{"model_name": "test_lr"}')
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "test_lr"
    assert r.headers["x-request-id"] == body["request_id"]


def test_predict_rejects_invalid_input(client, sample_patient):
    assert client.post("/predict", json={**sample_patient, "cp": 9}).status_code == 422
    missing = {k: v for k, v in sample_patient.items() if k != "age"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**sample_patient, "foo": 1}).status_code == 422


def test_predict_batch(client, sample_patient):
    r = client.post("/predict/batch",
                    json={"patients": [sample_patient, {**sample_patient, "age": 40}]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_metrics_exposed(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text
    assert 'endpoint="/predict"' in text


def test_model_info(client):
    assert client.get("/model-info").json()["model_name"] == "test_lr"
