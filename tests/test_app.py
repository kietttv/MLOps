"""Integration tests for the Flask prediction service."""

from __future__ import annotations

import json
from typing import Dict

import mlflow.pyfunc
import numpy as np
import pytest
from flask.testing import FlaskClient

from app.app import FEATURE_NAMES, create_app


class DummyModel(mlflow.pyfunc.PyFuncModel):
    """Stub model returning a deterministic prediction for testing."""

    def __init__(self) -> None:
        self.meta = None

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.ones(shape=(data.shape[0],), dtype=int)


@pytest.fixture(name="client")
def fixture_client(monkeypatch: pytest.MonkeyPatch) -> FlaskClient:
    """Provide a Flask test client with the model loader patched."""

    monkeypatch.setattr("app.app._load_model", lambda: DummyModel())
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _make_payload() -> Dict[str, float]:
    return {feature: float(idx) for idx, feature in enumerate(FEATURE_NAMES, start=1)}


def test_health_endpoint(client: FlaskClient) -> None:
    """Ensure the health endpoint returns operational status."""

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok"}


def test_predict_valid_payload(client: FlaskClient) -> None:
    """Submitting a valid payload should return a prediction and label."""

    payload = _make_payload()
    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 200
    body = response.get_json()
    assert "prediction" in body
    assert "label" in body


def test_predict_missing_field(client: FlaskClient) -> None:
    """Missing one of the required features should result in HTTP 400."""

    payload = _make_payload()
    payload.pop(FEATURE_NAMES[0])

    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 400
    body = response.get_json()
    assert "error" in body

