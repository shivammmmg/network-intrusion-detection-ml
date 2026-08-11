from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from demo.api.app.main import app
from demo.api.app.settings import EXPECTED_THRESHOLDS, FROZEN_SOURCE_SHA, INPUT_CONTRACT


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "golden_examples.json"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["examples"][0]["record"]


def test_read_endpoints_expose_frozen_contract(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert health.json()["frozen_source_sha"] == FROZEN_SOURCE_SHA
    assert all(health.json()["models"].values())

    models = client.get("/models")
    assert models.status_code == 200
    assert models.json()["input_contract"] == INPUT_CONTRACT
    assert {item["id"] for item in models.json()["models"]} == set(EXPECTED_THRESHOLDS)

    schema = client.get("/schema")
    assert schema.status_code == 200
    assert len(schema.json()["fields"]) == 39
    assert {name: len(values) for name, values in schema.json()["categorical_categories"].items()} == {
        "proto": 133,
        "service": 13,
        "state": 9,
    }

    examples = client.get("/examples")
    assert examples.status_code == 200
    assert len(examples.json()["examples"]) == 6


def test_local_frontend_origin_is_permitted(client: TestClient) -> None:
    response = client.options(
        "/predict-all",
        headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"
    assert "POST" in response.headers["access-control-allow-methods"]


@pytest.mark.parametrize("model_id", list(EXPECTED_THRESHOLDS))
def test_predict_one_for_each_model(client: TestClient, record: dict, model_id: str) -> None:
    response = client.post(f"/predict/{model_id}", json=record)
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == model_id
    assert payload["threshold"] == EXPECTED_THRESHOLDS[model_id]
    assert payload["prediction"] == int(payload["attack_probability"] >= payload["threshold"])


def test_predict_all_uses_one_record_and_returns_four_results(client: TestClient, record: dict) -> None:
    response = client.post("/predict-all", json=record)
    assert response.status_code == 200
    payload = response.json()
    assert payload["input_contract"] == INPUT_CONTRACT
    assert [item["model"] for item in payload["results"]] == list(EXPECTED_THRESHOLDS)


@pytest.mark.parametrize("field,value", [("label", 1), ("attack_cat", "Normal"), ("sttl", 254), ("spkts", -1), ("is_ftp_login", 2), ("proto", "unknown")])
def test_security_and_schema_errors_are_rejected(client: TestClient, record: dict, field: str, value: object) -> None:
    payload = dict(record)
    payload[field] = value
    response = client.post("/predict-all", json=payload)
    assert response.status_code == 422
