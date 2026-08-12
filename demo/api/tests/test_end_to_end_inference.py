"""Regression coverage for the real browser-to-artifact inference contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from starlette.testclient import TestClient

from demo.api.app import repo_ml
from demo.api.app.main import app
from demo.api.app.registry import MODEL_SPECS
from demo.api.app.settings import CANONICAL_FIELDS, EXPECTED_THRESHOLDS


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "golden_examples.json"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def direct_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen files independently of the API registry."""
    preprocessors = {
        family: repo_ml.load_preprocessor(family) for family in repo_ml.PREPROCESSOR_PATHS
    }
    models: dict[str, Any] = {}
    for model_id, path in repo_ml.MODEL_PATHS.items():
        if model_id == "xgboost":
            model = xgb.XGBClassifier()
            model.load_model(path)
            models[model_id] = model
        else:
            models[model_id] = joblib.load(path)
    return preprocessors, models


def direct_probability(model_id: str, record: dict[str, Any], direct_artifacts: tuple[dict[str, Any], dict[str, Any]]) -> float:
    preprocessors, models = direct_artifacts
    spec = next(spec for spec in MODEL_SPECS if spec.identifier == model_id)
    frame = repo_ml.canonicalize_record(record)
    transformed = preprocessors[spec.family].transform(frame)
    return float(models[model_id].predict_proba(transformed)[:, 1][0])


@pytest.fixture(scope="module")
def records() -> list[dict[str, Any]]:
    features = pd.read_parquet(repo_ml.TEST_FEATURES_PATH)
    selected_indexes = np.random.default_rng(3404).choice(features.index.to_numpy(), size=5, replace=False)
    sampled = [features.loc[index, list(CANONICAL_FIELDS)].to_dict() for index in selected_indexes]
    for record in sampled:
        for key, value in tuple(record.items()):
            if isinstance(value, np.generic):
                record[key] = value.item()

    # A valid manually-entered edge record: zeroes, a tiny decimal, and a large
    # count, while retaining real fitted-vocabulary categories from a test row.
    manual = dict(sampled[0])
    manual.update({"dur": 0.0, "rate": 1e-12, "sbytes": 1_000_000_000, "dbytes": 0, "spkts": 0, "dpkts": 0})
    return [*sampled, manual]


@pytest.mark.parametrize("model_id", [spec.identifier for spec in MODEL_SPECS])
def test_api_matches_independently_loaded_frozen_artifacts_for_random_and_manual_records(
    client: TestClient, records: list[dict[str, Any]], direct_artifacts: tuple[dict[str, Any], dict[str, Any]], model_id: str
) -> None:
    for record in records:
        response = client.post(f"/predict/{model_id}", json=record)
        assert response.status_code == 200, response.text
        result = response.json()
        probability = direct_probability(model_id, record, direct_artifacts)
        assert result["attack_probability"] == pytest.approx(probability, abs=1e-6)
        assert result["threshold"] == EXPECTED_THRESHOLDS[model_id]
        assert result["prediction"] == int(probability >= EXPECTED_THRESHOLDS[model_id])
        assert result["label"] == ("attack" if result["prediction"] else "normal")


def test_predict_all_uses_the_same_record_for_all_four_independently_loaded_models(
    client: TestClient, records: list[dict[str, Any]], direct_artifacts: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    response = client.post("/predict-all", json=records[-1])
    assert response.status_code == 200, response.text
    results = {result["model"]: result for result in response.json()["results"]}
    assert set(results) == set(EXPECTED_THRESHOLDS)
    for model_id in EXPECTED_THRESHOLDS:
        probability = direct_probability(model_id, records[-1], direct_artifacts)
        assert results[model_id]["attack_probability"] == pytest.approx(probability, abs=1e-6)
        assert results[model_id]["prediction"] == int(probability >= EXPECTED_THRESHOLDS[model_id])


@pytest.fixture(scope="module")
def valid_record() -> dict[str, Any]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["examples"][0]["record"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.pop("dur"),
        lambda record: record.update({"unexpected": 1}),
        lambda record: record.update({"spkts": "one"}),
        lambda record: record.update({"dur": None}),
    ],
)
def test_invalid_manual_input_is_rejected_without_a_prediction(client: TestClient, valid_record: dict[str, Any], mutate: Any) -> None:
    record = dict(valid_record)
    mutate(record)
    response = client.post("/predict-all", json=record)
    assert response.status_code == 422
    assert "results" not in response.json()


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_numbers_are_rejected_without_a_prediction(
    client: TestClient, valid_record: dict[str, Any], non_finite: str
) -> None:
    body = json.dumps({**valid_record, "dur": float(non_finite)})
    response = client.post("/predict-all", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "results" not in response.json()


def test_malformed_request_and_unknown_model_are_controlled_4xx_responses(client: TestClient, valid_record: dict[str, Any]) -> None:
    malformed = client.post("/predict-all", content=b"{not json", headers={"content-type": "application/json"})
    assert malformed.status_code == 422
    unknown_model = client.post("/predict/not-a-model", json=valid_record)
    assert unknown_model.status_code == 404
    assert unknown_model.json() == {"detail": "unknown model"}
