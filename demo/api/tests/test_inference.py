from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.api.app.inference import normalized_frame, predict_all, predict_one
from demo.api.app.registry import UnknownCategoryError, get_registry
from demo.api.app.schemas import NetworkRecord
from demo.api.app.settings import CANONICAL_FIELDS, INPUT_CONTRACT


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "golden_examples.json"


@pytest.fixture(scope="module")
def example() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["examples"][2]


def test_one_normalized_record_feeds_all_four_models(example: dict) -> None:
    registry = get_registry()
    record = NetworkRecord.model_validate(example["record"])
    frame = normalized_frame(record, registry)
    assert list(frame.columns) == list(CANONICAL_FIELDS)
    results = predict_all(record, registry)
    assert [result.model for result in results] == [
        "logistic_regression",
        "neural_network",
        "random_forest",
        "xgboost",
    ]
    assert all(result.input_contract == INPUT_CONTRACT for result in results)
    assert all(result.prediction == int(result.attack_probability >= result.threshold) for result in results)


def test_unknown_category_is_rejected_after_normalization(example: dict) -> None:
    payload = dict(example["record"])
    payload["proto"] = "not-a-frozen-protocol"
    with pytest.raises(UnknownCategoryError):
        predict_one("random_forest", NetworkRecord.model_validate(payload), get_registry())


def test_all_models_use_locked_threshold_not_predict_method(example: dict) -> None:
    result = predict_one("neural_network", NetworkRecord.model_validate(example["record"]), get_registry())
    assert result.threshold != 0.5
    assert result.prediction == int(result.attack_probability >= result.threshold)
