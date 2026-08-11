from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from demo.api.app import repo_ml
from demo.api.app.inference import predict_one
from demo.api.app.registry import get_registry
from demo.api.app.schemas import NetworkRecord
from demo.api.app.settings import EXCLUDED_FIELDS, EXPECTED_THRESHOLDS, FROZEN_SOURCE_SHA


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "golden_examples.json"
TOLERANCES = {
    "logistic_regression": 1e-10,
    "neural_network": 1e-10,
    "random_forest": 1e-10,
    "xgboost": 1e-6,
}


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed_predictions() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path).set_index("sample_index") for name, path in repo_ml.PREDICTION_PATHS.items()}


def test_fixtures_match_frozen_held_out_predictors_and_labels(fixtures: dict) -> None:
    assert fixtures["frozen_source_sha"] == FROZEN_SOURCE_SHA
    features = pd.read_parquet(repo_ml.TEST_FEATURES_PATH)
    labels = pd.read_parquet(repo_ml.TEST_LABELS_PATH)["label"]
    for example in fixtures["examples"]:
        index = example["sample_index"]
        expected = example["record"]
        actual = features.loc[index, list(expected)].to_dict()
        assert actual == expected
        assert int(labels.loc[index]) == example["evaluation_metadata"]["known_held_out_label"]
        assert not (set(expected) & set(EXCLUDED_FIELDS))


def test_every_frozen_model_matches_committed_probabilities_and_locked_labels(
    fixtures: dict, committed_predictions: dict[str, pd.DataFrame]
) -> None:
    registry = get_registry()
    assert registry.thresholds == EXPECTED_THRESHOLDS
    for example in fixtures["examples"]:
        record = NetworkRecord.model_validate(example["record"])
        metadata = example["evaluation_metadata"]
        for model_id, expected_probability in metadata["expected_probabilities"].items():
            result = predict_one(model_id, record, registry)
            committed = float(committed_predictions[model_id].loc[example["sample_index"], "attack_probability"])
            assert result.attack_probability == pytest.approx(expected_probability, abs=TOLERANCES[model_id])
            assert result.attack_probability == pytest.approx(committed, abs=TOLERANCES[model_id])
            assert result.prediction == metadata["expected_locked_predictions"][model_id]
            assert result.prediction == int(result.attack_probability >= registry.thresholds[model_id])


def test_labels_are_evaluation_metadata_not_inference_input(fixtures: dict) -> None:
    for example in fixtures["examples"]:
        assert "label" not in example["record"]
        assert "known_held_out_label" not in example["record"]
