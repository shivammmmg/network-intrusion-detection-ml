"""Startup-loaded registry for the project's four immutable primary models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from . import repo_ml
from .settings import CATEGORICAL_FIELDS, EXPECTED_THRESHOLDS, FROZEN_SOURCE_SHA


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    display_name: str
    family: str
    transformed_features: int
    caveat: str


MODEL_SPECS = (
    ModelSpec("logistic_regression", "Logistic Regression", "linear", 66, "Linear baseline with one-hot categories."),
    ModelSpec("neural_network", "Neural Network", "linear", 66, "Saved artifact is served; fresh training is environment-sensitive."),
    ModelSpec("random_forest", "Random Forest", "tree", 39, "Largest artifact; uses ordinal categorical encoding."),
    ModelSpec("xgboost", "XGBoost", "tree", 39, "Saved joblib serialization is the Stage 0-verified serving artifact."),
)


class UnknownCategoryError(ValueError):
    """Raised when a normalized categorical value is outside frozen training vocabulary."""


class FrozenModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.preprocessors: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}
        self.categories: dict[str, set[str]] = {}
        self.ready = False

    def initialize(self) -> None:
        for path in (*repo_ml.MODEL_PATHS.values(), *repo_ml.PREPROCESSOR_PATHS.values(), repo_ml.THRESHOLDS_PATH):
            repo_ml.require_frozen_path(path)

        self.preprocessors = {
            family: repo_ml.load_preprocessor(family) for family in repo_ml.PREPROCESSOR_PATHS
        }
        self.thresholds = repo_ml.load_thresholds()
        if self.thresholds != EXPECTED_THRESHOLDS:
            raise RuntimeError("selected thresholds do not match the frozen verification snapshot")

        for spec in MODEL_SPECS:
            model = repo_ml.load_model(spec.identifier)
            if not hasattr(model, "predict_proba"):
                raise TypeError(f"{spec.identifier} does not expose predict_proba")
            if not np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])):
                raise ValueError(f"{spec.identifier} classes are not [0, 1]")
            transformed_width = len(self.preprocessors[spec.family].get_feature_names_out())
            if transformed_width != spec.transformed_features:
                raise ValueError(f"{spec.identifier} transformed width is {transformed_width}, not {spec.transformed_features}")
            self.models[spec.identifier] = model

        linear_categories = self.preprocessors["linear"].named_transformers_["cat"].named_steps["onehot"].categories_
        tree_categories = self.preprocessors["tree"].named_transformers_["cat"].named_steps["ordinal"].categories_
        self.categories = {
            name: set(map(str, values)) for name, values in zip(CATEGORICAL_FIELDS, linear_categories)
        }
        for name, values in zip(CATEGORICAL_FIELDS, tree_categories):
            if self.categories[name] != set(map(str, values)):
                raise ValueError(f"{name} category sets differ between frozen preprocessors")
        self.ready = True

    def assert_known_categories(self, record: dict[str, Any]) -> None:
        for field, accepted in self.categories.items():
            if record[field] not in accepted:
                raise UnknownCategoryError(f"unknown {field!r} category")

    def get_spec(self, model_id: str) -> ModelSpec:
        for spec in MODEL_SPECS:
            if spec.identifier == model_id:
                return spec
        raise KeyError(model_id)

    def model_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "id": spec.identifier,
                "display_name": spec.display_name,
                "family": spec.family,
                "threshold": self.thresholds[spec.identifier],
                "transformed_feature_count": spec.transformed_features,
                "probability_semantics": "P(attack) = predict_proba(... )[:, 1]",
                "caveat": spec.caveat,
            }
            for spec in MODEL_SPECS
        ]

    @property
    def frozen_source_sha(self) -> str:
        return FROZEN_SOURCE_SHA


@lru_cache(maxsize=1)
def get_registry() -> FrozenModelRegistry:
    registry = FrozenModelRegistry()
    registry.initialize()
    return registry
