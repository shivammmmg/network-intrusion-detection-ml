"""Single inference path used by every public prediction endpoint."""

from __future__ import annotations

from typing import Any

from .repo_ml import canonicalize_record
from .registry import MODEL_SPECS, FrozenModelRegistry, ModelSpec
from .schemas import NetworkRecord, PredictionResponse
from .settings import INPUT_CONTRACT


def normalized_frame(record: NetworkRecord, registry: FrozenModelRegistry):
    payload = record.model_dump(mode="python")
    registry.assert_known_categories(payload)
    return canonicalize_record(payload)


def predict_one_from_frame(
    model_id: str, frame: Any, registry: FrozenModelRegistry
) -> PredictionResponse:
    spec: ModelSpec = registry.get_spec(model_id)
    transformed = registry.preprocessors[spec.family].transform(frame)
    probability = float(registry.models[model_id].predict_proba(transformed)[:, 1][0])
    threshold = registry.thresholds[model_id]
    prediction = int(probability >= threshold)
    return PredictionResponse(
        model=model_id,
        attack_probability=probability,
        threshold=threshold,
        prediction=prediction,
        label="attack" if prediction else "normal",
        threshold_source="validation_selected_locked",
        input_contract=INPUT_CONTRACT,
    )


def predict_one(model_id: str, record: NetworkRecord, registry: FrozenModelRegistry) -> PredictionResponse:
    return predict_one_from_frame(model_id, normalized_frame(record, registry), registry)


def predict_all(record: NetworkRecord, registry: FrozenModelRegistry) -> list[PredictionResponse]:
    frame = normalized_frame(record, registry)
    return [predict_one_from_frame(spec.identifier, frame, registry) for spec in MODEL_SPECS]
