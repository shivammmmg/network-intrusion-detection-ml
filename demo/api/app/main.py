"""FastAPI entry point for read-only inference over frozen project artifacts."""

from __future__ import annotations

from contextlib import asynccontextmanager
import math
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .examples import load_examples
from .inference import predict_all, predict_one
from .registry import FrozenModelRegistry, UnknownCategoryError, get_registry
from .schemas import HealthResponse, ModelsResponse, NetworkRecord, PredictAllResponse, PredictionResponse, SchemaResponse
from .settings import (
    BINARY_FIELDS,
    CATEGORICAL_FIELDS,
    EXCLUDED_FIELDS,
    FLOAT_FIELDS,
    FROZEN_SOURCE_SHA,
    INPUT_CONTRACT,
    INTEGER_FIELDS,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.registry = None
    app.state.initialization_failed = False
    try:
        app.state.registry = get_registry()
    except Exception:
        app.state.initialization_failed = True
    yield


app = FastAPI(
    title="EECS 3404 Frozen Model Demo API",
    version="0.1.0",
    lifespan=lifespan,
)

# The Phase 2 UI is a separate local development server. Keep the scope narrow:
# this demo has no deployment configuration and only accepts the two local origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def json_safe_validation_detail(value: object) -> object:
    """Keep malformed JSON values from breaking FastAPI's validation response."""
    if isinstance(value, float) and not math.isfinite(value):
        return "non-finite number"
    if isinstance(value, dict):
        return {str(key): json_safe_validation_detail(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_validation_detail(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe_validation_detail(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, error: RequestValidationError) -> JSONResponse:
    """Return controlled JSON for all invalid inputs, including NaN/Infinity."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": json_safe_validation_detail(error.errors())},
    )


def require_registry(request: Request) -> FrozenModelRegistry:
    registry = request.app.state.registry
    if request.app.state.initialization_failed or registry is None or not registry.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="demo artifacts are not ready")
    return registry


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    registry = getattr(request.app.state, "registry", None)
    ready = bool(registry is not None and registry.ready and not request.app.state.initialization_failed)
    model_status = {
        name: bool(ready and name in registry.models)
        for name in ("logistic_regression", "neural_network", "random_forest", "xgboost")
    }
    return HealthResponse(ready=ready, frozen_source_sha=FROZEN_SOURCE_SHA, models=model_status)


@app.get("/models", response_model=ModelsResponse)
def models(registry: FrozenModelRegistry = Depends(require_registry)) -> ModelsResponse:
    return ModelsResponse(input_contract=INPUT_CONTRACT, models=registry.model_metadata())


@app.get("/schema", response_model=SchemaResponse)
def schema(registry: FrozenModelRegistry = Depends(require_registry)) -> SchemaResponse:
    fields = []
    for name in NetworkRecord.model_fields:
        kind = "categorical" if name in CATEGORICAL_FIELDS else "numeric"
        data_type = "string" if kind == "categorical" else ("integer" if name in INTEGER_FIELDS else "number")
        fields.append(
            {
                "name": name,
                "type": data_type,
                "kind": kind,
                "required": True,
                "binary": name in BINARY_FIELDS,
                "non_negative": kind == "numeric",
            }
        )
    return SchemaResponse(
        input_contract=INPUT_CONTRACT,
        fields=fields,
        categorical_categories={name: sorted(values) for name, values in registry.categories.items()},
        excluded_fields=list(EXCLUDED_FIELDS),
    )


@app.get("/examples")
def examples(registry: FrozenModelRegistry = Depends(require_registry)) -> dict:
    del registry
    return {"input_contract": INPUT_CONTRACT, "examples": load_examples()}


@app.post("/predict/{model_id}", response_model=PredictionResponse)
def predict(model_id: str, record: NetworkRecord, registry: FrozenModelRegistry = Depends(require_registry)) -> PredictionResponse:
    try:
        registry.get_spec(model_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown model") from error
    try:
        return predict_one(model_id, record, registry)
    except UnknownCategoryError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error


@app.post("/predict-all", response_model=PredictAllResponse)
def predict_every_model(record: NetworkRecord, registry: FrozenModelRegistry = Depends(require_registry)) -> PredictAllResponse:
    try:
        results = predict_all(record, registry)
    except UnknownCategoryError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    return PredictAllResponse(input_contract=INPUT_CONTRACT, results=results)
