"""Read-only adapter to the frozen project files outside demo/."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from .settings import CANONICAL_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
THRESHOLDS_PATH = EXPERIMENTS_DIR / "standardized_evaluation" / "selected_thresholds.json"
TEST_FEATURES_PATH = PROCESSED_DIR / "X_test.parquet"
TEST_LABELS_PATH = PROCESSED_DIR / "y_test.parquet"

# XGBoost is loaded from its own native JSON format (src/07_xgboost.py's
# model.save_model() output) instead of the pickled artifacts.joblib copy.
# Native JSON is XGBoost's cross-version/cross-platform serialization, so it
# avoids joblib/pickle load failures seen on some Windows environments; both
# forms reproduce identical predict_proba output (verified numerically).
XGBOOST_NATIVE_MODEL_PATH = EXPERIMENTS_DIR / "xgboost" / "xgboost_model.json"

MODEL_PATHS = {
    "logistic_regression": ARTIFACTS_DIR / "logistic_regression.joblib",
    "neural_network": ARTIFACTS_DIR / "neural_network.joblib",
    "random_forest": ARTIFACTS_DIR / "random_forest.joblib",
    "xgboost": XGBOOST_NATIVE_MODEL_PATH,
}
PREPROCESSOR_PATHS = {
    "linear": ARTIFACTS_DIR / "preprocess_linear.joblib",
    "tree": ARTIFACTS_DIR / "preprocess_tree.joblib",
}
PREDICTION_PATHS = {
    name: EXPERIMENTS_DIR / name / "test_predictions.csv" for name in MODEL_PATHS
}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# These imports deliberately use the project implementation; demo/ never copies it.
from preprocess import clean_df, load_artifact  # noqa: E402


def require_frozen_path(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen project file is missing: {path.name}")
    return path


def load_thresholds() -> dict[str, float]:
    with require_frozen_path(THRESHOLDS_PATH).open(encoding="utf-8") as handle:
        return {name: float(value) for name, value in json.load(handle).items()}


def load_model(name: str) -> Any:
    path = require_frozen_path(MODEL_PATHS[name])
    if name == "xgboost":
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model
    return joblib.load(path)


def load_preprocessor(family: str) -> Any:
    return load_artifact(require_frozen_path(PREPROCESSOR_PATHS[family]))


def canonicalize_record(record: dict[str, Any]) -> pd.DataFrame:
    """Apply frozen cleaning and enforce the canonical public feature order."""
    frame = pd.DataFrame([record], columns=CANONICAL_FIELDS)
    cleaned = clean_df(frame)
    return cleaned.loc[:, CANONICAL_FIELDS]
