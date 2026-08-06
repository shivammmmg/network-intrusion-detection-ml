from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_FOLDER / "src"))

from config import ARTIFACTS_DIR, PROCESSED_DIR  # noqa: E402
from preprocess import load_artifact  # noqa: E402


EXPERIMENT_DIR = ROOT_FOLDER / "experiments" / "neural_network"

MODEL_PATH = ARTIFACTS_DIR / "neural_network.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocess_linear.joblib"

METRICS_PATH = EXPERIMENT_DIR / "metrics.json"
VALIDATION_PREDICTIONS_PATH = EXPERIMENT_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = EXPERIMENT_DIR / "test_predictions.csv"


def load_split(split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Function loads one original processed split"""

    x_path = PROCESSED_DIR / f"x_{split_name}.parquet"

    if not x_path.exists():
        x_path = PROCESSED_DIR / f"X_{split_name}.parquet"

    y_path = PROCESSED_DIR / f"y_{split_name}.parquet"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing feature file: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing label file: {y_path}")

    x = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["label"].astype(int)

    if len(x) != len(y):
        raise RuntimeError(
            f"Feature and label lengths differ for {split_name}: "
            f"{len(x)} != {len(y)}"
        )

    return x, y


def calculate_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Function calculates metrics using the same rule as 08_neural_network.py"""

    probabilities = np.asarray(probabilities, dtype=float)
    predicted_labels = (probabilities >= 0.5).astype(int)

    raw_metrics = {
        "pr_auc": average_precision_score(labels, probabilities),
        "roc_auc": roc_auc_score(labels, probabilities),
        "accuracy": accuracy_score(labels, predicted_labels),
        "precision": precision_score(
            labels,
            predicted_labels,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predicted_labels,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predicted_labels,
            zero_division=0,
        ),
    }

    return {
        name: round(float(value), 6)
        for name, value in raw_metrics.items()
    }


def compare_metrics(
    expected: dict[str, Any],
    actual: dict[str, float],
    split_name: str,
) -> None:
    """Function compares recalculated metrics with metrics.json"""

    mismatches: list[str] = []

    for metric_name, actual_value in actual.items():
        expected_value = round(float(expected[metric_name]), 6)

        if actual_value != expected_value:
            mismatches.append(
                f"{metric_name}: expected {expected_value}, "
                f"recalculated {actual_value}"
            )

    if mismatches:
        raise RuntimeError(
            f"{split_name} metric mismatch:\n  "
            + "\n  ".join(mismatches)
        )

    print(f"PASS: {split_name} metrics match metrics.json")


def verify_split(
    *,
    display_name: str,
    raw_split_name: str,
    prediction_path: Path,
    metrics_key: str,
    metrics_payload: dict[str, Any],
    model,
    preprocessor,
) -> None:
    """Function verifies one exported prediction file"""

    predictions = pd.read_csv(prediction_path)

    required_columns = {
        "sample_index",
        "true_label",
        "attack_probability",
        "predicted_label",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise RuntimeError(
            f"{prediction_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    csv_labels = predictions["true_label"].to_numpy(dtype=int)
    csv_probabilities = predictions[
        "attack_probability"
    ].to_numpy(dtype=float)
    csv_predicted_labels = predictions[
        "predicted_label"
    ].to_numpy(dtype=int)

    # Check 1: Prediction CSV metrics versus metrics.json
    csv_metrics = calculate_metrics(csv_labels, csv_probabilities)

    compare_metrics(
        metrics_payload[metrics_key],
        csv_metrics,
        display_name,
    )

    # Check 2: Saved source labels versus prediction CSV labels
    x_raw, y = load_split(raw_split_name)

    if not np.array_equal(
        y.to_numpy(dtype=int),
        csv_labels,
    ):
        raise RuntimeError(
            f"{display_name} true labels do not match the saved split"
        )

    if not np.array_equal(
        y.index.to_numpy(),
        predictions["sample_index"].to_numpy(),
    ):
        raise RuntimeError(
            f"{display_name} sample indexes do not match the saved split"
        )

    print(f"PASS: {display_name} labels and indexes match the data split")

    # Check 3: Saved model probabilities versus prediction CSV
    transformed_features = preprocessor.transform(x_raw)
    artifact_probabilities = model.predict_proba(
        transformed_features
    )[:, 1]

    maximum_difference = float(
        np.max(
            np.abs(
                artifact_probabilities - csv_probabilities
            )
        )
    )

    if not np.allclose(
        artifact_probabilities,
        csv_probabilities,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"{display_name} artifact probabilities do not match the CSV. "
            f"Maximum difference: {maximum_difference}"
        )

    print(
        f"PASS: {display_name} model probabilities match the CSV "
        f"(maximum difference={maximum_difference:.3e})"
    )

    # Check 4: Saved model labels versus exported labels
    artifact_labels = (
        artifact_probabilities >= 0.5
    ).astype(int)

    if not np.array_equal(
        artifact_labels,
        csv_predicted_labels,
    ):
        raise RuntimeError(
            f"{display_name} artifact labels do not match the CSV"
        )

    print(f"PASS: {display_name} predicted labels match the artifact")

    # Check 5: Metrics calculated directly from the artifact
    artifact_metrics = calculate_metrics(
        y,
        artifact_probabilities,
    )

    compare_metrics(
        metrics_payload[metrics_key],
        artifact_metrics,
        f"{display_name} artifact",
    )


def main() -> None:
    required_paths = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METRICS_PATH,
        VALIDATION_PREDICTIONS_PATH,
        TEST_PREDICTIONS_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    with METRICS_PATH.open(encoding="utf-8") as handle:
        metrics_payload = json.load(handle)

    model = joblib.load(MODEL_PATH)
    preprocessor = load_artifact(PREPROCESSOR_PATH)

    verify_split(
        display_name="validation",
        raw_split_name="val",
        prediction_path=VALIDATION_PREDICTIONS_PATH,
        metrics_key="validation",
        metrics_payload=metrics_payload,
        model=model,
        preprocessor=preprocessor,
    )

    verify_split(
        display_name="test",
        raw_split_name="test",
        prediction_path=TEST_PREDICTIONS_PATH,
        metrics_key="test_reference_only",
        metrics_payload=metrics_payload,
        model=model,
        preprocessor=preprocessor,
    )

    print("\nAll Neural Network artifact consistency checks passed.")


if __name__ == "__main__":
    main()