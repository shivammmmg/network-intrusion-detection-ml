"""Staged Random Forest experiment for the fixed UNSW-NB15 splits.

Each tuning stage is intentionally a separate command so that no single
invocation has to run the whole search.  The test split is only opened by the
``final`` stage, after the best validation configuration has been selected.

Examples::

    python src/06_random_forest.py --stage baseline
    python src/06_random_forest.py --stage depth
    python src/06_random_forest.py --stage features
    python src/06_random_forest.py --stage estimators
    python src/06_random_forest.py --stage final
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
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

from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE  # noqa: E402
from preprocess import load_artifact  # noqa: E402


EXPERIMENT_DIR = ROOT_FOLDER / "experiments" / "random_forest"
TUNING_RESULTS_PATH = EXPERIMENT_DIR / "tuning_results.csv"
ROUND2_RESULTS_PATH = EXPERIMENT_DIR / "round2_joint_search.csv"
MODEL_PATH = ARTIFACTS_DIR / "random_forest.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocess_tree.joblib"

EXPECTED_FEATURES = 39
EXPECTED_SHAPES = {
    "train": (79685, EXPECTED_FEATURES),
    "val": (19922, EXPECTED_FEATURES),
    "test": (82332, EXPECTED_FEATURES),
}

TUNING_COLUMNS = [
    "stage",
    "config_id",
    "params_json",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "fit_seconds",
    "train_pr_auc",
    "train_roc_auc",
    "train_accuracy",
    "train_precision",
    "train_recall",
    "train_f1",
    "val_pr_auc",
    "val_roc_auc",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
]

ROUND2_COLUMNS = [
    "rank",
    "config_id",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "fit_seconds",
    "val_pr_auc",
    "val_roc_auc",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
]


def load_train_val() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and transform only the train and validation splits."""
    x_train_raw = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["label"].astype(int)
    x_val_raw = pd.read_parquet(PROCESSED_DIR / "X_val.parquet")
    y_val = pd.read_parquet(PROCESSED_DIR / "y_val.parquet")["label"].astype(int)

    preprocessor = load_artifact(PREPROCESSOR_PATH)
    x_train = preprocessor.transform(x_train_raw)
    x_val = preprocessor.transform(x_val_raw)

    if x_train.shape != EXPECTED_SHAPES["train"]:
        raise RuntimeError(f"Unexpected transformed train shape: {x_train.shape}")
    if x_val.shape != EXPECTED_SHAPES["val"]:
        raise RuntimeError(f"Unexpected transformed validation shape: {x_val.shape}")
    if len(y_train) != x_train.shape[0] or len(y_val) != x_val.shape[0]:
        raise RuntimeError("Feature and label row counts do not match")
    for split_name, labels in (("train", y_train), ("val", y_val)):
        attack_rate = float(labels.mean())
        if not 0.47 <= attack_rate <= 0.50:
            raise RuntimeError(f"Unexpected {split_name} attack rate: {attack_rate:.4f}")

    print(f"Transformed train shape: {x_train.shape}")
    print(f"Transformed validation shape: {x_val.shape}")
    return x_train, y_train, x_val, y_val


def raw_metric_dict(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Calculate metrics without rounding for validation-based ranking."""
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def metric_dict(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate metrics at stable six-decimal export precision."""
    return {
        key: round(value, 6)
        for key, value in raw_metric_dict(y_true, probabilities).items()
    }


def make_model(params: dict[str, Any] | None = None) -> RandomForestClassifier:
    """Create the requested model, preserving sklearn defaults for baseline."""
    if params is None:
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)


def json_value(value: Any) -> Any:
    """Convert numpy scalar values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_tuning_row(row: dict[str, Any]) -> None:
    """Append one fitted configuration to the shared tuning log."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not TUNING_RESULTS_PATH.exists()
    with TUNING_RESULTS_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TUNING_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in TUNING_COLUMNS})


def fit_and_log(
    stage: str,
    config_id: str,
    params: dict[str, Any] | None,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, Any]:
    """Fit one candidate, score both train and validation, and append its row."""
    model = make_model(params)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    train_probability = model.predict_proba(x_train)[:, 1]
    val_probability = model.predict_proba(x_val)[:, 1]
    train_metrics = metric_dict(y_train, train_probability)
    val_metrics = metric_dict(y_val, val_probability)

    relevant = {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "class_weight",
    }
    resolved_params = {
        key: json_value(value)
        for key, value in model.get_params().items()
        if key in relevant
    }
    row: dict[str, Any] = {
        "stage": stage,
        "config_id": config_id,
        "params_json": json.dumps(resolved_params, sort_keys=True),
        "n_estimators": resolved_params["n_estimators"],
        "max_depth": resolved_params["max_depth"],
        "min_samples_leaf": resolved_params["min_samples_leaf"],
        "max_features": resolved_params["max_features"],
        "class_weight": resolved_params["class_weight"],
        "fit_seconds": fit_seconds,
    }
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"val_{key}": value for key, value in val_metrics.items()})
    append_tuning_row(row)

    print(
        f"{stage}: {config_id} | fit={fit_seconds:.2f}s | "
        f"val PR-AUC={val_metrics['pr_auc']:.6f} | "
        f"val ROC-AUC={val_metrics['roc_auc']:.6f}"
    )
    return row


def read_tuning_results() -> pd.DataFrame:
    if not TUNING_RESULTS_PATH.exists():
        raise RuntimeError(
            f"Missing {TUNING_RESULTS_PATH}; run the earlier tuning stages first."
        )
    results = pd.read_csv(TUNING_RESULTS_PATH)
    if results.empty:
        raise RuntimeError("tuning_results.csv contains no fitted configurations")
    return results


def sort_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by validation PR-AUC, then validation ROC-AUC, descending."""
    return frame.sort_values(
        ["val_pr_auc", "val_roc_auc"], ascending=[False, False], kind="mergesort"
    )


def row_params(row: pd.Series) -> dict[str, Any]:
    parsed = json.loads(str(row["params_json"]))
    if parsed.get("max_depth") is not None:
        parsed["max_depth"] = int(parsed["max_depth"])
    parsed["n_estimators"] = int(parsed["n_estimators"])
    parsed["min_samples_leaf"] = int(parsed["min_samples_leaf"])
    if isinstance(parsed.get("max_features"), str):
        if parsed["max_features"] not in {"sqrt", "log2"}:
            parsed["max_features"] = float(parsed["max_features"])
    return parsed


def best_rows(stage: str, count: int) -> list[tuple[str, dict[str, Any]]]:
    results = read_tuning_results()
    stage_results = results[results["stage"] == stage]
    if stage_results.empty:
        raise RuntimeError(f"No rows for required stage {stage!r}")
    selected = sort_key(stage_results).head(count)
    return [
        (str(row["config_id"]), row_params(row))
        for _, row in selected.iterrows()
    ]


def run_baseline() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    row = fit_and_log(
        "baseline",
        "default",
        None,
        x_train,
        y_train,
        x_val,
        y_val,
    )

    baseline_path = ROOT_FOLDER / "docs" / "baseline.json"
    with baseline_path.open() as handle:
        baseline = json.load(handle)["most_frequent"]["val"]
    if float(row["val_pr_auc"]) <= float(baseline["pr_auc"]):
        raise RuntimeError("Default Random Forest did not beat the dummy PR-AUC baseline")
    print(
        "Default Random Forest beats the dummy baseline: "
        f"{row['val_pr_auc']:.6f} > {baseline['pr_auc']:.6f} PR-AUC"
    )


def run_depth_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    for max_depth in (None, 15, 25, 35):
        for min_samples_leaf in (1, 2, 5):
            params = {
                "n_estimators": 300,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": "sqrt",
                "class_weight": None,
            }
            depth_name = "none" if max_depth is None else str(max_depth)
            fit_and_log(
                "depth",
                f"depth_d{depth_name}_leaf{min_samples_leaf}",
                params,
                x_train,
                y_train,
                x_val,
                y_val,
            )


def run_features_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    depth_winners = best_rows("depth", 2)
    for rank, (depth_id, depth_params) in enumerate(depth_winners, start=1):
        for max_features in ("sqrt", 0.3, 0.5):
            for class_weight in (None, "balanced"):
                params = {
                    **depth_params,
                    "max_features": max_features,
                    "class_weight": class_weight,
                }
                feature_name = str(max_features).replace(".", "p")
                weight_name = "none" if class_weight is None else class_weight
                fit_and_log(
                    "features",
                    f"depthrank{rank}_{depth_id}_features{feature_name}_weight{weight_name}",
                    params,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                )


def run_estimators_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    feature_winners = best_rows("features", 2)
    for rank, (feature_id, feature_params) in enumerate(feature_winners, start=1):
        for n_estimators in (200, 400, 800):
            params = {**feature_params, "n_estimators": n_estimators}
            fit_and_log(
                "estimators",
                f"featurerank{rank}_{feature_id}_n{n_estimators}",
                params,
                x_train,
                y_train,
                x_val,
                y_val,
            )


def run_round2_joint_search() -> None:
    """Run the untouched-test, 48-cell Round 2 Cartesian grid."""
    if ROUND2_RESULTS_PATH.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing Round 2 output: {ROUND2_RESULTS_PATH}"
        )

    total_start = time.perf_counter()
    x_train, y_train, x_val, y_val = load_train_val()
    candidates = list(
        product(
            (400, 800, 1000),
            (25, 35),
            (1, 2),
            (0.3, 0.5),
            (None, "balanced"),
        )
    )
    if len(candidates) != 48:
        raise RuntimeError(f"Round 2 grid unexpectedly contains {len(candidates)} cells")

    ROUND2_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate_number, (
        n_estimators,
        max_depth,
        min_samples_leaf,
        max_features,
        class_weight,
    ) in enumerate(candidates, start=1):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
        }
        model = make_model(params)
        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        val_probability = model.predict_proba(x_val)[:, 1]
        raw_metrics = raw_metric_dict(y_val, val_probability)
        exported_metrics = {key: round(value, 6) for key, value in raw_metrics.items()}
        row: dict[str, Any] = {
            "rank": "",
            "config_id": f"round2_{candidate_number:02d}",
            **params,
            "class_weight": "None" if class_weight is None else class_weight,
            "fit_seconds": fit_seconds,
            "val_pr_auc": exported_metrics["pr_auc"],
            "val_roc_auc": exported_metrics["roc_auc"],
            "val_accuracy": exported_metrics["accuracy"],
            "val_precision": exported_metrics["precision"],
            "val_recall": exported_metrics["recall"],
            "val_f1": exported_metrics["f1"],
            "_raw_pr_auc": raw_metrics["pr_auc"],
            "_raw_roc_auc": raw_metrics["roc_auc"],
        }
        rows.append(row)
        print(
            f"Round 2 {candidate_number:02d}/48 | "
            f"n={n_estimators}, depth={max_depth}, leaf={min_samples_leaf}, "
            f"features={max_features}, weight={class_weight} | "
            f"fit={fit_seconds:.2f}s | val PR-AUC={raw_metrics['pr_auc']:.9f} | "
            f"val ROC-AUC={raw_metrics['roc_auc']:.9f}"
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (row["_raw_pr_auc"], row["_raw_roc_auc"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    with ROUND2_RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND2_COLUMNS)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({column: row[column] for column in ROUND2_COLUMNS})

    round1_best = sort_key(read_tuning_results()).iloc[0]
    round1_pr_auc = float(round1_best["val_pr_auc"])
    best = ranked_rows[0]
    pr_delta = float(best["_raw_pr_auc"]) - round1_pr_auc
    beats_round1 = float(best["_raw_pr_auc"]) > round1_pr_auc
    total_runtime = time.perf_counter() - total_start

    print(f"Round 2 complete: {len(ranked_rows)} configurations")
    print(f"Round 2 total runtime: {total_runtime:.3f}s")
    print(
        "Best Round 2 configuration: "
        f"n_estimators={best['n_estimators']}, max_depth={best['max_depth']}, "
        f"min_samples_leaf={best['min_samples_leaf']}, max_features={best['max_features']}, "
        f"class_weight={best['class_weight']}"
    )
    print(
        f"Best Round 2 validation PR-AUC={best['_raw_pr_auc']:.9f}; "
        f"Round 1 PR-AUC={round1_pr_auc:.6f}; "
        f"delta={pr_delta:+.9f}; beats_round1={beats_round1}"
    )
    print("Top 5 Round 2 configurations:")
    for row in ranked_rows[:5]:
        print(
            f"  rank={row['rank']} {row['config_id']} | "
            f"PR-AUC={row['_raw_pr_auc']:.9f} | ROC-AUC={row['_raw_roc_auc']:.9f}"
        )
    print(f"Round 2 results: {ROUND2_RESULTS_PATH}")


def prediction_frame(
    index: pd.Index,
    labels: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": index.to_numpy(),
            "true_label": labels.to_numpy(dtype=int),
            "attack_probability": probabilities.astype(float),
            "predicted_label": (probabilities >= 0.5).astype(int),
        }
    )


def load_test_after_lock() -> tuple[pd.DataFrame, pd.Series]:
    """Load the frozen test split; called only by the final locked stage."""
    x_test_raw = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["label"].astype(int)
    preprocessor = load_artifact(PREPROCESSOR_PATH)
    x_test = preprocessor.transform(x_test_raw)
    if x_test.shape != EXPECTED_SHAPES["test"]:
        raise RuntimeError(f"Unexpected transformed test shape: {x_test.shape}")
    if len(y_test) != x_test.shape[0]:
        raise RuntimeError("Test feature and label row counts do not match")
    print(f"Transformed test shape: {x_test.shape}")
    return x_test, y_test


def run_final() -> None:
    # The best row is selected from validation results before any test file is
    # opened. This is the explicit configuration-lock boundary.
    results = read_tuning_results()
    best = sort_key(results).iloc[0]
    params = row_params(best)
    print(f"LOCKED configuration: {json.dumps(params, sort_keys=True)}")

    x_train, y_train, x_val, y_val = load_train_val()
    model = make_model(params)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_runtime = time.perf_counter() - start

    val_probability = model.predict_proba(x_val)[:, 1]
    val_metrics = metric_dict(y_val, val_probability)
    validation_predictions = prediction_frame(
        y_val.index, y_val, val_probability
    )
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    validation_predictions.to_csv(
        EXPERIMENT_DIR / "validation_predictions.csv", index=False
    )
    joblib.dump(model, MODEL_PATH, compress=3)

    # Test is intentionally loaded only after the validation-selected config
    # was locked, the final train-only fit completed, and the model was saved.
    x_test, y_test = load_test_after_lock()
    test_probability = model.predict_proba(x_test)[:, 1]
    test_metrics = metric_dict(y_test, test_probability)
    test_predictions = prediction_frame(y_test.index, y_test, test_probability)
    test_predictions.to_csv(EXPERIMENT_DIR / "test_predictions.csv", index=False)

    preprocessor = load_artifact(PREPROCESSOR_PATH)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    if len(feature_names) != EXPECTED_FEATURES:
        raise RuntimeError(f"Unexpected feature-name count: {len(feature_names)}")

    config = {
        "model": "RandomForestClassifier",
        "selected_hyperparameters": params,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "preprocessing": {
            "artifact": "artifacts/preprocess_tree.joblib",
            "ttl_included": False,
            "n_features_out": EXPECTED_FEATURES,
        },
        "feature_names": feature_names,
        "training_runtime_seconds": float(training_runtime),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (EXPERIMENT_DIR / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    metrics = {
        "validation": val_metrics,
        "test_reference_only": test_metrics,
        "selected_hyperparameters": params,
        "random_state": RANDOM_STATE,
        "selection_metric": "validation average_precision_score (PR-AUC), then validation ROC-AUC",
    }
    with (EXPERIMENT_DIR / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    with (EXPERIMENT_DIR / "README.md").open("w") as handle:
        handle.write(
            "# Random Forest experiment\n\n"
            f"Selected on the fixed validation split by PR-AUC, then ROC-AUC. "
            f"The locked model uses `{json.dumps(params, sort_keys=True)}` with `random_state=42`.\n\n"
            "Preprocessing uses the fitted `artifacts/preprocess_tree.joblib` artifact "
            "(TTL-excluded, 39 features) with transform-only application. The test "
            "metrics in `metrics.json` are sanity/reference values; downstream "
            "cross-model comparison uses the prediction CSVs.\n\n"
            f"Final train-only refit runtime: `{training_runtime:.3f}` seconds.\n"
        )

    print(f"Final training runtime: {training_runtime:.3f}s")
    print(f"Validation metrics: {json.dumps(val_metrics, sort_keys=True)}")
    print(f"Test reference metrics: {json.dumps(test_metrics, sort_keys=True)}")


def run_final_round2() -> None:
    """Finalize the locked Round 2 winner, then evaluate the frozen test once."""
    if not ROUND2_RESULTS_PATH.exists():
        raise RuntimeError(f"Missing completed Round 2 results: {ROUND2_RESULTS_PATH}")

    # Confirm the requested configuration is the recorded Round 2 winner before
    # locking it. This reads only the validation search log, never test data.
    round2_results = pd.read_csv(ROUND2_RESULTS_PATH, keep_default_na=False)
    winner = round2_results[round2_results["rank"] == 1]
    if len(winner) != 1:
        raise RuntimeError("Round 2 results do not contain exactly one rank-1 winner")
    winner_row = winner.iloc[0]
    params = {
        "n_estimators": int(winner_row["n_estimators"]),
        "max_depth": int(winner_row["max_depth"]),
        "min_samples_leaf": int(winner_row["min_samples_leaf"]),
        "max_features": float(winner_row["max_features"]),
        "class_weight": None if winner_row["class_weight"] == "None" else winner_row["class_weight"],
    }
    print(f"LOCKED Round 2 configuration: {json.dumps(params, sort_keys=True)}")

    x_train, y_train, x_val, y_val = load_train_val()
    model = make_model(params)
    actual_model_params = model.get_params()
    for key, expected_value in {
        **params,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }.items():
        if actual_model_params[key] != expected_value:
            raise RuntimeError(
                f"Final model parameter mismatch for {key}: "
                f"{actual_model_params[key]!r} != {expected_value!r}"
            )

    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_runtime = time.perf_counter() - start

    # Validation outputs and model are written before the frozen test is opened.
    val_probability = model.predict_proba(x_val)[:, 1]
    val_metrics = metric_dict(y_val, val_probability)
    validation_predictions = prediction_frame(y_val.index, y_val, val_probability)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    validation_predictions.to_csv(
        EXPERIMENT_DIR / "validation_predictions.csv", index=False
    )
    joblib.dump(model, MODEL_PATH, compress=3)

    # Configuration is locked and validation outputs exist; only now touch test.
    x_test, y_test = load_test_after_lock()
    test_probability = model.predict_proba(x_test)[:, 1]
    test_metrics = metric_dict(y_test, test_probability)
    test_predictions = prediction_frame(y_test.index, y_test, test_probability)
    test_predictions.to_csv(EXPERIMENT_DIR / "test_predictions.csv", index=False)

    preprocessor = load_artifact(PREPROCESSOR_PATH)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    if len(feature_names) != EXPECTED_FEATURES:
        raise RuntimeError(f"Unexpected feature-name count: {len(feature_names)}")

    config = {
        "model": "RandomForestClassifier",
        "selected_hyperparameters": params,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "selection": {
            "round": "Round 2 joint tuning",
            "source": "experiments/random_forest/round2_joint_search.csv",
            "selected_rank": 1,
            "primary_metric": "validation PR-AUC",
            "tie_breaker": "validation ROC-AUC",
        },
        "preprocessing": {
            "artifact": "artifacts/preprocess_tree.joblib",
            "ttl_included": False,
            "n_features_out": EXPECTED_FEATURES,
        },
        "feature_names": feature_names,
        "training_runtime_seconds": float(training_runtime),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (EXPERIMENT_DIR / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    metrics = {
        "validation": val_metrics,
        "test_reference_only": test_metrics,
        "selected_hyperparameters": params,
        "random_state": RANDOM_STATE,
        "selection_metric": "validation average_precision_score (PR-AUC), then validation ROC-AUC",
        "test_metrics_note": "Reference output only; standardized cross-model comparison uses the prediction CSVs downstream.",
    }
    with (EXPERIMENT_DIR / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    with (EXPERIMENT_DIR / "README.md").open("w") as handle:
        handle.write(
            "# Random Forest experiment\n\n"
            "Final model selected from the Round 2 joint search by validation PR-AUC, "
            "with validation ROC-AUC as the tie-breaker. The locked configuration is "
            f"`{json.dumps(params, sort_keys=True)}` with `random_state=42`.\n\n"
            "Preprocessing uses the fitted `artifacts/preprocess_tree.joblib` artifact "
            "(TTL-excluded, 39 features) with transform-only application. Round 1 "
            "history remains in `tuning_results.csv`; Round 2 history remains in "
            "`round2_joint_search.csv`. Test metrics in `metrics.json` are sanity/reference "
            "values; downstream cross-model comparison uses the prediction CSVs.\n\n"
            f"Final train-only refit runtime: `{training_runtime:.3f}` seconds.\n"
        )

    print(f"Final Round 2 training runtime: {training_runtime:.3f}s")
    print(f"Final validation metrics: {json.dumps(val_metrics, sort_keys=True)}")
    print(f"Final test reference metrics: {json.dumps(test_metrics, sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "baseline",
            "depth",
            "features",
            "estimators",
            "final",
            "round2",
            "final_round2",
        ),
        help="Run exactly one staged experiment invocation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "baseline":
        run_baseline()
    elif args.stage == "depth":
        run_depth_stage()
    elif args.stage == "features":
        run_features_stage()
    elif args.stage == "estimators":
        run_estimators_stage()
    elif args.stage == "final":
        run_final()
    elif args.stage == "round2":
        run_round2_joint_search()
    elif args.stage == "final_round2":
        run_final_round2()


if __name__ == "__main__":
    main()
