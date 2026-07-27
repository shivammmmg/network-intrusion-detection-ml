"""Staged Neural Network experiment for the corrected UNSW-NB15 splits.

Each tuning stage is a separate invocation. The test split is loaded only by
the final stage, after the configuration and validation outputs are locked.

Examples:

    python src/08_neural_network.py --stage baseline
    python src/08_neural_network.py --stage architecture
    python src/08_neural_network.py --stage regularization
    python src/08_neural_network.py --stage learning_rate
    python src/08_neural_network.py --stage round2
    python src/08_neural_network.py --stage final
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier


ROOT_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_FOLDER / "src"))

from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE  # noqa: E402
from preprocess import load_artifact  # noqa: E402

EXPERIMENT_DIR = ROOT_FOLDER / "experiments" / "neural_network"
TUNING_RESULTS_PATH = EXPERIMENT_DIR / "tuning_results.csv"
ROUND2_RESULTS_PATH = EXPERIMENT_DIR / "round2_joint_search.csv"
VALIDATION_PREDICTIONS_PATH = EXPERIMENT_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = EXPERIMENT_DIR / "test_predictions.csv"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
README_PATH = EXPERIMENT_DIR / "README.md"
MODEL_PATH = ARTIFACTS_DIR / "neural_network.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocess_linear.joblib"

EXPECTED_FEATURES = 66
EXPECTED_SHAPES = {
    "train": (79685, EXPECTED_FEATURES),
    "val": (19922, EXPECTED_FEATURES),
    "test": (82332, EXPECTED_FEATURES),
}

MAX_ITER = 100
N_ITER_NO_CHANGE = 8
VALIDATION_FRACTION = 0.10

ROUND1_STAGES = ["baseline", "architecture", "regularization", "learning_rate"]

TUNING_COLUMNS = [
    "stage",
    "config_id",
    "params_json",
    "hidden_layer_sizes",
    "activation",
    "solver",
    "alpha",
    "learning_rate_init",
    "batch_size",
    "max_iter",
    "early_stopping",
    "validation_fraction",
    "n_iter_no_change",
    "n_iter",
    "converged",
    "stopped_early",
    "loss",
    "best_internal_validation_score",
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

ROUND2_COLUMNS = ["rank", *TUNING_COLUMNS]

#Joint search
ROUND2_GRID = {
    "hidden_layer_sizes": ((64,), (64, 32), (128, 64)),
    "alpha": (0.0001, 0.001),
    "learning_rate_init": (0.0005, 0.001),
    "batch_size": (128, 256),
}

BASE_PARAMETERS: dict[str, Any] = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 256,
    "learning_rate_init": 0.001,
}

def load_split(split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Function loads the raw features and labels for one fixed split"""
    x_path = PROCESSED_DIR / f"x_{split_name}.parquet"
    if not x_path.exists():
        x_path = PROCESSED_DIR / f"X_{split_name}.parquet"
    y_path = PROCESSED_DIR / f"y_{split_name}.parquet"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing processed feature file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing processed label file: {y_path}")
    x = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["label"].astype(int)
    if len(x) != len(y):
        raise RuntimeError(
            f"Feature and label row counts do not match for {split_name}: "
            f"{len(x)} != {len(y)}"
        )
    return x, y

def load_train_val() -> tuple[Any, pd.Series, Any, pd.Series]:
    """Function loads and transforms training and validation splits"""

    x_train_raw, y_train = load_split("train")
    x_val_raw, y_val = load_split("val")

    preprocessor = load_artifact(PREPROCESSOR_PATH)
    x_train = preprocessor.transform(x_train_raw)
    x_val = preprocessor.transform(x_val_raw)

    if x_train.shape != EXPECTED_SHAPES["train"]:
        raise RuntimeError(f"Unexpected transformed train shape: {x_train.shape}")
    if x_val.shape != EXPECTED_SHAPES["val"]:
        raise RuntimeError(f"Unexpected transformed validation shape: {x_val.shape}")
    if len(y_train) != x_train.shape[0] or len(y_val) != x_val.shape[0]:
        raise RuntimeError("Feature and label row counts do not match")

    print(f"Transformed train shape: {x_train.shape}")
    print(f"Transformed validation shape: {x_val.shape}")

    return x_train, y_train, x_val, y_val

def load_test_after_lock() -> tuple[Any, pd.Series]:
    """Function loads the frozen test split only after model selection is locked"""
    x_test_raw, y_test = load_split("test")
    preprocessor = load_artifact(PREPROCESSOR_PATH)
    x_test = preprocessor.transform(x_test_raw)
    if x_test.shape != EXPECTED_SHAPES["test"]:
        raise RuntimeError(f"Unexpected transformed test shape: {x_test.shape}")
    if len(y_test) != x_test.shape[0]:
        raise RuntimeError("Test feature and label row counts do not match")
    return x_test, y_test

def raw_metric_dict(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Function calculates binary classification metrics using a threshold of 0.5"""

    predicted = (probabilities >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }

def metric_dict(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Function returns metrics rounded to six decimal places for exported files"""

    return {
        name: round(value, 6)
        for name, value in raw_metric_dict(y_true, probabilities).items()
    }

def json_safe(value: Any) -> Any:
    """Function converts NumPy and tuple values into JSON compatible values"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value

def normalized_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Function adds settings shared by all tuning candidates"""

    return {
        **BASE_PARAMETERS,
        **parameters,
        "early_stopping": True,
        "validation_fraction": VALIDATION_FRACTION,
        "n_iter_no_change": N_ITER_NO_CHANGE,
        "max_iter": MAX_ITER,
        "random_state": RANDOM_STATE,
    }

def make_tuning_model(parameters: dict[str, Any]) -> MLPClassifier:
    """Function creates MLP with early stopping"""
    return MLPClassifier(**normalized_parameters(parameters))

def make_final_model(parameters: dict[str, Any], selected_epochs: int) -> MLPClassifier:
    """Function refits the MLP model using the trained dataset"""

    final_parameters = {
        **BASE_PARAMETERS,
        **parameters,
        "early_stopping": False,
        "max_iter": max(1, int(selected_epochs)),
        "random_state": RANDOM_STATE,
    }
    return MLPClassifier(**final_parameters)

def remove_existing_stage(stage: str) -> None:
    """Function replaces a run of a round 1 stage"""
    if not TUNING_RESULTS_PATH.exists():
        return
    existing = pd.read_csv(TUNING_RESULTS_PATH)
    if "stage" not in existing.columns:
        raise RuntimeError(f"Malformed tuning file: {TUNING_RESULTS_PATH}")
    remaining = existing[existing["stage"] != stage]
    if remaining.empty:
        TUNING_RESULTS_PATH.unlink()
    else:
        remaining.to_csv(TUNING_RESULTS_PATH, index=False)

def append_tuning_row(result_row: dict[str, Any]) -> None:
    """Function appends round 1 result"""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not TUNING_RESULTS_PATH.exists()
    with TUNING_RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TUNING_COLUMNS,
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {column: result_row.get(column, "") for column in TUNING_COLUMNS}
        )

def fit_candidate(
    stage: str,
    config_id: str,
    parameters: dict[str, Any],
    x_train,
    y_train: pd.Series,
    x_val,
    y_val: pd.Series,
    *,
    append: bool,
) -> dict[str, Any]:
    """Function fits one candidate and returns standardized result"""

    model = make_tuning_model(parameters)
    actual_parameters = normalized_parameters(parameters)

    print(f"{stage}: {config_id} | fitting Neural Network...", flush=True)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    converged = not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    train_probability = model.predict_proba(x_train)[:, 1]
    val_probability = model.predict_proba(x_val)[:, 1]
    train_metrics = metric_dict(y_train, train_probability)
    val_metrics = metric_dict(y_val, val_probability)

    n_iter = int(model.n_iter_)
    stopped_early = n_iter < int(actual_parameters["max_iter"])
    best_internal_score = getattr(model, "best_validation_score_", None)

    result_row: dict[str, Any] = {
        "stage": stage,
        "config_id": config_id,
        "params_json": json.dumps(json_safe(actual_parameters), sort_keys=True),
        "hidden_layer_sizes": json.dumps(
            json_safe(actual_parameters["hidden_layer_sizes"])
        ),
        "activation": actual_parameters["activation"],
        "solver": actual_parameters["solver"],
        "alpha": actual_parameters["alpha"],
        "batch_size": actual_parameters["batch_size"],
        "learning_rate_init": actual_parameters["learning_rate_init"],
        "early_stopping": actual_parameters["early_stopping"],
        "validation_fraction": actual_parameters["validation_fraction"],
        "n_iter_no_change": actual_parameters["n_iter_no_change"],
        "max_iter": actual_parameters["max_iter"],
        "n_iter": n_iter,
        "converged": converged,
        "stopped_early": stopped_early,
        "loss": round(float(model.loss_), 6),
        "best_internal_validation_score": (
            "" if best_internal_score is None else round(float(best_internal_score), 6)
        ),
        "fit_seconds": round(fit_seconds, 6),
    }
    result_row.update(
        {f"train_{name}": value for name, value in train_metrics.items()}
    )
    result_row.update({f"val_{name}": value for name, value in val_metrics.items()})

    if append:
        append_tuning_row(result_row)

    print(
        f"{stage}: {config_id} | fit={fit_seconds:.2f}s | "
        f"epochs={n_iter} | val PR-AUC={val_metrics['pr_auc']:.6f} | "
        f"val ROC-AUC={val_metrics['roc_auc']:.6f}"
    )

    if not converged:
        print(
            f"WARNING: {config_id} reached max_iter={actual_parameters['max_iter']} "
            "before convergence; the result was logged with converged=False."
        )

    return result_row


def read_tuning_results() -> pd.DataFrame:
    """Function reads all round 1 stages"""

    if not TUNING_RESULTS_PATH.exists():
        raise RuntimeError(
            f"Missing {TUNING_RESULTS_PATH}; run the earlier tuning stages first."
        )

    results = pd.read_csv(TUNING_RESULTS_PATH)
    if results.empty:
        raise RuntimeError("tuning_results.csv contains no fitted configurations")
    return results


def sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    """ Function ranks by PR-AUC and ROC-AUC"""
    return frame.sort_values(
        ["val_pr_auc", "val_roc_auc"],
        ascending=[False, False],
        kind="mergesort",
    )


def parameters_from_row(result_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Function parses and returns MLP hyperparameters from result row"""
    parsed = json.loads(str(result_row["params_json"]))
    hidden_sizes = parsed.get("hidden_layer_sizes", [64, 32])
    parsed["hidden_layer_sizes"] = tuple(int(size) for size in hidden_sizes)
    keep = {
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "batch_size",
        "learning_rate_init",
    }
    return {name: parsed[name] for name in keep}


def best_rows(stage: str, count: int) -> list[tuple[str, dict[str, Any]]]:
    """Function returns best saved candidate for one stage"""
    results = read_tuning_results()
    stage_results = results[results["stage"] == stage]
    if stage_results.empty:
        raise RuntimeError(
            f"No results exist for stage {stage!r}; run --stage {stage} first."
        )

    selected = sort_results(stage_results).head(count)
    return [
        (str(result_row["config_id"]), parameters_from_row(result_row))
        for _, result_row in selected.iterrows()
    ]

def load_dummy_baseline_pr_auc() -> float:
    """Read most frequent validation PR-AUC"""
    baseline_path = ROOT_FOLDER / "docs" / "baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Missing dummy-baseline file: {baseline_path}. "
            "Run the baseline preparation script first."
        )

    with baseline_path.open(encoding="utf-8") as handle:
        baseline = json.load(handle)

    try:
        return float(baseline["most_frequent"]["val"]["pr_auc"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Unexpected baseline.json structure: {baseline_path}") from error


def run_baseline() -> None:
    """Function fits default MLP and compares it with dummy baseline"""
    remove_existing_stage("baseline")
    x_train, y_train, x_val, y_val = load_train_val()
    result_row = fit_candidate(
        "baseline",
        "default",
        {},
        x_train,
        y_train,
        x_val,
        y_val,
        append=True,
    )
    dummy_pr_auc = load_dummy_baseline_pr_auc()
    model_pr_auc = float(result_row["val_pr_auc"])
    if model_pr_auc <= dummy_pr_auc:
        raise RuntimeError("Default Neural Network did not beat the dummy PR-AUC baseline")
    print(
        "Default Neural Network beats the dummy baseline: "
        f"{model_pr_auc:.6f} > {dummy_pr_auc:.6f} PR-AUC"
    )

def run_architecture_stage() -> None:
    """Function compares hidden architectures"""
    remove_existing_stage("architecture")
    x_train, y_train, x_val, y_val = load_train_val()
    architectures = (
        (32,),
        (64,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    )
    for hidden_sizes in architectures:
        name = "x".join(str(size) for size in hidden_sizes)
        fit_candidate(
            "architecture",
            f"hidden_{name}",
            {"hidden_layer_sizes": hidden_sizes},
            x_train,
            y_train,
            x_val,
            y_val,
            append=True,
        )
    print("STOP: Architecture stage complete. Review tuning_results.csv.")


def run_regularization_stage() -> None:
    """Function tunes L2 regularization for the two top performing MLP architectures"""
    remove_existing_stage("regularization")
    x_train, y_train, x_val, y_val = load_train_val()
    architecture_winners = best_rows("architecture", 2)

    for rank, (architecture_id, architecture_params) in enumerate(
        architecture_winners,
        start=1,
    ):
        for alpha in (0.00001, 0.0001, 0.001, 0.01):
            parameters = {**architecture_params, "alpha": alpha}
            fit_candidate(
                "regularization",
                f"archrank{rank}_{architecture_id}_alpha{alpha:g}",
                parameters,
                x_train,
                y_train,
                x_val,
                y_val,
                append=True,
            )
    print("STOP: Regularization stage complete. Review tuning_results.csv.")

def run_learning_rate_stage() -> None:
    """Function tunes learning rate and batch size"""
    remove_existing_stage("learning_rate")
    x_train, y_train, x_val, y_val = load_train_val()
    regularization_winners = best_rows("regularization", 2)
    for rank, (regularization_id, regularization_params) in enumerate(
        regularization_winners,
        start=1,
    ):
        for learning_rate_init, batch_size in product(
            (0.0003, 0.001, 0.003),
            (128, 256),
        ):
            parameters = {
                **regularization_params,
                "learning_rate_init": learning_rate_init,
                "batch_size": batch_size,
            }
            fit_candidate(
                "learning_rate",
                (
                    f"regrank{rank}_{regularization_id}_"
                    f"lr{learning_rate_init:g}_batch{batch_size}"
                ),
                parameters,
                x_train,
                y_train,
                x_val,
                y_val,
                append=True,
            )
    print_round1_review()


def print_round1_review() -> None:
    """Function prints best round 1 configurations"""
    results = read_tuning_results()
    missing = [
        stage for stage in ROUND1_STAGES if results[results["stage"] == stage].empty
    ]
    if missing:
        raise RuntimeError(f"Round 1 is incomplete; missing stages: {missing}")
    ranked = sort_results(results)
    print("\nTop 10 Round 1 configurations:")
    for rank, (_, result_row) in enumerate(ranked.head(10).iterrows(), start=1):
        print(
            f"  {rank}. {result_row['config_id']} | "
            f"PR-AUC={result_row['val_pr_auc']:.6f} | "
            f"ROC-AUC={result_row['val_roc_auc']:.6f} | "
            f"F1={result_row['val_f1']:.6f} | "
            f"epochs={int(result_row['n_iter'])} | "
            f"converged={result_row['converged']}"
        )
    print("Competitive candidates by stage (within 0.001 PR-AUC of winner):")
    for stage in ROUND1_STAGES:
        stage_results = results[results["stage"] == stage]
        stage_ranked = sort_results(stage_results)
        winner_pr_auc = float(stage_ranked.iloc[0]["val_pr_auc"])
        competitive = sort_results(
            stage_results[stage_results["val_pr_auc"] >= winner_pr_auc - 0.001]
        )
        print(
            f"  {stage}: winner PR-AUC={winner_pr_auc:.6f}; "
            f"{len(competitive)} competitive candidates"
        )
    print(f"Logged fit-time total: {results['fit_seconds'].sum():.3f}s")
    print("STOP: Round 1 review complete. Run --stage round2 next.")

def run_round2() -> None:
    """Function evaluates and ranks round 2 hyperparameter configurations"""
    results = read_tuning_results()
    missing = [
        stage for stage in ROUND1_STAGES if results[results["stage"] == stage].empty
    ]
    if missing:
        raise RuntimeError(f"Round 1 is incomplete; missing stages: {missing}")

    x_train, y_train, x_val, y_val = load_train_val()
    total_start = time.perf_counter()
    result_rows: list[dict[str, Any]] = []
    combinations = list(
        product(
            ROUND2_GRID["hidden_layer_sizes"],
            ROUND2_GRID["alpha"],
            ROUND2_GRID["learning_rate_init"],
            ROUND2_GRID["batch_size"],
        )
    )
    print(f"Round 2 joint search contains {len(combinations)} configurations.")

    for number, (hidden_sizes, alpha, learning_rate, batch_size) in enumerate(
        combinations,
        start=1,
    ):
        parameters = {
            "hidden_layer_sizes": hidden_sizes,
            "alpha": alpha,
            "learning_rate_init": learning_rate,
            "batch_size": batch_size,
        }
        name = "x".join(str(size) for size in hidden_sizes)
        config_id = (
            f"round2_{number:02d}_hidden{name}_alpha{alpha:g}_"
            f"lr{learning_rate:g}_batch{batch_size}"
        )
        result_rows.append(
            fit_candidate(
                "round2",
                config_id,
                parameters,
                x_train,
                y_train,
                x_val,
                y_val,
                append=False,
            )
        )
    ranked_rows = sorted(
        result_rows,
        key=lambda item: (float(item["val_pr_auc"]), float(item["val_roc_auc"])),
        reverse=True,
    )
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    with ROUND2_RESULTS_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ROUND2_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for rank, result_row in enumerate(ranked_rows, start=1):
            output_row: dict[str, Any] = {
                column: result_row.get(column, "")
                for column in ROUND2_COLUMNS
            }
            output_row["rank"] = rank
            writer.writerow(output_row)
    total_seconds = time.perf_counter() - total_start
    print(f"\nWrote {len(ranked_rows)} Round 2 rows to {ROUND2_RESULTS_PATH}")
    print(f"Total Round 2 runtime: {total_seconds:.3f}s")
    print("Top 10 Round 2 configurations:")
    for rank, result_row in enumerate(ranked_rows[:10], start=1):
        print(
            f"  {rank}. {result_row['config_id']} | "
            f"PR-AUC={result_row['val_pr_auc']:.6f} | "
            f"ROC-AUC={result_row['val_roc_auc']:.6f} | "
            f"F1={result_row['val_f1']:.6f} | "
            f"epochs={result_row['n_iter']} | "
            f"converged={result_row['converged']}"
        )
    best = ranked_rows[0]
    print(
        "STOP: Round 2 complete. The current winner is "
        f"{best['config_id']} with validation PR-AUC="
        f"{best['val_pr_auc']:.6f}. Review the CSV before --stage final."
    )

def read_round2_winner() -> pd.Series:
    """Function returns highest ranked round 2 config"""

    if not ROUND2_RESULTS_PATH.exists():
        raise RuntimeError(
            f"Missing {ROUND2_RESULTS_PATH}; run --stage round2 first."
        )

    results = pd.read_csv(ROUND2_RESULTS_PATH)
    if results.empty:
        raise RuntimeError("round2_joint_search.csv is empty")

    if "rank" in results.columns:
        ranked = results.sort_values("rank", kind="mergesort")
    else:
        ranked = sort_results(results)

    return ranked.iloc[0]


def create_prediction_frame(labels: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Function creates a prediction dataframe"""
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities):
        raise RuntimeError("Prediction and label row counts do not match")
    return pd.DataFrame(
        {
            "sample_index": labels.index.to_numpy(),
            "true_label": labels.to_numpy(dtype=int),
            "attack_probability": probabilities,
            "predicted_label": (probabilities >= 0.5).astype(int),
        }
    )

def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")

def run_final() -> None:
    """Function finalizes and evaluates the selected round 2 model"""
    winner = read_round2_winner()
    selected_parameters = parameters_from_row(winner)
    selected_epochs = int(winner["n_iter"])
    selected_config_id = str(winner["config_id"])
    print(f"Locked Round 2 winner: {selected_config_id}")
    print(
        "Selected parameters:",
        json.dumps(json_safe(selected_parameters), sort_keys=True),
    )
    print(f"Selected training epochs: {selected_epochs}")
    x_train, y_train, x_val, y_val = load_train_val()
    model = make_final_model(selected_parameters, selected_epochs)
    print("final: locked_winner | fitting Neural Network on full training split...", flush=True)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    final_converged = not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )

    validation_probability = model.predict_proba(x_val)[:, 1]
    validation_metrics = metric_dict(y_val, validation_probability)
    validation_predictions = create_prediction_frame(y_val, validation_probability)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Lock and save all selection-dependent outputs before opening the test set.
    joblib.dump(model, MODEL_PATH, compress=3)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    try:
        preprocessor = load_artifact(PREPROCESSOR_PATH)
        feature_names = [
            str(name) for name in preprocessor.get_feature_names_out()
        ]
    except (AttributeError, TypeError):
        feature_names = []

    configuration = {
        "model": "MLPClassifier",
        "selected_config_id": selected_config_id,
        "selected_hyperparameters": selected_parameters,
        "selected_tuning_epochs": selected_epochs,
        "final_refit": {
            "early_stopping": False,
            "max_iter": selected_epochs,
            "n_iter_completed": int(model.n_iter_),
            "converged": final_converged,
            "fit_seconds": round(fit_seconds, 6),
        },
        "selection": {
            "primary_metric": "validation PR-AUC",
            "tie_breaker": "validation ROC-AUC",
            "source": "experiments/neural_network/round2_joint_search.csv",
        },
        "preprocessing": {
            "artifact": "artifacts/preprocess_linear.joblib",
            "number_of_features": len(feature_names),
        },
        "feature_names": feature_names,
        "random_state": RANDOM_STATE,
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json(CONFIG_PATH, configuration)

    preliminary_metrics = {
        "validation": validation_metrics,
        "selected_config_id": selected_config_id,
        "selected_hyperparameters": selected_parameters,
        "selection_metric": (
            "validation average_precision_score (PR-AUC), then validation ROC-AUC"
        ),
        "test_set_used": False,
    }
    write_json(METRICS_PATH, preliminary_metrics)
    x_test, y_test = load_test_after_lock()
    test_probability = model.predict_proba(x_test)[:, 1]
    test_metrics = metric_dict(y_test, test_probability)
    test_predictions = create_prediction_frame(y_test, test_probability)
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    final_metrics = {
        **preliminary_metrics,
        "test_reference_only": test_metrics,
        "test_set_used": True,
        "test_usage_note": (
            "The test split was opened only after model selection, the final "
            "model artifact, validation predictions, and locked configuration "
            "had been saved. Test metrics were not used for model selection."
        ),
    }
    write_json(METRICS_PATH, final_metrics)

    readme = f"""# Neural Network final experiment

The final Neural Network model uses the Round 2 winner, selected by validation
PR-AUC with validation ROC-AUC as the tie-breaker. The locked configuration is
`{json.dumps(json_safe(selected_parameters), sort_keys=True)}`.

During tuning, `MLPClassifier` used early stopping with an internal holdout from
the training split. The winning candidate stopped after `{selected_epochs}`
epochs. The final model was refit on the complete training split for
`{selected_epochs}` epochs with early stopping disabled.

Validation PR-AUC: `{validation_metrics['pr_auc']:.6f}`
Validation ROC-AUC: `{validation_metrics['roc_auc']:.6f}`

Preprocessing uses the fitted `artifacts/preprocess_linear.joblib` artifact with
transform-only application. Round 1 history is stored in `tuning_results.csv`,
and the joint search is stored in `round2_joint_search.csv`. Validation and test
prediction CSVs support downstream cross-model comparison.

Test metrics are stored under `test_reference_only` in `metrics.json`. They were
computed only after model selection and were not used to choose the model.

Final train-only refit runtime: `{fit_seconds:.3f}` seconds.
"""
    README_PATH.write_text(readme, encoding="utf-8")

    print("Final validation metrics:", json.dumps(validation_metrics, sort_keys=True))
    print("Final test reference metrics:", json.dumps(test_metrics, sort_keys=True))
    print(f"Final train-only refit runtime: {fit_seconds:.3f}s")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Round 1 results: {TUNING_RESULTS_PATH}")
    print(f"Round 2 results: {ROUND2_RESULTS_PATH}")
    print(f"Validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    print(f"Test predictions: {TEST_PREDICTIONS_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Configuration: {CONFIG_PATH}")
    print(f"README: {README_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "baseline",
            "architecture",
            "regularization",
            "learning_rate",
            "round2",
            "final",
        ),
        help="Run exactly one staged Neural Network experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage == "baseline":
        run_baseline()
    elif args.stage == "architecture":
        run_architecture_stage()
    elif args.stage == "regularization":
        run_regularization_stage()
    elif args.stage == "learning_rate":
        run_learning_rate_stage()
    elif args.stage == "round2":
        run_round2()
    elif args.stage == "final":
        run_final()


if __name__ == "__main__":
    main()