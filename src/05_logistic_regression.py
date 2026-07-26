"""Logistic Regression hyperparameter-tuning experiment for the fixed UNSW-NB15 splits.

The model is trained using the fixed training split and hyperparameters are selected
using the validation split according to Validation PR-AUC and ROC-AUC.

The test split is only opened by the ''final'' stage after the best validation configuration has been selected
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
from sklearn.linear_model import LogisticRegression
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


EXPERIMENT_DIR = ROOT_FOLDER / "experiments" / "logistic_regression"
TUNING_RESULTS_PATH = EXPERIMENT_DIR / "tuning_results.csv"
ROUND2_RESULTS_PATH = EXPERIMENT_DIR / "round2_joint_search.csv"
MODEL_PATH = ARTIFACTS_DIR / "logistic_regression.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocess_linear.joblib"

EXPECTED_FEATURES = 66
EXPECTED_SHAPES = {
    "train": (79685, EXPECTED_FEATURES),
    "val": (19922, EXPECTED_FEATURES),
    "test": (82332, EXPECTED_FEATURES),
}

TUNING_COLUMNS = [
    "stage",
    "config_id",
    "params_json",
    "C",
    "penalty",
    "solver",
    "class_weight",
    "max_iter",
    "converged",
    "n_iter",
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

# Joint search used after earlier stages have been reviewed
ROUND2_GRID = {
    "C": (0.01, 0.1, 1.0, 10.0, 100.0),
    "class_weight": (None, "balanced"),
    "solver_penalty": (
        ("lbfgs", "l2"),
        ("liblinear", "l1"),
        ("liblinear", "l2"),
    ),
}


def load_split(split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Function loads the raw feature and label files for one split"""

    x_path = PROCESSED_DIR / f"x_{split_name}.parquet"
    if not x_path.exists():
        x_path = PROCESSED_DIR / f"X_{split_name}.parquet"

    x = pd.read_parquet(x_path)
    y = pd.read_parquet(PROCESSED_DIR / f"y_{split_name}.parquet")["label"].astype(int)

    if len(x) != len(y):
        raise RuntimeError(f"Feature and label row counts do not match for {split_name}")

    return x, y


def load_train_val() -> tuple[Any, pd.Series, Any, pd.Series]:
    """Function loads and transforms the training and validation splits"""

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
    """Function loads the frozen test split (only) in the final stage"""

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
    """Function calculates classification metrics using a threshold of 0.5"""

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
    """Function returns metrics for CSV output (rounded to 6 decimal places)"""

    return {
        name: round(value, 6)
        for name, value in raw_metric_dict(y_true, probabilities).items()
    }


def make_model(parameters: dict[str, Any]) -> LogisticRegression:
    """Function creates a Logistic Regression model (reproducibility settings fixed)"""
    return LogisticRegression(
        **parameters,
        max_iter=5000,
        tol=0.0001,
        random_state=RANDOM_STATE,
    )


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_tuning_row(row: dict[str, Any]) -> None:
    """Function appends one Round 1 result (Doesn't overwrite earlier stages)"""

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not TUNING_RESULTS_PATH.exists()

    with TUNING_RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TUNING_COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in TUNING_COLUMNS})


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
    """Function fits one candidate, returns a standardized result row and suppresses depreciation noise"""

    model = make_model(parameters)
    converged = True

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message=".*'penalty' was deprecated.*",
            category=FutureWarning,
        )
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        converged = False

    train_probability = model.predict_proba(x_train)[:, 1]
    val_probability = model.predict_proba(x_val)[:, 1]
    train_metrics = metric_dict(y_train, train_probability)
    val_metrics = metric_dict(y_val, val_probability)

    logged_params = {
        "C": json_value(parameters["C"]),
        "penalty": parameters["penalty"],
        "solver": parameters["solver"],
        "class_weight": parameters["class_weight"],
        "max_iter": 10000,
        "random_state": RANDOM_STATE,
    }

    row: dict[str, Any] = {
        "stage": stage,
        "config_id": config_id,
        "params_json": json.dumps(logged_params, sort_keys=True),
        "C": logged_params["C"],
        "penalty": logged_params["penalty"],
        "solver": logged_params["solver"],
        "class_weight": (
            "None" if logged_params["class_weight"] is None else logged_params["class_weight"]
        ),
        "max_iter": 10000,
        "converged": converged,
        "n_iter": int(np.max(model.n_iter_)),
        "fit_seconds": fit_seconds,
    }
    row.update({f"train_{name}": value for name, value in train_metrics.items()})
    row.update({f"val_{name}": value for name, value in val_metrics.items()})

    if append:
        append_tuning_row(row)

    print(
        f"{stage}: {config_id} | fit={fit_seconds:.2f}s | "
        f"val PR-AUC={val_metrics['pr_auc']:.6f} | "
        f"val ROC-AUC={val_metrics['roc_auc']:.6f}"
    )
    if not converged:
        print(
            f"WARNING: {config_id} reached max_iter={model.max_iter} before convergence\n"
            "the result was logged with converged=False"
        )

    return row


def read_tuning_results() -> pd.DataFrame:
    if not TUNING_RESULTS_PATH.exists():
        raise RuntimeError(f"Missing {TUNING_RESULTS_PATH}, run the earlier stages first")
    results = pd.read_csv(TUNING_RESULTS_PATH)
    if results.empty:
        raise RuntimeError("tuning_results.csv contains no fitted configurations")
    return results


def sort_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Function ranks by validation PR-AUC and then validation ROC-AUC"""

    return frame.sort_values(
        ["val_pr_auc", "val_roc_auc"],
        ascending=[False, False],
        kind="mergesort",
    )


def row_params(row: pd.Series) -> dict[str, Any]:
    parsed = json.loads(str(row["params_json"]))
    parsed["C"] = float(parsed["C"])
    parsed.pop("max_iter", None)
    parsed.pop("random_state", None)
    return parsed


def best_rows(stage: str, count: int) -> list[tuple[str, dict[str, Any]]]:
    results = read_tuning_results()
    stage_results = results[results["stage"] == stage]
    if stage_results.empty:
        raise RuntimeError(f"No rows found for required stage {stage!r}")
    selected = sort_key(stage_results).head(count)
    return [(str(row["config_id"]), row_params(row)) for _, row in selected.iterrows()]


def dummy_baseline_pr_auc() -> float:
    baseline_path = ROOT_FOLDER / "docs" / "baseline.json"
    with baseline_path.open(encoding="utf-8") as handle:
        return float(json.load(handle)["most_frequent"]["val"]["pr_auc"])


def run_baseline() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    row = fit_candidate(
        "baseline",
        "default",
        {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": None,
        },
        x_train,
        y_train,
        x_val,
        y_val,
        append=True,
    )

    baseline_pr_auc = dummy_baseline_pr_auc()
    if float(row["val_pr_auc"]) <= baseline_pr_auc:
        raise RuntimeError(
            f"Default Logistic Regression did not beat dummy PR-AUC: "
            f"{row['val_pr_auc']} <= {baseline_pr_auc}"
        )

    print(
        f"Default Logistic Regression beats the dummy baseline: "
        f"{row['val_pr_auc']:.6f} > {baseline_pr_auc:.6f} PR-AUC"
    )


def run_solver_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    candidates = (
        ("lbfgs_l2", {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": None}),
        ("liblinear_l1", {"C": 1.0, "penalty": "l1", "solver": "liblinear", "class_weight": None}),
        ("liblinear_l2", {"C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": None}),
    )
    for config_id, params in candidates:
        fit_candidate("solver", config_id, params, x_train, y_train, x_val, y_val, append=True)
    print("STOP: Solver stage complete. Review tuning_results.csv before continuing.")


def run_regularization_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    solver_winners = best_rows("solver", 2)

    for rank, (solver_id, solver_params) in enumerate(solver_winners, start=1):
        for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
            params = {**solver_params, "C": c_value, "class_weight": None}
            fit_candidate(
                "regularization",
                f"solverrank{rank}_{solver_id}_C{c_value}",
                params,
                x_train,
                y_train,
                x_val,
                y_val,
                append=True,
            )
    print("STOP: Regularization stage complete. Review tuning_results.csv before continuing.")


def run_class_weight_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    regularization_winners = best_rows("regularization", 2)

    for rank, (regularization_id, regularization_params) in enumerate(
        regularization_winners, start=1
    ):
        for class_weight in (None, "balanced"):
            params = {**regularization_params, "class_weight": class_weight}
            label = "none" if class_weight is None else "balanced"
            fit_candidate(
                "class_weight",
                f"regrank{rank}_{regularization_id}_weight{label}",
                params,
                x_train,
                y_train,
                x_val,
                y_val,
                append=True,
            )
    print("STOP: Class-weight stage complete. Review tuning_results.csv before continuing.")


def run_round2() -> None:
    """Function runs 30 cell joint grid using train or validation"""

    if ROUND2_RESULTS_PATH.exists():
        raise RuntimeError(f"Refusing to overwrite existing Round 2 results: {ROUND2_RESULTS_PATH}")

    x_train, y_train, x_val, y_val = load_train_val()
    candidates = list(
        product(
            ROUND2_GRID["C"],
            ROUND2_GRID["class_weight"],
            ROUND2_GRID["solver_penalty"],
        )
    )
    if len(candidates) != 30:
        raise RuntimeError(f"Round 2 grid unexpectedly contains {len(candidates)} candidates")

    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for c_value, class_weight, (solver, penalty) in candidates:
        weight_label = "none" if class_weight is None else "balanced"
        config_id = f"round2_{solver}_{penalty}_C{c_value}_weight{weight_label}"
        rows.append(
            fit_candidate(
                "round2",
                config_id,
                {
                    "C": c_value,
                    "penalty": penalty,
                    "solver": solver,
                    "class_weight": class_weight,
                },
                x_train,
                y_train,
                x_val,
                y_val,
                append=False,
            )
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (-float(row["val_pr_auc"]), -float(row["val_roc_auc"])),
    )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    with ROUND2_RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND2_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for rank, row in enumerate(ranked_rows, start=1):
            output_row = {column: row.get(column, "") for column in ROUND2_COLUMNS}
            output_row["rank"] = str(rank)
            writer.writerow(output_row)

    total_seconds = time.perf_counter() - total_start
    best = ranked_rows[0]
    print(f"\nWrote {len(ranked_rows)} Round 2 rows to {ROUND2_RESULTS_PATH}")
    print(f"Total Round 2 runtime: {total_seconds:.3f}s")
    print("Top 10 Round 2 configurations:")
    for rank, row in enumerate(ranked_rows[:10], start=1):
        print(
            f"  {rank}. {row['config_id']} | "
            f"PR-AUC={row['val_pr_auc']:.6f} | "
            f"ROC-AUC={row['val_roc_auc']:.6f} | "
            f"F1={row['val_f1']:.6f} | "
            f"converged={row['converged']}"
        )
    print(
        f"Best Round 2: {best['config_id']} | "
        f"PR-AUC={best['val_pr_auc']:.6f} | "
        f"ROC-AUC={best['val_roc_auc']:.6f}"
    )
    print("STOP: Round 2 review complete. No test evaluation was performed.")


def write_predictions(
    path: Path,
    sample_index: pd.Index,
    labels: pd.Series,
    probabilities: np.ndarray,
) -> None:
    predicted = (probabilities >= 0.5).astype(int)
    if not (
        np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise RuntimeError(f"Invalid probability output for {path}")

    pd.DataFrame(
        {
            "sample_index": sample_index.to_numpy(),
            "true_label": labels.to_numpy(dtype=int),
            "attack_probability": probabilities,
            "predicted_label": predicted,
        }
    ).to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_final() -> None:
    """Function fits the locked Round 2 winner on train only and evaluate the test once"""

    if not ROUND2_RESULTS_PATH.exists():
        raise RuntimeError(f"Missing completed Round 2 results: {ROUND2_RESULTS_PATH}")

    round2_results = pd.read_csv(ROUND2_RESULTS_PATH, keep_default_na=False)
    winner = round2_results[round2_results["rank"] == 1]
    if len(winner) != 1:
        raise RuntimeError("Round 2 results do not contain exactly one rank-1 winner")

    winner_row = winner.iloc[0]
    locked_params = row_params(winner_row)

    print("Locked Logistic Regression configuration (before test access):")
    print(json.dumps(locked_params, indent=2, sort_keys=True))

    x_train, y_train, x_val, y_val = load_train_val()
    model = make_model(locked_params)

    print("Final fit: training data only. Validation is used only for locked-model reporting.")
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*'penalty' was deprecated.*",
            category=FutureWarning,
        )
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    val_probability = model.predict_proba(x_val)[:, 1]
    validation_metrics = raw_metric_dict(y_val, val_probability)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    write_predictions(
        EXPERIMENT_DIR / "validation_predictions.csv",
        y_val.index,
        y_val,
        val_probability,
    )
    joblib.dump(model, MODEL_PATH, compress=3)

    preprocessor = load_artifact(PREPROCESSOR_PATH)
    try:
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    except AttributeError:
        feature_names = []

    configuration = {
        "model": "LogisticRegression",
        "selected_hyperparameters": {
            **locked_params,
            "max_iter": 10000,
            "random_state": RANDOM_STATE,
        },
        "selection": {
            "round": "Round 2 joint search",
            "source": "experiments/logistic_regression/round2_joint_search.csv",
            "selected_rank": 1,
            "primary_metric": "validation PR-AUC",
            "tie_breaker": "validation ROC-AUC",
        },
        "preprocessing": {
            "artifact": "artifacts/preprocess_linear.joblib",
            "mode": "transform_only",
            "n_features_out": len(feature_names),
        },
        "feature_names": feature_names,
        "final_training_runtime_seconds": fit_seconds,
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json(EXPERIMENT_DIR / "config.json", configuration)

    print("Final model and validation outputs saved. Loading frozen test now.")
    x_test, y_test = load_test_after_lock()
    test_probability = model.predict_proba(x_test)[:, 1]
    test_metrics = raw_metric_dict(y_test, test_probability)

    write_predictions(
        EXPERIMENT_DIR / "test_predictions.csv",
        y_test.index,
        y_test,
        test_probability,
    )

    metrics_payload = {
        "validation": validation_metrics,
        "test_reference_only": test_metrics,
        "notes": (
            "Test metrics are reference outputs only. no test result was used for model selection."
        ),
    }
    write_json(EXPERIMENT_DIR / "metrics.json", metrics_payload)

    with (EXPERIMENT_DIR / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Logistic Regression experiment\n\n"
            "Final model selected from the Round 2 joint search by validation PR-AUC, "
            "with validation ROC-AUC as the tie-breaker. The locked configuration is "
            f"`{json.dumps(configuration['selected_hyperparameters'], sort_keys=True)}`.\n\n"
            "Preprocessing uses the fitted `artifacts/preprocess_linear.joblib` artifact "
            "with transform-only application. Round 1 history remains in "
            "`tuning_results.csv`. Round 2 history remains in "
            "`round2_joint_search.csv`. Test metrics in `metrics.json` are "
            "sanity/reference values. downstream cross-model comparison uses the "
            "prediction CSVs.\n\n"
            f"Final train-only refit runtime: `{fit_seconds:.3f}` seconds.\n"
        )

    print("Final validation metrics:", json.dumps(validation_metrics, sort_keys=True))
    print("Final test reference metrics:", json.dumps(test_metrics, sort_keys=True))
    print(f"Final train-only refit runtime: {fit_seconds:.3f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "baseline",
            "solver",
            "regularization",
            "class_weight",
            "round2",
            "final",
        ),
        help="Run exactly one staged experiment invocation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "baseline":
        run_baseline()
    elif args.stage == "solver":
        run_solver_stage()
    elif args.stage == "regularization":
        run_regularization_stage()
    elif args.stage == "class_weight":
        run_class_weight_stage()
    elif args.stage == "round2":
        run_round2()
    elif args.stage == "final":
        run_final()


if __name__ == "__main__":
    main()