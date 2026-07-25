"""Staged XGBoost experiment for the corrected UNSW-NB15 splits.

Each tuning stage is a separate invocation. The test split is loaded only by
the final stage, after the configuration and validation outputs are locked.

Examples::

    python src/07_xgboost.py --stage baseline
    python src/07_xgboost.py --stage structure
    python src/07_xgboost.py --stage sampling
    python src/07_xgboost.py --stage regularization
    python src/07_xgboost.py --stage learning_rate
    python src/07_xgboost.py --stage round2
    python src/07_xgboost.py --stage final
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_FOLDER / "src"))

from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE  # noqa: E402
from preprocess import load_artifact  # noqa: E402


EXPERIMENT_DIR = ROOT_FOLDER / "experiments" / "xgboost"
TUNING_RESULTS_PATH = EXPERIMENT_DIR / "tuning_results.csv"
ROUND2_RESULTS_PATH = EXPERIMENT_DIR / "round2_joint_search.csv"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocess_tree.joblib"

EXPECTED_FEATURES = 39
EXPECTED_SHAPES = {
    "train": (79685, EXPECTED_FEATURES),
    "val": (19922, EXPECTED_FEATURES),
    "test": (82332, EXPECTED_FEATURES),
}

ROUND1_STAGES = ["baseline", "structure", "sampling", "regularization", "learning_rate"]

TUNING_COLUMNS = [
    "stage",
    "config_id",
    "params_json",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "gamma",
    "scale_pos_weight",
    "best_iteration",
    "best_val_aucpr",
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

ROUND2_GRID = {
    "max_depth": (8, 10),
    "min_child_weight": (1, 5),
    "subsample": (0.85, 1.0),
    "colsample_bytree": (0.6, 0.8),
    "reg_lambda": (1.0, 5.0),
    "reg_alpha": (0.0, 1.0),
    "learning_rate": (0.1,),
}


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
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    return {key: round(value, 6) for key, value in raw_metric_dict(y_true, probabilities).items()}


def make_model(
    params: dict[str, Any] | None = None,
    *,
    early_stopping: bool,
) -> XGBClassifier:
    """Create an XGBoost model with the plan's fixed settings."""
    fixed: dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "n_jobs": -1,
        "eval_metric": "aucpr",
        "scale_pos_weight": 1,
        "gamma": 0,
    }
    if early_stopping:
        fixed["n_estimators"] = 2000
        fixed["early_stopping_rounds"] = 50
    else:
        # The plan permits the library-default 100-tree baseline without early
        # stopping. This preserves a genuine default reference row.
        fixed["n_estimators"] = 100
    if params:
        fixed.update(params)
    return XGBClassifier(**fixed)


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_tuning_row(row: dict[str, Any]) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not TUNING_RESULTS_PATH.exists()
    with TUNING_RESULTS_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TUNING_COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in TUNING_COLUMNS})


def xgb_best_iteration_metrics(model: XGBClassifier) -> tuple[int, float]:
    evals = model.evals_result()
    values = evals["validation_0"]["aucpr"]
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = len(values) - 1
    return int(best_iteration), float(values[int(best_iteration)])


def fit_and_log(
    stage: str,
    config_id: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    early_stopping: bool = True,
) -> dict[str, Any]:
    model = make_model(params, early_stopping=early_stopping)
    start = time.perf_counter()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    fit_seconds = time.perf_counter() - start

    train_probability = model.predict_proba(x_train)[:, 1]
    val_probability = model.predict_proba(x_val)[:, 1]
    train_metrics = metric_dict(y_train, train_probability)
    val_metrics = metric_dict(y_val, val_probability)
    best_iteration, best_val_aucpr = xgb_best_iteration_metrics(model)

    model_params = model.get_params()
    logged_params = {
        "n_estimators": json_value(model_params["n_estimators"]),
        "learning_rate": json_value(model_params["learning_rate"]),
        "max_depth": json_value(model_params["max_depth"]),
        "min_child_weight": json_value(model_params["min_child_weight"]),
        "subsample": json_value(model_params["subsample"]),
        "colsample_bytree": json_value(model_params["colsample_bytree"]),
        "reg_lambda": json_value(model_params["reg_lambda"]),
        "reg_alpha": json_value(model_params["reg_alpha"]),
        "gamma": json_value(model_params["gamma"]),
        "scale_pos_weight": json_value(model_params["scale_pos_weight"]),
        "tree_method": model_params["tree_method"],
        "eval_metric": model_params["eval_metric"],
        "early_stopping_rounds": model_params.get("early_stopping_rounds"),
        "random_state": model_params["random_state"],
        "n_jobs": model_params["n_jobs"],
    }
    row: dict[str, Any] = {
        "stage": stage,
        "config_id": config_id,
        "params_json": json.dumps(logged_params, sort_keys=True),
        **{key: logged_params[key] for key in TUNING_COLUMNS if key in logged_params},
        "best_iteration": best_iteration,
        "best_val_aucpr": best_val_aucpr,
        "fit_seconds": fit_seconds,
    }
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"val_{key}": value for key, value in val_metrics.items()})
    append_tuning_row(row)

    print(
        f"{stage}: {config_id} | fit={fit_seconds:.2f}s | "
        f"best_iteration={best_iteration} | val PR-AUC={val_metrics['pr_auc']:.6f} | "
        f"val ROC-AUC={val_metrics['roc_auc']:.6f}"
    )
    return row


def read_tuning_results() -> pd.DataFrame:
    if not TUNING_RESULTS_PATH.exists():
        raise RuntimeError(f"Missing {TUNING_RESULTS_PATH}; run earlier stages first.")
    results = pd.read_csv(TUNING_RESULTS_PATH)
    if results.empty:
        raise RuntimeError("tuning_results.csv contains no fitted configurations")
    return results


def sort_key(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["val_pr_auc", "val_roc_auc"],
        ascending=[False, False],
        kind="mergesort",
    )


def row_params(row: pd.Series) -> dict[str, Any]:
    parsed = json.loads(str(row["params_json"]))
    for key in ("n_estimators", "max_depth", "min_child_weight"):
        if parsed.get(key) is not None:
            parsed[key] = int(parsed[key])
    for key in (
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "gamma",
        "scale_pos_weight",
    ):
        if parsed.get(key) is not None:
            parsed[key] = float(parsed[key])
    parsed.pop("n_estimators", None)
    parsed.pop("early_stopping_rounds", None)
    parsed.pop("tree_method", None)
    parsed.pop("eval_metric", None)
    parsed.pop("random_state", None)
    parsed.pop("n_jobs", None)
    parsed.pop("gamma", None)
    parsed.pop("scale_pos_weight", None)
    return parsed


def best_rows(stage: str, count: int) -> list[tuple[str, dict[str, Any]]]:
    results = read_tuning_results()
    stage_results = results[results["stage"] == stage]
    if stage_results.empty:
        raise RuntimeError(f"No rows for required stage {stage!r}")
    selected = sort_key(stage_results).head(count)
    return [(str(row["config_id"]), row_params(row)) for _, row in selected.iterrows()]


def run_baseline() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    row = fit_and_log(
        "baseline",
        "default",
        {
            "learning_rate": 0.3,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
        x_train,
        y_train,
        x_val,
        y_val,
        early_stopping=False,
    )
    with (ROOT_FOLDER / "docs" / "baseline.json").open() as handle:
        baseline_pr_auc = float(json.load(handle)["most_frequent"]["val"]["pr_auc"])
    if float(row["val_pr_auc"]) <= baseline_pr_auc:
        raise RuntimeError(
            f"Default XGBoost did not beat dummy PR-AUC: {row['val_pr_auc']} <= {baseline_pr_auc}"
        )
    print(
        f"Default XGBoost beats dummy baseline: {row['val_pr_auc']:.6f} > "
        f"{baseline_pr_auc:.6f} PR-AUC"
    )


def run_structure_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    for max_depth in (4, 6, 8, 10):
        for min_child_weight in (1, 5, 10):
            params = {
                "learning_rate": 0.1,
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
            }
            fit_and_log(
                "structure",
                f"structure_d{max_depth}_child{min_child_weight}",
                params,
                x_train,
                y_train,
                x_val,
                y_val,
            )


def run_sampling_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    structure_winners = best_rows("structure", 2)
    for rank, (structure_id, structure_params) in enumerate(structure_winners, start=1):
        for subsample in (0.7, 0.85, 1.0):
            for colsample_bytree in (0.6, 0.8, 1.0):
                params = {
                    **structure_params,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                }
                fit_and_log(
                    "sampling",
                    f"structurerank{rank}_{structure_id}_sub{subsample}_col{colsample_bytree}",
                    params,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                )


def run_regularization_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    sampling_winners = best_rows("sampling", 2)
    for rank, (sampling_id, sampling_params) in enumerate(sampling_winners, start=1):
        for reg_lambda in (1.0, 5.0, 10.0):
            for reg_alpha in (0.0, 1.0):
                params = {
                    **sampling_params,
                    "reg_lambda": reg_lambda,
                    "reg_alpha": reg_alpha,
                }
                fit_and_log(
                    "regularization",
                    f"samplingrank{rank}_{sampling_id}_lambda{reg_lambda}_alpha{reg_alpha}",
                    params,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                )


def print_round1_review() -> None:
    results = read_tuning_results()
    ranked = sort_key(results)
    print("\n=== XGBoost Round 1 review ===")
    print(f"Total Round 1 configurations: {len(results)}")
    print("Top 10 configurations across Round 1:")
    for rank, (_, row) in enumerate(ranked.head(10).iterrows(), start=1):
        print(
            f"  {rank}. {row['stage']} / {row['config_id']} | "
            f"PR-AUC={row['val_pr_auc']:.6f} | ROC-AUC={row['val_roc_auc']:.6f} | "
            f"best_iteration={int(row['best_iteration'])} | "
            f"train-val PR-AUC gap={row['train_pr_auc'] - row['val_pr_auc']:+.6f} | "
            f"params={row['params_json']}"
        )

    print("Competitive candidates by stage (within 0.001 PR-AUC of that stage winner):")
    for stage in ROUND1_STAGES:
        stage_results = results[results["stage"] == stage]
        winner = sort_key(stage_results).iloc[0]
        competitive = sort_key(
            stage_results[stage_results["val_pr_auc"] >= winner["val_pr_auc"] - 0.001]
        )
        print(
            f"  {stage}: winner PR-AUC={winner['val_pr_auc']:.6f}; "
            f"{len(competitive)} competitive candidates"
        )
        for _, row in competitive.iterrows():
            print(
                f"    {row['config_id']} | PR-AUC={row['val_pr_auc']:.6f} | "
                f"ROC-AUC={row['val_roc_auc']:.6f} | params={row['params_json']}"
            )

    print("Train-vs-validation PR-AUC gaps for the top configurations:")
    for _, row in ranked.head(10).iterrows():
        print(
            f"  {row['config_id']}: train={row['train_pr_auc']:.6f}, "
            f"val={row['val_pr_auc']:.6f}, gap={row['train_pr_auc'] - row['val_pr_auc']:+.6f}"
        )

    print("Learning-rate best_iteration patterns:")
    learning_results = sort_key(results[results["stage"] == "learning_rate"])
    for _, row in learning_results.iterrows():
        params = json.loads(row["params_json"])
        print(
            f"  {row['config_id']}: learning_rate={params['learning_rate']} | "
            f"best_iteration={int(row['best_iteration'])} | "
            f"val PR-AUC={row['val_pr_auc']:.6f}"
        )

    print(f"Logged fit-time total: {results['fit_seconds'].sum():.3f}s")
    print("STOP: Round 1 review complete. Round 2 must be designed from this evidence.")


def run_learning_rate_stage() -> None:
    x_train, y_train, x_val, y_val = load_train_val()
    regularization_winners = best_rows("regularization", 2)
    for rank, (regularization_id, regularization_params) in enumerate(
        regularization_winners, start=1
    ):
        for learning_rate in (0.03, 0.05, 0.1):
            params = {**regularization_params, "learning_rate": learning_rate}
            fit_and_log(
                "learning_rate",
                f"regrank{rank}_{regularization_id}_lr{learning_rate}",
                params,
                x_train,
                y_train,
                x_val,
                y_val,
            )
    print_round1_review()


def load_test_after_lock() -> tuple[pd.DataFrame, pd.Series]:
    """Reserved for the final stage; Round 1 never calls this function."""
    x_test_raw = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["label"].astype(int)
    preprocessor = load_artifact(PREPROCESSOR_PATH)
    x_test = preprocessor.transform(x_test_raw)
    if x_test.shape != EXPECTED_SHAPES["test"]:
        raise RuntimeError(f"Unexpected transformed test shape: {x_test.shape}")
    return x_test, y_test


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
    frame = pd.DataFrame(
        {
            "sample_index": sample_index.to_numpy(),
            "true_label": labels.to_numpy(dtype=int),
            "attack_probability": probabilities,
            "predicted_label": predicted,
        }
    )
    if not np.array_equal(
        frame["predicted_label"].to_numpy(),
        (frame["attack_probability"].to_numpy() >= 0.5).astype(int),
    ):
        raise RuntimeError(f"Threshold-label mismatch in {path}")
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fit_round2_candidate(
    config_id: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, Any]:
    """Fit one Round 2 candidate without touching the Round 1 CSV."""
    model = make_model(params, early_stopping=True)
    start = time.perf_counter()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    fit_seconds = time.perf_counter() - start

    train_probability = model.predict_proba(x_train)[:, 1]
    val_probability = model.predict_proba(x_val)[:, 1]
    if not (
        np.isfinite(train_probability).all()
        and np.isfinite(val_probability).all()
        and ((train_probability >= 0) & (train_probability <= 1)).all()
        and ((val_probability >= 0) & (val_probability <= 1)).all()
    ):
        raise RuntimeError(f"Invalid probability output for Round 2 config {config_id}")

    train_metrics = metric_dict(y_train, train_probability)
    val_metrics = metric_dict(y_val, val_probability)
    best_iteration, best_val_aucpr = xgb_best_iteration_metrics(model)
    model_params = model.get_params()
    logged_params = {
        "n_estimators": json_value(model_params["n_estimators"]),
        "learning_rate": json_value(model_params["learning_rate"]),
        "max_depth": json_value(model_params["max_depth"]),
        "min_child_weight": json_value(model_params["min_child_weight"]),
        "subsample": json_value(model_params["subsample"]),
        "colsample_bytree": json_value(model_params["colsample_bytree"]),
        "reg_lambda": json_value(model_params["reg_lambda"]),
        "reg_alpha": json_value(model_params["reg_alpha"]),
        "gamma": json_value(model_params["gamma"]),
        "scale_pos_weight": json_value(model_params["scale_pos_weight"]),
        "tree_method": model_params["tree_method"],
        "eval_metric": model_params["eval_metric"],
        "early_stopping_rounds": model_params.get("early_stopping_rounds"),
        "random_state": model_params["random_state"],
        "n_jobs": model_params["n_jobs"],
    }
    row: dict[str, Any] = {
        "stage": "round2",
        "config_id": config_id,
        "params_json": json.dumps(logged_params, sort_keys=True),
        **{key: logged_params[key] for key in TUNING_COLUMNS if key in logged_params},
        "best_iteration": best_iteration,
        "best_val_aucpr": best_val_aucpr,
        "fit_seconds": fit_seconds,
        "_raw_val_pr_auc": val_metrics["pr_auc"],
        "_raw_val_roc_auc": val_metrics["roc_auc"],
    }
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"val_{key}": value for key, value in val_metrics.items()})

    if not all(np.isfinite(float(row[column])) for column in ROUND2_COLUMNS if column in row and column not in {"rank", "stage", "config_id", "params_json"}):
        raise RuntimeError(f"Non-finite metric or timing value for Round 2 config {config_id}")

    print(
        f"round2: {config_id} | fit={fit_seconds:.2f}s | "
        f"best_iteration={best_iteration} | val PR-AUC={val_metrics['pr_auc']:.6f} | "
        f"val ROC-AUC={val_metrics['roc_auc']:.6f}"
    )
    return row


def run_round2() -> None:
    """Run the exact 64-cell joint grid using train/validation only."""
    if ROUND2_RESULTS_PATH.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing Round 2 results: {ROUND2_RESULTS_PATH}"
        )

    x_train, y_train, x_val, y_val = load_train_val()
    candidates = list(
        product(
            ROUND2_GRID["max_depth"],
            ROUND2_GRID["min_child_weight"],
            ROUND2_GRID["subsample"],
            ROUND2_GRID["colsample_bytree"],
            ROUND2_GRID["reg_lambda"],
            ROUND2_GRID["reg_alpha"],
        )
    )
    if len(candidates) != 64:
        raise RuntimeError(f"Round 2 grid unexpectedly contains {len(candidates)} candidates")

    total_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, reg_alpha in candidates:
        params = {
            "learning_rate": ROUND2_GRID["learning_rate"][0],
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
        }
        config_id = (
            f"round2_d{max_depth}_child{min_child_weight}_sub{subsample}"
            f"_col{colsample_bytree}_lambda{reg_lambda}_alpha{reg_alpha}"
        )
        rows.append(
            fit_round2_candidate(
                config_id, params, x_train, y_train, x_val, y_val
            )
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (-row["_raw_val_pr_auc"], -row["_raw_val_roc_auc"]),
    )
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    with ROUND2_RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND2_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for rank, row in enumerate(ranked_rows, start=1):
            output_row = {column: row.get(column, "") for column in ROUND2_COLUMNS}
            output_row["rank"] = rank
            writer.writerow(output_row)

    round1_best = sort_key(read_tuning_results()).iloc[0]
    best = ranked_rows[0]
    total_seconds = time.perf_counter() - total_start
    print(f"\nWrote {len(ranked_rows)} Round 2 rows to {ROUND2_RESULTS_PATH}")
    print(f"Total Round 2 runtime: {total_seconds:.3f}s")
    print("Top 10 Round 2 configurations:")
    for rank, row in enumerate(ranked_rows[:10], start=1):
        print(
            f"  {rank}. d={row['max_depth']} child={row['min_child_weight']} "
            f"sub={row['subsample']} col={row['colsample_bytree']} "
            f"lambda={row['reg_lambda']} alpha={row['reg_alpha']} "
            f"lr={row['learning_rate']} best_iteration={int(row['best_iteration'])} "
            f"val PR-AUC={row['val_pr_auc']:.6f} "
            f"val ROC-AUC={row['val_roc_auc']:.6f} val F1={row['val_f1']:.6f} "
            f"train-val gap={row['train_pr_auc'] - row['val_pr_auc']:+.6f}"
        )
    print(
        "Best Round 2: "
        f"{best['config_id']} | PR-AUC={best['val_pr_auc']:.6f} | "
        f"ROC-AUC={best['val_roc_auc']:.6f}"
    )
    print(
        "Round 1 comparison: "
        f"PR-AUC={round1_best['val_pr_auc']:.6f}; "
        f"Round 2 difference={best['val_pr_auc'] - round1_best['val_pr_auc']:+.6f}"
    )
    print("STOP: Round 2 review complete; no final selection, refit, or test evaluation performed.")


def run_final() -> None:
    """Fit the locked winner on train only, then evaluate the frozen test once."""
    locked_params: dict[str, Any] = {
        "n_estimators": 283,
        "learning_rate": 0.1,
        "max_depth": 10,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 0.6,
        "reg_lambda": 1.0,
        "reg_alpha": 1.0,
        "scale_pos_weight": 1,
        "gamma": 0,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if locked_params["random_state"] != 42:
        raise RuntimeError("Final XGBoost random_state must be 42")

    x_train, y_train, x_val, y_val = load_train_val()
    total_fit_start = time.perf_counter()
    model = make_model(
        {
            **locked_params,
        },
        early_stopping=False,
    )
    model_params = model.get_params()
    if model_params["n_estimators"] != 283:
        raise RuntimeError("Final XGBoost n_estimators is not locked to 283")
    if model_params.get("early_stopping_rounds") is not None:
        raise RuntimeError("Final XGBoost fit unexpectedly has early stopping enabled")

    print("Locked XGBoost configuration (before test access):")
    print(json.dumps(locked_params, indent=2, sort_keys=True))
    print("Final fit: training data only; no validation eval_set and no early stopping.")
    model.fit(x_train, y_train, verbose=False)
    fit_seconds = time.perf_counter() - total_fit_start

    val_probability = model.predict_proba(x_val)[:, 1]
    validation_metrics = raw_metric_dict(y_val, val_probability)
    write_predictions(
        EXPERIMENT_DIR / "validation_predictions.csv",
        y_val.index,
        y_val,
        val_probability,
    )

    # These artifacts and the validation outputs are written before the frozen
    # test split is loaded.
    model_path = ARTIFACTS_DIR / "xgboost.joblib"
    native_model_path = EXPERIMENT_DIR / "xgboost_model.json"
    joblib.dump(model, model_path)
    model.save_model(native_model_path)

    metadata = {
        "selected_hyperparameters": locked_params,
        "n_estimators": 283,
        "source_best_iteration": 282,
        "random_state": RANDOM_STATE,
        "preprocessing_artifact": "artifacts/preprocess_tree.joblib",
        "preprocessing_mode": "transform_only",
        "feature_count": EXPECTED_FEATURES,
        "train_rows": EXPECTED_SHAPES["train"][0],
        "validation_rows": EXPECTED_SHAPES["val"][0],
        "test_rows": EXPECTED_SHAPES["test"][0],
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "final_training_runtime_seconds": fit_seconds,
        "selection_metric": "validation PR-AUC (sklearn average_precision_score)",
        "tie_breaker": "validation ROC-AUC",
        "round1_fits": 49,
        "round2_fits": 64,
        "total_tuning_fits": 113,
        "selected_from": "Round 2 joint search",
        "test_evaluation": "reference only after configuration lock and artifact save",
    }
    write_json(EXPERIMENT_DIR / "config.json", metadata)

    print("Final model and validation outputs saved; loading frozen test now.")
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
        "notes": "Test metrics are reference outputs only; no test result was used for model selection.",
    }
    write_json(EXPERIMENT_DIR / "metrics.json", metrics_payload)

    readme = f"""# XGBoost final experiment

The final XGBoost model uses the Round 2 winner, selected by validation PR-AUC
with validation ROC-AUC as the tie-breaker. It was refit on the training split
only with `n_estimators=283` (the selected `best_iteration=282` plus one),
without validation `eval_set` or early stopping.

Validation PR-AUC: `{validation_metrics['pr_auc']:.6f}`
Validation ROC-AUC: `{validation_metrics['roc_auc']:.6f}`
Test metrics are stored under `test_reference_only` in `metrics.json` and were
computed only after the configuration, validation outputs, and model artifacts
were saved.

Tuning history is preserved in `tuning_results.csv` (49 Round 1 fits) and
`round2_joint_search.csv` (64 Round 2 fits).
"""
    (EXPERIMENT_DIR / "README.md").write_text(readme)

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
            "structure",
            "sampling",
            "regularization",
            "learning_rate",
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
    elif args.stage == "structure":
        run_structure_stage()
    elif args.stage == "sampling":
        run_sampling_stage()
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
