"""Stage 3 TTL ablation using the finalized XGBoost configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ARTIFACTS_DIR, RANDOM_STATE
from diagnostics_lib import (
    load_model,
    load_split,
    load_thresholds,
    transform,
    verify_outputs,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parent.parent
TTL_DIR = ROOT / "experiments" / "diagnostics" / "ttl_ablation"
FIGURES_DIR = TTL_DIR / "figures"
TTL_COLUMNS = {"sttl", "dttl", "ct_state_ttl"}
VERIFY_TOLERANCE = 1e-6


def pr_auc(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Use positive-class probabilities for every permutation score."""
    return float(average_precision_score(y, estimator.predict_proba(X)[:, 1]))


def _importance(model: Any, X: pd.DataFrame, y: pd.Series, arm: str) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring=pr_auc,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "arm": arm,
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(["importance_mean", "feature"], ascending=[False, True], ignore_index=True)
    table.insert(3, "rank", np.arange(1, len(table) + 1))
    return table


def _metrics(arm: str, y: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    return {
        "arm": arm,
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "f1_locked_threshold": float(f1_score(y, predicted)),
        "precision_locked_threshold": float(precision_score(y, predicted, zero_division=0)),
        "recall_locked_threshold": float(recall_score(y, predicted, zero_division=0)),
        "accuracy_locked_threshold": float(accuracy_score(y, predicted)),
        "locked_threshold": threshold,
        "threshold_note": (
            "Validation-selected no-TTL XGBoost threshold, applied unchanged to both arms "
            "for like-for-like secondary operating-point metrics."
        ),
    }


def _rank_shift(no_ttl: pd.DataFrame, with_ttl: pd.DataFrame) -> pd.DataFrame:
    left = no_ttl[["feature", "rank", "importance_mean"]].rename(
        columns={"rank": "no_ttl_rank", "importance_mean": "no_ttl_importance_mean"}
    )
    right = with_ttl[["feature", "rank", "importance_mean"]].rename(
        columns={"rank": "with_ttl_rank", "importance_mean": "with_ttl_importance_mean"}
    )
    return left.merge(right, on="feature", how="outer").sort_values(
        ["with_ttl_rank", "no_ttl_rank", "feature"], na_position="last", ignore_index=True
    )


def _row_fingerprints(X: pd.DataFrame, y: pd.Series) -> set[int]:
    table = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return set(pd.util.hash_pandas_object(table, index=False).astype("uint64").tolist())


def _leakage_assertions(
    X_train_ttl: pd.DataFrame,
    X_val_ttl: pd.DataFrame,
    X_test_ttl: pd.DataFrame,
    X_train_no_ttl: pd.DataFrame,
    X_test_no_ttl: pd.DataFrame,
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> dict[str, Any]:
    fingerprints = {name: _row_fingerprints(*split) for name, split in splits.items()}
    overlaps = {
        "train_val": len(fingerprints["train"] & fingerprints["val"]),
        "train_test": len(fingerprints["train"] & fingerprints["test"]),
        "val_test": len(fingerprints["val"] & fingerprints["test"]),
    }
    ttl_presence = {
        "with_ttl_train": sorted(TTL_COLUMNS & set(X_train_ttl.columns)),
        "with_ttl_val": sorted(TTL_COLUMNS & set(X_val_ttl.columns)),
        "with_ttl_test": sorted(TTL_COLUMNS & set(X_test_ttl.columns)),
        "no_ttl_train": sorted(TTL_COLUMNS & set(X_train_no_ttl.columns)),
        "no_ttl_test": sorted(TTL_COLUMNS & set(X_test_no_ttl.columns)),
    }
    return {
        "ttl_columns": sorted(TTL_COLUMNS),
        "ttl_column_presence": ttl_presence,
        "ttl_only_in_with_ttl": (
            all(len(ttl_presence[key]) == 3 for key in ("with_ttl_train", "with_ttl_val", "with_ttl_test"))
            and not ttl_presence["no_ttl_train"]
            and not ttl_presence["no_ttl_test"]
        ),
        "with_ttl_feature_counts": {
            "train": int(X_train_ttl.shape[1]),
            "val": int(X_val_ttl.shape[1]),
            "test": int(X_test_ttl.shape[1]),
            "expected": 42,
            "pass": all(matrix.shape[1] == 42 for matrix in (X_train_ttl, X_val_ttl, X_test_ttl)),
        },
        "split_rows_pairwise_disjoint": {
            "overlap_counts": overlaps,
            "pass": not any(overlaps.values()),
        },
    }


def _json_table(value: Any) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def visit(current: Any, path: str) -> None:
        if isinstance(current, dict):
            for key in sorted(current):
                visit(current[key], f"{path}.{key}" if path else str(key))
        elif isinstance(current, list):
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")
        else:
            rows.append({"path": path, "value": json.dumps(current, sort_keys=True, default=str)})

    visit(value, "")
    return pd.DataFrame(rows)


def _read_json_table(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        return _json_table(json.load(handle))


def _verify(
    csv_outputs: dict[Path, tuple[pd.DataFrame, list[str]]],
    json_outputs: dict[Path, pd.DataFrame],
) -> bool:
    passed = True
    for path, (current, keys) in csv_outputs.items():
        if not path.exists():
            print(f"VERIFY {path.relative_to(ROOT)}: missing prior output")
            passed = False
            continue
        result = verify_outputs(current, path, VERIFY_TOLERANCE, keys)
        print(f"VERIFY {path.relative_to(ROOT)}: {'PASS' if result['pass'] else 'FAIL'}")
        passed = passed and result["pass"]
    for path, current in json_outputs.items():
        if not path.exists():
            print(f"VERIFY {path.relative_to(ROOT)}: missing prior output")
            passed = False
            continue
        result = verify_outputs(current, _read_json_table(path), VERIFY_TOLERANCE, ["path"])
        print(f"VERIFY {path.relative_to(ROOT)}: {'PASS' if result['pass'] else 'FAIL'}")
        passed = passed and result["pass"]
    return passed


def _generate_figure(metrics: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    persisted = metrics.set_index("arm")[["pr_auc", "roc_auc"]]
    figure, axis = plt.subplots(figsize=(7, 5))
    persisted.plot.bar(ax=axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Threshold-independent score")
    axis.set_title("TTL ablation: probability-ranking metrics")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "ttl_ranking_metrics.png", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="recompute and compare without writing")
    args = parser.parse_args()

    X_train_raw, y_train = load_split("train")
    X_val_raw, y_val = load_split("val")
    X_test_raw, y_test = load_split("test")
    frozen_model = load_model("xgboost")
    frozen_params = frozen_model.get_params()
    locked_estimators = (
        int(frozen_model.best_iteration) + 1
        if getattr(frozen_model, "best_iteration", None) is not None
        else int(frozen_params["n_estimators"])
    )
    fit_params = dict(frozen_params)
    fit_params.update({"n_estimators": locked_estimators, "early_stopping_rounds": None})
    ttl_model = xgb.XGBClassifier(**fit_params)

    X_train_ttl = transform(ARTIFACTS_DIR / "preprocess_tree_with_ttl.joblib", X_train_raw)
    X_val_ttl = transform(ARTIFACTS_DIR / "preprocess_tree_with_ttl.joblib", X_val_raw)
    X_test_ttl = transform(ARTIFACTS_DIR / "preprocess_tree_with_ttl.joblib", X_test_raw)
    X_train_no_ttl = transform(ARTIFACTS_DIR / "preprocess_tree.joblib", X_train_raw)
    X_test_no_ttl = transform(ARTIFACTS_DIR / "preprocess_tree.joblib", X_test_raw)

    ttl_model.fit(X_train_ttl, y_train)
    threshold = load_thresholds()["xgboost"]
    no_ttl_probabilities = frozen_model.predict_proba(X_test_no_ttl)[:, 1]
    with_ttl_probabilities = ttl_model.predict_proba(X_test_ttl)[:, 1]
    metrics = pd.DataFrame(
        [
            _metrics("no_ttl_frozen", y_test, no_ttl_probabilities, threshold),
            _metrics("with_ttl_refit", y_test, with_ttl_probabilities, threshold),
        ]
    )
    no_ttl_importance = _importance(frozen_model, X_test_no_ttl, y_test, "no_ttl_frozen")
    with_ttl_importance = _importance(ttl_model, X_test_ttl, y_test, "with_ttl_refit")
    importance = pd.concat([no_ttl_importance, with_ttl_importance], ignore_index=True)
    rank_shift = _rank_shift(no_ttl_importance, with_ttl_importance)
    leakage = _leakage_assertions(
        X_train_ttl,
        X_val_ttl,
        X_test_ttl,
        X_train_no_ttl,
        X_test_no_ttl,
        {"train": (X_train_raw, y_train), "val": (X_val_raw, y_val), "test": (X_test_raw, y_test)},
    )
    config = {
        "random_state": RANDOM_STATE,
        "model": "XGBClassifier",
        "parameter_source": "artifacts/xgboost.joblib.get_params()",
        "frozen_model_params": frozen_params,
        "n_estimators": {
            "value": locked_estimators,
            "source": "frozen n_estimators because best_iteration was unavailable",
        },
        "early_stopping": {"value": None, "source": "dropped for fixed train-only ablation fit"},
        "preprocessors": {
            "with_ttl": "artifacts/preprocess_tree_with_ttl.joblib",
            "no_ttl": "artifacts/preprocess_tree.joblib",
        },
        "fit_data": "data/processed/X_train.parquet and y_train.parquet only",
        "evaluation_data": "frozen test split only",
        "threshold_policy": metrics.loc[0, "threshold_note"],
    }
    csv_outputs = {
        TTL_DIR / "ttl_metrics_comparison.csv": (metrics, ["arm"]),
        TTL_DIR / "ttl_permutation_importance.csv": (importance, ["arm", "feature"]),
        TTL_DIR / "ttl_rank_shift.csv": (rank_shift, ["feature"]),
    }
    json_outputs = {
        TTL_DIR / "ttl_ablation_config.json": _json_table(config),
        TTL_DIR / "ttl_leakage_assertions.json": _json_table(leakage),
    }
    if args.verify:
        return 0 if _verify(csv_outputs, json_outputs) else 1

    for path, (table, _) in csv_outputs.items():
        write_csv(table, path)
    write_json(config, TTL_DIR / "ttl_ablation_config.json")
    write_json(leakage, TTL_DIR / "ttl_leakage_assertions.json")
    persisted_metrics = pd.read_csv(TTL_DIR / "ttl_metrics_comparison.csv")
    _generate_figure(persisted_metrics)
    print("TTL ablation fit and evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
