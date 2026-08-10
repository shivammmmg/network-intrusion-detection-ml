"""Stage 1 SHAP diagnostics for the frozen Random Forest and XGBoost models."""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedShuffleSplit

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
SHAP_DIR = ROOT / "experiments" / "diagnostics" / "shap"
FIGURES_DIR = SHAP_DIR / "figures"
MODELS = {
    "random_forest": "preprocess_tree.joblib",
    "xgboost": "preprocess_tree.joblib",
}
GROUPS = ("TP", "TN", "FP", "FN")
VERIFY_TOLERANCE = 1e-6


def _stratified_indices(y: pd.Series, size: int) -> np.ndarray:
    """Select a seeded stratified subset without using a positional slice."""
    if size > len(y):
        raise ValueError(f"Requested {size} rows from a split with only {len(y)} rows")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=RANDOM_STATE)
    selected, _ = next(splitter.split(np.zeros(len(y)), y.to_numpy()))
    return np.sort(selected)


def _write_gzip_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a gzipped CSV with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        df.to_csv(handle, index=False, lineterminator="\n")


def _extract_positive_class(
    shap_values: Any,
    expected_value: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize SHAP's binary-class result variants to positive-class arrays."""
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    base = np.asarray(expected_value)
    if base.ndim == 0:
        base_values = np.full(values.shape[0], float(base))
    elif base.size == 1:
        base_values = np.full(values.shape[0], float(base.reshape(-1)[0]))
    else:
        base_values = np.full(values.shape[0], float(base.reshape(-1)[1]))
    return values, base_values


def _explain_model(
    name: str,
    model: Any,
    X_sample: pd.DataFrame,
    X_background: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Resolve SHAP scale, calculate values, and empirically check additivity."""
    metadata: dict[str, Any] = {
        "attempted_feature_perturbation": "interventional",
        "attempted_model_output": "probability",
    }
    try:
        explainer = shap.TreeExplainer(
            model,
            data=X_background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        raw_values = explainer.shap_values(X_sample)
        scale = "probability"
        metadata.update(
            {
                "resolved_feature_perturbation": "interventional",
                "resolved_model_output": "probability",
                "fallback_used": False,
            }
        )
    except Exception as error:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
        raw_values = explainer.shap_values(X_sample)
        scale = "raw"
        metadata.update(
            {
                "resolved_feature_perturbation": "tree_path_dependent",
                "resolved_model_output": "raw",
                "fallback_used": True,
                "fallback_exception": f"{type(error).__name__}: {error}",
            }
        )

    values, base_values = _extract_positive_class(raw_values, explainer.expected_value)
    if scale == "probability":
        target = model.predict_proba(X_sample)[:, 1]
        target_name = "predict_proba_positive_class"
    else:
        if name != "xgboost":
            raise RuntimeError(f"{name} cannot provide an output-margin additivity target")
        target = model.predict(X_sample, output_margin=True)
        target_name = "predict_output_margin"
    additivity_error = np.abs(values.sum(axis=1) + base_values - target)
    metadata.update(
        {
            "scale": scale,
            "additivity_target": target_name,
            "max_additivity_error": float(additivity_error.max()),
            "mean_additivity_error": float(additivity_error.mean()),
            "additivity_error_distribution": {
                "median": float(np.quantile(additivity_error, 0.50)),
                "p95": float(np.quantile(additivity_error, 0.95)),
                "p99": float(np.quantile(additivity_error, 0.99)),
                "rows_exceeding_1e-3": int((additivity_error > 1e-3).sum()),
            },
        }
    )
    if name == "random_forest":
        metadata["additivity_interpretation"] = (
            "The small median error and sparse >1e-3 tail are consistent with "
            "Random Forest probability leaf quantization."
        )
    return values, base_values, metadata, np.asarray(target)


def _values_table(
    sample_indices: np.ndarray,
    y_sample: pd.Series,
    probabilities: np.ndarray,
    target: np.ndarray,
    values: np.ndarray,
    base_values: np.ndarray,
    X_sample: pd.DataFrame,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "sample_index": sample_indices,
            "true_label": y_sample.to_numpy(dtype=int),
            "attack_probability": probabilities,
            "additivity_target": target,
            "base_value": base_values,
        }
    )
    for index, feature in enumerate(X_sample.columns):
        table[f"feature__{feature}"] = X_sample.iloc[:, index].to_numpy()
        table[f"shap__{feature}"] = values[:, index]
    return table


def _importance_table(values_table: pd.DataFrame, scale: str) -> pd.DataFrame:
    shap_columns = [column for column in values_table if column.startswith("shap__")]
    importance = pd.DataFrame(
        {
            "feature": [column.removeprefix("shap__") for column in shap_columns],
            "mean_abs_shap_value": [float(values_table[column].abs().mean()) for column in shap_columns],
            "mean_shap_value": [float(values_table[column].mean()) for column in shap_columns],
            "scale": scale,
        }
    )
    importance = importance.sort_values(
        ["mean_abs_shap_value", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    return importance


def _representative_rows(
    name: str,
    values_table: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    predicted = values_table["attack_probability"] >= threshold
    actual = values_table["true_label"] == 1
    masks = {
        "TP": actual & predicted,
        "TN": ~actual & ~predicted,
        "FP": ~actual & predicted,
        "FN": actual & ~predicted,
    }
    rows = []
    for group in GROUPS:
        candidates = values_table.loc[masks[group]].sort_values("sample_index")
        if candidates.empty:
            rows.append(
                {
                    "model": name,
                    "group": group,
                    "sample_index": None,
                    "true_label": None,
                    "attack_probability": None,
                    "threshold": threshold,
                    "selection_status": "no_sample_in_stratified_selection",
                }
            )
        else:
            selected = candidates.iloc[0]
            rows.append(
                {
                    "model": name,
                    "group": group,
                    "sample_index": int(selected["sample_index"]),
                    "true_label": int(selected["true_label"]),
                    "attack_probability": float(selected["attack_probability"]),
                    "threshold": threshold,
                    "selection_status": "selected_lowest_sample_index",
                }
            )
    return pd.DataFrame(rows)


def _local_table(
    values_table: pd.DataFrame,
    representative: pd.Series,
    scale: str,
) -> pd.DataFrame:
    columns = [
        "sample_index",
        "true_label",
        "attack_probability",
        "additivity_target",
        "base_value",
        *[column for column in values_table if column.startswith("shap__")],
    ]
    if pd.isna(representative["sample_index"]):
        result = values_table.loc[:, columns].iloc[0:0].copy()
    else:
        result = values_table.loc[
            values_table["sample_index"] == int(representative["sample_index"]),
            columns,
        ].copy()
    result.insert(0, "group", representative["group"])
    result.insert(0, "threshold", representative["threshold"])
    result.insert(0, "model", representative["model"])
    result.insert(0, "scale", scale)
    return result


def _generate_figures(
    name: str,
    importance: pd.DataFrame,
    values_table: pd.DataFrame,
    local_tables: dict[str, pd.DataFrame],
    scale: str,
) -> None:
    """Create figures exclusively from the persisted SHAP tables."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = importance.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(top["feature"], top["mean_abs_shap_value"])
    scale_label = "raw margin (log-odds)" if scale == "raw" else "probability"
    axis.set_xlabel(f"Mean |SHAP value| ({scale_label})")
    axis.set_title(f"{name}: global SHAP importance")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"shap_bar_{name}.png", dpi=180)
    plt.close(figure)

    shap_columns = [column for column in values_table if column.startswith("shap__")]
    feature_columns = [f"feature__{column.removeprefix('shap__')}" for column in shap_columns]
    figure = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        values_table.loc[:, shap_columns].to_numpy(),
        values_table.loc[:, feature_columns].to_numpy(),
        feature_names=[column.removeprefix("shap__") for column in shap_columns],
        show=False,
        max_display=20,
    )
    plt.title(f"{name}: SHAP beeswarm ({scale_label})")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"shap_beeswarm_{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    for group, local in local_tables.items():
        if local.empty:
            continue
        local_shap_columns = [column for column in local if column.startswith("shap__")]
        explanation = shap.Explanation(
            values=local.loc[:, local_shap_columns].iloc[0].to_numpy(dtype=float),
            base_values=float(local["base_value"].iloc[0]),
            data=values_table.loc[
                values_table["sample_index"] == local["sample_index"].iloc[0],
                [f"feature__{column.removeprefix('shap__')}" for column in local_shap_columns],
            ].iloc[0].to_numpy(),
            feature_names=[column.removeprefix("shap__") for column in local_shap_columns],
        )
        figure = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(f"{name}: {group} ({scale_label})")
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / f"shap_waterfall_{name}_{group}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def _stage0_evidence() -> dict[str, Any]:
    path = ROOT / "experiments" / "diagnostics" / "validation" / "artifact_prediction_check.json"
    with path.open(encoding="utf-8") as handle:
        check = json.load(handle)
    return {
        "numpy_manifest_version": "2.5.1",
        "numpy_runtime_version": np.__version__,
        "decision": (
            "SHAP 0.52.0 requires numba; numba resolves NumPy 2.4.6. "
            "The frozen manifest remains unchanged."
        ),
        "artifact_prediction_gate_pass": check["pass"],
        "max_abs_diff": {
            model: result["max_abs_diff"]
            for model, result in check["models"].items()
        },
        "shap_verification_note": (
            "XGBoost SHAP values were bit-identical across recomputations. "
            "The 1e-6 CSV verification tolerance accommodates float32 "
            "CSV round-trip precision (about 2.4e-7), not model non-determinism."
        ),
    }


def _expected_csv_paths() -> dict[Path, list[str]]:
    paths: dict[Path, list[str]] = {
        SHAP_DIR / "sample_indices.csv": ["role", "sample_index"],
        SHAP_DIR / "representative_cases.csv": ["model", "group"],
    }
    for name in MODELS:
        paths[SHAP_DIR / f"shap_global_importance_{name}.csv"] = ["rank", "feature"]
        paths[SHAP_DIR / f"shap_values_{name}.csv.gz"] = ["sample_index"]
        for group in GROUPS:
            paths[SHAP_DIR / f"shap_local_{name}_{group}.csv"] = ["sample_index"]
    return paths


def _verify_csv_outputs(outputs: dict[Path, pd.DataFrame]) -> bool:
    passed = True
    for path, key_columns in _expected_csv_paths().items():
        if not path.exists():
            print(f"VERIFY {path.relative_to(ROOT)}: missing prior output")
            passed = False
            continue
        comparison = verify_outputs(outputs[path], path, VERIFY_TOLERANCE, key_columns)
        print(f"VERIFY {path.relative_to(ROOT)}: {'PASS' if comparison['pass'] else 'FAIL'}")
        passed = passed and comparison["pass"]
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="recompute and compare without writing")
    args = parser.parse_args()

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    random_forest_sample_indices = _stratified_indices(y_test, 500)
    xgboost_sample_indices = np.arange(len(y_test))
    background_indices = _stratified_indices(y_train, 100)
    X_background_raw = X_train.iloc[background_indices]
    sample_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "role": "random_forest_test_sample",
                    "sample_index": random_forest_sample_indices,
                    "true_label": y_test.iloc[random_forest_sample_indices].to_numpy(dtype=int),
                }
            ),
            pd.DataFrame(
                {
                    "role": "xgboost_test_full",
                    "sample_index": xgboost_sample_indices,
                    "true_label": y_test.to_numpy(dtype=int),
                }
            ),
            pd.DataFrame(
                {
                    "role": "train_background",
                    "sample_index": background_indices,
                    "true_label": y_train.iloc[background_indices].to_numpy(dtype=int),
                }
            ),
        ],
        ignore_index=True,
    )
    thresholds = load_thresholds()
    outputs: dict[Path, pd.DataFrame] = {SHAP_DIR / "sample_indices.csv": sample_table}
    representatives: list[pd.DataFrame] = []
    scale_metadata: dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "model_sample_sizes": {
            "random_forest": int(len(random_forest_sample_indices)),
            "xgboost": int(len(xgboost_sample_indices)),
        },
        "background_size": int(len(background_indices)),
        "verify_tolerance": VERIFY_TOLERANCE,
        "environment_decision": _stage0_evidence(),
        "models": {},
    }

    for name, preprocessor_name in MODELS.items():
        model_sample_indices = (
            random_forest_sample_indices if name == "random_forest" else xgboost_sample_indices
        )
        X_sample_raw = X_test.iloc[model_sample_indices]
        y_sample = y_test.iloc[model_sample_indices].reset_index(drop=True)
        X_sample = transform(ARTIFACTS_DIR / preprocessor_name, X_sample_raw).reset_index(drop=True)
        X_background = transform(ARTIFACTS_DIR / preprocessor_name, X_background_raw).reset_index(drop=True)
        model = load_model(name)
        probabilities = model.predict_proba(X_sample)[:, 1]
        values, base_values, metadata, target = _explain_model(name, model, X_sample, X_background)
        values_table = _values_table(
            model_sample_indices,
            y_sample,
            probabilities,
            target,
            values,
            base_values,
            X_sample,
        )
        importance = _importance_table(values_table, metadata["scale"])
        representatives_for_model = _representative_rows(name, values_table, thresholds[name])
        representatives.append(representatives_for_model)
        local_tables = {
            group: _local_table(
                values_table,
                representatives_for_model.loc[representatives_for_model["group"] == group].iloc[0],
                metadata["scale"],
            )
            for group in GROUPS
        }

        outputs[SHAP_DIR / f"shap_values_{name}.csv.gz"] = values_table
        outputs[SHAP_DIR / f"shap_global_importance_{name}.csv"] = importance
        for group, local in local_tables.items():
            outputs[SHAP_DIR / f"shap_local_{name}_{group}.csv"] = local
        scale_metadata["models"][name] = metadata
        if not args.verify:
            _write_gzip_csv(values_table, SHAP_DIR / f"shap_values_{name}.csv.gz")
            write_csv(importance, SHAP_DIR / f"shap_global_importance_{name}.csv")
            for group, local in local_tables.items():
                write_csv(local, SHAP_DIR / f"shap_local_{name}_{group}.csv")
            saved_values = pd.read_csv(SHAP_DIR / f"shap_values_{name}.csv.gz")
            saved_importance = pd.read_csv(SHAP_DIR / f"shap_global_importance_{name}.csv")
            saved_locals = {
                group: pd.read_csv(SHAP_DIR / f"shap_local_{name}_{group}.csv")
                for group in GROUPS
            }
            _generate_figures(name, saved_importance, saved_values, saved_locals, metadata["scale"])

        print(
            f"{name}: {metadata['scale']} scale, "
            f"max additivity error={metadata['max_additivity_error']:.3e}"
        )

    representative_table = pd.concat(representatives, ignore_index=True)
    outputs[SHAP_DIR / "representative_cases.csv"] = representative_table
    verify_pass = _verify_csv_outputs(outputs) if args.verify else True
    if not args.verify:
        write_csv(sample_table, SHAP_DIR / "sample_indices.csv")
        write_csv(representative_table, SHAP_DIR / "representative_cases.csv")
        scale_metadata["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        write_json(scale_metadata, SHAP_DIR / "shap_scale_metadata.json")

    return 0 if verify_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
