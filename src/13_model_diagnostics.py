"""Stage 2 diagnostics: importance, errors, calibration, and drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss

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
DIAGNOSTICS_DIR = ROOT / "experiments" / "diagnostics"
IMPORTANCE_DIR = DIAGNOSTICS_DIR / "importance"
ERRORS_DIR = DIAGNOSTICS_DIR / "errors"
CALIBRATION_DIR = DIAGNOSTICS_DIR / "calibration"
DRIFT_DIR = DIAGNOSTICS_DIR / "drift"
FIGURES_DIR = CALIBRATION_DIR / "figures"
MODELS = {
    "logistic_regression": "preprocess_linear.joblib",
    "neural_network": "preprocess_linear.joblib",
    "random_forest": "preprocess_tree.joblib",
    "xgboost": "preprocess_tree.joblib",
}
GROUPS = ("TP", "TN", "FP", "FN")
VERIFY_TOLERANCE = 1e-6


def pr_auc(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Probability-space PR-AUC scorer shared by every model."""
    return float(average_precision_score(y, estimator.predict_proba(X)[:, 1]))


def _importance_table(model: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(["importance_mean", "feature"], ascending=[False, True], ignore_index=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table[["feature", "importance_mean", "importance_std", "rank"]]


def _groups(y: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    actual = y.to_numpy(dtype=int) == 1
    predicted = probabilities >= threshold
    return {
        "TP": actual & predicted,
        "TN": ~actual & ~predicted,
        "FP": ~actual & predicted,
        "FN": actual & ~predicted,
    }


def _error_profile(X: pd.DataFrame, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        selected = X.loc[groups[group]]
        means = selected.mean(axis=0) if not selected.empty else pd.Series(np.nan, index=X.columns)
        rows.extend(
            {
                "group": group,
                "feature": feature,
                "mean_feature_value": float(value) if pd.notna(value) else np.nan,
                "group_count": int(len(selected)),
            }
            for feature, value in means.items()
        )
    return pd.DataFrame(rows)


def _error_summary(groups_by_model: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}}
    for name, groups in groups_by_model.items():
        counts = {group: int(mask.sum()) for group, mask in groups.items()}
        negatives = counts["TN"] + counts["FP"]
        positives = counts["TP"] + counts["FN"]
        summary["models"][name] = {
            "group_counts": counts,
            "false_positive_rate": float(counts["FP"] / negatives) if negatives else None,
            "false_negative_rate": float(counts["FN"] / positives) if positives else None,
        }
    return summary


def _calibration_bins(probabilities: np.ndarray, y: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({"probability": probabilities, "true_label": y.to_numpy(dtype=int)})
    try:
        quantiles = pd.qcut(table["probability"], q=10, duplicates="drop")
    except ValueError:
        quantiles = pd.Series(["all"] * len(table), index=table.index)
    grouped = table.assign(quantile_bin=quantiles).groupby("quantile_bin", observed=True)
    bins = grouped.agg(
        mean_predicted=("probability", "mean"),
        observed_frequency=("true_label", "mean"),
        count=("true_label", "size"),
    ).reset_index(drop=True)
    bins.insert(0, "bin", np.arange(1, len(bins) + 1))
    bins["gap"] = bins["observed_frequency"] - bins["mean_predicted"]
    return bins[["bin", "mean_predicted", "observed_frequency", "count", "gap"]]


def _calibration_summary(
    bins_by_model: dict[str, pd.DataFrame],
    probabilities_by_model: dict[str, np.ndarray],
    y: pd.Series,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}}
    for name, bins in bins_by_model.items():
        absolute_gap = bins["gap"].abs()
        worst_index = int(absolute_gap.idxmax())
        ece = float((absolute_gap * bins["count"]).sum() / bins["count"].sum())
        summary["models"][name] = {
            "brier_score": float(brier_score_loss(y, probabilities_by_model[name])),
            "expected_calibration_error": ece,
            "worst_bin": {
                "bin": int(bins.loc[worst_index, "bin"]),
                "gap": float(bins.loc[worst_index, "gap"]),
                "absolute_gap": float(absolute_gap.loc[worst_index]),
                "count": int(bins.loc[worst_index, "count"]),
            },
        }
    return summary


def _drift_table(X_train: pd.DataFrame, X_test: pd.DataFrame, requested_bins: int = 10) -> pd.DataFrame:
    rows = []
    for feature in X_train.columns:
        train = X_train[feature].to_numpy(dtype=float)
        test = X_test[feature].to_numpy(dtype=float)
        unique_values = np.unique(np.concatenate((train, test)))
        if len(unique_values) <= 50:
            binning_strategy = "value"
            if len(unique_values) >= 2:
                interior_edges = (unique_values[:-1] + unique_values[1:]) / 2
                edges = np.concatenate(([-np.inf], interior_edges, [np.inf]))
            else:
                edges = np.array([-np.inf, np.inf])
        else:
            binning_strategy = "quantile"
            quantile_edges = np.unique(np.quantile(train, np.linspace(0, 1, requested_bins + 1)))
            edges = np.concatenate(([-np.inf], quantile_edges[1:-1], [np.inf]))
        bins_effective = len(edges) - 1
        if bins_effective < 2:
            psi = np.nan
        else:
            train_counts, _ = np.histogram(train, bins=edges)
            test_counts, _ = np.histogram(test, bins=edges)
            train_share = np.maximum(train_counts / len(train), 1e-6)
            test_share = np.maximum(test_counts / len(test), 1e-6)
            psi = float(np.sum((test_share - train_share) * np.log(test_share / train_share)))
            bins_effective = len(edges) - 1
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "ks_statistic": float(ks_2samp(train, test).statistic),
                "n_bins_effective": bins_effective,
                "binning_strategy": binning_strategy,
                "psi_degenerate": bins_effective < 2,
                "psi_low_resolution": 2 <= bins_effective < 5,
            }
        )
    return pd.DataFrame(rows).sort_values(["psi", "feature"], ascending=[False, True], ignore_index=True)


def _drift_overlap(
    importance_by_model: dict[str, pd.DataFrame],
    drift: pd.DataFrame,
) -> pd.DataFrame:
    by_psi = drift.sort_values(["psi", "feature"], ascending=[False, True], na_position="last").copy()
    by_psi.insert(0, "drift_rank_by_psi", np.arange(1, len(by_psi) + 1))
    by_ks = drift.sort_values(["ks_statistic", "feature"], ascending=[False, True]).copy()
    by_ks.insert(0, "drift_rank_by_ks", np.arange(1, len(by_ks) + 1))
    drift_ranks = by_psi.merge(
        by_ks[["feature", "drift_rank_by_ks"]],
        on="feature",
        how="left",
    )
    top_features = set(by_psi.head(10)["feature"]) | set(by_ks.head(10)["feature"])
    rows = []
    for name, importance in importance_by_model.items():
        top_importance = importance.loc[importance["rank"] <= 10]
        joined = top_importance.merge(drift_ranks, on="feature", how="inner")
        joined = joined.loc[joined["feature"].isin(top_features)]
        rows.extend(
            {
                "model": name,
                "feature": row.feature,
                "permutation_rank": int(row.rank),
                "drift_rank_by_psi": int(row.drift_rank_by_psi),
                "drift_rank_by_ks": int(row.drift_rank_by_ks),
                "psi": float(row.psi) if pd.notna(row.psi) else np.nan,
                "ks_statistic": float(row.ks_statistic),
                "psi_degenerate": bool(row.psi_degenerate),
                "psi_low_resolution": bool(row.psi_low_resolution),
            }
            for row in joined.itertuples(index=False)
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "feature",
            "permutation_rank",
            "drift_rank_by_psi",
            "drift_rank_by_ks",
            "psi",
            "ks_statistic",
            "psi_degenerate",
            "psi_low_resolution",
        ],
    )


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
            rows.append({"path": path, "value": json.dumps(current, sort_keys=True)})

    visit(value, "")
    return pd.DataFrame(rows)


def _read_json_table(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        return _json_table(json.load(handle))


def _generate_calibration_figures(bins_by_model: dict[str, pd.DataFrame]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, bins in bins_by_model.items():
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.plot([0, 1], [0, 1], "--", color="black", label="perfect calibration")
        axis.plot(
            bins["mean_predicted"],
            bins["observed_frequency"],
            marker="o",
            label=name,
        )
        axis.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title=f"{name} reliability")
        axis.legend()
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / f"reliability_{name}.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="black", label="perfect calibration")
    for name, bins in bins_by_model.items():
        axis.plot(bins["mean_predicted"], bins["observed_frequency"], marker="o", label=name)
    axis.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Reliability comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "reliability_overlay.png", dpi=180)
    plt.close(figure)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="recompute and compare without writing")
    args = parser.parse_args()

    X_train_raw, y_train = load_split("train")
    X_test_raw, y_test = load_split("test")
    thresholds = load_thresholds()
    transformed = {
        "linear": (
            transform(ARTIFACTS_DIR / "preprocess_linear.joblib", X_train_raw),
            transform(ARTIFACTS_DIR / "preprocess_linear.joblib", X_test_raw),
        ),
        "tree": (
            transform(ARTIFACTS_DIR / "preprocess_tree.joblib", X_train_raw),
            transform(ARTIFACTS_DIR / "preprocess_tree.joblib", X_test_raw),
        ),
    }
    importance_by_model: dict[str, pd.DataFrame] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}
    groups_by_model: dict[str, dict[str, np.ndarray]] = {}
    bins_by_model: dict[str, pd.DataFrame] = {}
    csv_outputs: dict[Path, tuple[pd.DataFrame, list[str]]] = {}

    for name, preprocessor_name in MODELS.items():
        X_test = transformed["linear" if preprocessor_name.startswith("preprocess_linear") else "tree"][1]
        model = load_model(name)
        probabilities = model.predict_proba(X_test)[:, 1]
        importance = _importance_table(model, X_test, y_test)
        groups = _groups(y_test, probabilities, thresholds[name])
        profile = _error_profile(X_test, groups)
        bins = _calibration_bins(probabilities, y_test)
        importance_by_model[name] = importance
        probabilities_by_model[name] = probabilities
        groups_by_model[name] = groups
        bins_by_model[name] = bins
        csv_outputs[IMPORTANCE_DIR / f"permutation_importance_{name}.csv"] = (
            importance,
            ["feature"],
        )
        csv_outputs[ERRORS_DIR / f"error_profiles_{name}.csv"] = (
            profile,
            ["group", "feature"],
        )
        csv_outputs[CALIBRATION_DIR / f"calibration_bins_{name}.csv"] = (bins, ["bin"])
        print(f"{name}: permutation importance complete")

    drift = _drift_table(*transformed["tree"])
    overlap = _drift_overlap(importance_by_model, drift)
    csv_outputs[DRIFT_DIR / "drift_psi_ks.csv"] = (drift, ["feature"])
    csv_outputs[DRIFT_DIR / "drift_importance_overlap.csv"] = (overlap, ["model", "feature"])
    errors = _error_summary(groups_by_model)
    calibration = _calibration_summary(bins_by_model, probabilities_by_model, y_test)
    json_outputs = {
        ERRORS_DIR / "error_summary.json": _json_table(errors),
        CALIBRATION_DIR / "calibration_summary.json": _json_table(calibration),
    }

    if args.verify:
        return 0 if _verify(csv_outputs, json_outputs) else 1

    for path, (table, _) in csv_outputs.items():
        write_csv(table, path)
    write_json(errors, ERRORS_DIR / "error_summary.json")
    write_json(calibration, CALIBRATION_DIR / "calibration_summary.json")
    persisted_bins = {
        name: pd.read_csv(CALIBRATION_DIR / f"calibration_bins_{name}.csv")
        for name in MODELS
    }
    _generate_calibration_figures(persisted_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
