"""Post-training analysis for the finalized Random Forest and XGBoost tracks.

This script never fits a model. It reads the finalized tuning logs, saved
models, saved prediction files, and the frozen test features for inference
timing only. All model-selection summaries are validation-only; test outputs
are used only for descriptive comparison and paired bootstrap uncertainty.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE  # noqa: E402
from preprocess import load_artifact  # noqa: E402


ANALYSIS_DIR = ROOT / "experiments" / "model_analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"
RF_DIR = ROOT / "experiments" / "random_forest"
XGB_DIR = ROOT / "experiments" / "xgboost"

RF_CONFIG_COLUMNS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "class_weight",
]
XGB_CONFIG_COLUMNS = [
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
]
METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def level_key(value: Any) -> tuple[int, Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return (0, "None")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (1, float(value))
    if isinstance(value, str):
        try:
            return (1, float(value))
        except ValueError:
            pass
    return (2, str(value))


def display_level(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "None"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_rf_value(column: str, value: Any) -> Any:
    if pd.isna(value) or str(value).lower() in {"nan", "none"}:
        return None
    if column in {"n_estimators", "max_depth", "min_samples_leaf"}:
        return int(value)
    if column == "max_features":
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def normalize_xgb_value(column: str, value: Any) -> Any:
    if pd.isna(value):
        return None
    if column in {"n_estimators", "max_depth", "min_child_weight"}:
        return int(value)
    return float(value)


def config_key(row: pd.Series, columns: list[str], normalizer) -> str:
    values = {column: normalizer(column, row[column]) for column in columns}
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def load_tuning_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rf_round1 = pd.read_csv(RF_DIR / "tuning_results.csv")
    rf_round1["source_round"] = "round1"
    rf_round1["source_stage"] = rf_round1["stage"].astype(str)
    rf_round2 = pd.read_csv(RF_DIR / "round2_joint_search.csv")
    rf_round2["source_round"] = "round2"
    rf_round2["source_stage"] = "round2_joint"
    rf = pd.concat([rf_round1, rf_round2], ignore_index=True, sort=False)
    rf["config_key"] = rf.apply(
        lambda row: config_key(row, RF_CONFIG_COLUMNS, normalize_rf_value), axis=1
    )

    xgb_round1 = pd.read_csv(XGB_DIR / "tuning_results.csv")
    xgb_round1["source_round"] = "round1"
    xgb_round1["source_stage"] = xgb_round1["stage"].astype(str)
    xgb_round2 = pd.read_csv(XGB_DIR / "round2_joint_search.csv")
    xgb_round2["source_round"] = "round2"
    xgb_round2["source_stage"] = "round2_joint"
    xgb_all = pd.concat([xgb_round1, xgb_round2], ignore_index=True, sort=False)
    xgb_all["config_key"] = xgb_all.apply(
        lambda row: config_key(row, XGB_CONFIG_COLUMNS, normalize_xgb_value), axis=1
    )
    return rf, xgb_all


def duplicate_report(
    frame: pd.DataFrame, model: str, metric_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = frame["config_key"].value_counts()
    frame = frame.copy()
    frame["config_occurrences"] = frame["config_key"].map(counts)
    representatives = frame.drop_duplicates("config_key", keep="first").copy()

    duplicate_groups: list[dict[str, Any]] = []
    for key, group in frame.groupby("config_key", sort=False):
        if len(group) < 2:
            continue
        metric_ranges = {}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            metric_ranges[column] = float(values.max() - values.min()) if len(values) else None
        duplicate_groups.append(
            {
                "config_key": key,
                "occurrences": int(len(group)),
                "contexts": sorted(
                    f"{row.source_round}/{row.source_stage}"
                    for row in group.itertuples()
                ),
                "metric_ranges": metric_ranges,
            }
        )

    max_range = 0.0
    for group in duplicate_groups:
        for value in group["metric_ranges"].values():
            if value is not None:
                max_range = max(max_range, value)
    reproducible = max_range <= 1e-12
    report = {
        "model": model,
        "raw_fit_rows": int(len(frame)),
        "unique_full_configurations": int(len(representatives)),
        "duplicate_configuration_groups": int(len(duplicate_groups)),
        "duplicate_refit_occurrences_beyond_first": int(
            sum(group["occurrences"] - 1 for group in duplicate_groups)
        ),
        "duplicate_metrics_reproducible_to_csv_precision": bool(reproducible),
        "maximum_duplicate_metric_range": float(max_range),
        "duplicate_groups": duplicate_groups,
    }
    return representatives, report


def stability_row(
    model: str, raw: pd.DataFrame, representatives: pd.DataFrame, duplicates: dict[str, Any]
) -> dict[str, Any]:
    ranked = representatives.sort_values(
        ["val_pr_auc", "val_roc_auc"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)
    winner = float(ranked.iloc[0]["val_pr_auc"])
    winner_row = ranked.iloc[0]
    row: dict[str, Any] = {
        "model": model,
        "raw_fit_rows": len(raw),
        "unique_configurations": len(ranked),
        "winner_config_key": winner_row["config_key"],
        "winner_context": f"{winner_row['source_round']}/{winner_row['source_stage']}",
        "winner_pr_auc": winner,
        "second_pr_auc": float(ranked.iloc[1]["val_pr_auc"]),
        "fifth_pr_auc": float(ranked.iloc[4]["val_pr_auc"]),
        "tenth_pr_auc": float(ranked.iloc[9]["val_pr_auc"]),
        "top5_pr_auc_range": float(ranked.head(5)["val_pr_auc"].max() - ranked.head(5)["val_pr_auc"].min()),
        "top10_pr_auc_range": float(ranked.head(10)["val_pr_auc"].max() - ranked.head(10)["val_pr_auc"].min()),
        "within_0.0001": int((winner - ranked["val_pr_auc"] <= 0.0001 + 1e-15).sum()),
        "within_0.0005": int((winner - ranked["val_pr_auc"] <= 0.0005 + 1e-15).sum()),
        "within_0.0010": int((winner - ranked["val_pr_auc"] <= 0.0010 + 1e-15).sum()),
        "duplicate_configuration_groups": duplicates["duplicate_configuration_groups"],
        "duplicate_refit_occurrences_beyond_first": duplicates[
            "duplicate_refit_occurrences_beyond_first"
        ],
        "duplicate_metrics_reproducible": duplicates[
            "duplicate_metrics_reproducible_to_csv_precision"
        ],
    }
    return row


def stats_row(
    model: str,
    scope: str,
    source: str,
    parameter: str,
    level: str,
    group: pd.DataFrame,
    comparison_type: str,
    paired_count: int = 0,
    paired_delta: list[float] | None = None,
) -> dict[str, Any]:
    deltas = paired_delta or []
    row: dict[str, Any] = {
        "model": model,
        "scope": scope,
        "source": source,
        "parameter": parameter,
        "level": level,
        "comparison_type": comparison_type,
        "n_rows": int(len(group)),
        "n_unique_configurations": int(group["config_key"].nunique()),
        "mean_val_pr_auc": float(group["val_pr_auc"].mean()),
        "median_val_pr_auc": float(group["val_pr_auc"].median()),
        "min_val_pr_auc": float(group["val_pr_auc"].min()),
        "max_val_pr_auc": float(group["val_pr_auc"].max()),
        "mean_val_roc_auc": float(group["val_roc_auc"].mean()),
        "max_val_roc_auc": float(group["val_roc_auc"].max()),
        "mean_best_iteration": float(group["best_iteration"].mean())
        if "best_iteration" in group and group["best_iteration"].notna().any()
        else None,
        "paired_comparison_count": int(paired_count),
        "paired_delta_mean_pr_auc": float(np.mean(deltas)) if deltas else None,
        "paired_delta_min_pr_auc": float(np.min(deltas)) if deltas else None,
        "paired_delta_max_pr_auc": float(np.max(deltas)) if deltas else None,
    }
    best = group.loc[group["val_pr_auc"].idxmax()]
    row["best_config_id"] = str(best.get("config_id", ""))
    return row


def sensitivity_analysis(
    frame: pd.DataFrame, model: str, parameters: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Stage-level summaries are descriptive and retain the staged-search context.
    for source, stage_group in frame.groupby("source_stage", sort=False):
        for parameter in parameters:
            levels = stage_group[parameter].map(display_level).unique().tolist()
            if len(levels) < 2:
                continue
            for level, group in stage_group.groupby(
                stage_group[parameter].map(display_level), sort=False
            ):
                rows.append(
                    stats_row(
                        model,
                        "round1_staged_or_round2",
                        str(source),
                        parameter,
                        str(level),
                        group,
                        "descriptive_stage_aggregate",
                    )
                )

    # A full Cartesian Round 2 grid permits matched one-factor comparisons.
    round2 = frame[frame["source_round"] == "round2"].drop_duplicates("config_key").copy()
    for parameter in parameters:
        round2[f"__level_{parameter}"] = round2[parameter].map(display_level)
    controlled_summary: dict[str, Any] = {}
    for parameter in parameters:
        level_column = f"__level_{parameter}"
        other_parameters = [
            f"__level_{column}" for column in parameters if column != parameter
        ]
        deltas: list[float] = []
        paired_groups = 0
        levels = sorted(round2[level_column].drop_duplicates().tolist(), key=level_key)
        if len(levels) < 2:
            continue
        for _, matched in round2.groupby(other_parameters, dropna=False, sort=False):
            available = matched[level_column].drop_duplicates().tolist()
            if len(available) < 2:
                continue
            available = sorted(available, key=level_key)
            for low_index in range(len(available) - 1):
                for high_index in range(low_index + 1, len(available)):
                    low = matched[matched[level_column] == available[low_index]].iloc[0]
                    high = matched[matched[level_column] == available[high_index]].iloc[0]
                    delta = float(high["val_pr_auc"] - low["val_pr_auc"])
                    deltas.append(delta)
                    paired_groups += 1
                    rows.append(
                        stats_row(
                            model,
                            "round2_joint",
                            "round2_joint",
                            parameter,
                            f"{available[low_index]} -> {available[high_index]}",
                            pd.DataFrame([low, high]),
                            "controlled_one_factor",
                            paired_count=1,
                            paired_delta=[delta],
                        )
                    )
        controlled_summary[parameter] = {
            "levels": [display_level(value) for value in levels],
            "paired_comparisons": paired_groups,
            "mean_high_minus_low_pr_auc": float(np.mean(deltas)) if deltas else None,
            "min_high_minus_low_pr_auc": float(np.min(deltas)) if deltas else None,
            "max_high_minus_low_pr_auc": float(np.max(deltas)) if deltas else None,
        }

    return pd.DataFrame(rows), controlled_summary


def metric_dict(labels: np.ndarray, probabilities: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


def load_saved_predictions() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    predictions = {
        "Random Forest": pd.read_csv(RF_DIR / "test_predictions.csv"),
        "XGBoost": pd.read_csv(XGB_DIR / "test_predictions.csv"),
    }
    rf, xgb_frame = predictions.values()
    if not np.array_equal(rf["sample_index"].to_numpy(), xgb_frame["sample_index"].to_numpy()):
        raise RuntimeError("RF and XGBoost test sample_index columns do not align")
    if not np.array_equal(rf["true_label"].to_numpy(), xgb_frame["true_label"].to_numpy()):
        raise RuntimeError("RF and XGBoost test labels do not align")
    metrics = {}
    for name, frame in predictions.items():
        probabilities = frame["attack_probability"].to_numpy(dtype=float)
        predicted = frame["predicted_label"].to_numpy(dtype=int)
        if not (
            np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all()
            and np.array_equal(predicted, (probabilities >= 0.5).astype(int))
        ):
            raise RuntimeError(f"Invalid saved predictions for {name}")
        metrics[name] = metric_dict(
            frame["true_label"].to_numpy(dtype=int), probabilities, predicted
        )
    return predictions, metrics


def run_inference_benchmark() -> tuple[pd.DataFrame, dict[str, Any]]:
    x_test_raw = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    preprocessor = load_artifact(ARTIFACTS_DIR / "preprocess_tree.joblib")
    x_test = preprocessor.transform(x_test_raw)
    if x_test.shape != (82332, 39):
        raise RuntimeError(f"Unexpected transformed test shape for benchmark: {x_test.shape}")
    arr = x_test.to_numpy() if hasattr(x_test, "to_numpy") else np.asarray(x_test)
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite values in transformed benchmark matrix")

    models = {
        "Random Forest": joblib.load(ARTIFACTS_DIR / "random_forest.joblib"),
        "XGBoost": joblib.load(ARTIFACTS_DIR / "xgboost.joblib"),
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, model in models.items():
        for _ in range(2):
            model.predict_proba(x_test)
        timings = []
        for repetition in range(1, 11):
            start = time.perf_counter()
            probabilities = model.predict_proba(x_test)[:, 1]
            elapsed = time.perf_counter() - start
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"Invalid inference probabilities for {name}")
            timings.append(elapsed)
            rows.append(
                {
                    "model": name,
                    "repetition": repetition,
                    "seconds": elapsed,
                    "rows": len(x_test_raw),
                    "rows_per_second": len(x_test_raw) / elapsed,
                    "warmup_calls": 2,
                    "preprocessing_included": False,
                }
            )
        summary[name] = {
            "warmup_calls": 2,
            "timed_repetitions": 10,
            "median_seconds": float(np.median(timings)),
            "mean_seconds": float(np.mean(timings)),
            "total_seconds": float(np.sum(timings)),
            "median_rows_per_second": float(len(x_test_raw) / np.median(timings)),
            "preprocessing_included": False,
        }
    return pd.DataFrame(rows), summary


def run_bootstrap(
    predictions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rf = predictions["Random Forest"]
    xg = predictions["XGBoost"]
    labels = rf["true_label"].to_numpy(dtype=int)
    rf_prob = rf["attack_probability"].to_numpy(dtype=float)
    xg_prob = xg["attack_probability"].to_numpy(dtype=float)
    rf_pred = rf["predicted_label"].to_numpy(dtype=int)
    xg_pred = xg["predicted_label"].to_numpy(dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_values = {name: {metric: [] for metric in METRICS} for name in predictions}
    difference_values = {metric: [] for metric in ["pr_auc", "roc_auc", "f1", "precision", "recall"]}
    skipped = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[sampled]
        if np.unique(sampled_labels).size < 2:
            skipped += 1
            continue
        rf_metrics = metric_dict(sampled_labels, rf_prob[sampled], rf_pred[sampled])
        xg_metrics = metric_dict(sampled_labels, xg_prob[sampled], xg_pred[sampled])
        for metric in METRICS:
            metric_values["Random Forest"][metric].append(rf_metrics[metric])
            metric_values["XGBoost"][metric].append(xg_metrics[metric])
        for metric in difference_values:
            difference_values[metric].append(xg_metrics[metric] - rf_metrics[metric])

    observed = {
        name: metric_dict(
            frame["true_label"].to_numpy(dtype=int),
            frame["attack_probability"].to_numpy(dtype=float),
            frame["predicted_label"].to_numpy(dtype=int),
        )
        for name, frame in predictions.items()
    }
    metric_rows = []
    for model, values_by_metric in metric_values.items():
        for metric, values in values_by_metric.items():
            values_array = np.asarray(values)
            metric_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "observed": observed[model][metric],
                    "bootstrap_median": float(np.median(values_array)),
                    "ci_lower_2_5": float(np.percentile(values_array, 2.5)),
                    "ci_upper_97_5": float(np.percentile(values_array, 97.5)),
                    "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
                    "bootstrap_replicates_valid": len(values),
                    "bootstrap_replicates_skipped": skipped,
                    "seed": BOOTSTRAP_SEED,
                    "resampling_unit": "test rows with replacement",
                }
            )
    difference_rows = []
    observed_difference = {
        metric: observed["XGBoost"][metric] - observed["Random Forest"][metric]
        for metric in difference_values
    }
    for metric, values in difference_values.items():
        values_array = np.asarray(values)
        difference_rows.append(
            {
                "metric": metric,
                "observed_xgboost_minus_rf": observed_difference[metric],
                "bootstrap_median_difference": float(np.median(values_array)),
                "ci_lower_2_5": float(np.percentile(values_array, 2.5)),
                "ci_upper_97_5": float(np.percentile(values_array, 97.5)),
                "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
                "bootstrap_replicates_valid": len(values),
                "bootstrap_replicates_skipped": skipped,
                "seed": BOOTSTRAP_SEED,
                "paired_resampling": True,
            }
        )
    summary = {
        "replicates_requested": BOOTSTRAP_REPLICATES,
        "replicates_valid": BOOTSTRAP_REPLICATES - skipped,
        "replicates_skipped_one_class": skipped,
        "seed": BOOTSTRAP_SEED,
        "paired_resampling": True,
        "resampling_unit": "test rows with replacement",
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(difference_rows), summary


def make_figures(
    rf_representatives: pd.DataFrame,
    xgb_representatives: pd.DataFrame,
    rf_sensitivity: pd.DataFrame,
    xgb_sensitivity: pd.DataFrame,
    stability: pd.DataFrame,
    efficiency: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    paired: pd.DataFrame,
) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        path = FIGURES_DIR / name
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))

    def sensitivity_plot(frame: pd.DataFrame, params: list[str], name: str, title: str) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.ravel()
        for index, parameter in enumerate(params):
            ax = axes[index]
            values = []
            labels = []
            for level, group in frame.groupby(frame[parameter].map(display_level), sort=False):
                values.append(group["val_pr_auc"].to_numpy())
                labels.append(str(level))
            ax.boxplot(values, tick_labels=labels, showmeans=True)
            ax.set_title(parameter)
            ax.set_ylabel("Validation PR-AUC")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes[len(params) :]:
            ax.axis("off")
        fig.suptitle(title)
        save(fig, name)

    sensitivity_plot(
        rf_representatives,
        RF_CONFIG_COLUMNS,
        "rf_hyperparameter_sensitivity.png",
        "Random Forest configuration-level validation PR-AUC",
    )
    sensitivity_plot(
        xgb_representatives,
        XGB_CONFIG_COLUMNS[:6],
        "xgboost_hyperparameter_sensitivity.png",
        "XGBoost configuration-level validation PR-AUC",
    )

    lr = pd.read_csv(XGB_DIR / "tuning_results.csv")
    lr = lr[lr["stage"] == "learning_rate"].copy()
    lr["learning_rate"] = lr["learning_rate"].astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lr["learning_rate"], lr["best_iteration"], s=55)
    for _, row in lr.sort_values("learning_rate").iterrows():
        ax.annotate(f"{row['best_iteration']:.0f}", (row["learning_rate"], row["best_iteration"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Best iteration")
    ax.set_title("XGBoost learning rate and early-stopping best iteration")
    ax.grid(alpha=0.25)
    save(fig, "xgboost_learning_rate_best_iteration.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in [("Random Forest", rf_representatives), ("XGBoost", xgb_representatives)]:
        ranked = group.sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
        ax.plot(np.arange(1, len(ranked) + 1), ranked["val_pr_auc"], label=model)
    ax.set_xlabel("Unique configuration rank")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Configuration-level model-selection stability")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "model_selection_stability.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, title in zip(
        axes,
        ["test_pr_auc", "final_training_runtime_seconds"],
        ["Test PR-AUC", "Final training runtime (s)"],
    ):
        axes_values = efficiency[metric].astype(float)
        ax.bar(efficiency["model"], axes_values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Final performance and training efficiency")
    save(fig, "efficiency_performance.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    metric_order = METRICS
    x = np.arange(len(metric_order))
    width = 0.35
    for offset, model in [(-width / 2, "Random Forest"), (width / 2, "XGBoost")]:
        subset = bootstrap_metrics[bootstrap_metrics["model"] == model].set_index("metric").loc[metric_order]
        lower = subset["observed"] - subset["ci_lower_2_5"]
        upper = subset["ci_upper_97_5"] - subset["observed"]
        ax.errorbar(x + offset, subset["observed"], yerr=[lower, upper], fmt="o", capsize=4, label=model)
    ax.set_xticks(x, metric_order)
    ax.set_ylabel("Metric value")
    ax.set_title("Paired-test bootstrap 95% confidence intervals")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "bootstrap_metric_confidence_intervals.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    paired_order = paired["metric"].tolist()
    subset = paired.set_index("metric").loc[paired_order]
    y = np.arange(len(subset))
    lower = subset["bootstrap_median_difference"] - subset["ci_lower_2_5"]
    upper = subset["ci_upper_97_5"] - subset["bootstrap_median_difference"]
    ax.errorbar(subset["bootstrap_median_difference"], y, xerr=[lower, upper], fmt="o", capsize=4)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, paired_order)
    ax.set_xlabel("XGBoost minus Random Forest")
    ax.set_title("Paired bootstrap metric differences")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "bootstrap_paired_difference_intervals.png")
    return paths


def format_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(str(column) for column in view.columns) + " |"
    separator = "| " + " | ".join("---" for _ in view.columns) + " |"
    body = []
    for values in view.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if value is None or isinstance(value, (list, tuple, dict)):
                cells.append("" if value is None else str(value))
                continue
            missing = pd.isna(value)
            cells.append("" if isinstance(missing, (bool, np.bool_)) and missing else str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def make_report(
    stability: pd.DataFrame,
    efficiency: pd.DataFrame,
    inference: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    paired: pd.DataFrame,
    rf_sensitivity: pd.DataFrame,
    xgb_sensitivity: pd.DataFrame,
    rf_controlled: dict[str, Any],
    xgb_controlled: dict[str, Any],
    lr_summary: list[dict[str, Any]],
    duplicate_reports: dict[str, dict[str, Any]],
    bootstrap_summary: dict[str, Any],
    figure_paths: list[str],
    test_duplicate_rows: int,
) -> str:
    stability_columns = [
        "model",
        "unique_configurations",
        "winner_pr_auc",
        "second_pr_auc",
        "fifth_pr_auc",
        "tenth_pr_auc",
        "top5_pr_auc_range",
        "top10_pr_auc_range",
        "within_0.0001",
        "within_0.0005",
        "within_0.0010",
    ]
    efficiency_columns = [
        "model",
        "round1_fits",
        "round2_fits",
        "total_tuning_fits",
        "round1_logged_fit_seconds",
        "round2_logged_fit_seconds",
        "final_training_runtime_seconds",
        "joblib_size_bytes",
        "native_json_size_bytes",
        "validation_pr_auc",
        "test_pr_auc",
        "test_roc_auc",
        "test_f1",
        "test_precision",
        "test_recall",
    ]
    bootstrap_columns = ["model", "metric", "observed", "ci_lower_2_5", "ci_upper_97_5"]
    paired_columns = [
        "metric",
        "observed_xgboost_minus_rf",
        "bootstrap_median_difference",
        "ci_lower_2_5",
        "ci_upper_97_5",
    ]
    rf_effects = pd.DataFrame(
        [
            {"parameter": p, **values}
            for p, values in rf_controlled.items()
        ]
    )
    xgb_effects = pd.DataFrame(
        [
            {"parameter": p, **values}
            for p, values in xgb_controlled.items()
        ]
    )
    rf_n_estimators = rf_sensitivity[
        (rf_sensitivity["parameter"] == "n_estimators")
        & (rf_sensitivity["comparison_type"] == "controlled_one_factor")
    ].copy()
    inference_summary = inference.groupby("model", as_index=False).agg(
        median_seconds=("seconds", "median"),
        mean_seconds=("seconds", "mean"),
        total_seconds=("seconds", "sum"),
        median_rows_per_second=("rows_per_second", "median"),
    )
    duplicate_lines = []
    for model, report in duplicate_reports.items():
        duplicate_lines.append(
            f"- **{model}:** {report['raw_fit_rows']} raw fit rows, "
            f"{report['unique_full_configurations']} unique full configurations, "
            f"{report['duplicate_configuration_groups']} duplicate configuration groups, "
            f"{report['duplicate_refit_occurrences_beyond_first']} extra refit occurrences; "
            f"duplicate metrics reproducible to logged precision: `"
            f"{report['duplicate_metrics_reproducible_to_csv_precision']}`; "
            f"maximum logged metric range: `{report['maximum_duplicate_metric_range']:.6f}`."
        )
    lr_table = pd.DataFrame(lr_summary)
    return f"""# Advanced Random Forest and XGBoost Analysis

## Purpose

This is post-training analysis of the finalized Random Forest and XGBoost
experiments. It does not retune, reset thresholds, retrain models, or alter
the shared data foundation. Validation tuning outputs and frozen-test
descriptive outputs are kept separate throughout.

## Data and model status

Both models use the corrected shared splits and the transform-only
`artifacts/preprocess_tree.joblib` artifact, producing 39 tree features. The
saved final prediction files were not overwritten. The saved test set contains
`{test_duplicate_rows}` internally duplicated predictor rows; the primary
bootstrap therefore treats test rows as the sampling units and does not
deduplicate the official test set.

## Duplicate configuration handling

Stability counts use one representative row per unique full hyperparameter
configuration. Repeated staged/refit rows are not counted as separate
candidates:

{chr(10).join(duplicate_lines)}

Duplicate metrics and XGBoost `best_iteration` values were checked for
reproducibility. Sensitivity summaries retain stage context; Round 2 matched
comparisons are reported separately as controlled one-factor comparisons.

## Random Forest sensitivity

The RF analysis uses the 31 Round 1 rows and 48 Round 2 rows. The Round 2
Cartesian search supports matched comparisons in which one parameter changes
while the other Round 2 parameters are held fixed. These effects are
descriptive within the searched region and are not claimed to be causal
outside that design.

Controlled Round 2 mean PR-AUC changes (higher level minus lower level):

{format_table(rf_effects, ['parameter', 'levels', 'paired_comparisons', 'mean_high_minus_low_pr_auc', 'min_high_minus_low_pr_auc', 'max_high_minus_low_pr_auc']) if not rf_effects.empty else 'No controlled effects available.'}

The matched RF `n_estimators` comparisons are shown separately because the
Round 2 grid has three estimator levels:

{format_table(rf_n_estimators, ['level', 'paired_delta_mean_pr_auc', 'min_val_pr_auc', 'max_val_pr_auc']) if not rf_n_estimators.empty else 'No estimator comparisons available.'}

The selected RF depth is inside the broader searched depth set but is at the
lower boundary of the corrected-data Round 2 depth values. Nearby high-ranked
configurations and the stability table below should be read as evidence about
the observed plateau, not as a global optimum claim. `n_estimators` and the
other parameters are also summarized in
`rf_hyperparameter_sensitivity.csv`, with staged aggregate rows clearly
labelled as descriptive.

## XGBoost sensitivity

XGBoost uses 49 Round 1 fits and 64 Round 2 fits. The Round 2 grid jointly
varied depth, child weight, subsampling, column sampling, L2 regularization,
and L1 regularization at `learning_rate=0.1`. The controlled effects are:

{format_table(xgb_effects, ['parameter', 'levels', 'paired_comparisons', 'mean_high_minus_low_pr_auc', 'min_high_minus_low_pr_auc', 'max_high_minus_low_pr_auc']) if not xgb_effects.empty else 'No controlled effects available.'}

Round 1 learning-rate observations:

{format_table(lr_table, ['learning_rate', 'n_rows', 'mean_best_iteration', 'min_best_iteration', 'max_best_iteration', 'mean_val_pr_auc']) if not lr_table.empty else 'No learning-rate stage rows available.'}

Lower learning rates required more boosting iterations in this early-stopping
search, while the relationship is an empirical pattern over the tested
settings rather than an exact law. The strongest Round 2 region is concentrated
around depth 8–10, child weight 1, full subsampling, and column sampling 0.6–0.8.
The exact Round 1 winner was reproduced as the Round 2 winner, which supports
selection stability but does not establish global optimality.

## Model-selection stability

{format_table(stability, stability_columns)}

The `within_*` columns count unique configurations only. The top-five and
top-ten ranges are validation PR-AUC ranges, not confidence intervals. These
differences should not be interpreted as statistical significance.

## Computational efficiency and model size

{format_table(efficiency, efficiency_columns, digits=3)}

Model serialization size is a disk-size measurement, not a memory-footprint
measurement. The inference benchmark below times only `predict_proba` after
two warm-up calls on the same transformed test matrix, using ten repetitions
per model on this machine.

{format_table(inference_summary, ['model', 'median_seconds', 'mean_seconds', 'total_seconds', 'median_rows_per_second'], digits=4)}

These timings are local engineering measurements and may vary with laptop
load, threading, and library/runtime state.

## Bootstrap uncertainty on finalized test predictions

The analysis uses `{bootstrap_summary['replicates_requested']}` paired row-wise
bootstrap replicates with seed `{bootstrap_summary['seed']}`. The same sampled
test-row indices were applied to RF and XGBoost in each replicate. Valid
replicates: `{bootstrap_summary['replicates_valid']}`; skipped one-class
replicates: `{bootstrap_summary['replicates_skipped_one_class']}`.

### Individual model intervals

{format_table(bootstrap_metrics, bootstrap_columns)}

### Paired differences: XGBoost minus Random Forest

{format_table(paired, paired_columns)}

An interval that includes zero does not support a clear directional difference
under this paired bootstrap; an interval excluding zero is still an empirical
uncertainty result, not a license for test-based retuning. Because the frozen
UNSW-NB15 test set contains internal duplicate predictor rows, ordinary row
bootstrap treats duplicate rows as independent. A grouped/unique-vector
bootstrap could be a later sensitivity analysis, but it is not silently
substituted for this primary result.

## RF versus XGBoost interpretation

On the saved default-threshold outputs, XGBoost has higher test PR-AUC,
ROC-AUC, accuracy, precision, and F1, while RF has higher recall. This means
XGBoost currently appears stronger for ranking/separation and the fixed 0.5
operating point, whereas RF catches slightly more attacks with more false
positives. The project question— which model best balances malicious-traffic
detection against false-positive alerts—cannot be finalized solely from this
default-threshold snapshot. The standardized threshold analysis assigned to
the downstream workflow remains the appropriate basis for the eventual
operating-threshold decision.

## Figures

Generated figures:

{chr(10).join(f'- `{path}`' for path in figure_paths)}

## Limitations and handoff

- Sensitivity results are limited to the searched configurations and preserve
  staged-search context; pooled associations are not isolated causal effects.
- Stability differences are descriptive validation gaps, not significance
  tests.
- Runtime and serialization sizes are machine/runtime-specific.
- Test bootstrap intervals are post-hoc descriptive uncertainty estimates and
  inherit the frozen test set's internal duplicate-row limitation.
- Threshold optimization, FP/FN case analysis, SHAP/feature-importance
  interpretation, calibration, drift, TTL ablation, Logistic Regression, and
  Neural Network analysis remain deferred to the assigned workflows.

## Reproducibility

The analysis runner is `src/08_rf_xgboost_analysis.py`. It uses Python/library
versions recorded in `experiments/model_analysis/analysis_summary.json`,
`random_state=42`, and bootstrap seed 42. It reads the existing tuning logs,
saved models, saved predictions, and preprocessing artifact without fitting
or changing them.
"""


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rf_paths = [
        RF_DIR / "tuning_results.csv",
        RF_DIR / "round2_joint_search.csv",
        RF_DIR / "validation_predictions.csv",
        RF_DIR / "test_predictions.csv",
        ARTIFACTS_DIR / "random_forest.joblib",
    ]
    xgb_paths = [
        XGB_DIR / "tuning_results.csv",
        XGB_DIR / "round2_joint_search.csv",
        XGB_DIR / "validation_predictions.csv",
        XGB_DIR / "test_predictions.csv",
        ARTIFACTS_DIR / "xgboost.joblib",
        XGB_DIR / "xgboost_model.json",
    ]
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in rf_paths + xgb_paths}

    rf_all, xgb_all = load_tuning_logs()
    rf_representatives, rf_duplicates = duplicate_report(
        rf_all, "Random Forest", ["val_pr_auc", "val_roc_auc", "val_f1"]
    )
    xgb_representatives, xgb_duplicates = duplicate_report(
        xgb_all,
        "XGBoost",
        ["val_pr_auc", "val_roc_auc", "val_f1", "best_iteration"],
    )
    stability = pd.DataFrame(
        [
            stability_row("Random Forest", rf_all, rf_representatives, rf_duplicates),
            stability_row("XGBoost", xgb_all, xgb_representatives, xgb_duplicates),
        ]
    )
    rf_sensitivity, rf_controlled = sensitivity_analysis(
        rf_all, "Random Forest", RF_CONFIG_COLUMNS
    )
    xgb_sensitivity, xgb_controlled = sensitivity_analysis(
        xgb_all, "XGBoost", XGB_CONFIG_COLUMNS
    )
    rf_sensitivity.to_csv(ANALYSIS_DIR / "rf_hyperparameter_sensitivity.csv", index=False)
    xgb_sensitivity.to_csv(ANALYSIS_DIR / "xgboost_hyperparameter_sensitivity.csv", index=False)
    stability.to_csv(ANALYSIS_DIR / "model_selection_stability.csv", index=False)

    predictions, test_metrics = load_saved_predictions()
    inference, inference_summary = run_inference_benchmark()
    inference.to_csv(ANALYSIS_DIR / "inference_benchmark.csv", index=False)

    rf_config = json.loads((RF_DIR / "config.json").read_text())
    xgb_config = json.loads((XGB_DIR / "config.json").read_text())
    rf_metrics = json.loads((RF_DIR / "metrics.json").read_text())
    xgb_metrics = json.loads((XGB_DIR / "metrics.json").read_text())
    efficiency_rows = []
    for model, directory, config, metrics, joblib_path, native_path in [
        (
            "Random Forest",
            RF_DIR,
            rf_config,
            rf_metrics,
            ARTIFACTS_DIR / "random_forest.joblib",
            None,
        ),
        (
            "XGBoost",
            XGB_DIR,
            xgb_config,
            xgb_metrics,
            ARTIFACTS_DIR / "xgboost.joblib",
            XGB_DIR / "xgboost_model.json",
        ),
    ]:
        round1 = pd.read_csv(directory / "tuning_results.csv")
        round2 = pd.read_csv(directory / "round2_joint_search.csv")
        model_test = metrics["test_reference_only"]
        model_val = metrics["validation"]
        efficiency_rows.append(
            {
                "model": model,
                "round1_fits": len(round1),
                "round2_fits": len(round2),
                "total_tuning_fits": len(round1) + len(round2),
                "round1_logged_fit_seconds": float(round1["fit_seconds"].sum()),
                "round2_logged_fit_seconds": float(round2["fit_seconds"].sum()),
                "round2_wall_seconds_recorded": None,
                "final_training_runtime_seconds": config.get(
                    "training_runtime_seconds", config.get("final_training_runtime_seconds")
                ),
                "joblib_size_bytes": joblib_path.stat().st_size,
                "native_json_size_bytes": native_path.stat().st_size if native_path else None,
                "validation_pr_auc": model_val["pr_auc"],
                "validation_roc_auc": model_val["roc_auc"],
                "test_pr_auc": model_test["pr_auc"],
                "test_roc_auc": model_test["roc_auc"],
                "test_f1": model_test["f1"],
                "test_precision": model_test["precision"],
                "test_recall": model_test["recall"],
            }
        )
    efficiency = pd.DataFrame(efficiency_rows)
    efficiency.to_csv(ANALYSIS_DIR / "efficiency_comparison.csv", index=False)

    bootstrap_metrics, paired, bootstrap_summary = run_bootstrap(predictions)
    bootstrap_metrics.to_csv(ANALYSIS_DIR / "bootstrap_metrics.csv", index=False)
    paired.to_csv(ANALYSIS_DIR / "bootstrap_paired_differences.csv", index=False)

    test_x = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    test_duplicate_rows = int(test_x.duplicated().sum())

    lr_frame = pd.read_csv(XGB_DIR / "tuning_results.csv")
    lr_frame = lr_frame[lr_frame["stage"] == "learning_rate"]
    lr_summary = []
    for learning_rate, group in lr_frame.groupby("learning_rate", sort=True):
        lr_summary.append(
            {
                "learning_rate": float(learning_rate),
                "n_rows": len(group),
                "mean_best_iteration": float(group["best_iteration"].mean()),
                "min_best_iteration": int(group["best_iteration"].min()),
                "max_best_iteration": int(group["best_iteration"].max()),
                "mean_val_pr_auc": float(group["val_pr_auc"].mean()),
            }
        )

    figure_paths = make_figures(
        rf_representatives,
        xgb_representatives,
        rf_sensitivity,
        xgb_sensitivity,
        stability,
        efficiency,
        bootstrap_metrics,
        paired,
    )
    report = make_report(
        stability,
        efficiency,
        inference,
        bootstrap_metrics,
        paired,
        rf_sensitivity,
        xgb_sensitivity,
        rf_controlled,
        xgb_controlled,
        lr_summary,
        {"Random Forest": rf_duplicates, "XGBoost": xgb_duplicates},
        bootstrap_summary,
        figure_paths,
        test_duplicate_rows,
    )
    (ROOT / "docs" / "rf_xgboost_advanced_analysis.md").write_text(report)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": platform.platform(),
        "python_version": platform.python_version(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "random_state": RANDOM_STATE,
        "bootstrap": bootstrap_summary,
        "test_internal_duplicate_rows": test_duplicate_rows,
        "duplicate_configuration_reports": {
            "Random Forest": rf_duplicates,
            "XGBoost": xgb_duplicates,
        },
        "stability": stability.to_dict(orient="records"),
        "controlled_round2_effects": {
            "Random Forest": rf_controlled,
            "XGBoost": xgb_controlled,
        },
        "learning_rate_best_iteration": lr_summary,
        "test_metrics_recomputed": test_metrics,
        "inference_summary": inference_summary,
        "efficiency": efficiency.to_dict(orient="records"),
        "figures": figure_paths,
        "input_hashes_before_analysis": input_hashes,
        "outputs": [
            "experiments/model_analysis/rf_hyperparameter_sensitivity.csv",
            "experiments/model_analysis/xgboost_hyperparameter_sensitivity.csv",
            "experiments/model_analysis/model_selection_stability.csv",
            "experiments/model_analysis/efficiency_comparison.csv",
            "experiments/model_analysis/inference_benchmark.csv",
            "experiments/model_analysis/bootstrap_metrics.csv",
            "experiments/model_analysis/bootstrap_paired_differences.csv",
            "experiments/model_analysis/analysis_summary.json",
            "docs/rf_xgboost_advanced_analysis.md",
            *figure_paths,
        ],
        "safety": {
            "models_retrained": False,
            "test_based_retuning": False,
            "saved_predictions_overwritten": False,
            "preprocessing_fitted": False,
            "datasets_modified": False,
        },
    }
    write_json(ANALYSIS_DIR / "analysis_summary.json", summary)
    print(json.dumps({
        "stability": stability.to_dict(orient="records"),
        "bootstrap": bootstrap_summary,
        "inference": inference_summary,
        "test_internal_duplicate_rows": test_duplicate_rows,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
