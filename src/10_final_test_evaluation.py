"""Final frozen-test evaluation using validation-selected locked thresholds.

This script performs no training, tuning, threshold search, or preprocessing.
It evaluates saved test probabilities at threshold 0.50 and at thresholds read
directly from the committed validation-only selected_thresholds.json file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Use a writable cache when the user's default Matplotlib directory is read-only.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "network_intrusion_detection_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT_FOLDER = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = ROOT_FOLDER / "experiments" / "standardized_evaluation"
THRESHOLDS_PATH = STANDARDIZED_DIR / "selected_thresholds.json"
OUTPUT_DIR = STANDARDIZED_DIR / "final_test"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODEL_FILES = {
    "logistic_regression": ROOT_FOLDER
    / "experiments"
    / "logistic_regression"
    / "test_predictions.csv",
    "neural_network": ROOT_FOLDER
    / "experiments"
    / "neural_network"
    / "test_predictions.csv",
    "random_forest": ROOT_FOLDER
    / "experiments"
    / "random_forest"
    / "test_predictions.csv",
    "xgboost": ROOT_FOLDER
    / "experiments"
    / "xgboost"
    / "test_predictions.csv",
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "neural_network": "Neural Network",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

REQUIRED_COLUMNS = {
    "sample_index",
    "true_label",
    "attack_probability",
    "predicted_label",
}
EXPECTED_TEST_ROWS = 82_332
DEFAULT_THRESHOLD = 0.50
DELTA_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_positives",
    "false_negatives",
    "false_positive_rate",
    "false_negative_rate",
]


def load_locked_thresholds(path: Path) -> dict[str, float]:
    """Load the already-selected validation thresholds without recalculating them."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing locked threshold file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Locked threshold file is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Locked threshold file must contain a JSON object")

    expected_models = set(MODEL_FILES)
    actual_models = set(payload)
    if actual_models != expected_models:
        missing = sorted(expected_models.difference(actual_models))
        unexpected = sorted(actual_models.difference(expected_models))
        raise ValueError(
            f"Locked threshold model keys do not match; missing={missing}, "
            f"unexpected={unexpected}"
        )

    thresholds: dict[str, float] = {}
    for model in MODEL_FILES:
        value = payload[model]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Locked threshold for {model} must be numeric")
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Locked threshold for {model} must be between 0 and 1")
        thresholds[model] = threshold
    return thresholds


def load_test_predictions(model: str, path: Path) -> pd.DataFrame:
    """Load and validate one model's frozen test predictions."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing test predictions for {model}: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Test predictions for {model} are missing columns: {sorted(missing)}"
        )

    frame = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column {column!r} contains non-numeric values for {model}"
            ) from error

    numeric_values = frame.loc[:, sorted(REQUIRED_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"Test predictions contain missing or infinite values for {model}")

    for column in ["sample_index", "true_label", "predicted_label"]:
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Column {column!r} must contain integers for {model}")
        frame[column] = values.astype(np.int64)
    frame["attack_probability"] = frame["attack_probability"].astype(float)

    if len(frame) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS:,} test rows for {model}, found {len(frame):,}"
        )
    if not frame["sample_index"].is_unique:
        duplicate_count = int(frame["sample_index"].duplicated().sum())
        raise ValueError(f"Found {duplicate_count} duplicate sample indexes for {model}")

    for column in ["true_label", "predicted_label"]:
        invalid = sorted(set(frame[column].unique()).difference({0, 1}))
        if invalid:
            raise ValueError(f"Column {column!r} has non-binary values for {model}: {invalid}")

    if not frame["attack_probability"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"Attack probabilities must be between 0 and 1 for {model}")

    return frame


def validate_test_alignment(predictions: dict[str, pd.DataFrame]) -> None:
    """Verify that every model has the same ordered test samples and labels."""

    reference_model = next(iter(predictions))
    reference = predictions[reference_model]
    if set(reference["true_label"].unique()) != {0, 1}:
        raise ValueError("Test labels must contain both class 0 and class 1")

    for model, frame in predictions.items():
        if len(frame) != EXPECTED_TEST_ROWS:
            raise ValueError(f"Unexpected test row count for {model}: {len(frame):,}")
        if not np.array_equal(
            frame["sample_index"].to_numpy(), reference["sample_index"].to_numpy()
        ):
            raise ValueError(f"Test sample indexes are not aligned in order for {model}")
        if not np.array_equal(
            frame["true_label"].to_numpy(), reference["true_label"].to_numpy()
        ):
            raise ValueError(f"Test true labels are not aligned in order for {model}")


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate test metrics with attack label 1 as the positive class."""

    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    false_negative_rate = float(fn / (fn + tp)) if fn + tp else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(
            precision_score(y_true, predicted, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def make_delta_table(
    default_comparison: pd.DataFrame, locked_comparison: pd.DataFrame
) -> pd.DataFrame:
    """Return locked-minus-default changes for the requested metrics."""

    default_indexed = default_comparison.set_index("model")
    locked_indexed = locked_comparison.set_index("model")
    rows: list[dict[str, float | int | str]] = []
    for model in MODEL_FILES:
        row: dict[str, float | int | str] = {"model": model}
        row["default_threshold"] = float(default_indexed.loc[model, "threshold"])
        row["locked_threshold"] = float(locked_indexed.loc[model, "locked_threshold"])
        for metric in DELTA_METRICS:
            delta = locked_indexed.loc[model, metric] - default_indexed.loc[model, metric]
            row[f"delta_{metric}"] = int(delta) if metric in {
                "false_positives",
                "false_negatives",
            } else float(delta)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_roc_curves(predictions: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for model, frame in predictions.items():
        y_true = frame["true_label"].to_numpy()
        probabilities = frame["attack_probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, probabilities, pos_label=1)
        auc = roc_auc_score(y_true, probabilities)
        axis.plot(fpr, tpr, linewidth=2, label=f"{DISPLAY_NAMES[model]} (AUC={auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
    axis.set(title="Frozen Test ROC Curves", xlabel="False-Positive Rate", ylabel="True-Positive Rate")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curves(
    predictions: dict[str, pd.DataFrame], path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for model, frame in predictions.items():
        y_true = frame["true_label"].to_numpy()
        probabilities = frame["attack_probability"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, probabilities, pos_label=1)
        average_precision = average_precision_score(y_true, probabilities)
        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{DISPLAY_NAMES[model]} (AP={average_precision:.4f})",
        )
    prevalence = predictions[next(iter(predictions))]["true_label"].mean()
    axis.axhline(prevalence, linestyle="--", color="gray", linewidth=1, label="Prevalence")
    axis.set(
        title="Frozen Test Precision-Recall Curves",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    model: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    predicted = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        title=f"{DISPLAY_NAMES[model]} Frozen Test Confusion Matrix\nLocked threshold={threshold:.6g}",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal (0)", "Attack (1)"],
        yticklabels=["Normal (0)", "Attack (1)"],
    )
    cutoff = matrix.max() / 2.0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > cutoff else "black",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_locked_metric_comparison(comparison: pd.DataFrame, path: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    x_positions = np.arange(len(comparison))
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 6))
    for offset, metric in enumerate(metrics):
        axis.bar(
            x_positions + (offset - 1) * width,
            comparison[metric],
            width,
            label=metric.capitalize(),
        )
    axis.set(
        title="Locked-Threshold Frozen Test Metrics",
        ylabel="Score",
        ylim=(0, 1),
        xticks=x_positions,
        xticklabels=[DISPLAY_NAMES[model] for model in comparison["model"]],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_count_comparison(comparison: pd.DataFrame, path: Path) -> None:
    x_positions = np.arange(len(comparison))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        x_positions - width / 2,
        comparison["false_positives"],
        width,
        label="False positives",
    )
    axis.bar(
        x_positions + width / 2,
        comparison["false_negatives"],
        width,
        label="False negatives",
    )
    axis.set(
        title="Locked-Threshold Frozen Test Error Counts",
        ylabel="Count",
        xticks=x_positions,
        xticklabels=[DISPLAY_NAMES[model] for model in comparison["model"]],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_default_vs_locked(
    default_comparison: pd.DataFrame,
    locked_comparison: pd.DataFrame,
    path: Path,
) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    x_positions = np.arange(len(default_comparison))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for axis, metric in zip(axes.flat, metrics):
        axis.bar(
            x_positions - width / 2,
            default_comparison[metric],
            width,
            label="Default 0.50",
        )
        axis.bar(
            x_positions + width / 2,
            locked_comparison[metric],
            width,
            label="Locked validation threshold",
        )
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [DISPLAY_NAMES[model] for model in default_comparison["model"]],
            rotation=15,
            ha="right",
        )
    axes.flat[0].legend()
    fig.suptitle("Default vs. Locked Thresholds on the Frozen Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def identify_strongest_model(locked_comparison: pd.DataFrame) -> str:
    """Use F1 as the primary balance metric; PR-AUC and recall break ties."""

    ranked = locked_comparison.sort_values(
        [
            "f1",
            "pr_auc",
            "recall",
            "precision",
            "false_positive_rate",
            "false_negative_rate",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["model"])


def write_summary(
    thresholds: dict[str, float],
    mismatch_counts: dict[str, int],
    default_comparison: pd.DataFrame,
    locked_comparison: pd.DataFrame,
    deltas: pd.DataFrame,
    strongest_model: str,
) -> None:
    payload = {
        "methodology": {
            "positive_class": 1,
            "default_threshold": DEFAULT_THRESHOLD,
            "locked_threshold_source": str(THRESHOLDS_PATH.relative_to(ROOT_FOLDER)),
            "threshold_selection_data": "validation_only",
            "test_threshold_optimization_performed": False,
            "training_or_refitting_performed": False,
        },
        "test_alignment": {
            "passed": True,
            "model_count": len(MODEL_FILES),
            "rows_per_model": EXPECTED_TEST_ROWS,
            "sample_index_order_aligned": True,
            "true_label_order_aligned": True,
        },
        "locked_thresholds": thresholds,
        "saved_label_mismatch_counts_at_0_50": mismatch_counts,
        "default_threshold_results": records_for_json(default_comparison),
        "locked_threshold_results": records_for_json(locked_comparison),
        "locked_minus_default_deltas": records_for_json(deltas),
        "strongest_model": {
            "model": strongest_model,
            "basis": (
                "Highest locked-threshold F1, with PR-AUC, recall, precision, "
                "false-positive rate, and false-negative rate considered; "
                "accuracy was not used alone."
            ),
        },
        "statistical_significance_claim_made": False,
    }
    (OUTPUT_DIR / "final_test_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    thresholds = load_locked_thresholds(THRESHOLDS_PATH)
    print(f"Locked validation thresholds loaded from: {THRESHOLDS_PATH}")
    for model, threshold in thresholds.items():
        print(f"  {DISPLAY_NAMES[model]}: {threshold:.16g}")
    print("No test threshold optimization will be performed.")

    predictions = {
        model: load_test_predictions(model, path)
        for model, path in MODEL_FILES.items()
    }
    validate_test_alignment(predictions)
    print(
        f"Test alignment passed for all {len(predictions)} models "
        f"({EXPECTED_TEST_ROWS:,} rows each)."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mismatch_counts: dict[str, int] = {}
    default_rows: list[dict[str, float | int | str]] = []
    locked_rows: list[dict[str, float | int | str]] = []
    for model, frame in predictions.items():
        y_true = frame["true_label"].to_numpy()
        probabilities = frame["attack_probability"].to_numpy()
        saved_predictions = frame["predicted_label"].to_numpy()
        default_predictions = (probabilities >= DEFAULT_THRESHOLD).astype(np.int64)
        mismatch_count = int(np.count_nonzero(saved_predictions != default_predictions))
        mismatch_counts[model] = mismatch_count
        print(
            f"{DISPLAY_NAMES[model]}: saved predicted_label mismatches at "
            f"threshold 0.50 = {mismatch_count}"
        )

        default_rows.append(
            {
                "model": model,
                "threshold": DEFAULT_THRESHOLD,
                **classification_metrics(y_true, probabilities, DEFAULT_THRESHOLD),
                "saved_predicted_label_mismatch_count": mismatch_count,
            }
        )
        locked_threshold = thresholds[model]
        locked_rows.append(
            {
                "model": model,
                "locked_threshold": locked_threshold,
                **classification_metrics(y_true, probabilities, locked_threshold),
            }
        )
        plot_confusion_matrix(
            model,
            y_true,
            probabilities,
            locked_threshold,
            FIGURES_DIR / f"{model}_test_locked_threshold_confusion_matrix.png",
        )

    default_comparison = pd.DataFrame(default_rows)
    locked_comparison = pd.DataFrame(locked_rows)
    deltas = make_delta_table(default_comparison, locked_comparison)
    strongest_model = identify_strongest_model(locked_comparison)

    default_comparison.to_csv(
        OUTPUT_DIR / "test_default_threshold_comparison.csv", index=False
    )
    locked_comparison.to_csv(
        OUTPUT_DIR / "test_locked_threshold_comparison.csv", index=False
    )
    deltas.to_csv(OUTPUT_DIR / "default_vs_locked_threshold_deltas.csv", index=False)
    write_summary(
        thresholds,
        mismatch_counts,
        default_comparison,
        locked_comparison,
        deltas,
        strongest_model,
    )

    plot_roc_curves(predictions, FIGURES_DIR / "test_roc_curves.png")
    plot_precision_recall_curves(
        predictions, FIGURES_DIR / "test_precision_recall_curves.png"
    )
    plot_locked_metric_comparison(
        locked_comparison,
        FIGURES_DIR / "locked_threshold_model_metric_comparison.png",
    )
    plot_error_count_comparison(
        locked_comparison,
        FIGURES_DIR / "false_positive_false_negative_comparison.png",
    )
    plot_default_vs_locked(
        default_comparison,
        locked_comparison,
        FIGURES_DIR / "default_vs_locked_threshold_comparison.png",
    )

    print("\nDefault-threshold frozen-test comparison:")
    print(default_comparison.to_string(index=False))
    print("\nLocked-threshold frozen-test comparison:")
    print(locked_comparison.to_string(index=False))
    print("\nLocked-minus-default threshold deltas:")
    print(deltas.to_string(index=False))
    print(
        f"\nStrongest frozen-test model: {DISPLAY_NAMES[strongest_model]} "
        "(selected by locked-threshold F1 with the full metric profile considered)."
    )
    print(f"Outputs saved under: {OUTPUT_DIR}")
    print(
        "Final frozen-test evaluation complete; thresholds came directly from "
        "validation and were not optimized on test data."
    )


if __name__ == "__main__":
    main()
