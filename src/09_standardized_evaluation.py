"""Standardized validation-only evaluation for all finalized model tracks.

This script reads saved validation predictions, checks that every model was
evaluated on the same ordered samples, compares the models at threshold 0.50,
and selects a separate F1-maximizing validation threshold for each model. It
does not train models or access frozen test predictions.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

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
OUTPUT_DIR = ROOT_FOLDER / "experiments" / "standardized_evaluation"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODEL_FILES = {
    "logistic_regression": ROOT_FOLDER
    / "experiments"
    / "logistic_regression"
    / "validation_predictions.csv",
    "neural_network": ROOT_FOLDER
    / "experiments"
    / "neural_network"
    / "validation_predictions.csv",
    "random_forest": ROOT_FOLDER
    / "experiments"
    / "random_forest"
    / "validation_predictions.csv",
    "xgboost": ROOT_FOLDER
    / "experiments"
    / "xgboost"
    / "validation_predictions.csv",
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
DEFAULT_THRESHOLD = 0.50


def load_validation_predictions(model: str, path: Path) -> pd.DataFrame:
    """Load and validate one model's saved validation predictions."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing validation predictions for {model}: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Validation predictions for {model} are missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column {column!r} contains non-numeric values for {model}"
            ) from error

    values = frame.loc[:, sorted(REQUIRED_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Validation predictions contain missing or infinite values for {model}")

    integer_columns = ["sample_index", "true_label", "predicted_label"]
    for column in integer_columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Column {column!r} must contain integers for {model}")
        frame[column] = values.astype(np.int64)
    frame["attack_probability"] = frame["attack_probability"].astype(float)

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


def validate_alignment(predictions: dict[str, pd.DataFrame]) -> None:
    """Verify row counts, sample order, and labels match across every model."""

    reference_model = next(iter(predictions))
    reference = predictions[reference_model]
    if set(reference["true_label"].unique()) != {0, 1}:
        raise ValueError("Validation labels must contain both class 0 and class 1")

    for model, frame in predictions.items():
        if len(frame) != len(reference):
            raise ValueError(
                f"Row-count mismatch: {reference_model} has {len(reference)} rows, "
                f"but {model} has {len(frame)}"
            )
        if not np.array_equal(
            frame["sample_index"].to_numpy(), reference["sample_index"].to_numpy()
        ):
            raise ValueError(f"Sample indexes are not aligned in order for {model}")
        if not np.array_equal(
            frame["true_label"].to_numpy(), reference["true_label"].to_numpy()
        ):
            raise ValueError(f"True labels are not aligned in order for {model}")


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate binary metrics with attack label 1 as the positive class."""

    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    false_negative_rate = float(fn / (fn + tp)) if fn + tp else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
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


def threshold_analysis(
    y_true: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    """Calculate precision, recall, and F1 at every distinct score threshold."""

    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities, pos_label=1
    )
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0.0,
    )
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def select_threshold(analysis: pd.DataFrame) -> pd.Series:
    """Select by F1, then precision, recall, and threshold, all descending."""

    if analysis.empty:
        raise ValueError("Cannot select a threshold from an empty analysis")
    ranked = analysis.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return ranked.iloc[0]


def plot_roc_curves(predictions: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for model, frame in predictions.items():
        y_true = frame["true_label"].to_numpy()
        probabilities = frame["attack_probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, probabilities, pos_label=1)
        auc = roc_auc_score(y_true, probabilities)
        axis.plot(fpr, tpr, linewidth=2, label=f"{DISPLAY_NAMES[model]} (AUC={auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
    axis.set(title="Validation ROC Curves", xlabel="False-Positive Rate", ylabel="True-Positive Rate")
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
        title="Validation Precision-Recall Curves",
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


def plot_threshold_curve(
    model: str, analysis: pd.DataFrame, selected_threshold: float, path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for metric in ["precision", "recall", "f1"]:
        axis.plot(analysis["threshold"], analysis[metric], label=metric.capitalize())
    axis.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Selected ({selected_threshold:.6g})",
    )
    axis.set(
        title=f"{DISPLAY_NAMES[model]} Validation Threshold Analysis",
        xlabel="Threshold",
        ylabel="Score",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.25)
    axis.legend()
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
        title=f"{DISPLAY_NAMES[model]} Validation Confusion Matrix\nThreshold={threshold:.6g}",
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = {
        model: load_validation_predictions(model, path)
        for model, path in MODEL_FILES.items()
    }
    validate_alignment(predictions)
    row_count = len(next(iter(predictions.values())))
    print(f"Validation alignment passed for all 4 models ({row_count:,} rows each).")

    default_rows: list[dict[str, float | int | str]] = []
    selected_rows: list[dict[str, float | int | str]] = []
    selected_thresholds: dict[str, float] = {}

    for model, frame in predictions.items():
        y_true = frame["true_label"].to_numpy()
        probabilities = frame["attack_probability"].to_numpy()
        saved_predictions = frame["predicted_label"].to_numpy()
        default_predictions = (probabilities >= DEFAULT_THRESHOLD).astype(np.int64)
        mismatch_count = int(np.count_nonzero(saved_predictions != default_predictions))

        default_rows.append(
            {
                "model": model,
                "threshold": DEFAULT_THRESHOLD,
                **classification_metrics(y_true, probabilities, DEFAULT_THRESHOLD),
                "saved_predicted_label_mismatch_count": mismatch_count,
            }
        )
        print(
            f"{DISPLAY_NAMES[model]}: saved predicted_label mismatches at "
            f"threshold 0.50 = {mismatch_count}"
        )

        analysis = threshold_analysis(y_true, probabilities)
        analysis_path = OUTPUT_DIR / f"{model}_validation_threshold_analysis.csv"
        analysis.to_csv(analysis_path, index=False)
        selected = select_threshold(analysis)
        selected_threshold = float(selected["threshold"])
        selected_thresholds[model] = selected_threshold
        selected_rows.append(
            {
                "model": model,
                "selected_threshold": selected_threshold,
                **classification_metrics(y_true, probabilities, selected_threshold),
            }
        )

        plot_threshold_curve(
            model,
            analysis,
            selected_threshold,
            FIGURES_DIR / f"{model}_validation_threshold_curve.png",
        )
        plot_confusion_matrix(
            model,
            y_true,
            probabilities,
            selected_threshold,
            FIGURES_DIR / f"{model}_validation_selected_threshold_confusion_matrix.png",
        )

    default_comparison = pd.DataFrame(default_rows)
    selected_comparison = pd.DataFrame(selected_rows)
    default_path = OUTPUT_DIR / "validation_default_threshold_comparison.csv"
    selected_path = OUTPUT_DIR / "validation_selected_threshold_comparison.csv"
    thresholds_path = OUTPUT_DIR / "selected_thresholds.json"
    default_comparison.to_csv(default_path, index=False)
    selected_comparison.to_csv(selected_path, index=False)
    thresholds_path.write_text(
        json.dumps(selected_thresholds, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plot_roc_curves(predictions, FIGURES_DIR / "validation_roc_curves.png")
    plot_precision_recall_curves(
        predictions, FIGURES_DIR / "validation_precision_recall_curves.png"
    )

    print("\nDefault-threshold validation comparison:")
    print(default_comparison.to_string(index=False))
    print("\nSelected-threshold validation comparison:")
    print(selected_comparison.to_string(index=False))
    print(f"\nOutputs saved under: {OUTPUT_DIR}")
    print("Validation-only evaluation complete; test files were not opened.")


if __name__ == "__main__":
    main()
