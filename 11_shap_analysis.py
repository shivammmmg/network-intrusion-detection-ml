"""SHAP explainability for XGBoost and Random Forest.

This script creates bar plots, beeswarm plots, and waterfall plots for SHAP values of the trained models.

Outputs are saved to experiments/explainability_and_diagnostics/shap_analysis/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

ROOT_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_FOLDER / "src"))

from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE
from preprocess import load_artifact

OUTPUT_DIR = ROOT_FOLDER / "experiments" / "explainability_and_diagnostics" /"shap_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocess_tree.joblib"
RF_MODEL_PATH = ARTIFACTS_DIR / "random_forest.joblib"
XGB_MODEL_PATH = ARTIFACTS_DIR / "xgboost.joblib"

THRESHOLDS_PATH = ROOT_FOLDER / "experiments" / "standardized_evaluation" / "selected_thresholds.json"
locked_thresholds = pd.read_json(THRESHOLDS_PATH, typ="series")

N_BACKGROUND = 100


def load_data():
    """Loads the train/test data, apply the saved preprocessor, and return the transformed numeric feature matrices and labels."""
    X_train_raw = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["label"].astype(int)
    X_test_raw = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["label"].astype(int)

    pre = load_artifact(PREPROCESSOR_PATH)
    X_train = pre.transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    print(f"Transformed train shape: {X_train.shape}")
    print(f"Transformed test shape:  {X_test.shape}")
    feature_names = list(X_train.columns)
    return X_train, y_train, X_test, y_test, feature_names


def load_models():
    """Loads the trained Random Forest and XGBoost models from disk."""
    rf = joblib.load(RF_MODEL_PATH)
    xgb = joblib.load(XGB_MODEL_PATH)
    return rf, xgb


def compute_shap_values(model, X_background_pool, X_sample, model_name):
    """Computes SHAP values with TreeExplainer for a given model and sample of data.
    The background reference is drawn from `X_background_pool` (supposed to be
    training data) since the background should represent the training distribution rather
    than the exact points being explained. Returns a shap.Explanation object.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_background = min(N_BACKGROUND, len(X_background_pool))
    bg_indices = rng.choice(len(X_background_pool), size=n_background, replace=False)
    X_background = X_background_pool.iloc[bg_indices]

    print(f"Building TreeExplainer for {model_name} with {n_background} background samples...")

    try:
        explainer = shap.TreeExplainer(
            model,
            X_background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        shap_values = explainer(X_sample, check_additivity=False)
        output_space = "probability"
    except Exception as exc:
        # Fallback for models that do not support interventional/probability mode
        print(f"  {model_name}: interventional/probability mode unsupported "
              f"({type(exc).__name__}: {exc}). Falling back to "
              f"feature_perturbation='tree_path_dependent', model_output='raw' ")
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
        shap_values = explainer(X_sample, check_additivity=False)
        output_space = "raw (log-odds)"

    values = np.asarray(shap_values.values)
    base_values = np.asarray(shap_values.base_values)

    # Extracts only the positive class so the plotting functions work.
    if values.ndim == 3:
        class_idx = 1 if values.shape[2] > 1 else 0
        values = values[:, :, class_idx]
        if base_values.ndim == 2:
            base_values = base_values[:, class_idx]

    # Ensures the waterfall plotting function works if the baseline is a single scalar.
    if base_values.ndim == 0:
        base_values = np.full(values.shape[0], float(base_values))

    feature_names = X_sample.columns.tolist()
    shap_values = shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_sample.values,
        feature_names=feature_names,
    )

    print(f"  SHAP values shape: {shap_values.values.shape}")
    print(f"  Base values shape: {np.asarray(shap_values.base_values).shape}")
    print(f"  SHAP value units: {output_space}")
    return shap_values


def plot_global_bar(shap_values, model_name, output_path, max_display=20):
    """Plots a global bar plot with feature importance."""
    n_features = shap_values.values.shape[1]
    max_display = min(max_display, n_features)

    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"Global feature importance (top {max_display}) - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")
    print(f"  Global bar saved: {output_path}")


def plot_beeswarm(shap_values, model_name, output_path, max_display=20):
    """Plots a beeswarm summary plot of SHAP values."""
    n_features = shap_values.values.shape[1]
    max_display = min(max_display, n_features)

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP beeswarm - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")
    print(f"  Beeswarm saved: {output_path}")


def plot_waterfall(shap_values, instance_idx, model_name, output_path):
    """Plots a waterfall plot for a single instance using shap.plots.waterfall."""
    n_samples = shap_values.values.shape[0]
    if not (0 <= instance_idx < n_samples):
        print(f"  Warning: instance {instance_idx} out of range (0-{n_samples-1}); using 0 instead.")
        instance_idx = 0

    instance_exp = shap_values[instance_idx]

    shap.plots.waterfall(instance_exp, show=False)
    plt.title(f"{model_name} - instance {instance_idx}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")
    print(f"  Waterfall saved: {output_path}")


def main():
    """Loads the data and models, computes SHAP values, and generates the three plots for both Random Forest and XGBoost."""
    X_train, y_train, X_test, y_test, feature_names = load_data()

    sample_size = min(500, len(X_test))
    X_test_sample = X_test.iloc[:sample_size].copy()
    y_test_sample = y_test.iloc[:sample_size].copy()

    rf_model, xgb_model = load_models()
    models = {"Random Forest": rf_model, "XGBoost": xgb_model}

    for display_name, model in models.items():
        model_key = "random_forest" if "Random Forest" in display_name else "xgboost"
        print(f"\n{display_name}: ")

        shap_values = compute_shap_values(model, X_train, X_test_sample, display_name)

        imp_path = FIGURES_DIR / f"{model_key}_shap_importance.png"
        plot_global_bar(shap_values, display_name, imp_path)

        beeswarm_path = FIGURES_DIR / f"{model_key}_shap_beeswarm.png"
        plot_beeswarm(shap_values, display_name, beeswarm_path)

        y_sample = y_test_sample.values
        threshold = float(locked_thresholds[model_key])
        proba_sample = model.predict_proba(X_test_sample)[:, 1]
        pred_sample = (proba_sample >= threshold).astype(int)
        print(f"  Using locked threshold for {display_name}: {threshold:.4f}")
        # These four groups are chosen to diagnose why the model gets some cases right and others wrong.
        labels = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}
        positions = {}
        for name, (true, pred) in labels.items():
            pos_list = np.where((y_sample == true) & (pred_sample == pred))[0]
            if len(pos_list) > 0:
                positions[name] = pos_list[0]
            else:
                # If there is no case, skip the waterfall plot to prevent mislabelling.
                print(f"  No {name} case found in sample of size {sample_size}; skipping its waterfall plot.")

        # Generates a waterfall plot for Random Forest and XGBoost for each of the four cases found.
        for name, pos in positions.items():
            true = y_sample[pos]
            pred = pred_sample[pos]
            out_path = FIGURES_DIR / f"{model_key}_waterfall_{name}.png"
            plot_waterfall(
                shap_values,
                pos,
                f"{display_name} (true={true}, pred={pred})",
                out_path,
            )

    print("\nSHAP analysis is finished. All figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()