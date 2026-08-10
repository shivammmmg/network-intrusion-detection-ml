"""Final analysis: permutation importance, FP/FN analysis, calibration, drift, TTL ablation.

This script performs the following analysis on the finalized models:
- Permutation feature importance for Logistic Regression, Neural Network, Random Forest, and XGBoost.
- Analysis of false positives and false negatives using feature profiles (for LR/NN) and SHAP (for trees).
- Probability calibration, reliability curves, and Brier score for all models.
- Distribution-drift analysis (PSI, KS) between training and test (data‑only).
- TTL-feature ablation using the TTL-included preprocessing artifact (XGBoost only).

It does not alter models or data; it only reads existing artifacts and places analysis outputs in experiments/final_analysis/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE
from preprocess import load_artifact

OUTPUT_DIR = ROOT / "experiments" / "explainability_and_diagnostics" / "final_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODEL_PATHS = {
    "Logistic Regression": ARTIFACTS_DIR / "logistic_regression.joblib",
    "Neural Network": ARTIFACTS_DIR / "neural_network.joblib",
    "Random Forest": ARTIFACTS_DIR / "random_forest.joblib",
    "XGBoost": ARTIFACTS_DIR / "xgboost.joblib",
}
PREPROCESSOR_PATHS = {
    "Logistic Regression": ARTIFACTS_DIR / "preprocess_linear.joblib",
    "Neural Network": ARTIFACTS_DIR / "preprocess_linear.joblib",
    "Random Forest": ARTIFACTS_DIR / "preprocess_tree.joblib",
    "XGBoost": ARTIFACTS_DIR / "preprocess_tree.joblib",
}

THRESHOLDS_PATH = ROOT / "experiments" / "standardized_evaluation" / "selected_thresholds.json"
locked_thresholds = pd.read_json(THRESHOLDS_PATH, typ="series")
THRESHOLDS = {
    "Logistic Regression": float(locked_thresholds["logistic_regression"]),
    "Neural Network": float(locked_thresholds["neural_network"]),
    "Random Forest": float(locked_thresholds["random_forest"]),
    "XGBoost": float(locked_thresholds["xgboost"]),
}

native_path = ROOT / "experiments" / "xgboost" / "xgboost_model.json"

def load_data(preprocessor_path):
    """Load raw splits, transform, and return DataFrames with feature names"""
    X_train_raw = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["label"].astype(int)
    X_test_raw = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["label"].astype(int)

    pre = load_artifact(preprocessor_path)
    X_train = pre.transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    # Get feature names
    if hasattr(pre, "get_feature_names_out"):
        feature_names = pre.get_feature_names_out()
    else:
        feature_names = X_train.columns

    # Ensure DataFrames with correct column names
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_train.columns = feature_names
        X_test.columns = feature_names

    return X_train, y_train, X_test, y_test, list(feature_names)

def load_model_and_predictions(model_name, X_test, y_test):
    """Load model and its test prediction CSV"""
    if model_name == "XGBoost":
        # Load from native JSON
        if not native_path.exists():
            raise FileNotFoundError(f"Native XGBoost model not found: {native_path}")
        model = XGBClassifier()
        model.load_model(native_path)
    else: 
        model = joblib.load(MODEL_PATHS[model_name])
    
    proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy(dtype=int) if hasattr(y_test, "to_numpy") else np.asarray(y_test, dtype=int)
    return model, y_true, proba


def pr_auc_scorer(estimator, X, y):
    """PR-AUC scorer for permutation importance"""
    return average_precision_score(y, estimator.predict_proba(X)[:, 1])


def compute_psi(train_vals, test_vals, bins=10):
    """Calculates the Population Stability Index"""
    combined = np.concatenate([train_vals, test_vals])
    edges = np.percentile(combined, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.nan
    train_hist, _ = np.histogram(train_vals, bins=edges)
    test_hist, _ = np.histogram(test_vals, bins=edges)
    train_pct = train_hist / len(train_vals) + 1e-6
    test_pct = test_hist / len(test_vals) + 1e-6
    return np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))


def profile_errors(model_name, X_test, y_true, proba, threshold):
    """Mean feature values per confusion group for a given model and threshold. """
    pred = (proba >= threshold).astype(int)
    groups = {
        "FP": (y_true == 0) & (pred == 1),
        "FN": (y_true == 1) & (pred == 0),
        "TP": (y_true == 1) & (pred == 1),
        "TN": (y_true == 0) & (pred == 0),
    }
    profiles = {}
    for name, mask in groups.items():
        if mask.sum() > 0:
            profiles[name] = X_test.loc[mask].mean(axis=0)
        else:
            profiles[name] = pd.Series(index=X_test.columns, dtype=float).fillna(0)
    df = pd.DataFrame(profiles).T
    df.to_csv(OUTPUT_DIR / f"error_profiles_{model_name.lower().replace(' ', '_')}.csv")
    return df


def plot_psi_ranking(psi_df, save_path, top_n=20):
    """Plots a bar chart of the top PSI features."""
    psi_df = psi_df.sort_values("PSI", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(psi_df["feature"], psi_df["PSI"], color="steelblue")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title(f"Top {top_n} features by distribution drift (PSI)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_ttl_comparison(imp_no_ttl, imp_with_ttl, features_no_ttl, features_with_ttl, save_path):
    """Plots the comparison of permutation importance with/without TTL features."""
    df_no = pd.DataFrame({
        "feature": features_no_ttl,
        "importance_no_ttl": imp_no_ttl["importances_mean"],
        "std_no_ttl": imp_no_ttl["importances_std"],
    })
    df_with = pd.DataFrame({
        "feature": features_with_ttl,
        "importance_with_ttl": imp_with_ttl["importances_mean"],
        "std_with_ttl": imp_with_ttl["importances_std"],
    })
    merged = df_no.merge(df_with, on="feature", how="outer")
    ttl_cols = ["sttl", "dttl", "ct_state_ttl"]
    merged["is_ttl"] = merged["feature"].apply(lambda x: any(t in x for t in ttl_cols))
    merged = merged.sort_values("importance_with_ttl", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(merged))
    colors = ["red" if is_ttl else "steelblue" for is_ttl in merged["is_ttl"]]
    ax.barh(y_pos, merged["importance_with_ttl"], xerr=merged["std_with_ttl"],
            color=colors, alpha=0.7, label="With TTL")
    ax.scatter(merged["importance_no_ttl"], y_pos, color="black", s=30, label="Without TTL")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["feature"])
    ax.set_xlabel("Permutation importance (mean decrease in PR-AUC)")
    ax.set_title("TTL Ablation: Feature importance with and without TTL")
    legend_elements = [
        Patch(facecolor="red", label="TTL features"),
        Patch(facecolor="steelblue", label="Other features"),
        plt.Line2D([0], [0], marker="o", color="w", label="No TTL",
                   markerfacecolor="black", markersize=8)
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for both families
    X_train_tree, y_train, X_test_tree, y_test, features_tree = load_data(PREPROCESSOR_PATHS["XGBoost"])
    X_train_lin, _, X_test_lin, _, features_lin = load_data(PREPROCESSOR_PATHS["Logistic Regression"])

    print("\nComputing Permutation importance for all models:")
    perm_importances = {}
    for model_name in MODEL_PATHS.keys():
        print(f"  {model_name}...")
        X_test = X_test_lin if model_name in ["Logistic Regression", "Neural Network"] else X_test_tree
        model, _, _ = load_model_and_predictions(model_name, X_test, y_test)
        imp = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE,
            scoring=pr_auc_scorer, n_jobs=1
        )
        perm_importances[model_name] = imp
        df = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": imp.importances_mean,
            "importance_std": imp.importances_std,
        }).sort_values("importance_mean", ascending=False)
        df.to_csv(OUTPUT_DIR / f"permutation_importance_{model_name.lower().replace(' ', '_')}.csv", index=False)

    # Calibration for all models (individual + combined)
    print("\nComputing Calibration curves and Brier scores for all models:")
    calibration_data = {}
    for model_name in MODEL_PATHS.keys():
        X_test = X_test_lin if model_name in ["Logistic Regression", "Neural Network"] else X_test_tree
        _, y_true, proba = load_model_and_predictions(model_name, X_test, y_test)
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_true, proba)
        calibration_data[model_name] = {"prob_true": prob_true, "prob_pred": prob_pred, "brier": brier}

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=model_name)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability curve - {model_name}\nBrier score = {brier:.4f}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"calibration_curve_{model_name.lower().replace(' ', '_')}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Combined calibration plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, data in calibration_data.items():
        ax.plot(data["prob_pred"], data["prob_true"], marker="o", linewidth=2,
                label=f"{model_name} (Brier={data['brier']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves for all models")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curves_all_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Error profiling for Logistic Regression and Neural Network
    print("\nError profiling for Logistic Regression and Neural Network:")
    for model_name in ["Logistic Regression", "Neural Network"]:
        _, y_true, proba = load_model_and_predictions(model_name, X_test_lin, y_test) 
        X_test = X_test_lin
        threshold = THRESHOLDS[model_name]
        profile_errors(model_name, X_test, y_true, proba, threshold)

    # Drift analysis (PSI, KS)
    print("\nDrift analysis (PSI, KS)")
    drift_records = []
    for col in features_tree:
        train_vals = X_train_tree[col].to_numpy()
        test_vals = X_test_tree[col].to_numpy()
        psi = compute_psi(train_vals, test_vals, bins=10)
        # Inline KS: direct call to ks_2samp
        ks_stat, _ = ks_2samp(train_vals, test_vals)
        drift_records.append({"feature": col, "PSI": psi, "KS": ks_stat})
    drift_df = pd.DataFrame(drift_records)
    drift_df.to_csv(OUTPUT_DIR / "drift_psi_ks.csv", index=False)
    plot_psi_ranking(drift_df, FIGURES_DIR / "psi_top_features.png", top_n=20)

    # TTL ablation for XGBoost
    print("\nConducting TTL ablation for XGBoost...")
    # Load TTL-included data
    X_train_ttl, _, X_test_ttl, _, features_ttl = load_data(
        PREPROCESSOR_PATHS["XGBoost"].with_name("preprocess_tree_with_ttl.joblib")
    )

    # Load the finalized XGBoost hyperparameters (nested under selected_hyperparameters)
    xgb_config_path = ROOT / "experiments" / "xgboost" / "config.json"
    if not xgb_config_path.exists():
        raise FileNotFoundError(f"Missing finalized XGBoost config: {xgb_config_path}")
    with open(xgb_config_path) as f:
        xgb_config = json.load(f)

    ttl_params = {**xgb_config["selected_hyperparameters"]}
    ttl_params["random_state"] = RANDOM_STATE

    model_ttl = XGBClassifier(**ttl_params)
    print("Fitting XGBoost with TTL features on training data...")
    model_ttl.fit(X_train_ttl, y_train)
    
    # Metrics for TTL model
    proba_ttl = model_ttl.predict_proba(X_test_ttl)[:, 1]
    pred_ttl = (proba_ttl >= THRESHOLDS["XGBoost"]).astype(int)
    ttl_metrics = {
        "pr_auc": average_precision_score(y_test, proba_ttl),
        "roc_auc": roc_auc_score(y_test, proba_ttl),
        "f1": f1_score(y_test, pred_ttl),
        "precision": precision_score(y_test, pred_ttl),
        "recall": recall_score(y_test, pred_ttl),
        "accuracy": accuracy_score(y_test, pred_ttl),
    }
    pd.DataFrame([ttl_metrics]).to_csv(OUTPUT_DIR / "ttl_ablation_metrics.csv", index=False)

    # Load official no‑TTL test predictions (from the finalized experiment)
    xgb_test_pred_path = ROOT / "experiments" / "xgboost" / "test_predictions.csv"
    if not xgb_test_pred_path.exists():
        raise FileNotFoundError(f"Missing XGBoost test predictions: {xgb_test_pred_path}")
    xgb_pred_df = pd.read_csv(xgb_test_pred_path)
    proba_orig = xgb_pred_df["attack_probability"].to_numpy()
    pred_orig = xgb_pred_df["predicted_label"].to_numpy()
    # Ensure alignment with the test set
    if len(proba_orig) != len(y_test):
        raise ValueError("Test prediction length does not match y_test")
    original_metrics = {
        "pr_auc": average_precision_score(y_test, proba_orig),
        "roc_auc": roc_auc_score(y_test, proba_orig),
        "f1": f1_score(y_test, pred_orig),
        "precision": precision_score(y_test, pred_orig),
        "recall": recall_score(y_test, pred_orig),
        "accuracy": accuracy_score(y_test, pred_orig),
    }
    pd.DataFrame([original_metrics]).to_csv(OUTPUT_DIR / "original_metrics_no_ttl.csv", index=False)

    # Permutation importance for TTL model
    print("Calculating permutation importance for TTL model...")
    perm_imp_ttl = permutation_importance(
        model_ttl, X_test_ttl, y_test, n_repeats=5, random_state=RANDOM_STATE,
        scoring=pr_auc_scorer, n_jobs=-1
    )
    imp_ttl_df = pd.DataFrame({
        "feature": features_ttl,
        "importance_mean": perm_imp_ttl.importances_mean,
        "importance_std": perm_imp_ttl.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp_ttl_df.to_csv(OUTPUT_DIR / "permutation_importance_with_ttl.csv", index=False)
    
    # Plot TTL comparison
    imp_no_ttl = perm_importances["XGBoost"]
    plot_feature_importance_ttl_comparison(
        imp_no_ttl, perm_imp_ttl, features_tree, features_ttl,
        FIGURES_DIR / "ttl_ablation_feature_importance.png"
    )

    print(f"\nAll outputs saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()