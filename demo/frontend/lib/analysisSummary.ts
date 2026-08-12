/**
 * Static content for the Analysis page: captions, source citations, and
 * (build-time copies of) committed diagnostics figures. Nothing here is
 * computed — every figure is a direct copy of an already-generated PNG from
 * experiments/, and every number is read from the same committed CSV/JSON
 * outputs cited in lib/projectSummary.ts.
 *
 * Figures are duplicated (not moved) into public/diagnostics/ because Next.js
 * can only serve static assets from public/; the originals under experiments/
 * are untouched and remain the source of truth. Re-copy from experiments/ if
 * diagnostics are ever regenerated.
 */

export interface FigureEntry {
  src: string;
  alt: string;
  caption: string;
  width: number;
  height: number;
  sourcePath: string;
}

export interface AnalysisSection {
  id: string;
  title: string;
  summary: string;
  figures?: FigureEntry[];
}

export const ANALYSIS_SECTIONS: AnalysisSection[] = [
  {
    id: "evaluation",
    title: "Frozen test evaluation",
    summary:
      "ROC and precision-recall curves are threshold-independent; the confusion matrices and the false-positive/false-negative comparison use each model's own locked, validation-selected threshold on the 82,332-row frozen test split.",
    figures: [
      {
        src: "/diagnostics/evaluation/test_roc_curves.png",
        alt: "ROC curves for all four models on the frozen test split",
        caption: "ROC curves, frozen test split",
        width: 1421, height: 1061,
        sourcePath: "experiments/standardized_evaluation/final_test/figures/test_roc_curves.png"
      },
      {
        src: "/diagnostics/evaluation/test_precision_recall_curves.png",
        alt: "Precision-recall curves for all four models on the frozen test split",
        caption: "Precision-recall curves, frozen test split",
        width: 1421, height: 1060,
        sourcePath: "experiments/standardized_evaluation/final_test/figures/test_precision_recall_curves.png"
      },
      {
        src: "/diagnostics/evaluation/locked_threshold_model_metric_comparison.png",
        alt: "Bar chart comparing locked-threshold metrics across all four models",
        caption: "Locked-threshold metric comparison",
        width: 1782, height: 1060,
        sourcePath: "experiments/standardized_evaluation/final_test/figures/locked_threshold_model_metric_comparison.png"
      },
      {
        src: "/diagnostics/evaluation/false_positive_false_negative_comparison.png",
        alt: "Bar chart comparing false positive and false negative counts across all four models",
        caption: "False-positive / false-negative comparison",
        width: 1781, height: 1060,
        sourcePath: "experiments/standardized_evaluation/final_test/figures/false_positive_false_negative_comparison.png"
      },
      {
        src: "/diagnostics/evaluation/logistic_regression_test_locked_threshold_confusion_matrix.png",
        alt: "Logistic Regression confusion matrix at its locked threshold",
        caption: "Logistic Regression",
        width: 978, height: 804,
        sourcePath: "experiments/standardized_evaluation/final_test/figures/logistic_regression_test_locked_threshold_confusion_matrix.png"
      },
      {
        src: "/diagnostics/evaluation/neural_network_test_locked_threshold_confusion_matrix.png",
        alt: "Neural Network confusion matrix at its locked threshold",
        caption: "Neural Network",
        width: 978, height: 804,
        sourcePath: "experiments/standardized_evaluation/final_test/figures/neural_network_test_locked_threshold_confusion_matrix.png"
      },
      {
        src: "/diagnostics/evaluation/random_forest_test_locked_threshold_confusion_matrix.png",
        alt: "Random Forest confusion matrix at its locked threshold",
        caption: "Random Forest",
        width: 978, height: 804,
        sourcePath: "experiments/standardized_evaluation/final_test/figures/random_forest_test_locked_threshold_confusion_matrix.png"
      },
      {
        src: "/diagnostics/evaluation/xgboost_test_locked_threshold_confusion_matrix.png",
        alt: "XGBoost confusion matrix at its locked threshold",
        caption: "XGBoost",
        width: 978, height: 804,
        sourcePath: "experiments/standardized_evaluation/final_test/figures/xgboost_test_locked_threshold_confusion_matrix.png"
      }
    ]
  },
  {
    id: "calibration",
    title: "Calibration",
    summary:
      "Brier score (lower is better): Random Forest 0.0843 · XGBoost 0.0888 · Neural Network 0.0976 · Logistic Regression 0.1406. Every model under-predicts in its worst calibration bin — a low aggregate Brier score does not rule out local miscalibration.",
    figures: [
      {
        src: "/diagnostics/calibration/reliability_overlay.png",
        alt: "Reliability diagram overlaying calibration curves for all four models",
        caption: "Reliability diagram, all four models",
        width: 1260, height: 1080,
        sourcePath: "experiments/diagnostics/calibration/figures/reliability_overlay.png"
      }
    ]
  },
  {
    id: "explainability",
    title: "Explainability (SHAP)",
    summary:
      "Random Forest SHAP values are computed on a 500-row stratified sample; XGBoost covers the full 82,332-row test set. Both models rank ackdat as the top contributing feature, but the scales differ (Random Forest: probability scale; XGBoost: raw-margin scale, via SHAP's tree-path-dependent explainer) — rankings are comparable across the two models, magnitudes are not.",
    figures: [
      {
        src: "/diagnostics/explainability/shap_beeswarm_random_forest.png",
        alt: "SHAP beeswarm plot for Random Forest, probability scale",
        caption: "Random Forest — SHAP beeswarm (probability scale)",
        width: 1394, height: 1691,
        sourcePath: "experiments/diagnostics/shap/figures/shap_beeswarm_random_forest.png"
      },
      {
        src: "/diagnostics/explainability/shap_beeswarm_xgboost.png",
        alt: "SHAP beeswarm plot for XGBoost, raw margin scale",
        caption: "XGBoost — SHAP beeswarm (raw-margin scale)",
        width: 1394, height: 1691,
        sourcePath: "experiments/diagnostics/shap/figures/shap_beeswarm_xgboost.png"
      }
    ]
  },
  {
    id: "drift",
    title: "Train → test drift",
    summary:
      "Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics comparing each feature's train-split and test-split distributions. 29 of the 39 primary features exceed the PSI 0.2 shift threshold — the test split is deliberately not identically distributed to training."
  },
  {
    id: "ttl-ablation",
    title: "TTL leakage ablation",
    summary:
      "Refitting XGBoost with the excluded TTL fields included gains only +0.0011 PR-AUC over the production (no-TTL) configuration, while sttl jumps from unranked to the #1 importance rank and ackdat — the top feature everywhere else in this analysis — collapses from rank 1 to rank 33. This is evidence that the model can rely heavily on TTL-related dataset artifacts / shortcut features, which is why they stay excluded from the primary input contract.",
    figures: [
      {
        src: "/diagnostics/ttl-ablation/ttl_ranking_metrics.png",
        alt: "Feature importance ranking comparison with and without TTL fields",
        caption: "Feature importance rank shift, with vs. without TTL fields",
        width: 1260, height: 900,
        sourcePath: "experiments/diagnostics/ttl_ablation/figures/ttl_ranking_metrics.png"
      }
    ]
  },
  {
    id: "uncertainty",
    title: "Uncertainty & selection stability",
    summary:
      "A 2,000-replicate paired row-wise bootstrap (seed 42, 0 skipped replicates) compares XGBoost and Random Forest on the frozen test set. The PR-AUC gap in XGBoost's favor is small but its 95% confidence interval excludes zero; the recall gap favors Random Forest. Separately, re-ranking the top validation configurations for each model shows the selected configuration is not a fragile fluke: the top-10 PR-AUC range is 0.00011 for Random Forest and 0.000275 for XGBoost.",
    figures: [
      {
        src: "/diagnostics/uncertainty/bootstrap_paired_difference_intervals.png",
        alt: "Bootstrap confidence intervals for XGBoost minus Random Forest metric differences",
        caption: "Bootstrap CIs, XGBoost − Random Forest",
        width: 1415, height: 880,
        sourcePath: "experiments/model_analysis/figures/bootstrap_paired_difference_intervals.png"
      },
      {
        src: "/diagnostics/uncertainty/model_selection_stability.png",
        alt: "Validation PR-AUC of the top candidate configurations for Random Forest and XGBoost",
        caption: "Model-selection stability, top validation configurations",
        width: 1421, height: 880,
        sourcePath: "experiments/model_analysis/figures/model_selection_stability.png"
      }
    ]
  }
];

export interface DriftFeature {
  feature: string;
  psi: number;
  ks: number;
}

// Source: experiments/diagnostics/drift/drift_psi_ks.csv, top 5 rows by PSI.
export const TOP_DRIFT_FEATURES: DriftFeature[] = [
  { feature: "ct_dst_sport_ltm", psi: 0.8332, ks: 0.2811 },
  { feature: "dmean", psi: 0.5394, ks: 0.3188 },
  { feature: "dpkts", psi: 0.5233, ks: 0.3070 },
  { feature: "state", psi: 0.5102, ks: 0.3064 },
  { feature: "dbytes", psi: 0.5031, ks: 0.3200 }
];
