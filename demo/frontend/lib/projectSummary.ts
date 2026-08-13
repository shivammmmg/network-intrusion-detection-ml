/**
 * Static, presentation-only summary of the finalized ML project, shown on the
 * Overview page. Every value here is copied from committed repository outputs
 * (never computed, estimated, or re-derived) — each block cites its exact
 * source file so a future maintainer can re-verify or refresh it.
 *
 * This module carries no inference or threshold logic. Live predictions,
 * thresholds actually used for a verdict, and per-request model metadata
 * still come from the backend via lib/api.ts — the LOCKED_THRESHOLDS array
 * below is a display-only copy for the Overview page, shown before/without
 * calling the API.
 */

// Source: docs/split-manifest.json (raw_train_rows, raw_test_rows,
// train_same_label_duplicate_rows_removed, train_conflicting_predictor_vectors_removed,
// train_conflicting_predictor_rows_removed, train_rows_also_in_test_removed)
export const DATA_AUDIT = {
  originalTrainRows: 175_341,
  originalTestRows: 82_332,
  originalTotalRows: 257_673,
  duplicateTrainRowsRemoved: 74_072,
  ambiguousPredictorVectorsRemoved: 229,
  ambiguousPredictorRowsRemoved: 458,
  trainRowsOverlappingTestRemoved: 1_204
};

// Source: docs/split-manifest.json (splits.train.n, splits.val.n, splits.test.n)
export const SPLITS = {
  train: 79_685,
  validation: 19_922,
  test: 82_332,
  total: 181_939
};

// Source: experiments/diagnostics/validation/leakage_assertions.json
// (split_rows_pairwise_disjoint.overlap_counts) — re-verified independently
// of the original split script by the read-only diagnostics gate.
export const PREDICTOR_OVERLAP_CHECKS = {
  trainValidation: 0,
  trainTest: 0,
  validationTest: 0
};

// Source: experiments/diagnostics/validation/leakage_assertions.json
// (preprocessors_fitted_and_feature_counts_match_manifest.artifacts)
export const PREPROCESSING = {
  primaryInputFeatures: 39,
  linearTransformedFeatures: 66,
  treeTransformedFeatures: 39,
  ttlInclusiveLinearFeatures: 69,
  ttlInclusiveTreeFeatures: 42
};

export interface ModelTuningSummary {
  id: string;
  name: string;
  round1Configurations: number;
  round2Configurations: number;
}

// Source: row counts (minus header) of experiments/<model>/tuning_results.csv
// (Round 1) and experiments/<model>/round2_joint_search.csv (Round 2),
// cross-checked against the "Round 1" / "Round 2 winner" language in each
// experiments/<model>/README.md. Counts are per model — they are not equal
// across models and are not implied to be. These are recorded tuning fit
// rows, not asserted-unique hyperparameter configurations — do not relabel
// them as "total configurations" or similar in the UI.
export const MODEL_TUNING: ModelTuningSummary[] = [
  { id: "logistic_regression", name: "Logistic Regression", round1Configurations: 18, round2Configurations: 30 },
  { id: "neural_network", name: "Neural Network", round1Configurations: 26, round2Configurations: 24 },
  { id: "random_forest", name: "Random Forest", round1Configurations: 31, round2Configurations: 48 },
  { id: "xgboost", name: "XGBoost", round1Configurations: 49, round2Configurations: 64 }
];

export interface ModelThresholdDisplay {
  id: string;
  name: string;
  value: number;
}

// Source: experiments/standardized_evaluation/selected_thresholds.json
// (also the exact snapshot demo/api/app/settings.py cross-checks at startup).
export const LOCKED_THRESHOLDS: ModelThresholdDisplay[] = [
  { id: "logistic_regression", name: "Logistic Regression", value: 0.4631204833813834 },
  { id: "neural_network", name: "Neural Network", value: 0.3454528735910796 },
  { id: "random_forest", name: "Random Forest", value: 0.492968259796183 },
  { id: "xgboost", name: "XGBoost", value: 0.46306103 }
];

export interface FinalModelMetrics {
  id: string;
  name: string;
  prAuc: number;
  rocAuc: number;
  f1: number;
  recall: number;
}

// Source: experiments/standardized_evaluation/final_test/final_test_summary.json
// (locked_threshold_results). PR-AUC/ROC-AUC are threshold-independent ranking
// metrics; F1/recall here are evaluated at each model's locked threshold.
export const FINAL_TEST_METRICS: FinalModelMetrics[] = [
  { id: "logistic_regression", name: "Logistic Regression", prAuc: 0.927520, rocAuc: 0.906538, f1: 0.837039, recall: 0.953256 },
  { id: "neural_network", name: "Neural Network", prAuc: 0.970199, rocAuc: 0.959934, f1: 0.855610, recall: 0.969205 },
  { id: "random_forest", name: "Random Forest", prAuc: 0.982272, rocAuc: 0.977464, f1: 0.890862, recall: 0.984867 },
  { id: "xgboost", name: "XGBoost", prAuc: 0.985927, rocAuc: 0.980900, f1: 0.888731, recall: 0.980521 }
];

// Source: experiments/standardized_evaluation/final_test/final_test_summary.json
// (strongest_model). Quoted, not paraphrased into a stronger claim.
export const STRONGEST_MODEL_NOTE =
  "Random Forest — highest locked-threshold F1, with PR-AUC, recall, precision, and false-positive/false-negative rate considered; accuracy was not used alone.";

export interface DiagnosticItem {
  title: string;
  detail: string;
}

// Sources, in order: docs/diagnostics_report.md (Tables 1, 2, 4) and
// experiments/diagnostics/{shap,importance,calibration}/; final_test_summary.json
// locked_threshold_results; docs/diagnostics_report.md drift section and
// experiments/diagnostics/drift/drift_psi_ks.csv; experiments/diagnostics/
// ttl_ablation/{ttl_metrics_comparison,ttl_rank_shift}.csv;
// docs/rf_xgboost_advanced_analysis.md and experiments/model_analysis/
// bootstrap_metrics.csv; experiments/model_analysis/model_selection_stability.csv;
// experiments/diagnostics/validation/{artifact_prediction_check,leakage_assertions}.json
export const DIAGNOSTICS: DiagnosticItem[] = [
  {
    title: "SHAP explainability",
    detail:
      "Random Forest (500-row stratified sample) and XGBoost (full 82,332-row test set); both rank ackdat as the top contributing feature."
  },
  {
    title: "Permutation importance",
    detail: "Computed for all four models; both tree models are led by ackdat, matching the SHAP ranking."
  },
  {
    title: "Calibration (Brier score)",
    detail: "Random Forest 0.0843 (best) · XGBoost 0.0888 · Neural Network 0.0976 · Logistic Regression 0.1406."
  },
  {
    title: "Error profiling",
    detail: "False positives/negatives counted at each model's locked threshold, e.g. Random Forest: 686 FN / 10,253 FP on the 82,332-row test set."
  },
  {
    title: "Train→test drift (PSI / KS)",
    detail: "29 of 39 primary features exceed PSI 0.2; ct_dst_sport_ltm shows the largest shift (PSI 0.8332)."
  },
  {
    title: "TTL leakage ablation",
    detail:
      "Refitting XGBoost with TTL fields included gains only +0.0011 PR-AUC while sttl jumps to the top importance rank and ackdat collapses from rank 1 to rank 33 — evidence that the model can rely heavily on TTL-related dataset artifacts / shortcut features."
  },
  {
    title: "Bootstrap uncertainty",
    detail: "2,000 paired row-wise bootstrap replicates (seed 42) comparing XGBoost and Random Forest on the frozen test set; 0 skipped/degenerate replicates."
  },
  {
    title: "Model-selection stability",
    detail: "Top-10 validation PR-AUC range across near-winning configurations: Random Forest 0.00011, XGBoost 0.000275 — the selected configuration is not a fragile fluke."
  },
  {
    title: "Artifact & prediction provenance",
    detail: "Reloading each frozen model and recomputing test predictions reproduces the committed outputs (largest deviation 2.98e-8, XGBoost)."
  },
  {
    title: "Leakage / integrity assertions",
    detail: "Train/validation/test predictor overlap: 0 / 0 / 0; all splits pass binary-label and row-count checks."
  }
];

export interface VerificationCheck {
  label: string;
  detail: string;
}

// Source: this session's verification runs — `python -m pytest tests/ -q`,
// `python -m pytest demo/api/tests/ -q`, `npm run test` (demo/frontend), and
// `python src/11_diagnostics_verify.py --verify` — cross-checked against
// experiments/diagnostics/validation/*.json. Refresh this list if the suites
// change size; do not adjust counts without re-running the commands.
export const VERIFICATION_CHECKS: VerificationCheck[] = [
  { label: "Root pytest suite", detail: "15 tests — PASS" },
  { label: "Backend API test suite", detail: "48 tests — PASS" },
  { label: "Frontend test suite", detail: "13 tests — PASS" },
  { label: "Artifact → prediction reproducibility gate", detail: "4 / 4 models — PASS" },
  { label: "Leakage / integrity assertions", detail: "PASS" }
];

// Condensed synthesis of the project's headline numbers, drawn from the
// blocks above (SPLITS.total, "4" finalized models, PREPROCESSING.primaryInputFeatures,
// PREDICTOR_OVERLAP_CHECKS) for the Overview's 4-stat executive summary.
export interface HeadlineStat {
  value: string;
  label: string;
}

export const EXECUTIVE_STATS: HeadlineStat[] = [
  { value: SPLITS.total.toLocaleString(), label: "Final samples" },
  { value: "4", label: "Finalized models" },
  { value: String(PREPROCESSING.primaryInputFeatures), label: "Primary features" },
  { value: String(PREDICTOR_OVERLAP_CHECKS.trainTest), label: "Cross-split overlap" }
];

// Verbatim sentence approved for the Overview's model-performance takeaway.
// Source facts: experiments/standardized_evaluation/final_test/final_test_summary.json
// (locked_threshold_results ranking + strongest_model). Not a statistical-
// significance claim — see the paired-bootstrap uncertainty section for that.
export const PERFORMANCE_TAKEAWAY =
  "XGBoost leads PR-AUC and ROC-AUC. Random Forest achieves the highest locked-threshold F1 and recall and was selected as the strongest overall model under the project evaluation criteria.";

export interface EvidenceHighlight {
  category: string;
  headline: string;
  lines: string[];
  /** Optional numerator/denominator for a small proportion bar (e.g. drift). */
  fraction?: { numerator: number; denominator: number };
}

// Condensed, cautious-wording versions of the DIAGNOSTICS entries above, for
// the Overview's 6-block evidence grid. Full detail stays in DIAGNOSTICS /
// the Analysis page; sources are identical to the corresponding DIAGNOSTICS
// entries.
export const EVIDENCE_HIGHLIGHTS: EvidenceHighlight[] = [
  {
    category: "Explainability",
    headline: "SHAP + permutation importance",
    lines: ["Tree models both rank ackdat #1"]
  },
  {
    category: "Drift",
    headline: "29 / 39",
    lines: ["features exceed PSI 0.2", "Largest PSI: 0.8332 (ct_dst_sport_ltm)"],
    fraction: { numerator: 29, denominator: 39 }
  },
  {
    category: "Calibration",
    headline: "0.0843",
    lines: ["Random Forest Brier score", "Lowest among the four models"]
  },
  {
    category: "TTL ablation",
    headline: "+0.0011 PR-AUC",
    lines: ["when TTL fields are added", "evidence the model can rely heavily on TTL-related dataset artifacts / shortcut features"]
  },
  {
    category: "Uncertainty",
    headline: "2,000",
    lines: ["paired bootstrap replicates", "seed 42"]
  },
  {
    category: "Verification",
    headline: "76 tests passing",
    lines: ["4 / 4 artifact reproduction gate", "0 cross-split overlap"]
  }
];
