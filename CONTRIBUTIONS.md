# Contributions

EECS 3404 major project — *Explainable and Drift-Aware Machine Learning for
Network Intrusion Detection*.

This document records what each team member contributed to the work present on
`main`. Assigned ownership is defined in
[`TEAM_RESPONSIBILITIES.md`](TEAM_RESPONSIBILITIES.md); this document records
what was actually delivered.

## Isaac — data preparation and shared pipeline

- Raw UNSW-NB15 acquisition with integrity and row-count validation
  (`src/00_download.py`).
- Exploratory data analysis, including class balance, missingness, duplicates
  and the TTL leakage investigation (`src/01_eda.py`,
  [`docs/eda-summary.md`](docs/eda-summary.md)).
- Cleaning, deduplication and leakage removal (`src/02_clean_split.py`):
  removal of `id` and `attack_cat`, same-label predictor deduplication,
  explicit removal of predictor vectors carrying conflicting labels, and
  removal of training rows whose predictor values occur in the test set.
- The fixed train/validation/test split (79,685 / 19,922 / 82,332) with
  `RANDOM_STATE = 42`, documented in
  [`docs/split-manifest.json`](docs/split-manifest.json).
- Train-only fitted preprocessing artifacts for the linear and tree model
  families, plus the TTL-included variants reserved for the ablation
  (`src/03_pipelines.py`, `artifacts/preprocess_*.joblib`).
- Dummy-classifier baselines (`src/04_baseline.py`,
  [`docs/baseline.md`](docs/baseline.md)).
- Dataset documentation and reproducibility manifests
  ([`docs/DATA_CARD.md`](docs/DATA_CARD.md), `artifacts/manifest.json`).

## May — Logistic Regression and Neural Network

- Logistic Regression implementation, two-round hyperparameter search, and
  validation-only model selection (`src/05_logistic_regression.py`).
- Neural Network implementation (`MLPClassifier`, 128×64, Adam), two-round
  search, and the fixed validation-selected epoch-budget refit procedure
  (`src/08_neural_network.py`).
- Final artifacts, configurations, metrics, tuning histories, and
  validation/test predictions for both models (`artifacts/`,
  `experiments/logistic_regression/`, `experiments/neural_network/`).
- The per-model experiment notes in `experiments/logistic_regression/README.md`
  and `experiments/neural_network/README.md`, including the Round 1 logging
  correction and the fixed-epoch-budget refit rationale.

## Shivam — tree models and diagnostics

- Random Forest implementation, tuning and validation-only selection
  (`src/06_random_forest.py`,
  [`docs/random_forest_results.md`](docs/random_forest_results.md)).
- XGBoost implementation, tuning and validation-only selection
  (`src/07_xgboost.py`, [`docs/xgboost_results.md`](docs/xgboost_results.md)).
- Supplemental Random Forest and XGBoost analysis: bootstrap confidence
  intervals, hyperparameter sensitivity, model-selection stability and inference
  efficiency (`src/08_rf_xgboost_analysis.py`, `experiments/model_analysis/`,
  [`docs/rf_xgboost_advanced_analysis.md`](docs/rf_xgboost_advanced_analysis.md)).
- The read-only Neural Network artifact verification script, which confirms the
  saved model still reproduces the committed predictions and metrics
  (`src/08_neural_network_verify.py`).
- The explainability and advanced diagnostics implementation used by the team
  (`src/diagnostics_lib.py`, `src/11_diagnostics_verify.py` through
  `src/15_build_report_tables.py`, `experiments/diagnostics/`,
  [`docs/diagnostics_report.md`](docs/diagnostics_report.md)). This covers
  provenance and artifact-to-prediction verification, leakage assertions, SHAP
  global and local explanations, permutation importance, error analysis,
  calibration, distribution drift, the TTL ablation, and deterministic
  report-table generation, together with unit tests (`tests/`) and a `--verify`
  mode on every stage.

## Sharwin — standardized evaluation and model comparison

- Validation-only decision-threshold analysis and threshold selection for all
  four models (`src/09_standardized_evaluation.py`,
  `experiments/standardized_evaluation/selected_thresholds.json`).
- Frozen-test evaluation at both the default 0.50 threshold and the locked
  validation-selected thresholds (`src/10_final_test_evaluation.py`,
  `experiments/standardized_evaluation/final_test/`).
- Standardized metrics, confusion matrices, ROC and precision-recall curves, and
  comparison tables and figures across all four models.
- The final cross-model comparison and its conclusions, including the scope
  limits of those conclusions
  ([`docs/standardized_model_evaluation.md`](docs/standardized_model_evaluation.md)).

## Paul — exploratory diagnostics work

Paul contributed exploratory work toward the explainability and advanced
diagnostics component. After review, the final diagnostics implementation used
by the team was developed and verified by Shivam and is the version merged into
`main`.

## Shared team work

The following work was completed collaboratively across the team:

- Final written report — completed.
- Presentation and video walkthrough — completed.
- Cross-review of one another's outputs, and integration of the model, evaluation, and diagnostics stages into a single consistent pipeline — completed.
- Agreement on shared conventions: the fixed split, the transform-only preprocessing rule, `1 = attack`, `attack_probability = P(y = 1)`, the common prediction-file schema, and `RANDOM_STATE = 42` — completed.
