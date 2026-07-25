# Random Forest — End-to-End Implementation Plan

**Owner:** Shivam (tree-model tracks, see `TEAM_RESPONSIBILITIES.md`)
**Model:** Random Forest (binary classification: `0` = normal, `1` = attack)
**Dataset:** UNSW-NB15, using Isaac's fixed splits and preprocessing artifacts

## Ground rules baked into the plan

- Fit models on **train only**, select using **val only**.
- The frozen test set is **not loaded or transformed at all** until the model
  configuration is finalized and locked; it is then used once.
- Never refit preprocessing — only call `.transform()` on the loaded artifact.
- `RANDOM_STATE = 42` everywhere; `1` = attack; `attack_probability` = P(y=1).
- Sharwin owns thresholds — export probabilities plus default-0.5 labels and
  stop there.

## Workflow at a glance

```text
Train candidate config on training data
  → evaluate config on validation data
  → select best hyperparameters (PR-AUC primary, ROC-AUC tie-breaker)
  → lock configuration
  → refit final RF on training data only
  → generate final validation predictions
  → only then load/transform the frozen test set
  → generate test predictions once
  → hand outputs to Sharwin and Paul
```

## Phase 0 — Scaffolding

Create one script following the repo's numbering: `src/06_random_forest.py`
(May's Logistic Regression is `05`), plus the output folder
`experiments/random_forest/`.

Load pattern (mirroring `src/05_logistic_regression.py` but with two fixes):

- Load **only** `X_train.parquet`, `X_val.parquet`, `y_train.parquet`, and
  `y_val.parquet`. The test set is not touched in Phases 0–2.
- Read the files with a **capital X** — that's the actual filename on disk. The
  LR script uses lowercase `x_train.parquet`, which only works because macOS is
  case-insensitive; copying the pattern exactly would break on Linux.
- Load `artifacts/preprocess_tree.joblib` via `load_artifact()` (it hard-errors
  on a scikit-learn version mismatch, which we want) and transform **train and
  validation only**.

Sanity checks before any modelling: shapes should be **84,814 × 39 (train)**
and **21,204 × 39 (val)** per `docs/split-manifest.json`, and class balance
~51% attack in train/val.

## Phase 1 — Default baseline experiment

Train `RandomForestClassifier(random_state=42, n_jobs=-1)` with all defaults,
timed with `time.perf_counter()`. Record validation PR-AUC, ROC-AUC, accuracy,
precision, recall, F1.

Two purposes:

1. A reference row in the tuning log.
2. A sanity gate — it should comfortably beat the dummy baseline in
   `docs/baseline.json`.

Also compare train vs. val score to gauge how much default RF (unlimited depth)
overfits here.

## Phase 2 — Hyperparameter tuning on the fixed validation split

**Why a manual loop instead of `GridSearchCV`/`RandomizedSearchCV`:**
cross-validation confined to training data would not inherently cause test
leakage. However, this project has explicitly standardized model selection
around Isaac's fixed train/validation split. A manual configuration loop that
fits on train and scores on the shared validation set is therefore preferred:
it is transparent, consistent across team members, and produces a
straightforward `tuning_results.csv`.

Search space (coarse first, then refine around the winner):

| Parameter          | Values             |
| ------------------ | ------------------ |
| `n_estimators`     | 200, 400, 800      |
| `max_depth`        | None, 15, 25, 35   |
| `min_samples_leaf` | 1, 2, 5            |
| `max_features`     | 'sqrt', 0.3, 0.5   |
| `class_weight`     | None, 'balanced'   |

The full cross-product is ~200 fits, which is fine at 85k rows but slow; do a
**staged search** instead:

1. Tune `max_depth` + `min_samples_leaf` at fixed `n_estimators=300` (these
   matter most for RF).
2. Then tune `max_features` and `class_weight`.
3. Then confirm the winner is stable at higher `n_estimators`.

Roughly 30–40 fits total.

**Selection metric:** primary is validation **PR-AUC**, with validation
**ROC-AUC** as the tie-breaker. The project README explicitly prioritizes
precision, recall, F1, and PR-AUC because the research question focuses on
detecting attacks while controlling false positives — and since threshold
selection belongs to Sharwin, hyperparameter selection must use
threshold-independent metrics. F1 is deliberately **not** the selection metric
because it depends on a decision threshold.

Log every config with its params, all val metrics, and fit time into
`tuning_results.csv`.

## Phase 3 — Lock, finalize, and (only now) touch test

- Pick the winning config from validation results only, refit it on **train
  only** (this keeps the exported validation predictions honest — they must
  come from the same fitted model), and record the final training runtime.
- Generate the final validation predictions from this refit model.
- Save the fitted model to `artifacts/random_forest.joblib` (matching where the
  LR model was saved).
- **Only after the configuration is locked**: load `X_test.parquet` /
  `y_test.parquet`, transform test with the already-fitted
  `preprocess_tree.joblib` (expected shape **82,332 × 39**), and generate test
  predictions once. No going back to tune after seeing test outputs.

Test metrics computed here are a **sanity/reference output only** — they are
not the official final comparison. Sharwin recomputes the standardized final
metrics from the prediction files so that Logistic Regression, Neural Network,
Random Forest, and XGBoost are all evaluated identically.

## Phase 4 — Handoff exports to `experiments/random_forest/`

Per the standard handoff contract in `TEAM_RESPONSIBILITIES.md`:

- `config.json` — selected hyperparameters, `random_state`, preprocessing
  artifact used (`preprocess_tree.joblib`, TTL-excluded), the 39 feature names
  (from `preprocessor.get_feature_names_out()`), training runtime in seconds,
  and library versions (should match `artifacts/manifest.json`:
  scikit-learn 1.9.0).
- `validation_predictions.csv` and `test_predictions.csv` — columns:
  `sample_index` (the parquet row index, so Paul can match errors back to the
  split), `true_label`, `attack_probability` (`predict_proba[:, 1]`),
  `predicted_label` (0.5 threshold — handoff labels only; Sharwin owns
  threshold selection).
- `metrics.json` — validation metrics for the final model, plus the
  sanity/reference test metrics (clearly labelled as reference, not the
  official comparison).
- `tuning_results.csv` — the full search log from Phase 2.
- A short experiment summary (a few lines in `metrics.json` or a small
  `README.md` in the folder): chosen config, artifact variant, and anything
  notable.

## Phase 5 — Verification pass

- Rerun the script end-to-end and confirm identical metrics (seed
  reproducibility).
- Check prediction files: row counts match split sizes, probabilities in
  [0, 1], `predicted_label == (attack_probability >= 0.5)`.
- Confirm nothing in the flow ever calls `.fit()` on the preprocessor, and that
  the test set is never loaded or transformed before the configuration lock in
  Phase 3.

## Explicitly out of scope

- Threshold selection and the official cross-model comparison (Sharwin).
- SHAP, calibration, drift, and the TTL ablation (Paul).
- Any extra experiments beyond RF + XGBoost.

The XGBoost track will reuse Phases 0–5 wholesale, so structuring the
loader/export code as reusable functions now pays off there.
