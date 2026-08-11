# Explainable and Drift-Aware Machine Learning for Network Intrusion Detection

EECS 3404 major project exploring binary network-intrusion detection with the
UNSW-NB15 dataset. The project compares machine-learning models by their ability
to detect attack traffic while limiting false-positive alerts, with particular
attention to data leakage, class imbalance, explainability, calibration, and
distribution drift.

> **Project status:** complete. The shared data workflow, leakage audit, fixed
> train/validation/test split, preprocessing artifacts, and baselines are
> implemented; all four models (Logistic Regression, Neural Network, Random
> Forest, XGBoost) are finalized; standardized threshold selection and frozen-test
> evaluation are done; and the explainability and advanced diagnostics pipeline
> is complete and independently verifiable.

The original proposal is available in
[`EECS 3404 Major Project Idea.pdf`](./EECS%203404%20Major%20Project%20Idea.pdf).

## Research question

Which machine-learning model provides the best balance between detecting
malicious network traffic and minimizing false-positive alerts?

## Scope

The task is binary classification:

- `0` - normal traffic
- `1` - attack traffic

The intended model comparison is:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Neural Network

Models are selected on validation data and evaluated on the frozen test set only
after the modelling choices are finalized. Accuracy is reported, but never used
alone, because it can hide missed attacks or a poor false-positive rate.

### Model finalization and handoff

- All four models are finalized. Each has a saved model in `artifacts/` and a
  configuration, metrics file, tuning history, and validation/test predictions in
  its `experiments/<model>/` folder.
- Hyperparameters were selected on validation data only. The frozen test set was
  used once, after every modelling choice and decision threshold was locked.
- XGBoost is strongest at the default `0.50` threshold, while Random Forest is
  strongest at the validation-selected locked thresholds. See
  [`docs/standardized_model_evaluation.md`](docs/standardized_model_evaluation.md)
  for the full comparison and its scope.
- Exact fresh Neural Network training can vary across Python, operating-system,
  and BLAS environments. The committed artifact verification
  (`src/08_neural_network_verify.py`) remains reproducible with the saved model
  and preprocessing artifact.

## Dataset

This project uses the pre-partitioned **UNSW-NB15** network-traffic dataset from
the Australian Centre for Cyber Security (Moustafa & Slay). The repository's
download script retrieves and validates these CSV files:

- `UNSW_NB15_training-set.csv` - 175,341 rows
- `UNSW_NB15_testing-set.csv` - 82,332 rows

The raw files are downloaded into `data/raw/`. The label is `label`, where
`0` is normal traffic and `1` is attack traffic. See
[`docs/DATA_CARD.md`](docs/DATA_CARD.md) for the dataset card, cleaning rules,
leakage findings, split sizes, and limitations.

## What is implemented

### Reproducible data workflow

- Raw-data download and integrity checks.
- Exploratory data analysis for class balance, missingness, duplicates,
  categorical values, and TTL leakage.
- Deterministic cleaning and a shared train/validation/test split using
  `RANDOM_STATE = 42`.
- Train-only preprocessing artifacts for linear and tree-based models.
- Dummy-classifier baselines on validation and test data.
- Stored manifests, processed Parquet files, and preprocessing artifacts.

### Leakage-aware preprocessing

The project deliberately handles several UNSW-NB15 issues:

- `id` is removed because it is a row identifier.
- `attack_cat` is removed because it reveals the binary target.
- Same-label predictor duplicates are removed from the binary development pool;
  predictor vectors with conflicting binary labels are removed explicitly.
- Training rows whose predictor values occur in the test set are removed from
  training data; the test set itself remains untouched.
- Preprocessing is fit on `X_train` only. Validation and test data are
  transformed without refitting imputers, scalers, or encoders.
- TTL fields (`sttl`, `dttl`, `ct_state_ttl`) are treated as a dataset artifact.
  The primary preprocessing artifacts exclude them; `*_with_ttl` artifacts are
  reserved for the leakage/ablation comparison.

### Current data split

| Split | Rows | Normal | Attack |
|---|---:|---:|---:|
| Train | 79,685 | 51.6% | 48.4% |
| Validation | 19,922 | 51.6% | 48.4% |
| Test (frozen) | 82,332 | 44.9% | 55.1% |

The validation split is a stratified 20% slice of the cleaned training data.
The GitHub Issue #1 fix removes predictor vectors with conflicting binary labels
before the split, so no complete predictor vector can appear in both train and
validation.

## Quick start

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The numbered scripts run in order from the project root. The first block rebuilds
the shared data foundation; the download step needs internet access.

```bash
# 1. Shared data workflow (Isaac)
python src/00_download.py        # raw download and integrity checks
python src/01_eda.py             # EDA summary and figures
python src/02_clean_split.py     # cleaning, deduplication, leakage removal, splits
python src/03_pipelines.py       # train-only preprocessing artifacts
python src/04_baseline.py        # dummy-classifier baselines
```

```bash
# 2. Model experiments (May: 05/08, Shivam: 06/07)
python src/05_logistic_regression.py
python src/06_random_forest.py
python src/07_xgboost.py
python src/08_neural_network.py
python src/08_neural_network_verify.py   # verify saved NN artifact reloads correctly
python src/08_rf_xgboost_analysis.py     # supplemental RF/XGBoost analysis
```

```bash
# 3. Standardized evaluation and thresholds (Sharwin)
python src/09_standardized_evaluation.py  # validation-only threshold selection
python src/10_final_test_evaluation.py    # frozen-test evaluation at locked thresholds
```

```bash
# 4. Explainability and advanced diagnostics
python src/11_diagnostics_verify.py    # provenance, artifact gate, leakage assertions
python src/12_shap_diagnostics.py      # SHAP global and local explanations
python src/13_model_diagnostics.py     # permutation importance, errors, calibration, drift
python src/14_ttl_ablation.py          # TTL-feature ablation
python src/15_build_report_tables.py   # report-ready tables
```

Re-running the model or diagnostics stages is not required to inspect the
results: every output is committed. Each diagnostics stage also accepts
`--verify`, which recomputes its outputs and compares them against the committed
files without writing anything.

```bash
python src/11_diagnostics_verify.py --verify
python src/15_build_report_tables.py --verify
python -m pytest tests/ -q
```

## Using the shared data and preprocessors

Do not create a new split or fit preprocessing on validation/test data. Load the
artifact that matches the model family. Fit candidate models on transformed
training data, use validation performance to select hyperparameters and decision
thresholds, and use the frozen test set only after all modelling choices are
finalized.

```python
import sys
import pandas as pd

sys.path.insert(0, "src")

from preprocess import load_artifact

preprocessor = load_artifact("artifacts/preprocess_linear.joblib")

X_train = preprocessor.transform(
    pd.read_parquet("data/processed/X_train.parquet")
)
X_val = preprocessor.transform(
    pd.read_parquet("data/processed/X_val.parquet")
)
y_train = pd.read_parquet("data/processed/y_train.parquet")["label"]
y_val = pd.read_parquet("data/processed/y_val.parquet")["label"]
```

| Model family | Primary artifact | Feature handling |
|---|---|---|
| Logistic Regression / Neural Network | `artifacts/preprocess_linear.joblib` | median imputation, RobustScaler, one-hot categorical encoding |
| Random Forest / XGBoost | `artifacts/preprocess_tree.joblib` | median imputation, ordinal categorical encoding |
| TTL ablation only | `artifacts/preprocess_*_with_ttl.joblib` | Same as above, with TTL features included |

The corrected primary linear artifact produces 66 features and the primary tree
artifact produces 39. The TTL-included variants produce 69 and 42 respectively.
The artifact loader treats a scikit-learn version mismatch as an error to prevent
silent preprocessing differences.

## Evaluation

Each model is evaluated using:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- PR-AUC
- confusion matrix

Each model owner selected and tuned that model's hyperparameters using validation
performance. Standardized decision-threshold analysis and cross-model evaluation
were then run on the finalized model outputs. The test set was not used for
hyperparameter or threshold selection.

The comparison focuses on precision, recall, F1, and PR-AUC. These metrics
better represent the cost of false alarms and missed attacks than accuracy alone.

### Existing baseline

The current baseline uses `DummyClassifier`. After the [Issue #1](https://github.com/shivammmmg/network-intrusion-detection-ml/issues/1) split fix, the
most-frequent strategy predicts the normal class because the development pool is
51.6% normal:

| Split | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 0.5157 | 0.0000 | 0.0000 | 0.0000 | 0.4843 |
| Test | 0.4494 | 0.0000 | 0.0000 | 0.0000 | 0.5506 |

See [`docs/baseline.md`](docs/baseline.md) for both baseline strategies. A
trained model should exceed these baselines on F1 and PR-AUC, not simply on
accuracy.

## Final results and key outputs

| Document | Contents |
|---|---|
| [`docs/standardized_model_evaluation.md`](docs/standardized_model_evaluation.md) | Validation-only threshold selection, frozen-test comparison across all four models, and the final model conclusion |
| [`docs/diagnostics_report.md`](docs/diagnostics_report.md) | SHAP explainability, permutation importance, error analysis, calibration, distribution drift, and the TTL ablation |
| [`docs/logistic_regression_results.md`](docs/logistic_regression_results.md) | Logistic Regression tuning history and finalized configuration |
| [`docs/neural_network_results.md`](docs/neural_network_results.md) | Neural Network architecture, training setup, and artifact verification |
| [`docs/random_forest_results.md`](docs/random_forest_results.md) | Random Forest tuning history and finalized configuration |
| [`docs/xgboost_results.md`](docs/xgboost_results.md) | XGBoost tuning history and finalized configuration |

Supporting output directories:

- `experiments/standardized_evaluation/` - threshold analyses, comparison tables,
  and validation/test figures.
- `experiments/diagnostics/` - provenance, SHAP values, permutation importance,
  error profiles, calibration, drift, TTL ablation, and generated report tables.
- `experiments/model_analysis/` - supplemental Random Forest and XGBoost
  analysis (bootstrap confidence intervals, hyperparameter sensitivity,
  selection stability, and inference efficiency).

### Reading the prediction files

Each `experiments/<model>/test_predictions.csv` contains `sample_index`,
`true_label`, `attack_probability`, and `predicted_label`. The
`attack_probability` column is `P(y = 1)`, and rows are aligned with
`data/processed/y_test.parquet`.

> **Important:** the `predicted_label` column is generated at the **default
> `0.50` threshold**, not at the validation-selected locked threshold. Any
> locked-threshold operating-point metric - accuracy, precision, recall, F1,
> or confusion-matrix counts - must be taken from the standardized evaluation
> outputs in `experiments/standardized_evaluation/final_test/`, or recomputed
> from `attack_probability` using
> `experiments/standardized_evaluation/selected_thresholds.json`. Do not read
> operating-point results out of `predicted_label`.

## Repository layout

```text
.
├── src/
│   ├── 00_download.py             # Download and validate raw UNSW-NB15 files
│   ├── 01_eda.py                  # EDA report and figures
│   ├── 02_clean_split.py          # Cleaning, deduplication, leakage removal, splits
│   ├── 03_pipelines.py            # Train-only preprocessing artifacts
│   ├── 04_baseline.py             # Dummy-classifier baselines
│   ├── 05_logistic_regression.py  # Final Logistic Regression experiment
│   ├── 06_random_forest.py        # Random Forest experiment
│   ├── 07_xgboost.py              # XGBoost experiment
│   ├── 08_neural_network.py       # Final Neural Network experiment
│   ├── 08_rf_xgboost_analysis.py  # Supplemental Random Forest/XGBoost analysis
│   ├── 08_neural_network_verify.py # Neural Network artifact verification
│   ├── 09_standardized_evaluation.py # Validation-only threshold selection
│   ├── 10_final_test_evaluation.py   # Frozen-test evaluation at locked thresholds
│   ├── 11_diagnostics_verify.py   # Provenance, artifact gate, leakage assertions
│   ├── 12_shap_diagnostics.py     # SHAP global and local explanations
│   ├── 13_model_diagnostics.py    # Permutation importance, errors, calibration, drift
│   ├── 14_ttl_ablation.py         # TTL-feature ablation
│   ├── 15_build_report_tables.py  # Report-ready table generation
│   ├── config.py                  # Paths, seed, feature roles
│   ├── diagnostics_lib.py         # Shared read-only diagnostics helpers
│   └── preprocess.py              # Cleaning and preprocessing builders
├── tests/                         # Unit tests for the diagnostics library and tables
├── data/processed/                # Shared X/y Parquet splits
├── artifacts/                     # Fitted models, preprocessors, and manifest
├── experiments/
│   ├── logistic_regression/       # Config, metrics, tuning history, predictions
│   ├── neural_network/            # Config, metrics, tuning history, predictions
│   ├── random_forest/             # Config, metrics, tuning history, predictions
│   │   └── pre_split_fix/         # Historical pre-split-fix results; not final
│   ├── xgboost/                   # Config, metrics, tuning history, predictions
│   ├── model_analysis/            # Supplemental RF/XGBoost analysis and figures
│   ├── standardized_evaluation/   # Thresholds, comparison tables, figures
│   └── diagnostics/               # Verified explainability and diagnostics outputs
├── docs/                          # Data card, EDA, baselines, results, reports
├── TEAM_RESPONSIBILITIES.md        # Team ownership, deliverables, and handoffs
├── CONTRIBUTIONS.md                # What each team member contributed
├── SUBMISSION_CHECKLIST.md         # Packaging notes for the final submission
├── requirements.txt
└── EECS 3404 Major Project Idea.pdf
```

`experiments/random_forest/pre_split_fix/` preserves Random Forest results
produced before the train/validation predictor-overlap correction. It is kept
for audit and comparison only and is **not** part of the final results; the
current Random Forest results are in `experiments/random_forest/`.

## Analysis and diagnostics

The agreed post-modelling work is complete:

- Decision-threshold analysis for the attack-detection versus false-positive
  trade-off, selected on validation data only.
- ROC and precision-recall curves, and standardized comparison tables across all
  four finalized models.
- False-positive and false-negative error analysis, including per-model error
  rates and representative cases.
- SHAP global and local explanations for the two tree models, with explicit
  scale metadata.
- Permutation feature importance for all four models.
- Probability calibration analysis with reliability curves, Brier score,
  expected calibration error, and worst-bin detection.
- Train-to-test distribution-drift analysis (PSI and KS), including which
  high-importance features also drift.
- TTL-feature ablation investigating dataset-specific shortcut signal.

Results are written up in
[`docs/standardized_model_evaluation.md`](docs/standardized_model_evaluation.md)
and [`docs/diagnostics_report.md`](docs/diagnostics_report.md). Team ownership
and handoff rules are defined in
[TEAM_RESPONSIBILITIES.md](TEAM_RESPONSIBILITIES.md).

### Contributions to this component

Paul contributed exploratory work toward the explainability and advanced
diagnostics component. After review, the final diagnostics implementation used
by the team was developed and verified by Shivam and is the version merged into
`main` and described above.

## Limitations

- UNSW-NB15 uses synthetic/testbed traffic and may not represent production
  networks.
- Some attack examples may be old or simulated; unseen attack types may not be
  recognized.
- TTL and connection-count features can reflect dataset construction rather
  than genuine attack behaviour.
- The class balance does not represent real-world attack prevalence.
- Encrypted traffic can remove useful inspection features.
- Any threshold or calibration conclusion from this dataset must be reassessed
  before deployment.

## Reproducibility

Dependencies are pinned in `requirements.txt`; the main scripts use the fixed
seed in `src/config.py`; and `artifacts/manifest.json` records package versions,
input hashes, preprocessing variants, and output feature counts. Generated
documentation records the shared split and baseline measurements. Exact fresh
Neural Network training may vary by Python, operating-system, and BLAS
environment; use the committed artifact verification for the saved handoff.

### Verified diagnostics

The diagnostics stages are read-only with respect to the finalized models: they
load frozen artifacts and apply already-fitted preprocessors, and the only model
fitting anywhere in the stage is the TTL ablation's train-only refit. Stage 0
records SHA-256 hashes of every model, split, and prediction file it consumes,
and asserts that the splits are disjoint and that no diagnostics source file
contains a stray fit call.

Every stage accepts `--verify`, which recomputes its outputs and compares them
against the committed files without writing. This makes stale results detectable
rather than silent, so any number quoted in the reports can be re-derived on
demand.

### A note on the NumPy version

The frozen training manifest records **NumPy 2.5.1**, the version under which the
model artifacts were produced. The final environment pins **NumPy 2.4.6**, because
SHAP depends on Numba, which requires NumPy below 2.5.

The Stage 0 artifact-to-prediction gate resolves this: it reloads each finalized
model, recomputes test probabilities under the diagnostics environment, and
compares them with the committed prediction files. All four models match within
tolerance (largest deviation about `3e-8`), so the finalized predictions are
unchanged by the version difference.

`artifacts/manifest.json` is deliberately **not** edited. It is a record of the
environment that produced the artifacts, and rewriting it would destroy that
provenance rather than fix anything.

## Academic use

This repository is an EECS 3404 course project. The work is for research and
educational evaluation only; it is not a production intrusion-detection system.
