# Explainable and Drift-Aware Machine Learning for Network Intrusion Detection

EECS 3404 major project exploring binary network-intrusion detection with the
UNSW-NB15 dataset. The project compares machine-learning models by their ability
to detect attack traffic while limiting false-positive alerts, with particular
attention to data leakage, class imbalance, explainability, calibration, and
distribution drift.

> **Project status:** the shared data workflow, exploratory analysis, leakage
> audit, fixed train/validation/test split, fitted preprocessing artifacts, and
> baseline classifiers are implemented. Logistic Regression and Neural Network
> are finalized; standardized evaluation and advanced analysis remain in
> progress.

The original proposal is available in
[`EECS 3404 Major Project Idea.pdf`](./EECS%203404%20Major%20Project%20Idea.pdf).

## Research question

Which machine-learning model provides the best balance between detecting
malicious network traffic and minimizing false-positive alerts?

## Scope

The initial task is binary classification:

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

### Final Logistic Regression and Neural Network handoff

- Logistic Regression is finalized, with its saved model, configuration, metrics,
  and validation/test predictions available in `artifacts/` and
  `experiments/logistic_regression/`.
- Neural Network is finalized with the corrected validation-selected winner:
  `alpha=0.0001` and a 28-epoch final refit. Its saved model, configuration,
  metrics, and validation/test predictions are available in `artifacts/` and
  `experiments/neural_network/`.
- The updated Neural Network artifact and prediction files pass the committed
  verification script. Standardized validation threshold selection and
  locked-threshold test evaluation have been rerun using the finalized Logistic
  Regression, Neural Network, Random Forest, and XGBoost outputs.
- XGBoost is strongest at the default `0.50` threshold, while Random Forest is
  strongest at the validation-selected locked thresholds. Sharwin's evaluation
  is finalized.
- Exact fresh Neural Network training can vary across Python, operating-system,
  and BLAS environments. The committed artifact verification remains reproducible
  with the saved model and preprocessing artifact.

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

To rebuild the currently implemented data workflow from scratch, run the
numbered scripts in order:

```bash
python src/00_download.py
python src/01_eda.py
python src/02_clean_split.py
python src/03_pipelines.py
python src/04_baseline.py
```

This regenerates the raw-data checks, EDA summary and figures, processed splits,
fitted preprocessors, manifests, and baseline reports. The download step needs
internet access.

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

Each model should report at least:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- PR-AUC
- confusion matrix

Each model owner selects and tunes that model's hyperparameters using validation
performance. Standardized decision-threshold analysis and cross-model evaluation
occur after finalized model outputs are available. The test set must not be used
for hyperparameter or threshold selection.

The required comparison should focus on precision, recall, F1, and PR-AUC.
These metrics better represent the cost of false alarms and missed attacks than
accuracy alone.

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
│   ├── 08_rf_xgboost_analysis.py  # Random Forest/XGBoost analysis
│   ├── 08_neural_network_verify.py # Neural Network artifact verification
│   ├── config.py                  # Paths, seed, feature roles
│   └── preprocess.py              # Cleaning and preprocessing builders
├── data/processed/                # Shared X/y Parquet splits
├── artifacts/                     # Fitted preprocessors and manifests
├── docs/                          # Data card, EDA, baselines, split manifest
├── TEAM_RESPONSIBILITIES.md        # Team ownership, deliverables, and handoffs
├── requirements.txt
└── EECS 3404 Major Project Idea.pdf
```

## Planned experiments

The agreed work after model development includes:

- Decision-threshold analysis for the attack-detection versus false-positive
  trade-off.
- False-positive and false-negative error analysis.
- Feature importance and SHAP-based explainability.
- Probability calibration analysis using reliability curves and Brier score.
- TTL-feature ablation to investigate possible dataset-specific shortcut
  signals.
- Train/test distribution-drift analysis.
- ROC and precision-recall curves.
- Standardized comparison tables across all four finalized models.
- Final report and presentation walkthrough.

Team ownership and handoff rules are defined in
[TEAM_RESPONSIBILITIES.md](TEAM_RESPONSIBILITIES.md).

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

## Academic use

This repository is an EECS 3404 course project. The work is for research and
educational evaluation only; it is not a production intrusion-detection system.
