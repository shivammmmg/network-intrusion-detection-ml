# Team Responsibilities

## Project

**Explainable and Drift-Aware Machine Learning for Network Intrusion Detection**  
**Dataset:** UNSW-NB15  
**Task:** binary classification (`0` = normal traffic, `1` = attack traffic)

This document defines who owns each part of the EECS 3404 project, what they
should deliver, and how work moves between team members. The goal is to keep
the experiments comparable and prevent data leakage or duplicated work.

| Owner | Primary responsibility | Main model/artifact |
|---|---|---|
| Isaac | Data preparation and shared ML pipeline | Shared splits, preprocessing, and baselines |
| May | Logistic Regression and Neural Network experiments | Logistic Regression and Neural Network; `artifacts/preprocess_linear.joblib` |
| Shivam | Tree-model experiments | Random Forest and XGBoost; `artifacts/preprocess_tree.joblib` |
| Sharwin | Standardized evaluation and comparison | Finalized model outputs from May and Shivam |
| Paul | Explainability and advanced diagnostics | Finalized models, predictions, and evaluation outputs |

## Isaac - Data Preparation and Shared ML Pipeline

Isaac owns the common data foundation that every model uses. This work is
implemented in scripts `00` through `04`.

### Owns

- Dataset download and integrity validation.
- Exploratory data analysis.
- Cleaning and duplicate handling.
- Leakage investigation and removal.
- Fixed train/validation/test splits.
- Shared preprocessing pipelines and saved artifacts.
- TTL-excluded and TTL-included preprocessing variants.
- Dummy/baseline classifiers.
- Dataset documentation and reproducibility manifests.

### Deliverables

- Shared Parquet splits in `data/processed/`.
- Preprocessing artifacts in `artifacts/`.
- Data card, EDA summary, split manifest, and baseline results in `docs/`.
- A documented leakage decision: the primary artifacts exclude the TTL features;
  TTL-included variants are for the ablation comparison only.

### Boundary and handoff

All other team members must use Isaac's fixed splits and preprocessing artifacts.
No one should create their own split or refit preprocessing with validation or
test data.

## May - Logistic Regression and Neural Network

May owns the Logistic Regression and Neural Network model tracks.

### Owns

- Logistic Regression implementation.
- Neural Network implementation.
- Initial/default experiments for both models.
- Hyperparameter tuning for both models.
- Model selection using validation data only.
- Saving finalized fitted models.
- Exporting validation probabilities and predicted labels.
- Exporting final test probabilities and predicted labels only after the model
  configuration is locked.
- Recording selected hyperparameters and training runtime.

### Required input

Use:

```text
artifacts/preprocess_linear.joblib
```

This is the primary, TTL-excluded preprocessing artifact for Logistic Regression
and the Neural Network.

### Boundary and handoff

May owns model tuning, not the final cross-model comparison. Sharwin owns the
standardized evaluation and model comparison after May hands off finalized
outputs.

## Shivam - Random Forest and XGBoost

Shivam owns the two tree-model tracks.

### Owns

- Random Forest implementation.
- XGBoost implementation.
- Initial/default experiments for both models.
- Hyperparameter tuning for both models.
- Model selection using validation data only.
- Saving finalized fitted models.
- Exporting validation probabilities and predicted labels.
- Exporting final test probabilities and predicted labels only after the model
  configuration is locked.
- Recording selected hyperparameters and training runtime.

### Required input

Use:

```text
artifacts/preprocess_tree.joblib
```

This is the primary, TTL-excluded preprocessing artifact for Random Forest and
XGBoost.

### Boundary and handoff

Shivam owns model tuning, not the final cross-model comparison. Sharwin owns the
standardized evaluation and model comparison after Shivam hands off finalized
outputs.

## Sharwin - Evaluation, Threshold Analysis, and Model Comparison

Sharwin owns standardized evaluation for all four finalized models.

### Owns

- Accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
- Confusion matrices, ROC curves, and precision-recall curves.
- False-positive and false-negative rates.
- Validation-set decision-threshold analysis.
- Comparison of the default threshold with the selected operating threshold.
- The final comparison table across Logistic Regression, Neural Network, Random
  Forest, and XGBoost.
- Identifying the strongest model for the research objective.
- Analysis of the attack-detection versus false-positive tradeoff.

### Boundary and handoff

Model owners tune hyperparameters. Sharwin evaluates/selects decision thresholds
from validation probabilities. Sharwin must not use test results to tune
hyperparameters, select thresholds, or make further modelling decisions. After
all model configurations and thresholds are locked using training/validation
data, the frozen test set is used once for the final reported comparison across
models. Test results must not be used to go back and modify the models.

## Paul - Explainability and Advanced Diagnostics

Paul owns interpretation and reliability analysis of finalized models, with a
focus on the strongest tree-based models where appropriate.

### Owns

- SHAP global feature importance.
- SHAP beeswarm and bar plots.
- Local SHAP explanations.
- Representative true-positive, true-negative, false-positive, and
  false-negative explanations.
- Permutation feature importance.
- Analysis of why false positives and false negatives occur.
- Probability calibration analysis, reliability/calibration curves, and Brier
  score.
- Distribution-drift analysis between training and test data.
- Identification of important features that also show distribution shift.
- TTL-feature ablation on a selected finalized model using the corresponding
  TTL-included preprocessing artifact, followed by comparison and interpretation
  of whether TTL features provide potentially dataset-specific shortcut signal.

The TTL ablation should reuse an already-selected model configuration where
practical rather than starting a separate large hyperparameter search.

### Questions to answer

- Why is the model making these predictions?
- Can its probabilities be trusted?
- Is it relying on unstable or dataset-specific features?
- Why does it fail on certain samples?

SHAP and feature-importance results describe model behaviour; they do **not**
prove that a feature causes an attack.

## Shared Rules

1. Everyone uses the exact shared train/validation/test splits.
2. Preprocessing must never be refit using validation or test data.
3. Candidate models are fitted on training data. Hyperparameter configurations
   are selected using validation performance.
4. The frozen test set is used only after modelling decisions are finalized.
5. Model owners are responsible for tuning their own models.
6. Sharwin owns standardized evaluation and threshold analysis.
7. Paul owns explainability and advanced diagnostics.
8. All model outputs should use a common format so evaluation scripts can consume
   them.
9. Use `RANDOM_STATE = 42` whenever applicable.
10. Do not overwrite another team member's work without coordination.

## Standard Model Handoff

All four model owners/outputs must follow the same column names, class definition
(`1` = attack), probability convention (`attack_probability` = `P(y=1)`), and
sample identifiers. This prevents evaluation inconsistencies across models.

For each finalized model, May and Shivam should provide Sharwin and Paul with:

- Saved fitted model.
- Selected hyperparameters.
- Training runtime.
- Validation predicted probabilities.
- Validation predicted labels.
- Test predicted probabilities.
- Test predicted labels.
- Feature names used by the model.
- Short experiment summary, including the chosen configuration and any relevant
  preprocessing/artifact variant.

Test predictions are handed off only after the model configuration is locked.
The shared convention is a target structure for new experiments; it does not
require retrofitting files that are not yet implemented in this repository.

```text
experiments/
├── logistic_regression/
├── neural_network/
├── random_forest/
└── xgboost/
```

Each model folder may contain:

```text
config.json
validation_predictions.csv
test_predictions.csv
metrics.json
tuning_results.csv
```

For prediction files, the recommended common fields are a stable sample index,
the true label, the attack probability, and the predicted label. This lets
Sharwin reuse one evaluator for every model and lets Paul match error cases back
to the relevant split.

## Workflow and Handoffs

```text
Isaac
Data + preprocessing + fixed splits + baseline

        |
        v

May ---------------- Shivam
LR + NN              RF + XGBoost
Model tuning         Model tuning

        \              /
         \            /
          v          v

             Sharwin
Evaluation + thresholds + comparison

                 |
                 v

                Paul
Explainability + calibration + drift + advanced diagnostics
```

Some work can happen in parallel. In particular, Paul can begin error analysis,
feature-importance work, and calibration preparation as soon as finalized
prediction files and models become available; Sharwin does not need to finish
the final comparison table first.

## Workload Note

This division is intended to keep workloads reasonably balanced:

- Isaac owns the foundational data-engineering work.
- May and Shivam each own two models and their tuning.
- Sharwin owns standardized evaluation, threshold selection, and comparative
  analysis.
- Paul owns explainability, calibration, drift, and advanced diagnostics.

Do not add experiments beyond this scope unless the team explicitly agrees to
them first.
