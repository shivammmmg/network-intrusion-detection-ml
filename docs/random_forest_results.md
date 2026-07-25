# Random Forest Experiment

## 1. Objective

Random Forest was included as a nonlinear, tree-based baseline for the binary
UNSW-NB15 intrusion-detection comparison. It provides a useful complement to
linear and neural models because it can represent feature interactions and
nonlinear decision boundaries without requiring feature scaling or a parametric
linear form.

The positive class is `label=1` (attack); `label=0` represents normal traffic.

## 2. Data Foundation

The official experiment uses the corrected shared data foundation:

| Split | Rows |
|---|---:|
| Train | 79,685 |
| Validation | 19,922 |
| Test | 82,332 |

Random Forest uses the fitted, TTL-excluded tree preprocessing artifact:

```text
artifacts/preprocess_tree.joblib
```

The transformed representation contains 39 features. The preprocessor was fit
on the training split only and then applied with `transform()` to validation and
test. `RANDOM_STATE=42` was used consistently.

The corrected data foundation has zero exact full-predictor overlap across all
split pairs:

- Train ↔ validation: 0
- Train ↔ test: 0
- Validation ↔ test: 0

An earlier split was found to contain 1,565 validation rows (7.38%) whose full
predictor vector also appeared in training. That overlap was identified by an
independent audit and corrected before the official Random Forest rerun.

## 3. Why TTL Features Were Excluded

The primary model excludes `sttl`, `dttl`, and `ct_state_ttl`. These features
can reflect the way the benchmark traffic was generated rather than robust
network behavior, creating an unrealistic shortcut for classification.

TTL-inclusive preprocessing variants are retained for a later ablation analysis,
but the TTL-excluded 39-feature representation is the primary Random Forest
result.

## 4. Random Forest Method

Random Forest combines many decision trees trained on bootstrap samples of the
training data. Each tree also considers a random subset of features when making
splits. This bagging and feature-randomization process reduces the variance of
individual trees while retaining the ability to model nonlinear relationships.

For classification, the forest aggregates tree outputs into class probabilities;
the attack probability is `predict_proba()[:, 1]`. The default predicted label
uses a fixed threshold of 0.5. This structure is suitable for tabular intrusion
data containing heterogeneous numeric and categorical-derived predictors and
potentially complex feature interactions.

## 5. Hyperparameters Tuned

- `n_estimators`: number of trees in the ensemble; more trees generally reduce
  variance at increased computational cost.
- `max_depth`: maximum depth of each tree; this controls model complexity.
- `min_samples_leaf`: minimum number of training samples in a leaf; larger
  values regularize small, highly specific regions.
- `max_features`: number or fraction of predictors considered at each split;
  this controls tree diversity and split strength.
- `class_weight`: optional class weighting; `balanced` compensates for class
  frequency differences during fitting.

All candidates were fit on training data and ranked using validation results.
The test split was not opened during tuning.

## 6. Round 1 Staged Search

Round 1 tested 31 configurations through separate stages:

1. Baseline: the default Random Forest configuration.
2. Depth stage: 12 configurations varying `max_depth` across `None`, 15, 25,
   and 35 and `min_samples_leaf` across 1, 2, and 5, with 300 trees,
   `max_features="sqrt"`, and no class weighting.
3. Features stage: 12 configurations based on the two best depth-stage rows,
   varying `max_features` across `"sqrt"`, 0.3, and 0.5 and
   `class_weight` across `None` and `balanced`.
4. Estimators stage: 6 configurations based on the two best feature-stage rows,
   varying `n_estimators` across 200, 400, and 800.

The best Round 1 configuration was:

```text
n_estimators=400
max_depth=35
min_samples_leaf=2
max_features=0.5
class_weight=None
```

Its validation results were:

| Metric | Value |
|---|---:|
| Accuracy | 0.934846 |
| Precision | 0.919269 |
| Recall | 0.948803 |
| F1 | 0.933803 |
| ROC-AUC | 0.986014 |
| PR-AUC | 0.984321 |

Staged tuning is computationally efficient, but it does not fully evaluate
interactions among all hyperparameters simultaneously.

## 7. Round 2 Joint Search

Round 2 addressed those possible interactions with a full Cartesian grid:

```text
n_estimators     = [400, 800, 1000]
max_depth        = [25, 35]
min_samples_leaf = [1, 2]
max_features     = [0.3, 0.5]
class_weight     = [None, "balanced"]
```

This produced `3 × 2 × 2 × 2 × 2 = 48` configurations. The best Round 2
configuration was:

```text
n_estimators=800
max_depth=25
min_samples_leaf=1
max_features=0.3
class_weight=balanced
```

Its validation results were:

| Metric | Value |
|---|---:|
| Accuracy | 0.934645 |
| Precision | 0.916808 |
| Recall | 0.951394 |
| F1 | 0.933781 |
| ROC-AUC | 0.985987 |
| PR-AUC | 0.984366 |

Round 2 improved validation PR-AUC over the best Round 1 row by approximately
`+0.000044517`.

## 8. Final Model Selection

The selection rule was:

1. Maximize validation PR-AUC (`average_precision_score`).
2. Use validation ROC-AUC as the tie-breaker.

Accuracy, F1, and test performance were not used for hyperparameter selection.
The Round 2 winner was locked before the final test evaluation. A fresh model
was then fit on the corrected training split only.

The official final configuration is:

```text
n_estimators=800
max_depth=25
min_samples_leaf=1
max_features=0.3
class_weight=balanced
random_state=42
n_jobs=-1
```

The corrected-data experiment contains 79 tuning fits (31 Round 1 plus 48
Round 2) and one final refit, for 80 Random Forest fits in total. The final
train-only refit took 22.573 seconds.

## 9. Validation Results

| Metric | Value |
|---|---:|
| Accuracy | 0.934645 |
| Precision | 0.916808 |
| Recall | 0.951394 |
| F1 | 0.933781 |
| ROC-AUC | 0.985987 |
| PR-AUC | 0.984366 |

## 10. Test Results

| Metric | Value |
|---|---:|
| Accuracy | 0.868848 |
| Precision | 0.815760 |
| Recall | 0.984051 |
| F1 | 0.892037 |
| ROC-AUC | 0.977464 |
| PR-AUC | 0.982272 |

These are reference metrics from the finalized model. The standardized
cross-model comparison should be recomputed downstream from the saved
prediction CSVs.

## 11. Interpretation

The final model has very high test recall (0.984051), meaning it identifies the
vast majority of attack rows. Precision is lower (0.815760), so the model also
labels a meaningful number of normal rows as attacks. This reflects the
project's operational trade-off between detecting attacks and limiting false
alarms.

The reported predictions use the default 0.5 threshold. Later threshold
analysis can examine whether a different operating point improves the balance
between recall and false positives; that analysis is outside this experiment.

## 12. Validation vs Test Gap

| Metric | Validation | Test |
|---|---:|---:|
| PR-AUC | 0.984366 | 0.982272 |
| Accuracy | 0.934645 | 0.868848 |

Ranking performance remains strong, with only a modest PR-AUC difference. The
threshold-based accuracy and precision changed more substantially. Differences
in the development and test distributions may contribute, but this experiment
does not establish a single cause. Detailed drift and error analysis is
reserved for later work.

## 13. Reproducibility

The experiment can be reconstructed from:

- `RANDOM_STATE=42`.
- The fixed corrected train/validation/test parquet files.
- `artifacts/preprocess_tree.joblib`, used with transform-only application.
- `experiments/random_forest/tuning_results.csv`.
- `experiments/random_forest/round2_joint_search.csv`.
- `experiments/random_forest/config.json`.
- `artifacts/random_forest.joblib`.
- The saved validation and test prediction CSVs.
- The historical pre-split-fix archive at
  `experiments/random_forest/pre_split_fix/`.

The test split was used only after the final validation-selected configuration
was locked and the final training-only refit was complete.

## 14. Files Produced

Official corrected-data outputs:

```text
artifacts/random_forest.joblib
experiments/random_forest/config.json
experiments/random_forest/metrics.json
experiments/random_forest/validation_predictions.csv
experiments/random_forest/test_predictions.csv
experiments/random_forest/tuning_results.csv
experiments/random_forest/round2_joint_search.csv
experiments/random_forest/README.md
```

Historical pre-split-fix outputs are preserved under:

```text
experiments/random_forest/pre_split_fix/
```

## 15. Limitations / Future Analysis

Only the selected Random Forest hyperparameters were tuned. The search was
substantial but was not exhaustive over every possible Random Forest parameter.
The default threshold of 0.5 is not necessarily optimal for the final operating
objective.

Threshold analysis, SHAP, calibration, drift analysis, and TTL ablation are
handled as later project work and are not part of this finalized experiment.
