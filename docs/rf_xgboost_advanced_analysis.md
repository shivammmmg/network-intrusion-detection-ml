# Advanced Random Forest and XGBoost Analysis

## Purpose

This is post-training analysis of the finalized Random Forest and XGBoost
experiments. It does not retune, reset thresholds, retrain models, or alter
the shared data foundation. Validation tuning outputs and frozen-test
descriptive outputs are kept separate throughout.

## Data and model status

Both models use the corrected shared splits and the transform-only
`artifacts/preprocess_tree.joblib` artifact, producing 39 tree features. The
saved final prediction files were not overwritten. The frozen test set contains
6,318 predictor vectors that occur more than once, involving 34,704 rows in
total. Equivalently, there are 28,386 duplicate row occurrences beyond the
first instance of each unique predictor vector. These are internal test-set
duplicates; train–validation, train–test, and validation–test predictor
overlap remains zero. The official test rows were not deduplicated.

## Duplicate configuration handling

Stability counts use one representative row per unique full hyperparameter
configuration. Repeated staged/refit rows are not counted as separate
candidates:

- **Random Forest:** 79 raw fit rows, 75 unique full configurations, 4 duplicate configuration groups, 4 extra refit occurrences; duplicate metrics reproducible to logged precision: `False`; maximum logged metric range: `0.000001`.
- **XGBoost:** 113 raw fit rows, 95 unique full configurations, 14 duplicate configuration groups, 18 extra refit occurrences; duplicate metrics reproducible: `True`.

Duplicate metrics and XGBoost `best_iteration` values were checked for
reproducibility. Sensitivity summaries retain stage context; Round 2 matched
comparisons are reported separately as controlled one-factor comparisons.

## Random Forest sensitivity

The RF analysis uses the 31 Round 1 rows and 48 Round 2 rows. The Round 2
Cartesian search supports matched comparisons in which one parameter changes
while the other Round 2 parameters are held fixed. These effects are
descriptive within the searched region and are not claimed to be causal
outside that design.

Controlled Round 2 mean PR-AUC changes (higher level minus lower level):

| parameter | levels | paired_comparisons | mean_high_minus_low_pr_auc | min_high_minus_low_pr_auc | max_high_minus_low_pr_auc |
| --- | --- | --- | --- | --- | --- |
| n_estimators | ['400', '800', '1000'] | 48 | 0.000105 | -0.000109 | 0.000565 |
| max_depth | ['25', '35'] | 24 | -0.000050 | -0.000399 | 0.000225 |
| min_samples_leaf | ['1', '2'] | 24 | 0.000053 | -0.000276 | 0.000660 |
| max_features | ['0.3', '0.5'] | 24 | 0.000146 | -0.000213 | 0.000602 |
| class_weight | ['None', 'balanced'] | 24 | 0.000110 | -0.000097 | 0.000361 |

The matched RF `n_estimators` comparisons are shown separately because the
Round 2 grid has three estimator levels:

| level | paired_delta_mean_pr_auc | min_val_pr_auc | max_val_pr_auc |
| --- | --- | --- | --- |
| 400 -> 800 | 0.000162 | 0.984204 | 0.984366 |
| 400 -> 1000 | 0.000157 | 0.984204 | 0.984361 |
| 800 -> 1000 | -0.000005 | 0.984361 | 0.984366 |
| 400 -> 800 | -0.000013 | 0.984308 | 0.984321 |
| 400 -> 1000 | -0.000029 | 0.984292 | 0.984321 |
| 800 -> 1000 | -0.000016 | 0.984292 | 0.984308 |
| 400 -> 800 | 0.000231 | 0.984013 | 0.984244 |
| 400 -> 1000 | 0.000290 | 0.984013 | 0.984303 |
| 800 -> 1000 | 0.000059 | 0.984244 | 0.984303 |
| 400 -> 800 | 0.000248 | 0.983978 | 0.984226 |
| 400 -> 1000 | 0.000295 | 0.983978 | 0.984273 |
| 800 -> 1000 | 0.000047 | 0.984226 | 0.984273 |
| 400 -> 800 | -0.000024 | 0.984231 | 0.984255 |
| 400 -> 1000 | 0.000008 | 0.984255 | 0.984263 |
| 800 -> 1000 | 0.000032 | 0.984231 | 0.984263 |
| 400 -> 800 | 0.000470 | 0.983661 | 0.984131 |
| 400 -> 1000 | 0.000565 | 0.983661 | 0.984226 |
| 800 -> 1000 | 0.000095 | 0.984131 | 0.984226 |
| 400 -> 800 | 0.000193 | 0.984011 | 0.984204 |
| 400 -> 1000 | 0.000211 | 0.984011 | 0.984222 |
| 800 -> 1000 | 0.000018 | 0.984204 | 0.984222 |
| 400 -> 800 | -0.000094 | 0.984125 | 0.984219 |
| 400 -> 1000 | -0.000109 | 0.984110 | 0.984219 |
| 800 -> 1000 | -0.000015 | 0.984110 | 0.984125 |
| 400 -> 800 | -0.000002 | 0.984205 | 0.984207 |
| 400 -> 1000 | -0.000010 | 0.984197 | 0.984207 |
| 800 -> 1000 | -0.000008 | 0.984197 | 0.984205 |
| 400 -> 800 | 0.000016 | 0.984137 | 0.984153 |
| 400 -> 1000 | 0.000037 | 0.984137 | 0.984174 |
| 800 -> 1000 | 0.000021 | 0.984153 | 0.984174 |
| 400 -> 800 | 0.000268 | 0.983842 | 0.984110 |
| 400 -> 1000 | 0.000315 | 0.983842 | 0.984157 |
| 800 -> 1000 | 0.000047 | 0.984110 | 0.984157 |
| 400 -> 800 | 0.000181 | 0.983893 | 0.984074 |
| 400 -> 1000 | 0.000198 | 0.983893 | 0.984091 |
| 800 -> 1000 | 0.000017 | 0.984074 | 0.984091 |
| 400 -> 800 | -0.000022 | 0.984031 | 0.984053 |
| 400 -> 1000 | 0.000007 | 0.984053 | 0.984060 |
| 800 -> 1000 | 0.000029 | 0.984031 | 0.984060 |
| 400 -> 800 | 0.000398 | 0.983617 | 0.984015 |
| 400 -> 1000 | 0.000389 | 0.983617 | 0.984006 |
| 800 -> 1000 | -0.000009 | 0.984006 | 0.984015 |
| 400 -> 800 | 0.000080 | 0.983887 | 0.983967 |
| 400 -> 1000 | 0.000107 | 0.983887 | 0.983994 |
| 800 -> 1000 | 0.000027 | 0.983967 | 0.983994 |
| 400 -> 800 | -0.000054 | 0.983768 | 0.983822 |
| 400 -> 1000 | 0.000096 | 0.983822 | 0.983918 |
| 800 -> 1000 | 0.000150 | 0.983768 | 0.983918 |

The selected RF depth is inside the broader searched depth set but is at the
lower boundary of the corrected-data Round 2 depth values. Nearby high-ranked
configurations and the stability table below should be read as evidence about
the observed plateau, not as a global optimum claim. `n_estimators` and the
other parameters are also summarized in
`rf_hyperparameter_sensitivity.csv`, with staged aggregate rows clearly
labelled as descriptive.

## XGBoost sensitivity

XGBoost uses 49 Round 1 fits and 64 Round 2 fits. The Round 2 grid jointly
varied depth, child weight, subsampling, column sampling, L2 regularization,
and L1 regularization at `learning_rate=0.1`. The controlled effects are:

| parameter | levels | paired_comparisons | mean_high_minus_low_pr_auc | min_high_minus_low_pr_auc | max_high_minus_low_pr_auc |
| --- | --- | --- | --- | --- | --- |
| max_depth | ['8', '10'] | 32 | 0.000142 | -0.000305 | 0.000622 |
| min_child_weight | ['1', '5'] | 32 | -0.000191 | -0.000601 | 0.000266 |
| subsample | ['0.85', '1'] | 32 | 0.000327 | -0.000060 | 0.000769 |
| colsample_bytree | ['0.6', '0.8'] | 32 | -0.000021 | -0.000521 | 0.000313 |
| reg_lambda | ['1', '5'] | 32 | -0.000139 | -0.000540 | 0.000294 |
| reg_alpha | ['0', '1'] | 32 | -0.000052 | -0.000365 | 0.000476 |

Round 1 learning-rate observations:

| learning_rate | n_rows | mean_best_iteration | min_best_iteration | max_best_iteration | mean_val_pr_auc |
| --- | --- | --- | --- | --- | --- |
| 0.030000 | 2 | 947.500000 | 695 | 1200 | 0.985645 |
| 0.050000 | 2 | 639.000000 | 608 | 670 | 0.985720 |
| 0.100000 | 2 | 318.000000 | 282 | 354 | 0.985834 |

Lower learning rates required more boosting iterations in this early-stopping
search, while the relationship is an empirical pattern over the tested
settings rather than an exact law. The strongest Round 2 region is concentrated
around depth 8–10, child weight 1, full subsampling, and column sampling 0.6–0.8.
The exact Round 1 winner was reproduced as the Round 2 winner, which supports
selection stability but does not establish global optimality.

## Model-selection stability

| model | unique_configurations | winner_pr_auc | second_pr_auc | fifth_pr_auc | tenth_pr_auc | top5_pr_auc_range | top10_pr_auc_range | within_0.0001 | within_0.0005 | within_0.0010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 75 | 0.984366 | 0.984361 | 0.984303 | 0.984256 | 0.000063 | 0.000110 | 7 | 52 | 67 |
| XGBoost | 95 | 0.985938 | 0.985923 | 0.985729 | 0.985663 | 0.000209 | 0.000275 | 3 | 26 | 81 |

The `within_*` columns count unique configurations only. The top-five and
top-ten ranges are validation PR-AUC ranges, not confidence intervals. These
differences should not be interpreted as statistical significance.

## Computational efficiency and model size

| model | round1_fits | round2_fits | total_tuning_fits | round1_logged_fit_seconds | round2_logged_fit_seconds | final_training_runtime_seconds | joblib_size_bytes | native_json_size_bytes | validation_pr_auc | test_pr_auc | test_roc_auc | test_f1 | test_precision | test_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 31 | 48 | 79 | 273.365 | 1369.817 | 22.573 | 84012461 | nan | 0.984 | 0.982 | 0.977 | 0.892 | 0.816 | 0.984 |
| XGBoost | 49 | 64 | 113 | 147.012 | 181.647 | 1.579 | 3646034 | 5880165.000 | 0.986 | 0.986 | 0.981 | 0.893 | 0.822 | 0.978 |

Model serialization size is a disk-size measurement, not a memory-footprint
measurement. The inference benchmark below times only `predict_proba` after
two warm-up calls on the same transformed test matrix, using ten repetitions
per model on this machine.

| model | median_seconds | mean_seconds | total_seconds | median_rows_per_second |
| --- | --- | --- | --- | --- |
| Random Forest | 0.4080 | 0.4166 | 4.1665 | 201801.7825 |
| XGBoost | 0.0534 | 0.0556 | 0.5563 | 1543115.6369 |

These timings are local engineering measurements and may vary with laptop
load, threading, and library/runtime state.

## Bootstrap uncertainty on finalized test predictions

The analysis uses `2000` paired row-wise
bootstrap replicates with seed `42`. The same sampled
test-row indices were applied to RF and XGBoost in each replicate. Valid
replicates: `2000`; skipped one-class
replicates: `0`.

### Individual model intervals

| model | metric | observed | ci_lower_2_5 | ci_upper_97_5 |
| --- | --- | --- | --- | --- |
| Random Forest | pr_auc | 0.982272 | 0.981642 | 0.982925 |
| Random Forest | roc_auc | 0.977464 | 0.976720 | 0.978224 |
| Random Forest | f1 | 0.892037 | 0.889991 | 0.894046 |
| Random Forest | precision | 0.815760 | 0.812484 | 0.819003 |
| Random Forest | recall | 0.984051 | 0.982875 | 0.985163 |
| Random Forest | accuracy | 0.868848 | 0.866576 | 0.871205 |
| XGBoost | pr_auc | 0.985927 | 0.985394 | 0.986444 |
| XGBoost | roc_auc | 0.980900 | 0.980241 | 0.981580 |
| XGBoost | f1 | 0.893122 | 0.891082 | 0.895167 |
| XGBoost | precision | 0.821921 | 0.818677 | 0.825162 |
| XGBoost | recall | 0.977830 | 0.976501 | 0.979164 |
| XGBoost | accuracy | 0.871144 | 0.868885 | 0.873452 |

### Paired differences: XGBoost minus Random Forest

| metric | observed_xgboost_minus_rf | bootstrap_median_difference | ci_lower_2_5 | ci_upper_97_5 |
| --- | --- | --- | --- | --- |
| pr_auc | 0.003656 | 0.003656 | 0.003279 | 0.004002 |
| roc_auc | 0.003436 | 0.003434 | 0.003034 | 0.003800 |
| f1 | 0.001085 | 0.001084 | 0.000087 | 0.002078 |
| precision | 0.006161 | 0.006147 | 0.004749 | 0.007623 |
| recall | -0.006221 | -0.006214 | -0.007301 | -0.005149 |

An interval that includes zero does not support a clear directional difference
under this paired bootstrap; an interval excluding zero is still an empirical
uncertainty result, not a license for test-based retuning. The row-wise
bootstrap samples the 82,332 test rows as observations, although 34,704 rows
belong to repeated predictor-vector groups. Therefore, the reported intervals
describe row-wise bootstrap uncertainty and may not fully represent uncertainty
over independent predictor patterns. A grouped/unique-vector bootstrap could
be a later sensitivity analysis, but it is not silently substituted for this
primary result.

## RF versus XGBoost interpretation

On the saved default-threshold outputs, XGBoost has higher test PR-AUC,
ROC-AUC, accuracy, precision, and F1, while RF has higher recall. This means
XGBoost currently appears stronger for ranking/separation and the fixed 0.5
operating point, whereas RF catches slightly more attacks with more false
positives. The project question— which model best balances malicious-traffic
detection against false-positive alerts—cannot be finalized solely from this
default-threshold snapshot. The standardized threshold analysis assigned to
the downstream workflow remains the appropriate basis for the eventual
operating-threshold decision.

## Figures

Generated figures:

- `experiments/model_analysis/figures/rf_hyperparameter_sensitivity.png`
- `experiments/model_analysis/figures/xgboost_hyperparameter_sensitivity.png`
- `experiments/model_analysis/figures/xgboost_learning_rate_best_iteration.png`
- `experiments/model_analysis/figures/model_selection_stability.png`
- `experiments/model_analysis/figures/efficiency_performance.png`
- `experiments/model_analysis/figures/bootstrap_metric_confidence_intervals.png`
- `experiments/model_analysis/figures/bootstrap_paired_difference_intervals.png`

## Limitations and handoff

- Sensitivity results are limited to the searched configurations and preserve
  staged-search context; pooled associations are not isolated causal effects.
- Stability differences are descriptive validation gaps, not significance
  tests.
- Runtime and serialization sizes are machine/runtime-specific.
- Test bootstrap intervals are post-hoc descriptive uncertainty estimates and
  inherit the frozen test set's internal duplicate-row limitation.
- Threshold optimization, FP/FN case analysis, SHAP/feature-importance
  interpretation, calibration, drift, TTL ablation, Logistic Regression, and
  Neural Network analysis remain deferred to the assigned workflows.

## Reproducibility

The analysis runner is `src/08_rf_xgboost_analysis.py`. It uses Python/library
versions recorded in `experiments/model_analysis/analysis_summary.json`,
`random_state=42`, and bootstrap seed 42. It reads the existing tuning logs,
saved models, saved predictions, and preprocessing artifact without fitting
or changing them.
