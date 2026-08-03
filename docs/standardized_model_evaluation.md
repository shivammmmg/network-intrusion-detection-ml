# Standardized Model Evaluation

## Purpose and research question

This evaluation compares the finalized Logistic Regression, Neural Network,
Random Forest, and XGBoost classifiers under one consistent binary protocol:
normal traffic is class 0, attack traffic is class 1, and
`attack_probability` is interpreted as \(P(y=1)\). The research question is
which model best detects attacks while controlling false-positive alerts on a
previously unseen, frozen test set.

Accuracy is reported but is not used alone. Precision, recall, F1, PR-AUC,
ROC-AUC, false positives, false negatives, false-positive rate, and
false-negative rate are considered because intrusion detection must balance
missed attacks against operational alert burden.

## Validation-only threshold selection

Threshold selection was completed before this final test evaluation using only
the saved validation predictions. For each model, every distinct validation
score threshold was evaluated and the threshold maximizing validation F1 was
selected. Exact F1 ties were resolved by higher precision, then higher recall,
then higher threshold. The frozen test predictions were not opened during that
selection stage.

The resulting thresholds were saved in
`experiments/standardized_evaluation/selected_thresholds.json` and locked:

| Model | Locked validation-selected threshold |
|---|---:|
| Logistic Regression | 0.4631204833813834 |
| Neural Network | 0.4577027505449918 |
| Random Forest | 0.4929682597961830 |
| XGBoost | 0.4630610300000000 |

The final test script loaded these exact values directly from JSON. It did not
search, optimize, recalculate, or alter thresholds using test performance, and
it did not train, refit, or tune any model. Test results were used only for
final evaluation; they were not used to revise thresholds, models,
preprocessing, or hyperparameters.

## Frozen-test integrity checks

All four prediction files contained 82,332 rows and the required numeric
columns. Sample indexes were unique, probabilities were within [0, 1], labels
were binary, and sample-index and true-label order matched across models. For
every model, the saved `predicted_label` exactly matched
`attack_probability >= 0.50`; all mismatch counts were zero.

The common test set contains 45,332 attack samples and 37,000 normal samples.

## Default-threshold test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.794321 | 0.747844 | 0.945116 | 0.834987 | 0.906538 | 0.927520 | 22,554 | 14,446 | 2,488 | 42,844 | 0.390432 | 0.054884 |
| Neural Network | 0.848224 | 0.804466 | 0.956940 | 0.874103 | 0.964204 | 0.972269 | 26,456 | 10,544 | 1,952 | 43,380 | 0.284973 | 0.043060 |
| Random Forest | 0.868848 | 0.815760 | 0.984051 | 0.892037 | 0.977464 | 0.982272 | 26,925 | 10,075 | 723 | 44,609 | 0.272297 | 0.015949 |
| XGBoost | 0.871144 | 0.821921 | 0.977830 | 0.893122 | 0.980900 | 0.985927 | 27,396 | 9,604 | 1,005 | 44,327 | 0.259568 | 0.022170 |

At the common 0.50 threshold, XGBoost has the highest F1, precision, PR-AUC,
and ROC-AUC and produces the fewest false positives. Random Forest has the
highest recall and fewest false negatives.

## Locked-threshold test results

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.463120 | 0.795632 | 0.746081 | 0.953256 | 0.837039 | 0.906538 | 0.927520 | 22,293 | 14,707 | 2,119 | 43,213 | 0.397486 | 0.046744 |
| Neural Network | 0.457703 | 0.838289 | 0.790566 | 0.960844 | 0.867427 | 0.964204 | 0.972269 | 25,461 | 11,539 | 1,775 | 43,557 | 0.311865 | 0.039156 |
| Random Forest | 0.492968 | 0.867136 | 0.813239 | 0.984867 | 0.890862 | 0.977464 | 0.982272 | 26,747 | 10,253 | 686 | 44,646 | 0.277108 | 0.015133 |
| XGBoost | 0.463061 | 0.864816 | 0.812655 | 0.980521 | 0.888731 | 0.980900 | 0.985927 | 26,753 | 10,247 | 883 | 44,449 | 0.276946 | 0.019479 |

ROC-AUC and PR-AUC do not change between the default and locked-threshold
tables because they assess probability ranking rather than one operating
threshold.

## Default-versus-locked threshold changes

All values below are locked threshold minus default threshold.

| Model | Delta Accuracy | Delta Precision | Delta Recall | Delta F1 | Delta FP | Delta FN | Delta FPR | Delta FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | +0.001312 | -0.001763 | +0.008140 | +0.002053 | +261 | -369 | +0.007054 | -0.008140 |
| Neural Network | -0.009935 | -0.013900 | +0.003905 | -0.006676 | +995 | -177 | +0.026892 | -0.003905 |
| Random Forest | -0.001713 | -0.002521 | +0.000816 | -0.001175 | +178 | -37 | +0.004811 | -0.000816 |
| XGBoost | -0.006328 | -0.009265 | +0.002691 | -0.004391 | +643 | -122 | +0.017378 | -0.002691 |

Lower validation-selected thresholds increase attack recall for every model and
reduce false negatives, but they also increase false-positive alerts and reduce
precision. Logistic Regression is the only model whose locked threshold also
improves test F1; the validation F1 improvement does not transfer to test F1
for Neural Network, Random Forest, or XGBoost. This is an important
generalization limitation, not an implementation error. It is reported as an
honest out-of-sample result and was not used to revise thresholds, models,
preprocessing, or hyperparameters.

## False-positive and false-negative interpretation

A false positive labels normal traffic as an attack. It adds investigation
load and can contribute to alert fatigue. A false negative labels an attack as
normal and represents an undetected threat. Lowering a threshold generally
trades more false alerts for fewer missed attacks, as the delta table shows.

At the locked thresholds, Logistic Regression misses 2,119 attacks and raises
14,707 false alerts, substantially more errors than the tree ensembles. Neural
Network improves both counts but still has 1,775 false negatives and 11,539
false positives. Random Forest and XGBoost are close on false positives (10,253
versus 10,247), while Random Forest misses 197 fewer attacks (686 versus 883).
Thus Random Forest achieves a stronger attack-detection result at essentially
the same false-alert burden as XGBoost at their locked operating points.

## Conclusion

Random Forest is the strongest model at the locked validation-selected
operating point. It has the highest locked-threshold F1 (0.890862), highest
recall (0.984867), lowest false-negative rate (0.015133), and fewest missed
attacks (686). Its precision (0.813239) is also marginally higher than XGBoost's
0.812655, while it produces only six more false positives. This combination
best answers the project's operational question of detecting attacks without a
material increase in false-alert burden.

XGBoost remains extremely competitive and has the strongest threshold-independent
ranking performance: it has the highest PR-AUC (0.985927) and highest ROC-AUC
(0.980900). At the locked threshold, XGBoost also produces six fewer false
positives than Random Forest (10,247 versus 10,253). It is strongest at the
common 0.50 threshold as well. The conclusion therefore applies specifically
to the precommitted locked operating thresholds; it does not imply that Random
Forest is universally superior to XGBoost or dominates it under every possible
threshold or deployment cost.

## Limitations

- Results come from one fixed train/validation/test partition of UNSW-NB15 and
  may not generalize to other networks, time periods, attack mixes, or class
  prevalence.
- The high locked-threshold false-positive rates (about 27.7% even for the two
  leading ensembles) may be operationally expensive.
- Validation-selected F1 thresholds assume equal emphasis on precision and
  recall; a deployment with explicit costs for missed attacks and false alerts
  could require a different predeclared objective and new validation procedure.
- Probability calibration, temporal drift, inference resource requirements,
  and robustness to novel attack families are not established by this table.
- This comparison is descriptive. No statistical significance claim is made
  from these final point estimates. Any separate uncertainty analysis must be
  interpreted according to its own assumptions and scope.
