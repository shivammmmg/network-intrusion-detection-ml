# Logistic Regression results

## Purpose

Logistic Regression is the interpretable linear baseline in the four-model
comparison. It provides a reference point for how much of the attack/normal
separation in UNSW-NB15 is reachable with a linear decision boundary, against
which the Neural Network and the two tree models are judged.

## Preprocessing

The model consumes the shared, train-only fitted linear artifact:

| Item | Value |
|---|---|
| Artifact | `artifacts/preprocess_linear.joblib` |
| Application mode | `transform_only` |
| Features out | 66 |
| TTL features | excluded (primary pipeline) |

The artifact is applied with `transform()` only; nothing is refit on validation
or test data. Steps are median imputation, `RobustScaler`, and one-hot encoding
of the categorical columns.

## Model selection

Selection used validation data only, in two rounds:

| Item | Value |
|---|---|
| Round 1 candidates | 18 rows (`tuning_results.csv`) |
| Round 2 candidates | 30 rows (`round2_joint_search.csv`) |
| Primary metric | validation PR-AUC |
| Tie-breaker | validation ROC-AUC |
| Selected rank | 1 |
| Parameters searched | `C`, `penalty`, `solver`, `class_weight` |

Top three Round 2 candidates, as recorded in `round2_joint_search.csv`:

| Rank | Config | Val PR-AUC | Val ROC-AUC | Converged | Iterations |
|---:|---|---:|---:|:--|---:|
| 1 | `round2_liblinear_l1_C100.0_weightnone` | 0.913102 | 0.936002 | yes | 25 |
| 2 | `round2_liblinear_l1_C10.0_weightnone` | 0.912992 | 0.935964 | yes | 24 |
| 3 | `round2_liblinear_l1_C1.0_weightnone` | 0.912161 | 0.935560 | yes | 25 |

20 of the 30 Round 2 candidates converged. The 10 that did not all ranked below
the selected winner, so non-convergence did not affect the choice.

The frozen test set was not used at any point in selection.

## Final configuration

```json
{
  "C": 100.0,
  "class_weight": null,
  "max_iter": 10000,
  "penalty": "l1",
  "random_state": 42,
  "solver": "liblinear",
  "tol": 0.0001
}
```

Round 2 candidates were fit with `max_iter=3000` and `tol=0.001` to keep the
search tractable. After the winner was locked, the final train-only refit used
`max_iter=10000` and `tol=0.0001` so the selected model was fit to completion.
These are optimiser settings only; no selected hyperparameter changed between
the search and the refit. The final model converged after 34 iterations.

This is why the winner's search-time validation PR-AUC (0.913102) differs
slightly from the finalized validation PR-AUC below (0.913401): they come from
two different fits of the same configuration.

## Validation metrics

From `experiments/logistic_regression/metrics.json`:

| Metric | Value |
|---|---:|
| PR-AUC | 0.913401 |
| ROC-AUC | 0.936151 |
| Accuracy | 0.886256 |
| Precision | 0.839261 |
| Recall | 0.946419 |
| F1 | 0.889625 |

## Test metrics at the default 0.50 threshold

Recorded under `test_reference_only` in `metrics.json`. These were computed only
after the configuration was locked and the artifact saved, and were not used for
selection:

| Metric | Value |
|---|---:|
| PR-AUC | 0.927520 |
| ROC-AUC | 0.906538 |
| Accuracy | 0.794321 |
| Precision | 0.747844 |
| Recall | 0.945116 |
| F1 | 0.834987 |

> **These are default-threshold reference values, not the reported operating
> point.** The `predicted_label` column in
> `experiments/logistic_regression/test_predictions.csv` is likewise generated at
> the default 0.50 threshold. The locked validation-selected operating point for
> Logistic Regression is `0.4631204833813834`, and all locked-threshold metrics
> and confusion-matrix counts come from the standardized evaluation, not from
> this table or that column. See
> [`docs/standardized_model_evaluation.md`](standardized_model_evaluation.md)
> and `experiments/standardized_evaluation/final_test/`.

## Saved outputs

| Item | Path |
|---|---|
| Fitted model | `artifacts/logistic_regression.joblib` |
| Configuration | `experiments/logistic_regression/config.json` |
| Metrics | `experiments/logistic_regression/metrics.json` |
| Round 1 history | `experiments/logistic_regression/tuning_results.csv` |
| Round 2 history | `experiments/logistic_regression/round2_joint_search.csv` |
| Validation predictions | `experiments/logistic_regression/validation_predictions.csv` |
| Test predictions | `experiments/logistic_regression/test_predictions.csv` |
| Training script | `src/05_logistic_regression.py` |

Prediction files use the shared schema: `sample_index`, `true_label`,
`attack_probability` (`P(y = 1)`), `predicted_label`.

## Reproducibility

| Item | Value |
|---|---|
| Random state | 42 |
| Final refit runtime | 5.552 seconds (train only) |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| numpy | 2.5.1 |
| joblib | 1.5.3 |

`liblinear` with a fixed `random_state` is deterministic for this problem, so the
saved artifact is reproducible from `src/05_logistic_regression.py` given the
same environment. The Stage 0 diagnostics gate independently confirms the saved
artifact still reproduces the committed test probabilities.

A note in `experiments/logistic_regression/README.md` records that 17 historical
Round 1 rows logged `max_iter=10000` and omitted `tol`, while actually being fit
with `max_iter=5000` and the scikit-learn default `tol=0.0001`. The original CSV
was preserved as the experiment record rather than rewritten after the fact.

## Interpretation

Logistic Regression is clearly better than the dummy baselines but is the
weakest of the four models on the default-threshold headline test metrics. Its
validation PR-AUC of
0.913401 sits roughly seven points below the tree models, which is consistent
with a linear boundary being unable to capture the feature interactions that the
tree ensembles exploit.

Its recall is high (0.946419 on validation, 0.945116 on test at 0.50) but
precision is comparatively low, so it flags a larger share of normal traffic than
the other models. The diagnostics work also finds it the least exposed to
train-to-test feature drift among the four, and it is the only model whose top
permutation feature (`ct_dst_sport_ltm`) dominates by a wide margin.

The model is most useful in this project as an interpretable reference: its
coefficients are directly readable, and the gap between it and the tree models
quantifies how much of the task is genuinely non-linear.

## Limitations

- A linear decision boundary cannot represent the feature interactions that
  drive the tree models' advantage.
- Test metrics in this document are default-threshold reference values only;
  the reported operating point is the validation-selected locked threshold.
- Validation metrics are computed on a single fixed 20% stratified split, not
  cross-validated, so they carry the variance of one split.
- Results are specific to UNSW-NB15 and its documented dataset artifacts; the
  cautions in [`docs/DATA_CARD.md`](DATA_CARD.md) apply.
