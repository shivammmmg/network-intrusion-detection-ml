# Neural Network results

## Purpose

The Neural Network is the non-linear counterpart to Logistic Regression on the
same 66-feature linear preprocessing. It tests how much of the attack/normal
separation is recoverable by a general function approximator, as distinct from
the axis-aligned splits used by the tree models.

## Architecture

`MLPClassifier` (scikit-learn), a fully connected feed-forward network:

| Item | Value |
|---|---|
| Hidden layers | 128, 64 |
| Activation | ReLU |
| Solver | Adam |
| Learning rate (initial) | 0.0005 |
| Batch size | 128 |
| L2 penalty (`alpha`) | 0.0001 |
| Output | binary, `P(y = 1)` |

## Preprocessing

| Item | Value |
|---|---|
| Artifact | `artifacts/preprocess_linear.joblib` |
| Application mode | transform only |
| Features in | 66 |
| TTL features | excluded (primary pipeline) |

The same fitted artifact used by Logistic Regression, applied with `transform()`
only. Nothing is refit on validation or test data.

## Model selection

Selection used validation data only:

| Item | Value |
|---|---|
| Round 1 candidates | 26 rows (`tuning_results.csv`) |
| Round 2 candidates | 24 rows (`round2_joint_search.csv`) |
| Primary metric | validation PR-AUC |
| Tie-breaker | validation ROC-AUC |
| Selected config | `round2_17_hidden128x64_alpha0.0001_lr0.0005_batch128` |

Top three Round 2 candidates, as recorded in `round2_joint_search.csv`:

| Rank | Config | Val PR-AUC | Val ROC-AUC | Epochs |
|---:|---|---:|---:|---:|
| 1 | `round2_17_hidden128x64_alpha0.0001_lr0.0005_batch128` | 0.971851 | 0.976078 | 28 |
| 2 | `round2_20_hidden128x64_alpha0.0001_lr0.001_batch256` | 0.970701 | 0.975342 | 46 |
| 3 | `round2_23_hidden128x64_alpha0.001_lr0.001_batch128` | 0.965119 | 0.973175 | 23 |

All 24 Round 2 candidates converged. The frozen test set was not used in
selection.

## Training setup and convergence

Tuning and the final refit deliberately use different training rules:

| Setting | Tuning runs | Final refit |
|---|---|---|
| Early stopping | enabled | disabled |
| Internal validation fraction | 0.1 (carved from train) | not used |
| `max_iter` | 100 | 28 (locked budget) |
| `n_iter_no_change` | 8 | 29 |
| `tol` | 0.0001 | 0.0 |
| Training data | training split | complete training split |
| `random_state` | 42 | 42 |

The winning candidate stopped after 28 epochs under early stopping. That epoch
count became the locked budget, and the final model was refit on the complete
training split for exactly 28 epochs with early stopping disabled. The rule is
recorded in the config as `fixed_validation_selected_epoch_budget`. Doing it this
way means the epoch count is a validation-selected hyperparameter rather than
something tuned against the test set.

The final refit completed all 28 epochs (`reached_selected_epoch_budget: true`).
scikit-learn emitted a `ConvergenceWarning` because the fixed budget was reached
before its usual convergence criterion; this is expected under a fixed budget
with `tol=0.0` and does not indicate a failed fit.

Because the tuning fit and the final refit are two different fits under different
rules, the winner's search-time validation PR-AUC (0.971851) is not the same as
the finalized validation PR-AUC below (0.959759). Both are recorded values from
their respective runs.

## Validation metrics

From `experiments/neural_network/metrics.json` (final refit):

| Metric | Value |
|---|---:|
| PR-AUC | 0.959759 |
| ROC-AUC | 0.958987 |
| Accuracy | 0.888314 |
| Precision | 0.900086 |
| Recall | 0.865478 |
| F1 | 0.882443 |

## Test metrics at the default 0.50 threshold

Recorded under `test_reference_only`. The test split was opened only after model
selection, the artifact, the validation predictions and the locked configuration
were saved:

| Metric | Value |
|---|---:|
| PR-AUC | 0.970199 |
| ROC-AUC | 0.959934 |
| Accuracy | 0.853216 |
| Precision | 0.815658 |
| Recall | 0.947565 |
| F1 | 0.876677 |

> **These are default-threshold reference values, not the reported operating
> point.** The `predicted_label` column in
> `experiments/neural_network/test_predictions.csv` is likewise generated at the
> default 0.50 threshold. The locked validation-selected operating point for the
> Neural Network is `0.3454528735910796` — the lowest of the four models — so the
> difference between the two is substantial here. All locked-threshold metrics
> and confusion-matrix counts come from the standardized evaluation. See
> [`docs/standardized_model_evaluation.md`](standardized_model_evaluation.md)
> and `experiments/standardized_evaluation/final_test/`.

## Saved outputs

| Item | Path |
|---|---|
| Fitted model | `artifacts/neural_network.joblib` |
| Configuration | `experiments/neural_network/config.json` |
| Metrics | `experiments/neural_network/metrics.json` |
| Round 1 history | `experiments/neural_network/tuning_results.csv` |
| Round 2 history | `experiments/neural_network/round2_joint_search.csv` |
| Validation predictions | `experiments/neural_network/validation_predictions.csv` |
| Test predictions | `experiments/neural_network/test_predictions.csv` |
| Training script | `src/08_neural_network.py` |
| Verification script | `src/08_neural_network_verify.py` |

Prediction files use the shared schema: `sample_index`, `true_label`,
`attack_probability` (`P(y = 1)`), `predicted_label`.

## Artifact verification

Because fresh neural-network training is the least reproducible step in the
project, the saved artifact ships with a dedicated read-only checker,
`src/08_neural_network_verify.py`. It reloads `artifacts/neural_network.joblib`,
re-applies the preprocessing artifact, and confirms that the artifact still
reproduces the committed predictions and metrics:

```
PASS: validation metrics match metrics.json
PASS: validation labels and indexes match the data split
PASS: validation model probabilities match the CSV (maximum difference=2.165e-15)
PASS: validation predicted labels match the artifact
PASS: validation artifact metrics match metrics.json
PASS: test metrics match metrics.json
PASS: test labels and indexes match the data split
PASS: test model probabilities match the CSV (maximum difference=2.220e-15)
PASS: test predicted labels match the artifact
PASS: test artifact metrics match metrics.json
```

The Stage 0 diagnostics gate independently reaches the same conclusion for this
model (maximum absolute difference 2.220e-15).

## Reproducibility

| Item | Value |
|---|---|
| Random state | 42 |
| `PYTHONHASHSEED` | 42 |
| Thread pinning | `OMP`, `MKL`, `OPENBLAS`, `NUMEXPR` all set to 1 |
| Final refit runtime | 25.540 seconds (train only) |
| Python | 3.13.5 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| numpy | 2.5.1 |
| joblib | 1.5.3 |
| Recorded platform | Windows-11-10.0.26200-SP0 |

The thread pinning and hash seed are recorded because Adam's minibatch ordering
and BLAS reduction order both affect the fitted weights. Even so, exact fresh
retraining can differ across Python versions, operating systems and BLAS
builds. The committed artifact plus `08_neural_network_verify.py` is therefore
the reproducible handoff for this model, rather than a promise that retraining
reproduces the weights bit-for-bit.

## Interpretation

The Neural Network sits between Logistic Regression and the tree models. It
improves substantially on the linear model — validation PR-AUC 0.959759 against
0.913401 — which confirms that a meaningful part of the task is non-linear.

Its validation profile is precision-leaning (precision 0.900086, recall
0.865478), which is the opposite balance to Logistic Regression. The
validation-selected threshold compensates: at 0.3455, the lowest locked
threshold of the four models, the standardized evaluation shows the operating
point moving decisively toward recall.

The diagnostics work finds this model the most distinctive of the four in what
it relies on: its permutation importance is dominated by `swin` and `sload`,
features that rank low for every other model. This gives it the largest measured
overlap between high-importance and high-drift features among the four models,
indicating greater potential sensitivity to the observed distribution shift,
even though its measured test performance is strong.

## Limitations

- Exact retraining is environment-sensitive; use the saved artifact and its
  verification script rather than expecting bit-identical refits.
- The epoch budget was fixed from a single validation-selected candidate rather
  than re-searched during the final refit.
- Test metrics in this document are default-threshold reference values only;
  the reported operating point is the validation-selected locked threshold.
- Validation metrics come from one fixed 20% stratified split, not
  cross-validation.
- The model's reliance on `swin`/`sload`, combined with the measured drift in
  those features, is a documented generalization risk. See
  [`docs/diagnostics_report.md`](diagnostics_report.md).
- Results are specific to UNSW-NB15 and its documented dataset artifacts; the
  cautions in [`docs/DATA_CARD.md`](DATA_CARD.md) apply.
