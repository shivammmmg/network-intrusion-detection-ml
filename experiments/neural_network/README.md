# Neural Network final experiment

The final Neural Network model uses the Round 2 winner, selected by validation
PR-AUC with validation ROC-AUC as the tie-breaker. The locked configuration is
`{"activation": "relu", "alpha": 0.0001, "batch_size": 128, "hidden_layer_sizes": [128, 64], "learning_rate_init": 0.0005, "solver": "adam"}`.

During tuning, `MLPClassifier` used early stopping with a fixed internal
holdout from the training split and `random_state=42`. The winning
candidate stopped after `28` epochs.

The final model was refit on the complete training split using the
validation-selected budget of `28` epochs. Early stopping was
disabled, `tol=0.0`, and
`n_iter_no_change=29` so the model completed the locked
epoch budget without using test results to revise the configuration.

The final refit completed `28` epochs.

Scikit-learn emitted a ConvergenceWarning because the fixed epoch budget was reached before its usual convergence condition was met. The training run still completed the complete locked epoch budget.

This warning does not indicate that training failed; the model completed the
locked training procedure and produced the saved validation and test predictions.

Validation PR-AUC: `0.959759`
Validation ROC-AUC: `0.958987`

Preprocessing uses the fitted `artifacts/preprocess_linear.joblib` artifact
with transform-only application. Round 1 history is stored in
`tuning_results.csv`, and the joint search is stored in
`round2_joint_search.csv`. Validation and test prediction CSVs support
downstream cross-model comparison.

Test metrics are stored under `test_reference_only` in `metrics.json`. They
were computed only after model selection and were not used to choose or
revise the model.

Final train-only refit runtime: `25.540` seconds.
