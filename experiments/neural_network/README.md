# Neural Network final experiment

The final Neural Network model uses the Round 2 winner, selected by validation
PR-AUC with validation ROC-AUC as the tie-breaker. The locked configuration is
`{"activation": "relu", "alpha": 0.001, "batch_size": 128, "hidden_layer_sizes": [128, 64], "learning_rate_init": 0.0005, "solver": "adam"}`.

During tuning, `MLPClassifier` used early stopping with an internal holdout from
the training split. The winning candidate stopped after `59`
epochs. The final model was refit on the complete training split for
`59` epochs with early stopping disabled.

Validation PR-AUC: `0.968057`
Validation ROC-AUC: `0.972019`

Preprocessing uses the fitted `artifacts/preprocess_linear.joblib` artifact with
transform-only application. Round 1 history is stored in `tuning_results.csv`,
and the joint search is stored in `round2_joint_search.csv`. Validation and test
prediction CSVs support downstream cross-model comparison.

Test metrics are stored under `test_reference_only` in `metrics.json`. They were
computed only after model selection and were not used to choose the model.

Final train-only refit runtime: `81.406` seconds.

# Final Convergence Status

During tuning, early stopping selected 59 training epochs for the winning configuration. The final
model was then refit on the full training split for exactly 59 epochs with early stopping disabled.
Scikit-learn reported that the optimization had not converged by the end of the 59 training epochs. Therefore, 
'converged' is recorded as 'false' in the final configuration. However, this does not mean that the training failed,
as the model completed the planned refit and produced valid validation and test predictions.