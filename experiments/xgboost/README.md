# XGBoost final experiment

The final XGBoost model uses the Round 2 winner, selected by validation PR-AUC
with validation ROC-AUC as the tie-breaker. It was refit on the training split
only with `n_estimators=283` (the selected `best_iteration=282` plus one),
without validation `eval_set` or early stopping.

Validation PR-AUC: `0.985938`
Validation ROC-AUC: `0.987283`
Test metrics are stored under `test_reference_only` in `metrics.json` and were
computed only after the configuration, validation outputs, and model artifacts
were saved.

Tuning history is preserved in `tuning_results.csv` (49 Round 1 fits) and
`round2_joint_search.csv` (64 Round 2 fits).
