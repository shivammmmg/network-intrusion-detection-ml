# Logistic Regression experiment

Final model selected from the Round 2 joint search by validation PR-AUC, with validation ROC-AUC as the tie-breaker. The locked configuration is `{"C": 100.0, "class_weight": null, "max_iter": 10000, "penalty": "l1", "random_state": 42, "solver": "liblinear", "tol": 0.0001}`.

Preprocessing uses the fitted `artifacts/preprocess_linear.joblib` artifact with transform-only application. Round 1 history remains in `tuning_results.csv`. Round 2 history remains in `round2_joint_search.csv`. Test metrics in `metrics.json` are sanity/reference values. downstream cross-model comparison uses the prediction CSVs.

Final train-only refit runtime: `5.552` seconds.
