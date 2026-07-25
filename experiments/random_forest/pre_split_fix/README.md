# Historical Random Forest experiment — pre-split-fix

These files are historical results produced before the train/validation
predictor-overlap correction. They are preserved for audit and comparison only;
they are not the official current Random Forest results.

Final model selected from the Round 2 joint search by validation PR-AUC, with validation ROC-AUC as the tie-breaker. The locked configuration is `{"class_weight": null, "max_depth": 25, "max_features": 0.3, "min_samples_leaf": 2, "n_estimators": 800}` with `random_state=42`.

Preprocessing uses the fitted `artifacts/preprocess_tree.joblib` artifact (TTL-excluded, 39 features) with transform-only application. Round 1 history remains in `tuning_results.csv`; Round 2 history remains in `round2_joint_search.csv`. Test metrics in `metrics.json` are sanity/reference values; downstream cross-model comparison uses the prediction CSVs.

Final train-only refit runtime: `20.236` seconds.
