# Random Forest experiment

This is the corrected-data Random Forest run after the train/validation
predictor-overlap fix. The processed splits contain 79,685 train rows, 19,922
validation rows, and 82,332 frozen test rows. The tree preprocessor produces 39
features.

Final model selected from the Round 2 joint search by validation PR-AUC, with validation ROC-AUC as the tie-breaker. The locked configuration is `{"class_weight": "balanced", "max_depth": 25, "max_features": 0.3, "min_samples_leaf": 1, "n_estimators": 800}` with `random_state=42`.

Preprocessing uses the fitted `artifacts/preprocess_tree.joblib` artifact (TTL-excluded, 39 features) with transform-only application. Round 1 history remains in `tuning_results.csv`; Round 2 history remains in `round2_joint_search.csv`. Test metrics in `metrics.json` are sanity/reference values; downstream cross-model comparison uses the prediction CSVs.

The pre-split-fix Random Forest model, predictions, metrics, configuration, and
tuning history are preserved in `experiments/random_forest/pre_split_fix/`.

Final train-only refit runtime: `22.573` seconds.
