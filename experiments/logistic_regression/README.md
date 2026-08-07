# Logistic Regression experiment

Final model selected from the Round 2 joint search by validation PR-AUC, with validation ROC-AUC as the tie-breaker. The locked configuration is `{"C": 100.0, "class_weight": null, "max_iter": 10000, "penalty": "l1", "random_state": 42, "solver": "liblinear", "tol": 0.0001}`.

Preprocessing uses the fitted `artifacts/preprocess_linear.joblib` artifact with transform-only application. Round 1 history remains in `tuning_results.csv`. Round 2 history remains in `round2_joint_search.csv`. Test metrics in `metrics.json` are sanity/reference values. downstream cross-model comparison uses the prediction CSVs.

Final train-only refit runtime: `5.552` seconds.

# Optimization settings

Hyperparameter selection was based on validation PR-AUC, with validation ROC-AUC used
as the tie-breaker. The selected model parameters were 'C', 'penalty', 'solver' and 'class_weight'.

During Round 2 tuning, candidates used 'tol=0.001' and 'max_iter=3000' to keep the search
manageable computationally. After the winning configuration was locked, the final train-only refit
used 'tol=0.0001' and 'max_iter=10000'. These optimization settings were only used to fit the model
more completely. The selected Logistic Regression hyperparameters were not changed.

## Round 1 configuration note

The 17 historical solver, regularization, and class-weight rows were fit with
`max_iter=5000` and scikit-learn's default `tol=0.0001`, but incorrectly logged
`max_iter=10000` and omitted `tol`. The later baseline row was fit and correctly
logged with `max_iter=3000`, `tol=0.001`. The historical CSV has been preserved
as the original experiment record rather than rewritten after the fact.

Round 2 used `max_iter=3000` and `tol=0.001`. After the winning Round 2 configuration was locked, the final train-only refit used
`max_iter=10000` and `tol=0.0001`. The final model converged after 34 iterations. Ten of the 30 Round 2 candidates did not converge, but all
ranked below the selected winner.
