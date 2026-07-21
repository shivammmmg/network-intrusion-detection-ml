"""Fit the preprocessing pipelines on train only and save them.

    .venv/bin/python src/03_pipelines.py

Saves four fitted transformers to artifacts/:
  preprocess_linear.joblib          LogReg + NN, no TTL  (this is the main one)
  preprocess_tree.joblib            RF + XGBoost, no TTL  (main one)
  preprocess_linear_with_ttl.joblib the inflated run, ablation only
  preprocess_tree_with_ttl.joblib   ablation only

Plus feature_names_linear.json and manifest.json (versions and input hashes, so
an artifact can be traced to the exact data and libraries that made it).

fit() is called on X_train only; val and test only ever get transform(). The
learned medians, IQR and one-hot vocabulary live inside the saved object, so a
teammate can't refit them on val/test by accident.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import joblib
import sklearn
import xgboost

from config import PROCESSED_DIR, ARTIFACTS_DIR, RANDOM_STATE, RAW_TRAIN_CSV, RAW_TEST_CSV
from preprocess import build_linear_preprocessor, build_tree_preprocessor

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
X_val = pd.read_parquet(PROCESSED_DIR / "X_val.parquet")
X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["label"]


def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


manifest = {
    "versions": {
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "xgboost": xgboost.__version__,
    },
    "random_state": RANDOM_STATE,
    "raw_input_sha256": {
        "training-set.csv": sha256(RAW_TRAIN_CSV),
        "testing-set.csv": sha256(RAW_TEST_CSV),
    },
    "artifacts": {},
}

builders = {"linear": build_linear_preprocessor, "tree": build_tree_preprocessor}

for variant, builder in builders.items():
    for include_ttl in (False, True):
        suffix = "_with_ttl" if include_ttl else ""
        name = f"preprocess_{variant}{suffix}"

        pre = builder(X_train, include_ttl=include_ttl)
        pre.fit(X_train, y_train)              # train only

        Xt_tr = pre.transform(X_train)
        Xt_va = pre.transform(X_val)
        Xt_te = pre.transform(X_test)

        # no NaN should get through, and the three splits should share columns
        assert not Xt_tr.isna().any().any(), f"{name}: NaN in transformed train"
        assert list(Xt_tr.columns) == list(Xt_va.columns) == list(Xt_te.columns), \
            f"{name}: column mismatch across splits"

        joblib.dump(pre, ARTIFACTS_DIR / f"{name}.joblib", compress=3)
        manifest["artifacts"][name] = {
            "n_features_out": int(Xt_tr.shape[1]),
            "ttl_included": include_ttl,
            "serves": {"linear": "LogisticRegression + NeuralNet",
                       "tree": "RandomForest + XGBoost"}[variant],
            "primary": not include_ttl,
        }
        print(f"[fit] {name:28s} -> {Xt_tr.shape[1]} features "
              f"(train {Xt_tr.shape[0]}, val {Xt_va.shape[0]}, test {Xt_te.shape[0]})")

pre_lin = joblib.load(ARTIFACTS_DIR / "preprocess_linear.joblib")
feat_names = list(pre_lin.get_feature_names_out())
(ARTIFACTS_DIR / "feature_names_linear.json").write_text(
    json.dumps(feat_names, indent=2), encoding="utf-8")

(ARTIFACTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"\n[written] {ARTIFACTS_DIR}/*.joblib + feature_names_linear.json + manifest.json")
print(f"  linear main feature count: {len(feat_names)}")
