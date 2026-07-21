"""Baselines. Every model has to beat these.

    .venv/bin/python src/04_baseline.py

The prof kept saying every model should be compared against a baseline. A
classifier that always guesses the majority class is the floor; if a real model
can't beat "always guess attack", it isn't worth anything. Reported on val and test.

The two splits have different attack rates (val ~51%, test ~55%), so majority-class
accuracy differs between them, which is exactly why the rule is to compare on
precision/recall/F1/PR-AUC, not raw accuracy.
"""

import json
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score,
)

from config import PROCESSED_DIR, DOCS_DIR, RANDOM_STATE

X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["label"]
splits = {
    "val": (pd.read_parquet(PROCESSED_DIR / "X_val.parquet"),
            pd.read_parquet(PROCESSED_DIR / "y_val.parquet")["label"]),
    "test": (pd.read_parquet(PROCESSED_DIR / "X_test.parquet"),
             pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["label"]),
}


def metrics(y_true, y_pred, y_score):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
    }


results = {}
for strat in ("most_frequent", "stratified"):
    clf = DummyClassifier(strategy=strat, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    results[strat] = {}
    for split, (Xs, ys) in splits.items():
        y_pred = clf.predict(Xs)
        y_score = clf.predict_proba(Xs)[:, 1]
        results[strat][split] = metrics(ys, y_pred, y_score)

(DOCS_DIR / "baseline.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

md = ["# Baseline results (the bar every model has to beat)\n",
      "Positive class = attack (1).\n"]
for strat, per_split in results.items():
    md.append(f"## DummyClassifier(strategy=\"{strat}\")\n")
    md.append("| split | accuracy | precision | recall | f1 | pr_auc |")
    md.append("|---|---|---|---|---|---|")
    for split, m in per_split.items():
        md.append(f"| {split} | {m['accuracy']} | {m['precision']} | "
                  f"{m['recall']} | {m['f1']} | {m['pr_auc']} |")
        print(f"{strat:14s} {split:5s} -> {m}")
    md.append("")
(DOCS_DIR / "baseline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"[written] {DOCS_DIR/'baseline.md'} + baseline.json")
