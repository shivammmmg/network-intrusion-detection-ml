# EECS 3404 project: data and preprocessing

This is the data-cleaning and preprocessing part of the intrusion-detection
project. The output everyone else builds on is in `data/processed/` (the split)
and `artifacts/` (the fitted pipelines). Details are in `docs/DATA_CARD.md`.

## Using it

Load the artifact that matches your model and use it to transform the splits.
Don't re-split, and don't refit the preprocessing. The scaler and encoders were
fit on train only and are saved inside the artifact, which is what keeps the
results clean.

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from preprocess import load_artifact

pre = load_artifact("artifacts/preprocess_linear.joblib")   # your variant, see table
X_train = pre.transform(pd.read_parquet("data/processed/X_train.parquet"))
X_val   = pre.transform(pd.read_parquet("data/processed/X_val.parquet"))
y_train = pd.read_parquet("data/processed/y_train.parquet")["label"]
y_val   = pd.read_parquet("data/processed/y_val.parquet")["label"]
# train on (X_train, y_train), tune on val, and only touch X_test once, at the end.
```

Which artifact:

| Training | Use |
|---|---|
| Logistic Regression | `preprocess_linear.joblib` |
| Neural Network | `preprocess_linear.joblib` |
| Random Forest | `preprocess_tree.joblib` |
| XGBoost | `preprocess_tree.joblib` (or native categoricals off `X_*.parquet`) |

`load_artifact` turns a scikit-learn version mismatch into an error rather than a
silent wrong transform, so if your sklearn differs you'll know right away. Install
the exact versions with `pip install -r requirements.txt` in a fresh venv.

## A few rules so our numbers line up
Use the split that's already here. If everyone makes their own split with
`train_test_split` we end up with different test sets and can't compare numbers.
`X_test` is frozen: score on it once, at the very end, no tuning against it.
Report precision/recall/F1/PR-AUC per split, not raw accuracy, because train
(51% attack) and test (55%) don't have the same attack rate. Beat the baseline in
`docs/baseline.md` (always-guess-attack gets F1 0.71) on F1/PR-AUC.

On TTL: the main artifacts leave the TTL features out on purpose. The `*_with_ttl`
ones are only for the ablation section, since `sttl` alone scores 92% and that's a
dataset artifact, not real detection. The without-TTL result is the one we report.

## Layout
```
data/raw/         the UNSW-NB15 CSVs (gitignored; re-fetch with src/00_download.py)
data/processed/   X_train/val/test.parquet + y_train/val/test.parquet
artifacts/        preprocess_{linear,tree}[_with_ttl].joblib + feature_names + manifest.json
docs/             DATA_CARD.md, baseline.md, split-manifest.json, eda-summary.md, eda/*.png
src/              config.py, preprocess.py, 00_download.py .. 04_baseline.py
```

## Rebuild from scratch
Make a venv, `pip install -r requirements.txt`, then run the numbered scripts in
`src/` in order (`00_download.py` through `04_baseline.py`). Everything is
seeded, so a rebuild gives the same split and the same numbers.
