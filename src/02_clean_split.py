"""Clean, de-duplicate, and split into train / val / test.

    .venv/bin/python src/02_clean_split.py

Steps (each one shows up in the data card):
  1. Clean both files (constant fixes, safe to apply the same way everywhere).
  2. For the binary task, collapse repeated predictor vectors with the same
     label, regardless of attack_cat. Remove predictor vectors that occur with
     both binary labels because they are inherently ambiguous.
  3. Drop train rows whose predictor features also appear in test. Test is the
     frozen hold-out; we don't touch it, but we do delete its twins from train so
     the model can't memorize a feature pattern it'll be scored on.
  4. Take a stratified validation slice out of the cleaned train (same seed).
  5. Write everything to parquet.

Test is cleaned but otherwise left alone: no dedup, no resampling.
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_TRAIN_CSV, RAW_TEST_CSV, PROCESSED_DIR, DOCS_DIR,
    TARGET, DROP_COLS, RANDOM_STATE, VAL_FRACTION,
)
from preprocess import read_raw, clean_df, make_xy

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
report = {}

train_raw = clean_df(read_raw(RAW_TRAIN_CSV))
test_raw = clean_df(read_raw(RAW_TEST_CSV))
report["raw_train_rows"] = int(len(train_raw))
report["raw_test_rows"] = int(len(test_raw))

# Did is_ftp_login actually have stray >1 values? Record the pre-clamp max.
report["is_ftp_login_max_before_clamp"] = int(read_raw(RAW_TRAIN_CSV)["is_ftp_login"].max())

# The binary-task deduplication key is the model input vector plus the binary
# label. attack_cat is deliberately excluded: it is dropped from X and must not
# let otherwise identical binary examples survive as separate records.
predictor_cols = [c for c in train_raw.columns if c not in DROP_COLS + [TARGET]]
dedup_cols = predictor_cols + [TARGET]

# The test-overlap key is the same predictor-only list. If a train row's features
# match a test row, the model can memorize a pattern it will later be scored on,
# even if the two copies have different labels. Test labels are never used for
# cleaning.

before = len(train_raw)
train_dedup = train_raw.drop_duplicates(subset=dedup_cols, keep="first").copy()
same_label_duplicates_removed = int(before - len(train_dedup))

# A predictor vector mapping to both binary labels is ambiguous. Remove every
# remaining row for such a vector before the train/validation split rather than
# silently choosing one label or allowing the vector to straddle the split.
label_counts_by_predictor = train_dedup.groupby(
    predictor_cols, sort=False, dropna=False
)[TARGET].nunique()
ambiguous_predictors = label_counts_by_predictor[label_counts_by_predictor > 1].index
dedup_index = train_dedup.set_index(predictor_cols).index
ambiguous_mask = dedup_index.isin(ambiguous_predictors)
ambiguous_vectors_removed = int(len(ambiguous_predictors))
ambiguous_rows_removed = int(ambiguous_mask.sum())
train_unambiguous = train_dedup.loc[~ambiguous_mask].copy()

report["train_same_label_duplicate_rows_removed"] = same_label_duplicates_removed
report["train_conflicting_predictor_vectors_removed"] = ambiguous_vectors_removed
report["train_conflicting_predictor_rows_removed"] = ambiguous_rows_removed
report["deduplication"] = {
    "key": dedup_cols,
    "predictor_columns": predictor_cols,
    "excluded_from_key": ["id", "attack_cat"],
    "same_label_duplicate_rows_removed": same_label_duplicates_removed,
    "conflicting_predictor_vectors_removed": ambiguous_vectors_removed,
    "conflicting_predictor_rows_removed": ambiguous_rows_removed,
}

# Left-merge on predictor features; drop any train row that finds a test match.
test_keys = test_raw[predictor_cols].drop_duplicates()
train_unambiguous = train_unambiguous.copy()
train_unambiguous["_source_row"] = range(len(train_unambiguous))
merged = train_unambiguous.merge(test_keys, on=predictor_cols, how="left", indicator=True)
overlap_mask = merged["_merge"].to_numpy() == "both"
overlap_source_rows = merged.loc[overlap_mask, "_source_row"]
report["train_rows_also_in_test_removed"] = int(overlap_source_rows.nunique())
train_clean = train_unambiguous.loc[
    ~train_unambiguous["_source_row"].isin(overlap_source_rows),
].drop(columns="_source_row").reset_index(drop=True)

if train_clean.duplicated(subset=predictor_cols).any():
    raise RuntimeError("Predictor duplicates remain before train/validation split")
report["train_rows_after_dedup_and_overlap_removal"] = int(len(train_clean))

X_all, y_all = make_xy(train_clean)
X_test, y_test = make_xy(test_raw)  # cleaned, not deduped, frozen
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=VAL_FRACTION,
    stratify=y_all,          # keep the same attack rate in both slices
    random_state=RANDOM_STATE,
)

def dump(df_or_series, name):
    obj = df_or_series.to_frame() if isinstance(df_or_series, pd.Series) else df_or_series
    obj.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

for name, obj in [
    ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
    ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
]:
    dump(obj, name)

def balance(y):
    vc = y.value_counts(normalize=True).sort_index()
    return {"n": int(len(y)), "normal_pct": round(float(vc.get(0, 0)), 4),
            "attack_pct": round(float(vc.get(1, 0)), 4)}

report["splits"] = {
    "train": balance(y_train), "val": balance(y_val), "test": balance(y_test),
}
report["random_state"] = RANDOM_STATE
report["val_fraction"] = VAL_FRACTION
report["feature_columns"] = list(X_train.columns)

(DOCS_DIR / "split-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print("=== clean + split done ===")
for k in ["raw_train_rows", "train_rows_also_in_test_removed",
          "train_rows_after_dedup_and_overlap_removal", "is_ftp_login_max_before_clamp"]:
    if k in report:
        print(f"  {k}: {report[k]}")
print(f"  train_same_label_duplicate_rows_removed: {report['train_same_label_duplicate_rows_removed']}")
print(f"  train_conflicting_predictor_vectors_removed: {report['train_conflicting_predictor_vectors_removed']}")
print(f"  train_conflicting_predictor_rows_removed: {report['train_conflicting_predictor_rows_removed']}")
print("  splits:", json.dumps(report["splits"]))
print(f"[written] {PROCESSED_DIR}/*.parquet  +  {DOCS_DIR/'split-manifest.json'}")
