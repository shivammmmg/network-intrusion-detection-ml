"""EDA for UNSW-NB15.

    .venv/bin/python src/01_eda.py

Writes docs/eda-summary.md and a couple of figures to docs/eda/. Read-only; it
doesn't change the data. The point is to look at the things the cleaning and
leakage decisions rest on: class balance, missing values, duplicates, category
counts, and the sttl leak.
"""

import matplotlib
matplotlib.use("Agg")  # headless: write figures, don't open windows
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    RAW_TRAIN_CSV, RAW_TEST_CSV, DOCS_DIR, TARGET, CATEGORICAL,
)
from preprocess import read_raw

EDA_DIR = DOCS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

lines = []  # collected and written to docs/eda-summary.md at the end
def say(s=""):
    print(s)
    lines.append(s)


train = read_raw(RAW_TRAIN_CSV)
test = read_raw(RAW_TEST_CSV)

say("# UNSW-NB15 EDA summary\n")
say(f"- train shape: {train.shape}")
say(f"- test shape:  {test.shape}")
say(f"- columns ({len(train.columns)}): {', '.join(train.columns)}\n")

# Class balance. Train and test don't have the same attack rate.
say("## Class balance (label: 0=normal, 1=attack)")
for name, df in [("train", train), ("test", test)]:
    vc = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index()
    say(f"- {name}: normal={vc.get(0,0)} ({pct.get(0,0):.1%})  "
        f"attack={vc.get(1,0)} ({pct.get(1,0):.1%})")
say("")

fig, ax = plt.subplots(figsize=(5, 3))
bal = pd.DataFrame({
    "train": train[TARGET].value_counts(normalize=True).sort_index(),
    "test": test[TARGET].value_counts(normalize=True).sort_index(),
})
bal.index = ["normal (0)", "attack (1)"]
bal.plot.bar(ax=ax, rot=0)
ax.set_ylabel("proportion"); ax.set_title("Class balance: train vs test")
fig.tight_layout(); fig.savefig(EDA_DIR / "class_balance.png", dpi=110); plt.close(fig)

say("## Column dtypes")
obj_cols = train.select_dtypes(include="object").columns.tolist()
say(f"- object/text columns: {obj_cols}")
say("  (proto/service/state are real categories; anything else showing up here "
    "is a column that needs converting back to numeric.)\n")

say("## Missing values (columns with any NaN)")
miss = train.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss) == 0:
    say("- none reported as NaN in train")
else:
    for c, n in miss.items():
        say(f"- {c}: {n} ({n/len(train):.2%})")
say("")

# Duplicates. id is a unique counter, so drop it before looking for repeated
# rows or every row looks unique.
feat_cols = [c for c in train.columns if c != "id"]
n_dup_train = train.duplicated(subset=feat_cols).sum()
n_dup_test = test.duplicated(subset=feat_cols).sum()
say("## Duplicates")
say(f"- within train (ignoring id): {n_dup_train} ({n_dup_train/len(train):.1%})")
say(f"- within test  (ignoring id): {n_dup_test} ({n_dup_test/len(test):.1%})")

# Rows that show up in both files. Identical flows in train and test let a model
# memorize a row it will be scored on.
merged = train[feat_cols].merge(test[feat_cols].drop_duplicates(), how="inner")
n_overlap = len(merged)
say(f"- rows in BOTH train and test (feature+label match): {n_overlap}")
say("")

say("## Categorical features")
for c in CATEGORICAL:
    tr = set(train[c].astype(str).str.strip().str.lower().unique())
    te = set(test[c].astype(str).str.strip().str.lower().unique())
    only_tr = tr - te
    only_te = te - tr
    say(f"- {c}: {len(tr)} levels in train, {len(te)} in test; "
        f"{len(only_tr)} only-in-train, {len(only_te)} only-in-test")
    if c == "service":
        say(f"    service levels: {sorted(tr)}")
say("  (only-in-test levels are why the encoders need handle_unknown.)\n")

say("## TTL leakage evidence (sttl by class)")
for cls, label in [(0, "normal"), (1, "attack")]:
    top = train.loc[train[TARGET] == cls, "sttl"].value_counts().head(4)
    pairs = ", ".join(f"{int(v)}:{n}" for v, n in top.items())
    say(f"- sttl | {label}: most common values -> {pairs}")
best_acc, best_t = 0.0, None
for t in [32, 63, 128, 200, 254]:
    pred = (train["sttl"] >= t).astype(int)
    acc = (pred == train[TARGET]).mean()
    if acc > best_acc:
        best_acc, best_t = acc, t
say(f"- the single rule 'sttl >= {best_t} => attack' already gets {best_acc:.1%} "
    f"training accuracy, so everything gets run with and without the TTL columns "
    f"and the without-TTL number is the one we report.\n")

fig, ax = plt.subplots(figsize=(6, 3))
for cls, label, color in [(0, "normal", "tab:blue"), (1, "attack", "tab:red")]:
    ax.hist(train.loc[train[TARGET] == cls, "sttl"], bins=60, alpha=0.6,
            label=label, color=color)
ax.set_xlabel("sttl"); ax.set_ylabel("count"); ax.set_yscale("log")
ax.set_title("sttl by class (log y)")
ax.legend(); fig.tight_layout()
fig.savefig(EDA_DIR / "sttl_leak.png", dpi=110); plt.close(fig)

(DOCS_DIR / "eda-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n[written] {DOCS_DIR/'eda-summary.md'} + figures in {EDA_DIR}/")
