# Data Card: UNSW-NB15 (binary intrusion detection)

Notes on the dataset and everything the preprocessing does to it. Every number
here comes out of the scripts in `src/`, so running them reproduces it.

## Source and license
UNSW-NB15, the pre-partitioned CSVs (`UNSW_NB15_training-set.csv`,
`UNSW_NB15_testing-set.csv`). Made by the Australian Centre for Cyber Security
(Moustafa & Slay), traffic generated with IXIA PerfectStorm. Free for academic
use; commercial use needs the authors' okay. Cite Moustafa & Slay 2015 (MilCIS)
and the 2016 Information Security Journal paper. Task is binary: `label` 0 = normal,
1 = attack.

## Shapes and class balance
| Stage | rows | normal | attack |
|---|--:|--:|--:|
| Raw train | 175,341 | 31.9% | 68.1% |
| Raw test | 82,332 | 44.9% | 55.1% |
| Processed train | 79,685 | 51.6% | 48.4% |
| Processed val | 19,922 | 51.6% | 48.4% |
| Processed test (frozen) | 82,332 | 44.9% | 55.1% |

The raw train and test files don't have the same attack rate (68% vs 55%). The
binary predictor/label deduplication and ambiguous-vector removal reduce the
development pool to 48.4% attack, while the frozen test remains 55.1% attack.

## Feature roles (42 predictors after drops)
Three text columns: `proto` (133 levels, long tail), `service` (13, and `-` means
"no service", which is a real value not a missing one), `state` (9). Three TTL
columns (`sttl`, `dttl`, `ct_state_ttl`) covered in the leakage section. Two 0/1
flags (`is_ftp_login`, `is_sm_ips_ports`). Everything else is numeric flow stats.
`id` and `attack_cat` get dropped: `id` is a row counter, and `attack_cat` gives
the label away.

## Cleaning (constant rules, same on every split)
- `is_ftp_login` had stray values up to 4 even though it's meant to be 0/1, so it's clamped to 1.
- `ct_ftp_cmd` is coerced to numeric (other UNSW-NB15 releases store it with whitespace strings; this file was already numeric).
- `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd` NaNs filled with 0. This file has none, so that's a guard.
- `proto`/`service`/`state` stripped and lowercased so casing can't split one category into two.

## Leakage audit
This is the part that took the most work, so read it before training anything.

`id` and `attack_cat` are dropped, per above (row counter, and a straight
give-away of the label).

The TTL columns are the big one. Because the benign and attack traffic were
generated on different machines, TTL basically encodes the class: 87% of attacks
have `sttl=254`, and the single rule "`sttl >= 32` means attack" scores 92.1% on
train by itself. That's an artifact of how the data was built, not a real signal.
We keep the columns but the main artifacts leave them out, and the number we
report is the without-TTL one. The `*_with_ttl` artifacts exist only to show how
much they inflate the result.

Train/validation predictor-overlap fix ([GitHub Issue #1](https://github.com/shivammmmg/network-intrusion-detection-ml/issues/1)): the binary-task
deduplication key is now all 42 model predictors plus `label`; `attack_cat` is
not allowed to distinguish otherwise identical binary examples. This removed
74,072 same-label predictor duplicates. After that step, 229 unique predictor
vectors (458 rows) mapped to both binary labels; all of those ambiguous rows
were removed rather than silently choosing a label. The remaining development
pool had no repeated predictor vector before its stratified split.

Test leakage is handled separately: 1,204 remaining development rows with
predictor values matching a test row were removed. That match uses features
only, not test labels; test itself is never modified. The regenerated split has
zero train/validation, train/test, and validation/test predictor overlap.

Class balance mismatch: train is 48.4% attack, test 55.1%, so majority-class
accuracy isn't comparable across the two. Compare models to the baseline on
precision/recall/F1/PR-AUC, not accuracy.

One thing we can't fix: the `ct_*` connection-count features were computed by the
dataset authors with sliding windows over the whole capture, before the
train/test partition. That's a form of leakage baked into the dataset itself, so
it belongs in the limitations section of the report.

## Split
Test is the frozen hold-out: cleaned only, touched once at the end. Train is
deduplicated on predictors plus the binary label, ambiguous predictor/label
conflicts are removed, test-overlapping predictors are removed, then the
remaining development pool is split 80/20 into train/val with `random_state=42`,
stratified so both slices keep the attack rate. One split, made once, shared with
everyone. Don't re-split.

## Preprocessing artifacts (fit on train only)
| Artifact | For | Encoding | Scaling | Features |
|---|---|---|---|--:|
| `preprocess_linear.joblib` | LogReg, NN | OneHot (rare folded) | RobustScaler | 66 |
| `preprocess_tree.joblib` | RF, XGBoost | OrdinalEncoder | none | 39 |
| `preprocess_linear_with_ttl.joblib` | ablation | " | " | 69 |
| `preprocess_tree_with_ttl.joblib` | ablation | " | " | 42 |

The corrected train split has fewer repeated categorical observations, so the
train-only `min_frequency`/`max_categories` rules fold more categorical levels;
the linear feature count is therefore 66 (69 with TTL), rather than the former
86/89. Two variants because scaling and one-hot help the linear/NN side but do nothing
for trees (and one-hotting a 133-level column just splinters them). The shared
tree artifact doesn't use XGBoost's native categorical mode; whoever owns XGBoost
can build that off the raw cleaned parquet if they want it.

## Baseline (see `baseline.md`)
`DummyClassifier(most_frequent)` now predicts normal because the corrected
development pool is 51.6% normal: validation accuracy 0.516, F1 0.000, PR-AUC
0.484; frozen-test accuracy 0.449, F1 0.000, PR-AUC 0.551. Real models have to
clear the baseline on F1/PR-AUC, not accuracy.

## Reproducibility
The final environment pins scikit-learn 1.9.0, pandas 3.0.3, numpy 2.4.6, and
xgboost 3.3.0 in `requirements.txt`; `artifacts/manifest.json` records NumPy
2.5.1 as the environment that produced the frozen artifacts. `RANDOM_STATE=42`
throughout. Raw-file SHA-256 in
`artifacts/manifest.json` (training `bec7dd5ec88dc2a0`, testing `734fe6642edf758f`).
Raw data mirror: `github.com/Nir-J/ML-Projects`, `UNSW-Network_Packet_Classification/`,
fetched and checked by `src/00_download.py`. Rebuild with `00` through `04` in order.

## Limitations
Synthetic attacks that may not match real traffic; the TTL and `ct_*` testbed
artifacts; no features for encrypted traffic; and a class balance that doesn't
reflect real base rates, so any threshold or calibration set here would need
re-checking before real use.
