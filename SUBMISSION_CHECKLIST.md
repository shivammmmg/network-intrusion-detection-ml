# Submission checklist

Preparation notes for packaging the final EECS 3404 submission. This is a
planning document: **no submission archive has been created, and the written
report and video are not finished.**

> The exact submission format (archive structure, page limit, video length and
> hosting, and whether processed data must be included) is **not recorded
> anywhere in this repository**. `EECS 3404 Major Project Idea.pdf` is the team's
> own project proposal, not the course submission specification. Confirm every
> requirement below against the official course instructions before building the
> final archive.

## Status of non-repository deliverables

| Deliverable | Status |
|---|---|
| Final written report (PDF) | **Not started** |
| Presentation / video | **Not started** |
| Contribution statements | Drafted — [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) |
| Reference list / citations | **Not started** (to live in the report) |
| Code, artifacts, outputs | Complete and verified on `main` |

## Include

Source code and configuration:

- `src/` — the complete 00–15 pipeline, including the additional Neural Network
  verification and RF/XGBoost analysis utilities, plus `config.py`,
  `preprocess.py`, and `diagnostics_lib.py`
- `tests/` — unit tests for the diagnostics library and report tables
- `requirements.txt`, `requirements-lock.txt`
- `.gitignore`

Data and artifacts:

- `data/processed/` — the six shared split Parquet files (16.5 MB). Include if
  the course permits processed data; otherwise rely on `src/00_download.py` plus
  `src/02_clean_split.py` to regenerate them.
- `artifacts/` — fitted preprocessors, the four finalized models, and
  `manifest.json` (84 MB, dominated by `random_forest.joblib`)
- Raw data is **not** included and is not tracked; acquisition is scripted in
  `src/00_download.py` with row-count validation.

Experiment outputs:

- `experiments/logistic_regression/`, `experiments/neural_network/`,
  `experiments/random_forest/`, `experiments/xgboost/` — configs, metrics,
  tuning histories, predictions. The historical
  `experiments/random_forest/pre_split_fix/` records may be kept; only the
  74.1 MB model binary inside it is excluded, as described below.
- `experiments/standardized_evaluation/` — thresholds, comparison tables,
  figures
- `experiments/diagnostics/` — provenance, SHAP, importance, errors,
  calibration, drift, TTL ablation, report tables
- `experiments/model_analysis/` — supplemental RF/XGBoost analysis

Documentation:

- `README.md`
- `TEAM_RESPONSIBILITIES.md`
- `CONTRIBUTIONS.md`
- `docs/` — data card, EDA summary, baselines, split manifest, the four
  per-model results documents, the standardized evaluation, the advanced
  RF/XGBoost analysis, and the diagnostics report
- Final report PDF — **add when written**
- Video file or link, per course requirement — **add when recorded**

## Exclude

Always exclude:

- `.git/`
- `.venv/`
- `__pycache__/` (any depth)
- `.pytest_cache/`
- `.DS_Store` and other OS metadata
- Any temporary, scratch or debug files
- `data/raw/` — not tracked, and regenerable via `src/00_download.py`

Exclude for size, with no loss of final results:

- `experiments/random_forest/pre_split_fix/random_forest.joblib` — see below

## The historical Random Forest model

`experiments/random_forest/pre_split_fix/` holds Random Forest results produced
**before** the train/validation predictor-overlap correction. Its own README
already labels it historical.

| Decision | Rationale |
|---|---|
| Keep in the repository and git history | Useful audit trail showing the split fix was made and what changed |
| **Exclude `experiments/random_forest/pre_split_fix/random_forest.joblib` from the submission archive** | 74.1 MB superseded binary with no role in the final results; excluding it removes a third of the archive size |
| Lightweight historical records may remain | The README, `config.json` and `metrics.json` are small and document the correction |
| Never cite as final results | Current Random Forest results are in `experiments/random_forest/` and [`docs/random_forest_results.md`](docs/random_forest_results.md) |

Its validation PR-AUC (0.987519) is **higher** than the corrected model's
(0.984366), because the pre-fix split allowed predictor overlap between train
and validation. Quoting it would overstate performance.

## File-size audit

Total tracked content: **221.5 MB**.

Files over 25 MB:

| Size | Path | Action |
|---:|---|---|
| 80.1 MB | `artifacts/random_forest.joblib` | **Include** — the finalized model |
| 74.1 MB | `experiments/random_forest/pre_split_fix/random_forest.joblib` | **Exclude** — superseded |

Largest remaining files, all included:

| Size | Path |
|---:|---|
| 16.8 MB | `experiments/diagnostics/shap/shap_values_xgboost.csv.gz` |
| 8.4 MB | `data/processed/X_train.parquet` |
| 5.9 MB | `data/processed/X_test.parquet` |
| 5.6 MB | `experiments/xgboost/xgboost_model.json` |
| 3.5 MB | `artifacts/xgboost.joblib` |
| 2.3 MB | `experiments/neural_network/test_predictions.csv` |
| 2.3 MB | `experiments/logistic_regression/test_predictions.csv` |

By directory: `experiments/` 120.1 MB, `artifacts/` 84.1 MB, `data/` 16.5 MB,
`src/` 0.4 MB, `docs/` 0.1 MB, `tests/` < 0.1 MB.

Excluding the historical binary brings the archive to roughly **147 MB**; also
omitting `data/processed/` would bring it to roughly **131 MB**. If a hard size
cap applies, `artifacts/random_forest.joblib` is the only remaining large item,
and it cannot be dropped without losing the finalized model — it would have to be
regenerated from `src/06_random_forest.py`.

Note that `experiments/xgboost/xgboost_model.json` and
`artifacts/xgboost.joblib` are two serializations of the **same** finalized
XGBoost model and produce numerically equivalent predictions within the Stage 0
verification tolerance. Both are kept deliberately;
either could be dropped if size becomes critical, but keeping both is what allows
the model to be loaded without a joblib/scikit-learn version dependency.

## Pre-submission verification

Run from the repository root before packaging:

```bash
python src/11_diagnostics_verify.py --verify     # expect 4/4 PASS
python src/12_shap_diagnostics.py --verify       # expect 14/14 PASS
python src/13_model_diagnostics.py --verify      # expect 16/16 PASS
python src/14_ttl_ablation.py --verify           # expect 5/5 PASS
python src/15_build_report_tables.py --verify    # expect 10/10 PASS
python src/08_neural_network_verify.py           # expect all checks PASS
python -m pytest tests/ -q
git status --short                               # expect empty
git diff --check                                 # expect silent
```

## Final checks

- [ ] Course submission format confirmed against the official instructions
- [ ] Report PDF written and within the page limit
- [ ] Every team member appears in the video and states their contribution
- [ ] `CONTRIBUTIONS.md` reviewed and agreed by the whole team
- [ ] Report figures and numbers taken from committed outputs, not retyped
- [ ] Reference list complete (UNSW-NB15 / Moustafa & Slay, scikit-learn,
      XGBoost, SHAP)
- [ ] Verification commands above all pass
- [ ] Excluded paths confirmed absent from the archive
- [ ] Archive opens cleanly and the README is visible at the top level
