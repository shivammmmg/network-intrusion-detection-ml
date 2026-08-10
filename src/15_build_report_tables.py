"""Stage 4 diagnostics: build deterministic report tables from verified outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from diagnostics_lib import verify_outputs, write_csv


ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS_DIR = ROOT / "experiments" / "diagnostics"
REPORT_TABLES_DIR = DIAGNOSTICS_DIR / "report_tables"
MARKDOWN_PATH = REPORT_TABLES_DIR / "tables.md"
MODELS = ("logistic_regression", "neural_network", "random_forest", "xgboost")
SHAP_MODELS = ("random_forest", "xgboost")
TTL_METRICS = (
    "pr_auc",
    "roc_auc",
    "f1_locked_threshold",
    "precision_locked_threshold",
    "recall_locked_threshold",
    "accuracy_locked_threshold",
)
TTL_FEATURES = ("sttl", "dttl", "ct_state_ttl")
VERIFY_TOLERANCE = 1e-12


def _require_file(path: Path) -> Path:
    """Return an existing source path or raise a clear error before reading."""
    if not path.is_file():
        raise FileNotFoundError(f"Required diagnostics source is missing: {path}")
    return path


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a source CSV after checking its required schema."""
    table = pd.read_csv(_require_file(path))
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if table.empty:
        raise ValueError(f"{path} is empty")
    return table


def _read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON source object."""
    with _require_file(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _required_sources() -> list[Path]:
    """List all verified Stage 1--3 sources expected by this report stage."""
    paths = [
        DIAGNOSTICS_DIR / "shap" / "representative_cases.csv",
        DIAGNOSTICS_DIR / "shap" / "shap_scale_metadata.json",
        DIAGNOSTICS_DIR / "errors" / "error_summary.json",
        DIAGNOSTICS_DIR / "calibration" / "calibration_summary.json",
        DIAGNOSTICS_DIR / "drift" / "drift_psi_ks.csv",
        DIAGNOSTICS_DIR / "drift" / "drift_importance_overlap.csv",
        DIAGNOSTICS_DIR / "ttl_ablation" / "ttl_metrics_comparison.csv",
        DIAGNOSTICS_DIR / "ttl_ablation" / "ttl_permutation_importance.csv",
        DIAGNOSTICS_DIR / "ttl_ablation" / "ttl_rank_shift.csv",
    ]
    paths.extend(DIAGNOSTICS_DIR / "shap" / f"shap_global_importance_{name}.csv" for name in SHAP_MODELS)
    paths.extend(DIAGNOSTICS_DIR / "importance" / f"permutation_importance_{name}.csv" for name in MODELS)
    paths.extend(DIAGNOSTICS_DIR / "errors" / f"error_profiles_{name}.csv" for name in MODELS)
    paths.extend(DIAGNOSTICS_DIR / "calibration" / f"calibration_bins_{name}.csv" for name in MODELS)
    return paths


def _assert_all_sources_exist() -> None:
    """Ensure generation cannot proceed from an incomplete diagnostics run."""
    for path in _required_sources():
        _require_file(path)


def _build_shap_global() -> pd.DataFrame:
    """Build T1 from the verified SHAP global-importance tables."""
    tables = []
    for name in SHAP_MODELS:
        source = _read_csv(
            DIAGNOSTICS_DIR / "shap" / f"shap_global_importance_{name}.csv",
            ["rank", "feature", "mean_abs_shap_value", "mean_shap_value", "scale"],
        )
        top = source.sort_values(["rank", "feature"], kind="stable").head(10).copy()
        if len(top) != 10 or top["scale"].isna().any() or (top["scale"].astype(str) == "").any():
            raise ValueError(f"SHAP table for {name} must have ten rows with non-null scales")
        top.insert(0, "model", name)
        tables.append(top[["model", "rank", "feature", "mean_abs_shap_value", "scale"]])
    return pd.concat(tables, ignore_index=True)


def _build_permutation_importance() -> pd.DataFrame:
    """Build T2 from the four verified permutation-importance tables."""
    tables = []
    for name in MODELS:
        source = _read_csv(
            DIAGNOSTICS_DIR / "importance" / f"permutation_importance_{name}.csv",
            ["feature", "importance_mean", "importance_std", "rank"],
        )
        top = source.sort_values(["rank", "feature"], kind="stable").head(10).copy()
        if len(top) != 10:
            raise ValueError(f"Permutation table for {name} has fewer than ten rows")
        top.insert(0, "model", name)
        tables.append(top[["model", "rank", "feature", "importance_mean", "importance_std"]])
    return pd.concat(tables, ignore_index=True)


def _build_error_summary(summary: dict[str, Any]) -> pd.DataFrame:
    """Build T3 from the verified error-summary JSON."""
    models = summary.get("models")
    if not isinstance(models, dict):
        raise ValueError("error_summary.json must contain models")
    rows = []
    for name in MODELS:
        model = models.get(name)
        if not isinstance(model, dict):
            raise ValueError(f"error_summary.json is missing {name}")
        counts = model.get("group_counts")
        if not isinstance(counts, dict) or any(group not in counts for group in ("TP", "TN", "FP", "FN")):
            raise ValueError(f"error_summary.json has incomplete group counts for {name}")
        rows.append(
            {
                "model": name,
                "TP": counts["TP"],
                "TN": counts["TN"],
                "FP": counts["FP"],
                "FN": counts["FN"],
                "false_positive_rate": model["false_positive_rate"],
                "false_negative_rate": model["false_negative_rate"],
            }
        )
    return pd.DataFrame(rows)


def _worst_bin_mean_predicted(model_name: str, worst_bin: int) -> float:
    """Look up the mean predicted probability of a model's worst calibration bin.

    Bin indices are per-model quantile bins, so the index alone is not
    comparable across models. Surfacing the bin's mean predicted probability
    gives the report a comparable location for each worst bin.
    """
    bins = _read_csv(
        DIAGNOSTICS_DIR / "calibration" / f"calibration_bins_{model_name}.csv",
        ["bin", "mean_predicted", "observed_frequency", "count", "gap"],
    )
    matches = bins.loc[bins["bin"] == worst_bin]
    if len(matches) != 1:
        raise ValueError(
            f"calibration_bins_{model_name}.csv does not contain exactly one row for bin {worst_bin}"
        )
    return float(matches.iloc[0]["mean_predicted"])


def _build_calibration_summary(summary: dict[str, Any]) -> pd.DataFrame:
    """Build T4 from verified calibration summary values."""
    models = summary.get("models")
    if not isinstance(models, dict):
        raise ValueError("calibration_summary.json must contain models")
    rows = []
    for name in MODELS:
        model = models.get(name)
        if not isinstance(model, dict) or not isinstance(model.get("worst_bin"), dict):
            raise ValueError(f"calibration_summary.json is incomplete for {name}")
        worst = model["worst_bin"]
        rows.append(
            {
                "model": name,
                "brier_score": model["brier_score"],
                "expected_calibration_error": model["expected_calibration_error"],
                "worst_bin": worst["bin"],
                "worst_bin_mean_predicted": _worst_bin_mean_predicted(name, int(worst["bin"])),
                "worst_bin_gap": worst["gap"],
                "worst_bin_count": worst["count"],
            }
        )
    return pd.DataFrame(rows).sort_values(["brier_score", "model"], ignore_index=True)


def _drift_summary_counts(drift: pd.DataFrame) -> dict[str, int | float]:
    """Return the T5 scalar evidence derived solely by counting drift rows."""
    required = ["psi", "psi_degenerate", "psi_low_resolution"]
    missing = [column for column in required if column not in drift]
    if missing:
        raise ValueError(f"Drift table is missing columns: {missing}")
    total = len(drift)
    if total == 0:
        raise ValueError("Drift table is empty")
    over_threshold = int((drift["psi"] > 0.2).sum())
    return {
        "total_feature_count": total,
        "psi_over_0_2_count": over_threshold,
        "psi_over_0_2_percentage": 100 * over_threshold / total,
        "psi_degenerate_count": int(drift["psi_degenerate"].astype(bool).sum()),
        "psi_low_resolution_count": int(drift["psi_low_resolution"].astype(bool).sum()),
    }


def _build_drift_summary(drift: pd.DataFrame) -> pd.DataFrame:
    """Build T5's PSI and KS top-ten sections from the verified drift table."""
    columns = ["section", "rank", "feature", "psi", "ks_statistic", "binning_strategy", "psi_low_resolution"]
    sections = []
    for section, sort_column in (("top_psi", "psi"), ("top_ks", "ks_statistic")):
        top = drift.sort_values([sort_column, "feature"], ascending=[False, True], na_position="last").head(10).copy()
        if len(top) != 10:
            raise ValueError("Drift table has fewer than ten rows")
        top.insert(0, "rank", range(1, len(top) + 1))
        top.insert(0, "section", section)
        sections.append(top[columns])
    return pd.concat(sections, ignore_index=True)


def _build_drift_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    """Build T6 by deterministically ordering the verified overlap table."""
    required = [
        "model", "feature", "permutation_rank", "drift_rank_by_psi", "drift_rank_by_ks",
        "psi", "ks_statistic", "psi_degenerate", "psi_low_resolution",
    ]
    missing = [column for column in required if column not in overlap]
    if missing:
        raise ValueError(f"Drift overlap table is missing columns: {missing}")
    return overlap[required].sort_values(["model", "permutation_rank", "feature"], ignore_index=True)


def _build_ttl_comparison(metrics: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    """Build T7 and return its shared threshold evidence for Markdown."""
    required = ["arm", *TTL_METRICS, "locked_threshold", "threshold_note"]
    missing = [column for column in required if column not in metrics]
    if missing:
        raise ValueError(f"TTL metrics table is missing columns: {missing}")
    indexed = metrics.set_index("arm")
    if not indexed.index.is_unique:
        raise ValueError("TTL metrics table must contain one row per arm")
    expected_arms = {"no_ttl_frozen", "with_ttl_refit"}
    if set(indexed.index) != expected_arms:
        raise ValueError("TTL metrics table must contain exactly no_ttl_frozen and with_ttl_refit")
    thresholds = indexed["locked_threshold"]
    notes = indexed["threshold_note"]
    if thresholds.nunique(dropna=False) != 1 or notes.nunique(dropna=False) != 1:
        raise ValueError("TTL arms must share one locked threshold and threshold note")
    rows = []
    for metric in TTL_METRICS:
        no_ttl = float(indexed.loc["no_ttl_frozen", metric])
        with_ttl = float(indexed.loc["with_ttl_refit", metric])
        rows.append(
            {
                "metric": metric,
                "no_ttl_frozen": no_ttl,
                "with_ttl_refit": with_ttl,
                "delta": with_ttl - no_ttl,
            }
        )
    return pd.DataFrame(rows), float(thresholds.iloc[0]), str(notes.iloc[0])


def _build_ttl_rank_shift(rank_shift: pd.DataFrame) -> pd.DataFrame:
    """Build T8's top ten rank shifts without calculating new importance values."""
    required = ["feature", "no_ttl_rank", "no_ttl_importance_mean", "with_ttl_rank", "with_ttl_importance_mean"]
    missing = [column for column in required if column not in rank_shift]
    if missing:
        raise ValueError(f"TTL rank-shift table is missing columns: {missing}")
    return rank_shift[required].sort_values(["no_ttl_rank", "feature"], na_position="last").head(10).reset_index(drop=True)


def _ttl_feature_evidence(importance: pd.DataFrame) -> pd.DataFrame:
    """Extract the three TTL shortcut-feature ranks from the with-TTL source."""
    required = ["arm", "feature", "importance_mean", "rank", "importance_std"]
    missing = [column for column in required if column not in importance]
    if missing:
        raise ValueError(f"TTL permutation table is missing columns: {missing}")
    evidence = importance.loc[
        (importance["arm"] == "with_ttl_refit") & importance["feature"].isin(TTL_FEATURES),
        ["feature", "rank", "importance_mean"],
    ].copy()
    if set(evidence["feature"]) != set(TTL_FEATURES):
        raise ValueError("TTL permutation table is missing a requested TTL feature")
    return evidence.sort_values(["rank", "feature"], ignore_index=True)


def _build_trust_questions(
    shap: pd.DataFrame,
    importance: pd.DataFrame,
    errors: pd.DataFrame,
    calibration: pd.DataFrame,
    drift_counts: dict[str, int | float],
    overlap: pd.DataFrame,
    ttl_features: pd.DataFrame,
    representative_cases: pd.DataFrame,
) -> pd.DataFrame:
    """Build T9 using source-derived headline evidence for each report question."""
    top_shap = "; ".join(
        f"{name} {row.feature}={row.mean_abs_shap_value:.4f} ({row.scale})"
        for name in SHAP_MODELS
        for row in [shap.loc[shap["model"] == name].sort_values("rank").iloc[0]]
    )
    top_permutation = "; ".join(
        f"{name} {row.feature}={row.importance_mean:.4f}"
        for name in MODELS
        for row in [importance.loc[importance["model"] == name].sort_values("rank").iloc[0]]
    )
    best_brier = calibration.iloc[0]
    overlap_counts = overlap.groupby("model", sort=True).size()
    overlap_text = ", ".join(f"{name}={int(count)}" for name, count in overlap_counts.items())
    ttl_text = ", ".join(
        f"{row.feature} rank {int(row.rank)} ({row.importance_mean:.4f})"
        for row in ttl_features.itertuples(index=False)
    )
    best_fpr = errors.sort_values(["false_positive_rate", "model"]).iloc[0]
    best_fnr = errors.sort_values(["false_negative_rate", "model"]).iloc[0]
    if len(representative_cases) == 0:
        raise ValueError("representative_cases.csv is empty")
    return pd.DataFrame(
        [
            {
                "question": "Why is the model making these predictions?",
                "headline_evidence": f"Top SHAP: {top_shap}; top permutation: {top_permutation}.",
                "source_file": "shap/shap_global_importance_{random_forest,xgboost}.csv; importance/permutation_importance_{models}.csv",
            },
            {
                "question": "Can its probabilities be trusted?",
                "headline_evidence": (
                    f"Best Brier: {best_brier.model}={best_brier.brier_score:.4f}; "
                    f"ECE={best_brier.expected_calibration_error:.4f}; "
                    f"worst-bin gap={best_brier.worst_bin_gap:.4f} (bin {int(best_brier.worst_bin)})."
                ),
                "source_file": "calibration/calibration_summary.json",
            },
            {
                "question": "Is it relying on unstable or dataset-specific features?",
                "headline_evidence": (
                    f"PSI>0.2: {drift_counts['psi_over_0_2_count']}/{drift_counts['total_feature_count']} "
                    f"({drift_counts['psi_over_0_2_percentage']:.2f}%); overlap counts: {overlap_text}; "
                    f"with-TTL: {ttl_text}."
                ),
                "source_file": "drift/drift_psi_ks.csv; drift/drift_importance_overlap.csv; ttl_ablation/ttl_rank_shift.csv; ttl_ablation/ttl_permutation_importance.csv",
            },
            {
                "question": "Why does it fail on certain samples?",
                "headline_evidence": (
                    f"Lowest FPR: {best_fpr.model}={best_fpr.false_positive_rate:.4f}; "
                    f"lowest FNR: {best_fnr.model}={best_fnr.false_negative_rate:.4f}; "
                    f"{len(representative_cases)} selected local-SHAP cases."
                ),
                "source_file": "errors/error_summary.json; shap/representative_cases.csv",
            },
        ]
    )


def _format_value(value: Any, column: str) -> str:
    """Format report Markdown cells with the fixed precision required by the brief."""
    if pd.isna(value):
        return ""
    if column == "delta":
        return f"{float(value):+.4f}"
    if isinstance(value, bool):
        return str(value).lower()
    if column in {"rank", "worst_bin", "worst_bin_count", "TP", "TN", "FP", "FN"} or column.endswith("_rank"):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(table: pd.DataFrame) -> str:
    """Render a deterministic GitHub-flavoured pipe table."""
    columns = list(table.columns)
    lines = [f"| {' | '.join(columns)} |", f"| {' | '.join('---' for _ in columns)} |"]
    for row in table.itertuples(index=False, name=None):
        lines.append(f"| {' | '.join(_format_value(value, column) for value, column in zip(row, columns))} |")
    return "\n".join(lines)


def _build_markdown(
    tables: dict[str, pd.DataFrame],
    drift_counts: dict[str, int | float],
    overlap: pd.DataFrame,
    threshold: float,
    threshold_note: str,
    ttl_features: pd.DataFrame,
) -> str:
    """Build byte-stable report Markdown from the in-memory CSV tables."""
    overlap_counts = overlap.groupby("model", sort=True).size()
    overlap_text = ", ".join(f"{name}: {int(count)}" for name, count in overlap_counts.items())
    parts = ["generated_from: experiments/diagnostics/", ""]
    headings = (
        ("T1", "SHAP global importance"),
        ("T2", "Permutation importance"),
        ("T3", "Error summary"),
        ("T4", "Calibration summary"),
        ("T5", "Drift summary"),
        ("T6", "Importance × drift overlap"),
        ("T7", "TTL comparison"),
        ("T8", "TTL rank shift"),
        ("T9", "Trust-question evidence map"),
    )
    for key, title in headings:
        parts.extend([f"## {key} — {title}", "", _markdown_table(tables[key]), ""])
        if key == "T1":
            parts.extend(["**Rankings are comparable across the two models; magnitudes are not.**", ""])
        elif key == "T5":
            parts.extend([
                (
                    f"Drift scalars: {drift_counts['total_feature_count']} total features; "
                    f"{drift_counts['psi_over_0_2_count']} ({drift_counts['psi_over_0_2_percentage']:.2f}%) with PSI > 0.2; "
                    f"{drift_counts['psi_degenerate_count']} PSI-degenerate; "
                    f"{drift_counts['psi_low_resolution_count']} low-resolution."
                ),
                "",
            ])
        elif key == "T6":
            parts.extend([f"Overlapping features per model: {overlap_text}.", ""])
        elif key == "T7":
            parts.extend([
                f"locked_threshold: {threshold}",
                f"threshold_note: {threshold_note}",
                "",
            ])
        elif key == "T8":
            parts.extend(["TTL shortcut-feature evidence (with_ttl_refit):", "", _markdown_table(ttl_features), ""])
    return "\n".join(parts)


def _generate() -> tuple[dict[Path, tuple[pd.DataFrame, list[str]]], str]:
    """Read verified sources and construct every report table in memory."""
    _assert_all_sources_exist()
    shap = _build_shap_global()
    permutation = _build_permutation_importance()
    errors = _build_error_summary(_read_json(DIAGNOSTICS_DIR / "errors" / "error_summary.json"))
    calibration = _build_calibration_summary(_read_json(DIAGNOSTICS_DIR / "calibration" / "calibration_summary.json"))
    drift = _read_csv(
        DIAGNOSTICS_DIR / "drift" / "drift_psi_ks.csv",
        ["feature", "psi", "ks_statistic", "n_bins_effective", "binning_strategy", "psi_degenerate", "psi_low_resolution"],
    )
    drift_counts = _drift_summary_counts(drift)
    drift_summary = _build_drift_summary(drift)
    overlap = _build_drift_overlap(_read_csv(
        DIAGNOSTICS_DIR / "drift" / "drift_importance_overlap.csv",
        ["model", "feature", "permutation_rank", "drift_rank_by_psi", "drift_rank_by_ks", "psi", "ks_statistic", "psi_degenerate", "psi_low_resolution"],
    ))
    ttl_comparison, threshold, threshold_note = _build_ttl_comparison(_read_csv(
        DIAGNOSTICS_DIR / "ttl_ablation" / "ttl_metrics_comparison.csv",
        ["arm", *TTL_METRICS, "locked_threshold", "threshold_note"],
    ))
    ttl_rank_shift = _build_ttl_rank_shift(_read_csv(
        DIAGNOSTICS_DIR / "ttl_ablation" / "ttl_rank_shift.csv",
        ["feature", "no_ttl_rank", "no_ttl_importance_mean", "with_ttl_rank", "with_ttl_importance_mean"],
    ))
    ttl_features = _ttl_feature_evidence(_read_csv(
        DIAGNOSTICS_DIR / "ttl_ablation" / "ttl_permutation_importance.csv",
        ["arm", "feature", "importance_mean", "rank", "importance_std"],
    ))
    representative_cases = _read_csv(
        DIAGNOSTICS_DIR / "shap" / "representative_cases.csv",
        ["model", "group", "sample_index", "true_label", "attack_probability", "threshold", "selection_status"],
    )
    trust_questions = _build_trust_questions(
        shap, permutation, errors, calibration, drift_counts, overlap, ttl_features, representative_cases
    )
    tables = {
        "T1": shap,
        "T2": permutation,
        "T3": errors,
        "T4": calibration,
        "T5": drift_summary,
        "T6": overlap,
        "T7": ttl_comparison,
        "T8": ttl_rank_shift,
        "T9": trust_questions,
    }
    outputs = {
        REPORT_TABLES_DIR / "t1_shap_global.csv": (shap, ["model", "rank"]),
        REPORT_TABLES_DIR / "t2_permutation_importance.csv": (permutation, ["model", "rank"]),
        REPORT_TABLES_DIR / "t3_error_summary.csv": (errors, ["model"]),
        REPORT_TABLES_DIR / "t4_calibration_summary.csv": (calibration, ["model"]),
        REPORT_TABLES_DIR / "t5_drift_summary.csv": (drift_summary, ["section", "rank"]),
        REPORT_TABLES_DIR / "t6_drift_importance_overlap.csv": (overlap, ["model", "permutation_rank", "feature"]),
        REPORT_TABLES_DIR / "t7_ttl_comparison.csv": (ttl_comparison, ["metric"]),
        REPORT_TABLES_DIR / "t8_ttl_rank_shift.csv": (ttl_rank_shift, ["feature"]),
        REPORT_TABLES_DIR / "t9_trust_questions.csv": (trust_questions, ["question"]),
    }
    return outputs, _build_markdown(tables, drift_counts, overlap, threshold, threshold_note, ttl_features)


def _display_path(path: Path) -> str:
    """Render a path relative to the repository root when it lies inside it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _markdown_document(markdown: str) -> str:
    """Return the exact on-disk representation of the report Markdown."""
    return markdown + "\n"


def _verify_markdown(markdown: str) -> bool:
    """Compare the rebuilt Markdown with the committed tables.md byte for byte.

    The Markdown is the paste source for the written report, so an edit made
    directly to it would otherwise pass verification even though it no longer
    reflects the diagnostics outputs.
    """
    if not MARKDOWN_PATH.exists():
        print(f"VERIFY {_display_path(MARKDOWN_PATH)}: FAIL")
        return False
    committed = MARKDOWN_PATH.read_text(encoding="utf-8")
    passed = committed == _markdown_document(markdown)
    print(f"VERIFY {_display_path(MARKDOWN_PATH)}: {'PASS' if passed else 'FAIL'}")
    return passed


def _verify(outputs: dict[Path, tuple[pd.DataFrame, list[str]]], markdown: str) -> bool:
    """Compare all in-memory report tables with their committed versions."""
    passed = True
    for path, (table, key_columns) in outputs.items():
        if not path.exists():
            print(f"VERIFY {_display_path(path)}: FAIL")
            passed = False
            continue
        result = verify_outputs(table, path, VERIFY_TOLERANCE, key_columns)
        print(f"VERIFY {_display_path(path)}: {'PASS' if result['pass'] else 'FAIL'}")
        passed = passed and result["pass"]
    return _verify_markdown(markdown) and passed


def main() -> int:
    """Generate report tables, or verify their committed CSV representations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="rebuild and compare without writing")
    args = parser.parse_args()
    outputs, markdown = _generate()
    if args.verify:
        return 0 if _verify(outputs, markdown) else 1
    for path, (table, _) in outputs.items():
        write_csv(table, path)
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MARKDOWN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_markdown_document(markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
