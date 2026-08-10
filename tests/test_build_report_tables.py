"""Focused unit coverage for Stage 4 report-table transformations."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_module():
    spec = importlib.util.spec_from_file_location("build_report_tables_test_module", SRC / "15_build_report_tables.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


class ReportTableTransformTests(unittest.TestCase):
    def test_ttl_delta_is_with_ttl_minus_no_ttl(self) -> None:
        source = pd.DataFrame(
            {
                "arm": ["no_ttl_frozen", "with_ttl_refit"],
                "pr_auc": [0.90, 0.9011],
                "roc_auc": [0.80, 0.80],
                "f1_locked_threshold": [0.70, 0.71],
                "precision_locked_threshold": [0.60, 0.60],
                "recall_locked_threshold": [0.50, 0.51],
                "accuracy_locked_threshold": [0.85, 0.86],
                "locked_threshold": [0.42, 0.42],
                "threshold_note": ["shared threshold", "shared threshold"],
            }
        )
        table, threshold, note = MODULE._build_ttl_comparison(source)
        self.assertAlmostEqual(table.loc[0, "delta"], 0.0011, places=12)
        self.assertEqual(threshold, 0.42)
        self.assertEqual(note, "shared threshold")

    def test_drift_threshold_counts_are_derived_from_rows(self) -> None:
        drift = pd.DataFrame(
            {
                "psi": [0.21, 0.20, 0.50, 0.01],
                "psi_degenerate": [False, True, False, False],
                "psi_low_resolution": [True, False, True, False],
            }
        )
        counts = MODULE._drift_summary_counts(drift)
        self.assertEqual(counts["total_feature_count"], 4)
        self.assertEqual(counts["psi_over_0_2_count"], 2)
        self.assertEqual(counts["psi_degenerate_count"], 1)
        self.assertEqual(counts["psi_low_resolution_count"], 2)
        self.assertEqual(counts["psi_over_0_2_percentage"], 50.0)

    def test_t1_retains_non_null_scale_for_every_row(self) -> None:
        source = pd.DataFrame(
            {
                "rank": range(1, 11),
                "feature": [f"feature_{index}" for index in range(10)],
                "mean_abs_shap_value": [0.1] * 10,
                "mean_shap_value": [0.0] * 10,
                "scale": ["probability"] * 10,
            }
        )
        original_read_csv = MODULE._read_csv
        try:
            MODULE._read_csv = lambda path, columns: source.copy()
            table = MODULE._build_shap_global()
        finally:
            MODULE._read_csv = original_read_csv
        self.assertIn("scale", table.columns)
        self.assertFalse(table["scale"].isna().any())


class WorstBinLookupTests(unittest.TestCase):
    BINS = pd.DataFrame(
        {
            "bin": [1, 2, 3],
            "mean_predicted": [0.05, 0.53, 0.94],
            "observed_frequency": [0.01, 0.21, 0.83],
            "count": [100, 200, 300],
            "gap": [-0.04, -0.32, -0.11],
        }
    )

    def _with_stub_bins(self, bins: pd.DataFrame):
        original = MODULE._read_csv
        MODULE._read_csv = lambda path, columns: bins.copy()
        self.addCleanup(lambda: setattr(MODULE, "_read_csv", original))

    def test_returns_mean_predicted_of_the_named_bin(self) -> None:
        self._with_stub_bins(self.BINS)
        self.assertAlmostEqual(MODULE._worst_bin_mean_predicted("any_model", 2), 0.53, places=12)

    def test_bin_index_alone_does_not_determine_probability(self) -> None:
        # Bin 2 sits at a different predicted probability under a different
        # binning, which is why the report cannot compare bin indices directly.
        self._with_stub_bins(self.BINS.assign(mean_predicted=[0.10, 0.86, 0.99]))
        self.assertAlmostEqual(MODULE._worst_bin_mean_predicted("other_model", 2), 0.86, places=12)

    def test_missing_bin_raises(self) -> None:
        self._with_stub_bins(self.BINS)
        with self.assertRaises(ValueError):
            MODULE._worst_bin_mean_predicted("any_model", 99)


class MarkdownVerificationTests(unittest.TestCase):
    def _run_against_file(self, on_disk: str, rebuilt: str) -> bool:
        original = MODULE.MARKDOWN_PATH
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tables.md"
            path.write_text(on_disk, encoding="utf-8", newline="\n")
            try:
                MODULE.MARKDOWN_PATH = path
                return MODULE._verify_markdown(rebuilt)
            finally:
                MODULE.MARKDOWN_PATH = original

    def test_matching_markdown_passes(self) -> None:
        body = "## T1 — SHAP global importance"
        self.assertTrue(self._run_against_file(MODULE._markdown_document(body), body))

    def test_hand_edited_markdown_fails(self) -> None:
        body = "| random_forest | 1 | ackdat | 0.0552 | probability |"
        edited = MODULE._markdown_document(body).replace("0.0552", "0.0667")
        self.assertFalse(self._run_against_file(edited, body))

    def test_missing_markdown_fails(self) -> None:
        original = MODULE.MARKDOWN_PATH
        with tempfile.TemporaryDirectory() as directory:
            try:
                MODULE.MARKDOWN_PATH = Path(directory) / "absent.md"
                self.assertFalse(MODULE._verify_markdown("anything"))
            finally:
                MODULE.MARKDOWN_PATH = original


if __name__ == "__main__":
    unittest.main()
