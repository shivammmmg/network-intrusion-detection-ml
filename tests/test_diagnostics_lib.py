"""Unit and regression coverage for Stage 0 diagnostics safeguards."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics_lib import write_json, verify_outputs


def _load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "diagnostics_verify_test_module",
        SRC / "11_diagnostics_verify.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerifyOutputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.committed = pd.DataFrame(
            {
                "sample_index": [0, 1, 2],
                "attack_probability": [0.1, 0.5, 0.9],
                "predicted_label": [0, 1, 1],
            }
        )

    def test_identical_input_passes(self) -> None:
        result = verify_outputs(
            self.committed.copy(), self.committed, tolerance=1e-6, key_columns=["sample_index"]
        )
        self.assertTrue(result["pass"])
        self.assertTrue(result["row_order_matches"])

    def test_value_drift_beyond_tolerance_fails(self) -> None:
        changed = self.committed.copy()
        changed.loc[1, "attack_probability"] += 0.01
        result = verify_outputs(changed, self.committed, tolerance=1e-6, key_columns=["sample_index"])
        self.assertFalse(result["pass"])
        self.assertFalse(result["per_column_pass"]["attack_probability"])

    def test_rank_inversion_fails(self) -> None:
        changed = self.committed.copy()
        changed.loc[0, "attack_probability"] = 0.9
        changed.loc[2, "attack_probability"] = 0.1
        result = verify_outputs(changed, self.committed, tolerance=1e-6, key_columns=["sample_index"])
        self.assertFalse(result["pass"])

    def test_dropped_row_fails(self) -> None:
        result = verify_outputs(
            self.committed.iloc[:-1], self.committed, tolerance=1e-6, key_columns=["sample_index"]
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["row_count_matches"])

    def test_extra_column_fails(self) -> None:
        changed = self.committed.assign(extra_column=1)
        result = verify_outputs(changed, self.committed, tolerance=1e-6, key_columns=["sample_index"])
        self.assertFalse(result["pass"])
        self.assertFalse(result["columns_match"])


class VerifyModeRegressionTests(unittest.TestCase):
    def test_tampered_output_fails_on_two_consecutive_verify_runs(self) -> None:
        module = _load_verify_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            diagnostics_dir = Path(temporary_directory)
            validation_dir = diagnostics_dir / "validation"
            expected_outputs = {
                diagnostics_dir / "provenance.json": {
                    "git_commit": "new-commit",
                    "timestamp_utc": "2026-08-10T00:00:00Z",
                },
                validation_dir / "artifact_prediction_check.json": {"pass": True, "models": {}},
                validation_dir / "leakage_assertions.json": {"pass": True, "assertions": {}},
                validation_dir / "environment_check.json": {"pass": True},
            }
            for path, output in expected_outputs.items():
                write_json(output, path)
            write_json({"pass": False}, validation_dir / "environment_check.json")

            split = (pd.DataFrame({"feature": [1]}), pd.Series([0], name="label"))
            with (
                patch.object(module, "ROOT", diagnostics_dir.parent),
                patch.object(module, "DIAGNOSTICS_DIR", diagnostics_dir),
                patch.object(module, "_read_manifest", return_value={"versions": {}, "artifacts": {}}),
                patch.object(module, "environment_check", return_value={"pass": True}),
                patch.object(module, "load_split", return_value=split),
                patch.object(module, "artifact_prediction_check", return_value=expected_outputs[
                    validation_dir / "artifact_prediction_check.json"
                ]),
                patch.object(module, "leakage_assertions", return_value=expected_outputs[
                    validation_dir / "leakage_assertions.json"
                ]),
                patch.object(module, "capture_provenance", return_value=expected_outputs[
                    diagnostics_dir / "provenance.json"
                ]),
                patch.object(sys, "argv", ["11_diagnostics_verify.py", "--verify"]),
            ):
                first_console = io.StringIO()
                with contextlib.redirect_stdout(first_console):
                    self.assertEqual(module.main(), 1)
                second_console = io.StringIO()
                with contextlib.redirect_stdout(second_console):
                    self.assertEqual(module.main(), 1)

            self.assertIn("VERIFY", first_console.getvalue())
            self.assertIn("FAIL", first_console.getvalue())
            self.assertIn("FAIL", second_console.getvalue())
            with (validation_dir / "environment_check.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"pass": False})


if __name__ == "__main__":
    unittest.main()
