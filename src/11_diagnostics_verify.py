"""Stage 0 verification for frozen intrusion-detection artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.utils.validation import check_is_fitted

from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE
from diagnostics_lib import (
    capture_provenance,
    load_model,
    load_split,
    load_thresholds,
    transform,
    write_json,
)


ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS_DIR = ROOT / "experiments" / "diagnostics"
MODEL_PREPROCESSORS = {
    "logistic_regression": "preprocess_linear.joblib",
    "neural_network": "preprocess_linear.joblib",
    "random_forest": "preprocess_tree.joblib",
    "xgboost": "preprocess_tree.joblib",
}
ALL_PREPROCESSORS = (
    "preprocess_linear.joblib",
    "preprocess_tree.joblib",
    "preprocess_linear_with_ttl.joblib",
    "preprocess_tree_with_ttl.joblib",
)
FIT_CALL_ALLOWLIST = {"14_ttl_ablation.py"}


def _error_result(error: Exception) -> dict[str, Any]:
    return {"pass": False, "error": f"{type(error).__name__}: {error}"}


def _read_manifest() -> dict[str, Any]:
    with (ARTIFACTS_DIR / "manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def environment_check(manifest: dict[str, Any]) -> dict[str, Any]:
    actual = {
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "xgboost": xgboost.__version__,
    }
    packages = {}
    for package, expected in manifest["versions"].items():
        matches = actual.get(package) == expected
        packages[package] = {
            "expected": expected,
            "actual": actual.get(package),
            "matches": matches,
            "severity": "error" if package == "scikit-learn" and not matches else "warning" if not matches else "ok",
        }
    return {
        "packages": packages,
        "random_state_expected": manifest.get("random_state"),
        "random_state_actual": RANDOM_STATE,
        "random_state_matches": manifest.get("random_state") == RANDOM_STATE,
        "pass": all(item["severity"] != "error" for item in packages.values()),
    }


def artifact_prediction_check(X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {"tolerance": 1e-6, "models": {}}
    for name, preprocessor_name in MODEL_PREPROCESSORS.items():
        try:
            committed = pd.read_csv(ROOT / "experiments" / name / "test_predictions.csv")
            X_transformed = transform(ARTIFACTS_DIR / preprocessor_name, X_test)
            probabilities = load_model(name).predict_proba(X_transformed)[:, 1]
            committed_probabilities = committed["attack_probability"].to_numpy(dtype=float)
            csv_labels = committed["true_label"].to_numpy()
            labels = y_test.to_numpy()
            same_length = len(probabilities) == len(committed_probabilities) == len(labels)
            if same_length:
                absolute_difference = np.abs(probabilities - committed_probabilities)
                max_abs_diff = float(absolute_difference.max())
                mean_abs_diff = float(absolute_difference.mean())
                labels_match = bool(np.array_equal(csv_labels, labels))
                sample_index_matches = bool(np.array_equal(committed["sample_index"].to_numpy(), np.arange(len(labels))))
            else:
                max_abs_diff = None
                mean_abs_diff = None
                labels_match = False
                sample_index_matches = False
            model_result = {
                "n_rows": int(len(probabilities)),
                "prediction_rows": int(len(committed)),
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "true_label_matches_y_test": labels_match,
                "sample_index_matches_test_order": sample_index_matches,
                "pass": bool(
                    same_length
                    and labels_match
                    and sample_index_matches
                    and max_abs_diff is not None
                    and max_abs_diff <= result["tolerance"]
                ),
            }
        except Exception as error:  # Record failures so the remaining models still run.
            model_result = _error_result(error)
        result["models"][name] = model_result
    result["pass"] = all(item["pass"] for item in result["models"].values())
    return result


def leakage_assertions(
    manifest: dict[str, Any],
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> dict[str, Any]:
    assertions: dict[str, Any] = {}

    fitted: dict[str, Any] = {}
    X_test = splits["test"][0]
    for filename in ALL_PREPROCESSORS:
        try:
            from preprocess import load_artifact

            preprocessor = load_artifact(ARTIFACTS_DIR / filename)
            check_is_fitted(preprocessor)
            transformed = transform(ARTIFACTS_DIR / filename, X_test)
            expected_features = manifest["artifacts"][Path(filename).stem]["n_features_out"]
            fitted[filename] = {
                "expected_n_features_out": expected_features,
                "actual_n_features_out": int(transformed.shape[1]),
                "pass": transformed.shape[1] == expected_features,
            }
        except Exception as error:
            fitted[filename] = _error_result(error)
    assertions["preprocessors_fitted_and_feature_counts_match_manifest"] = {
        "artifacts": fitted,
        "pass": all(item["pass"] for item in fitted.values()),
    }

    source_dir = ROOT / "src"
    source_files = [
        source_dir / "diagnostics_lib.py",
        *sorted(source_dir.glob("1[1-5]_*.py")),
    ]
    expected_diagnostics = {
        source_dir / "diagnostics_lib.py",
        *source_dir.glob("1[1-5]_*.py"),
    }
    covered_files = set(source_files)
    uncovered_files = sorted(
        str(path.relative_to(ROOT))
        for path in expected_diagnostics
        if path not in covered_files
    )
    fit_pattern = re.compile(r"\.(fit|fit_transform|partial_fit)\s*\(")
    files_with_fit_calls = [
        str(path.relative_to(ROOT))
        for path in source_files
        if (
            path.name not in FIT_CALL_ALLOWLIST
            and fit_pattern.search(path.read_text(encoding="utf-8"))
        )
    ]
    assertions["diagnostics_source_contains_no_fit_calls"] = {
        "files_checked": [str(path.relative_to(ROOT)) for path in source_files],
        "fit_call_allowlist": sorted(FIT_CALL_ALLOWLIST),
        "files_with_fit_calls": files_with_fit_calls,
        "uncovered_files": uncovered_files,
        "pass": not files_with_fit_calls and not uncovered_files,
    }

    # Parquet was deliberately written without its transient pandas index.
    # Use a deterministic row fingerprint as the persisted split identity.
    split_indices = {
        name: set(pd.util.hash_pandas_object(
            pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1),
            index=False,
        ).astype("uint64").tolist())
        for name, (X, y) in splits.items()
    }
    overlaps = {
        "train_val": len(split_indices["train"] & split_indices["val"]),
        "train_test": len(split_indices["train"] & split_indices["test"]),
        "val_test": len(split_indices["val"] & split_indices["test"]),
    }
    assertions["split_rows_pairwise_disjoint"] = {
        "index_identity": "hash of persisted feature and label values",
        "overlap_counts": overlaps,
        "pass": not any(overlaps.values()),
    }

    split_row_counts = {}
    for name, (X, y) in splits.items():
        split_row_counts[name] = {
            "X_rows": int(len(X)),
            "y_rows": int(len(y)),
            "pass": len(X) == len(y),
        }
    assertions["split_feature_and_label_row_counts_match"] = {
        "splits": split_row_counts,
        "pass": all(item["pass"] for item in split_row_counts.values()),
    }

    prediction_rows = {}
    test_rows = len(splits["test"][0])
    for name in MODEL_PREPROCESSORS:
        try:
            prediction = pd.read_csv(ROOT / "experiments" / name / "test_predictions.csv")
            prediction_rows[name] = {
                "test_rows": int(test_rows),
                "prediction_rows": int(len(prediction)),
                "pass": len(prediction) == test_rows,
            }
        except Exception as error:
            prediction_rows[name] = _error_result(error)
    assertions["prediction_row_counts_match_test_split"] = {
        "models": prediction_rows,
        "pass": all(item["pass"] for item in prediction_rows.values()),
    }

    label_values = {}
    for name, (_, y) in splits.items():
        values = sorted(int(value) for value in pd.unique(y.dropna()))
        label_values[name] = {"values": values, "pass": set(values).issubset({0, 1})}
    assertions["labels_are_binary"] = {
        "splits": label_values,
        "pass": all(item["pass"] for item in label_values.values()),
    }

    try:
        thresholds = load_thresholds()
        missing = sorted(set(MODEL_PREPROCESSORS) - set(thresholds))
        invalid = {name: value for name, value in thresholds.items() if not 0 < float(value) < 1}
        assertions["locked_thresholds_present_and_in_open_unit_interval"] = {
            "thresholds": thresholds,
            "missing_models": missing,
            "invalid_thresholds": invalid,
            "pass": not missing and not invalid,
        }
    except Exception as error:
        assertions["locked_thresholds_present_and_in_open_unit_interval"] = _error_result(error)

    return {"assertions": assertions, "pass": all(item["pass"] for item in assertions.values())}


def _without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    result.pop("timestamp_utc", None)
    result.pop("git_commit", None)
    return result


def _json_matches(observed: Any, expected: Any, tolerance: float = 1e-12) -> bool:
    """Compare deterministic report data while allowing machine-precision noise."""
    if isinstance(observed, dict) and isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            _json_matches(observed[key], expected[key], tolerance) for key in observed
        )
    if isinstance(observed, list) and isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_matches(left, right, tolerance) for left, right in zip(observed, expected)
        )
    if (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance)
    return observed == expected


def _verify_existing(outputs: dict[Path, dict[str, Any]]) -> bool:
    passed = True
    for path, current in outputs.items():
        if not path.exists():
            print(f"VERIFY {path.relative_to(ROOT)}: missing prior output")
            passed = False
            continue
        with path.open(encoding="utf-8") as handle:
            prior = json.load(handle)
        expected = _without_timestamp(prior) if path.name == "provenance.json" else prior
        observed = _without_timestamp(current) if path.name == "provenance.json" else current
        matches = _json_matches(observed, expected)
        print(f"VERIFY {path.relative_to(ROOT)}: {'PASS' if matches else 'FAIL'}")
        passed = passed and matches
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="compare rerun results with prior Stage 0 outputs")
    args = parser.parse_args()

    manifest: dict[str, Any]
    try:
        manifest = _read_manifest()
    except Exception as error:
        manifest = {"versions": {}, "artifacts": {}}
        environment = _error_result(error)
    else:
        environment = environment_check(manifest)

    splits: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    split_errors: dict[str, Exception] = {}
    for name in ("train", "val", "test"):
        try:
            splits[name] = load_split(name)
        except Exception as error:
            split_errors[name] = error

    if "test" in splits:
        artifact_check = artifact_prediction_check(*splits["test"])
    else:
        artifact_check = {"pass": False, "error": f"Could not load test split: {split_errors.get('test')}"}

    if len(splits) == 3:
        leakage = leakage_assertions(manifest, splits)
    else:
        leakage = {
            "assertions": {name: _error_result(error) for name, error in split_errors.items()},
            "pass": False,
        }

    try:
        provenance = capture_provenance()
    except Exception as error:
        provenance = _error_result(error)

    outputs = {
        DIAGNOSTICS_DIR / "provenance.json": provenance,
        DIAGNOSTICS_DIR / "validation" / "artifact_prediction_check.json": artifact_check,
        DIAGNOSTICS_DIR / "validation" / "leakage_assertions.json": leakage,
        DIAGNOSTICS_DIR / "validation" / "environment_check.json": environment,
    }
    verify_pass = _verify_existing(outputs) if args.verify else True
    if not args.verify:
        for path, output in outputs.items():
            write_json(output, path)

    print(f"Environment check: {'PASS' if environment['pass'] else 'FAIL'}")
    for name, result in artifact_check.get("models", {}).items():
        print(
            f"Artifact/prediction {name}: {'PASS' if result['pass'] else 'FAIL'} "
            f"(max_abs_diff={result.get('max_abs_diff')})"
        )
    print(f"Artifact/prediction gate: {'PASS' if artifact_check['pass'] else 'FAIL'}")
    print(f"Leakage/integrity assertions: {'PASS' if leakage['pass'] else 'FAIL'}")
    print(f"Provenance captured: {outputs[DIAGNOSTICS_DIR / 'provenance.json'].get('git_commit')}")
    return 0 if all((environment["pass"], artifact_check["pass"], leakage["pass"], verify_pass)) else 1


if __name__ == "__main__":
    sys.exit(main())
