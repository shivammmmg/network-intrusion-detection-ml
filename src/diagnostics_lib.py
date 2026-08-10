"""Shared read-only helpers for the diagnostics workstream."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.utils.validation import check_is_fitted


def _project_root() -> Path:
    """Return the repository root without reading any project files."""
    return Path(__file__).resolve().parent.parent


def _manifest() -> dict[str, Any]:
    from config import ARTIFACTS_DIR

    with (ARTIFACTS_DIR / "manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load one processed feature/label split by name."""
    if name not in {"train", "val", "test"}:
        raise ValueError("name must be one of: train, val, test")

    from config import PROCESSED_DIR, TARGET

    X = pd.read_parquet(PROCESSED_DIR / f"X_{name}.parquet")
    y_frame = pd.read_parquet(PROCESSED_DIR / f"y_{name}.parquet")
    if TARGET not in y_frame.columns:
        raise ValueError(f"{name} labels do not contain {TARGET!r}")
    return X, y_frame[TARGET]


def transform(preprocessor_path: str | Path, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply an already fitted preprocessor and validate its feature width."""
    from preprocess import load_artifact

    path = Path(preprocessor_path)
    preprocessor = load_artifact(path)
    check_is_fitted(preprocessor)
    transformed = preprocessor.transform(X_raw)
    feature_names = list(preprocessor.get_feature_names_out())
    if isinstance(transformed, pd.DataFrame):
        result = transformed
        actual_names = list(result.columns)
        if actual_names != feature_names:
            differences = [
                f"position {index}: expected {expected!r}, got {actual!r}"
                for index, (expected, actual) in enumerate(zip(feature_names, actual_names))
                if expected != actual
            ]
            if len(actual_names) != len(feature_names):
                differences.append(
                    f"column count: expected {len(feature_names)}, got {len(actual_names)}"
                )
            raise AssertionError(
                f"{path.name} transformed DataFrame columns do not match "
                f"get_feature_names_out(): {'; '.join(differences[:5])}"
            )
    else:
        result = pd.DataFrame(transformed, index=X_raw.index, columns=feature_names)

    artifact_name = path.stem
    expected_width = _manifest()["artifacts"][artifact_name]["n_features_out"]
    if result.shape[1] != expected_width:
        raise AssertionError(
            f"{artifact_name} produced {result.shape[1]} features; expected {expected_width}"
        )
    return result


def load_model(name: str):
    """Load one frozen primary model without modifying it."""
    model_names = {
        "logistic_regression",
        "neural_network",
        "random_forest",
        "xgboost",
    }
    if name not in model_names:
        raise ValueError(f"Unknown model: {name}")
    from config import ARTIFACTS_DIR

    return joblib.load(ARTIFACTS_DIR / f"{name}.joblib")


def load_thresholds() -> dict[str, float]:
    """Read the locked decision thresholds."""
    path = _project_root() / "experiments" / "standardized_evaluation" / "selected_thresholds.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_provenance() -> dict[str, Any]:
    """Capture the Stage 0 inputs and the runtime that consumed them."""
    from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE

    root = _project_root()
    input_paths = [
        ARTIFACTS_DIR / "manifest.json",
        *(ARTIFACTS_DIR / f"{name}.joblib" for name in (
            "logistic_regression",
            "neural_network",
            "random_forest",
            "xgboost",
            "preprocess_linear",
            "preprocess_tree",
            "preprocess_linear_with_ttl",
            "preprocess_tree_with_ttl",
        )),
        *(PROCESSED_DIR / f"{kind}_{split}.parquet" for split in ("train", "val", "test") for kind in ("X", "y")),
        *(root / "experiments" / name / "test_predictions.csv" for name in (
            "logistic_regression",
            "neural_network",
            "random_forest",
            "xgboost",
        )),
        root / "experiments" / "standardized_evaluation" / "selected_thresholds.json",
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    return {
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "random_state": RANDOM_STATE,
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "inputs": {
            str(path.relative_to(root)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_paths
        },
    }


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV with portable LF line endings."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, lineterminator="\n")


def write_json(obj: Any, path: str | Path) -> None:
    """Write formatted JSON with portable LF line endings."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def verify_outputs(
    new: pd.DataFrame,
    committed: pd.DataFrame | str | Path,
    tolerance: float,
    key_columns: Iterable[str],
) -> dict[str, Any]:
    """Compare fresh tabular results with a committed CSV deterministically."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    committed_df = pd.read_csv(committed) if isinstance(committed, (str, Path)) else committed.copy()
    fresh = new.reset_index(drop=True)
    expected = committed_df.reset_index(drop=True)
    keys = list(key_columns)
    missing_keys = [key for key in keys if key not in fresh or key not in expected]
    columns_match = list(fresh.columns) == list(expected.columns)
    row_count_matches = len(fresh) == len(expected)
    row_order_matches = not missing_keys and row_count_matches and all(
        fresh[key].equals(expected[key]) for key in keys
    )

    per_column_max_abs_diff: dict[str, float | None] = {}
    per_column_pass: dict[str, bool] = {}
    common_columns = [column for column in fresh.columns if column in expected.columns]
    for column in common_columns:
        left = fresh[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if len(left) != len(right):
                per_column_max_abs_diff[column] = None
                per_column_pass[column] = False
                continue
            difference = (left.astype(float) - right.astype(float)).abs()
            both_missing = left.isna() & right.isna()
            mismatch_missing = left.isna() ^ right.isna()
            per_column_max_abs_diff[column] = (
                float(difference.mask(both_missing).max()) if difference.notna().any() else 0.0
            )
            per_column_pass[column] = (
                not bool(mismatch_missing.any())
                and per_column_max_abs_diff[column] <= tolerance
            )
        else:
            per_column_max_abs_diff[column] = None
            per_column_pass[column] = len(left) == len(right) and left.equals(right)

    return {
        "columns_match": columns_match,
        "row_count_matches": row_count_matches,
        "missing_key_columns": missing_keys,
        "row_order_matches": row_order_matches,
        "per_column_max_abs_diff": per_column_max_abs_diff,
        "per_column_pass": per_column_pass,
        "pass": (
            columns_match
            and row_count_matches
            and row_order_matches
            and not missing_keys
            and all(per_column_pass.values())
        ),
    }
