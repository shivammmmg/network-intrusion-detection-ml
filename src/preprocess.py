"""Cleaning functions and the two preprocessing pipelines.

Two separate stages:

clean_df does constant fixes (fill a NaN with 0, clamp a bad value, lowercase a
string). None of that is learned from the data, so running it on train, val and
test the same way is fine.

The ColumnTransformers below do learn things from the data (imputation medians,
the scaler's spread, the one-hot vocabulary), so they get fit on train only and
applied to val/test, otherwise information from val/test leaks into training.
The learned values are stored inside the fitted object.

Two pipeline variants because the models want different inputs. LogReg and the
neural net want scaled numbers and one-hot categories, but the tree models split
on thresholds, so scaling does nothing for them and one-hotting a 130-level
column just fragments the trees. They get ordinal codes instead.
"""

import warnings

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.exceptions import InconsistentVersionWarning

from config import TARGET, DROP_COLS, CATEGORICAL, FILL_ZERO_COLS, numeric_columns


def read_raw(path) -> pd.DataFrame:
    """Read a raw CSV. utf-8-sig drops the byte-order-mark on these files;
    without it the first column reads as '﻿id' instead of 'id'."""
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known UNSW-NB15 quirks. Nothing here is learned from the data."""
    df = df.copy()

    # ct_ftp_cmd sometimes arrives as a whitespace string instead of a number,
    # which turns the whole column into text. Coerce it; junk becomes NaN and
    # gets filled with 0 next.
    if "ct_ftp_cmd" in df.columns:
        df["ct_ftp_cmd"] = pd.to_numeric(df["ct_ftp_cmd"], errors="coerce")

    # NaN in these three means "not HTTP/FTP", so fill with 0.
    for col in FILL_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # is_ftp_login should be 0/1 but has stray 2s and 4s. Clamp down to 1.
    if "is_ftp_login" in df.columns:
        df["is_ftp_login"] = df["is_ftp_login"].clip(upper=1).astype(int)

    # Lowercase and strip the text columns so "FIN" and "fin " don't count as two
    # categories. .where(notna) puts real NaNs back, since astype(str) would turn
    # a missing value into the string "nan" (a fake category the imputer can't
    # see). No NaN categoricals in this file, but another release could have them.
    for col in CATEGORICAL:
        if col in df.columns:
            notna = df[col].notna()
            df[col] = df[col].astype(str).str.strip().str.lower().where(notna)

    return df


def make_xy(df: pd.DataFrame):
    """Split a cleaned frame into X and y. Drops id and attack_cat. Keeps the TTL
    columns in X; the with/without-TTL choice happens inside the pipeline so both
    runs read the same X."""
    drop = [c for c in DROP_COLS + [TARGET] if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET].astype(int)
    return X, y


def _numeric_and_categorical(X: pd.DataFrame, include_ttl: bool):
    num = numeric_columns(X, include_ttl=include_ttl)
    cat = [c for c in CATEGORICAL if c in X.columns]
    return num, cat


def build_linear_preprocessor(X: pd.DataFrame, include_ttl: bool = False) -> ColumnTransformer:
    """For LogReg and the neural net: median-impute + RobustScaler on numerics,
    one-hot on the text columns.

    RobustScaler (median and IQR) rather than StandardScaler because features like
    sbytes and sload have a few huge values that would throw off a mean-based
    scaler. The one-hot encoder folds rare levels together, which keeps proto's
    130-value tail from blowing up, and handle_unknown lets a category that shows
    up only in test pass through instead of crashing.
    """
    num, cat = _numeric_and_categorical(X, include_ttl)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=10,
            max_categories=30,
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, num), ("cat", categorical_pipe, cat)],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_tree_preprocessor(X: pd.DataFrame, include_ttl: bool = False) -> ColumnTransformer:
    """For random forest and XGBoost: median-impute numerics (no scaling, trees
    don't care), ordinal-encode the text columns (integer codes, no width blowup).

    Those codes make XGBoost treat the categoricals as plain numbers. That's fine.
    It does not use XGBoost's native categorical mode (that needs a pandas category
    dtype and enable_categorical); whoever owns XGBoost can build that off the raw
    cleaned parquet if they want it.
    """
    num, cat = _numeric_and_categorical(X, include_ttl)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, num), ("cat", categorical_pipe, cat)],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def load_artifact(path):
    """Load a fitted pipeline, but make a scikit-learn version mismatch a hard
    error instead of a warning. A different sklearn can transform data slightly
    wrong, and we'd rather find out loudly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(path)
