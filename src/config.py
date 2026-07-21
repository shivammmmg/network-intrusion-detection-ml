"""Shared config: paths, the random seed, and which columns are what.

Keeping the column roles in one place means the EDA, cleaning, and pipeline
scripts can't disagree about them.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_TRAIN_CSV = RAW_DIR / "UNSW_NB15_training-set.csv"
RAW_TEST_CSV = RAW_DIR / "UNSW_NB15_testing-set.csv"

# One seed used everywhere so runs are repeatable.
RANDOM_STATE = 42
VAL_FRACTION = 0.20  # validation slice taken out of the (deduped) train set

TARGET = "label"  # 0 = normal, 1 = attack

# Dropped for the binary task:
#   id         is just a row number.
#   attack_cat is the attack family name, and it gives the label away
#              (attack_cat == "Normal" is the same as label 0). Only useful as
#              the target if we later do the multiclass version.
DROP_COLS = ["id", "attack_cat"]

# The three text columns.
CATEGORICAL = ["proto", "service", "state"]

# sttl/dttl/ct_state_ttl leak the label. UNSW-NB15's normal and attack traffic
# were generated on different machines, so the TTL values basically encode which
# machine (so which class) a flow came from: most normal flows have sttl 31,
# 87% of attacks have 254, and sttl on its own nearly separates the two (not
# perfectly, ~20% of normal flows also show 254). That's an artifact of how
# the dataset was built, not a real signal, so a model that leans on it looks
# great here and falls apart on real traffic. We keep these columns but let the
# builders switch them off, and the switched-off (no-TTL) run is the number we
# actually report. The with-TTL run is only there to show how much it inflates.
TTL_LEAK_COLS = ["sttl", "dttl", "ct_state_ttl"]

# NaN in these means the flow wasn't HTTP/FTP, so 0 is the right fill (the mean
# would invent FTP/HTTP activity that never happened).
FILL_ZERO_COLS = ["ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"]

# The 0/1 flags (is_ftp_login, is_sm_ips_ports) just ride through the numeric
# path and get scaled. Scaling a 0/1 column does nothing useful but doesn't hurt.


def numeric_columns(df, include_ttl: bool = True):
    """Numeric predictor columns present in df.

    Numeric means everything that isn't the target, isn't dropped, and isn't one
    of the three text columns. Built from the actual frame so a stray column
    can't sneak through. include_ttl=False also drops the TTL leak columns.
    """
    exclude = set(DROP_COLS) | {TARGET} | set(CATEGORICAL)
    if not include_ttl:
        exclude |= set(TTL_LEAK_COLS)
    return [c for c in df.columns if c not in exclude]
