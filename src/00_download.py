"""Fetch the raw UNSW-NB15 partitioned CSVs and check they're intact.

    .venv/bin/python src/00_download.py [--force]

Skips files that already look right; --force re-downloads. Uses only the standard
library so it runs before the pip install. Source is a public GitHub mirror of the
standard partitioned split (github.com/Nir-J/ML-Projects); the original is UNSW
Canberra (Moustafa & Slay, 2015/2016).
"""

import shutil
import subprocess
import sys
from pathlib import Path

from config import RAW_DIR, RAW_TRAIN_CSV, RAW_TEST_CSV

BASE = ("https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
        "UNSW-Network_Packet_Classification")

# (url, destination, expected data rows without header)
FILES = [
    (f"{BASE}/UNSW_NB15_training-set.csv", RAW_TRAIN_CSV, 175_341),
    (f"{BASE}/UNSW_NB15_testing-set.csv", RAW_TEST_CSV, 82_332),
]
EXPECTED_COLS = 45
HEADER_START = "id,dur,proto,service,state"


def verify(path: Path, expected_rows: int) -> bool:
    """Check the file is the real CSV: right row count, 45 columns, right header.
    utf-8-sig drops the BOM (same as read_raw)."""
    if not path.exists():
        return False
    with path.open(encoding="utf-8-sig") as fh:
        header = fh.readline().rstrip("\n")
        data_rows = sum(1 for _ in fh)
    ok_cols = header.count(",") + 1 == EXPECTED_COLS
    ok_head = header.startswith(HEADER_START) and header.endswith("attack_cat,label")
    ok_rows = data_rows == expected_rows
    if not (ok_cols and ok_head and ok_rows):
        print(f"  ! verify failed for {path.name}: "
              f"rows={data_rows}(want {expected_rows}) cols_ok={ok_cols} header_ok={ok_head}")
    return ok_cols and ok_head and ok_rows


def fetch(url: str, dest: Path):
    """Download with curl, falling back to wget. Not urllib, because urllib kept
    failing TLS certificate verification on my machine and curl/wget use the OS
    certificate store."""
    tmp = dest.with_suffix(".csv.part")
    if shutil.which("curl"):
        subprocess.run(["curl", "-fsSL", url, "-o", str(tmp)], check=True)
    elif shutil.which("wget"):
        subprocess.run(["wget", "-q", url, "-O", str(tmp)], check=True)
    else:
        sys.exit("[FAIL] need curl or wget on PATH to download.")
    tmp.replace(dest)


def main(force: bool):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url, dest, rows in FILES:
        if not force and verify(dest, rows):
            print(f"[skip] {dest.name} already present and valid")
            continue
        print(f"[get ] {dest.name} <- {url}")
        fetch(url, dest)
        if verify(dest, rows):
            print(f"[ok  ] {dest.name}: {rows} rows, {EXPECTED_COLS} cols")
        else:
            sys.exit(f"[FAIL] {dest.name} did not verify, don't use it.")
    print("All raw files present and verified.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
