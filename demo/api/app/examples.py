"""Read deterministic predictor-only examples and separate evaluation metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "golden_examples.json"


def load_examples() -> list[dict[str, Any]]:
    with FIXTURES_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["examples"]
