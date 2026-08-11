from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from demo.api.app.schemas import NetworkRecord
from demo.api.app.settings import EXCLUDED_FIELDS


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "golden_examples.json"


def valid_record() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["examples"][0]["record"]


def test_valid_record_accepts_all_39_active_fields() -> None:
    record = NetworkRecord.model_validate(valid_record())
    assert len(record.model_dump()) == 39


@pytest.mark.parametrize("field", ["dur", "proto", "ct_srv_dst"])
def test_missing_required_feature_is_rejected(field: str) -> None:
    payload = valid_record()
    payload.pop(field)
    with pytest.raises(ValidationError):
        NetworkRecord.model_validate(payload)


@pytest.mark.parametrize("field", EXCLUDED_FIELDS)
def test_excluded_or_extra_feature_is_rejected(field: str) -> None:
    payload = valid_record()
    payload[field] = 0
    with pytest.raises(ValidationError):
        NetworkRecord.model_validate(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value: float) -> None:
    payload = valid_record()
    payload["dur"] = value
    with pytest.raises(ValidationError):
        NetworkRecord.model_validate(payload)


def test_negative_counts_and_non_binary_flags_are_rejected() -> None:
    negative = valid_record()
    negative["spkts"] = -1
    with pytest.raises(ValidationError):
        NetworkRecord.model_validate(negative)

    flag = valid_record()
    flag["is_sm_ips_ports"] = 2
    with pytest.raises(ValidationError):
        NetworkRecord.model_validate(flag)


def test_categorical_values_normalize_before_registry_validation() -> None:
    payload = valid_record()
    payload["proto"] = " TCP "
    payload["service"] = " POP3 "
    payload["state"] = " FIN "
    record = NetworkRecord.model_validate(payload)
    assert (record.proto, record.service, record.state) == ("tcp", "pop3", "fin")
