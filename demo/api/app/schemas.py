"""Strict public request and response schemas for the demo API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .settings import INPUT_CONTRACT


class NetworkRecord(BaseModel):
    """The 39 active primary-model fields; all extra fields are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    dur: float = Field(ge=0)
    proto: str
    service: str
    state: str
    spkts: int = Field(ge=0)
    dpkts: int = Field(ge=0)
    sbytes: int = Field(ge=0)
    dbytes: int = Field(ge=0)
    rate: float = Field(ge=0)
    sload: float = Field(ge=0)
    dload: float = Field(ge=0)
    sloss: int = Field(ge=0)
    dloss: int = Field(ge=0)
    sinpkt: float = Field(ge=0)
    dinpkt: float = Field(ge=0)
    sjit: float = Field(ge=0)
    djit: float = Field(ge=0)
    swin: int = Field(ge=0)
    stcpb: int = Field(ge=0)
    dtcpb: int = Field(ge=0)
    dwin: int = Field(ge=0)
    tcprtt: float = Field(ge=0)
    synack: float = Field(ge=0)
    ackdat: float = Field(ge=0)
    smean: int = Field(ge=0)
    dmean: int = Field(ge=0)
    trans_depth: int = Field(ge=0)
    response_body_len: int = Field(ge=0)
    ct_srv_src: int = Field(ge=0)
    ct_dst_ltm: int = Field(ge=0)
    ct_src_dport_ltm: int = Field(ge=0)
    ct_dst_sport_ltm: int = Field(ge=0)
    ct_dst_src_ltm: int = Field(ge=0)
    is_ftp_login: int = Field(ge=0, le=1)
    ct_ftp_cmd: int = Field(ge=0)
    ct_flw_http_mthd: int = Field(ge=0)
    ct_src_ltm: int = Field(ge=0)
    ct_srv_dst: int = Field(ge=0)
    is_sm_ips_ports: int = Field(ge=0, le=1)

    @field_validator("proto", "service", "state")
    @classmethod
    def normalize_categorical(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("categorical values must not be empty")
        return normalized


class PredictionResponse(BaseModel):
    model: str
    attack_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    prediction: Literal[0, 1]
    label: Literal["normal", "attack"]
    threshold_source: Literal["validation_selected_locked"]
    input_contract: Literal[INPUT_CONTRACT]


class PredictAllResponse(BaseModel):
    input_contract: Literal[INPUT_CONTRACT]
    results: list[PredictionResponse]


class SchemaResponse(BaseModel):
    input_contract: Literal[INPUT_CONTRACT]
    fields: list[dict[str, Any]]
    categorical_categories: dict[str, list[str]]
    excluded_fields: list[str]


class HealthResponse(BaseModel):
    ready: bool
    frozen_source_sha: str
    models: dict[str, bool]


class ModelsResponse(BaseModel):
    input_contract: Literal[INPUT_CONTRACT]
    models: list[dict[str, Any]]
