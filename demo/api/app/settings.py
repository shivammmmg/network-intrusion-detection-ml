"""Immutable demo contract constants; no model state lives here."""

from __future__ import annotations

FROZEN_SOURCE_SHA = "4b8165aaaaaf00d4ed9a551e5f07ea38c7cb0072"
INPUT_CONTRACT = "unsw_nb15_primary_v1"

CATEGORICAL_FIELDS = ("proto", "service", "state")
FLOAT_FIELDS = (
    "dur",
    "rate",
    "sload",
    "dload",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "tcprtt",
    "synack",
    "ackdat",
)
INTEGER_FIELDS = (
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sloss",
    "dloss",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
)
BINARY_FIELDS = ("is_ftp_login", "is_sm_ips_ports")

CANONICAL_FIELDS = (
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
)

EXCLUDED_FIELDS = ("id", "label", "attack_cat", "sttl", "dttl", "ct_state_ttl")

# A verification snapshot only: inference always reads selected_thresholds.json.
EXPECTED_THRESHOLDS = {
    "logistic_regression": 0.4631204833813834,
    "neural_network": 0.3454528735910796,
    "random_forest": 0.492968259796183,
    "xgboost": 0.46306103,
}
