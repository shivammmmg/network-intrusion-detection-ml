# UNSW-NB15 EDA summary

- train shape: (175341, 45)
- test shape:  (82332, 45)
- columns (45): id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports, attack_cat, label

## Class balance (label: 0=normal, 1=attack)
- train: normal=56000 (31.9%)  attack=119341 (68.1%)
- test: normal=37000 (44.9%)  attack=45332 (55.1%)

## Column dtypes
- object/text columns: ['proto', 'service', 'state', 'attack_cat']
  (proto/service/state are real categories; anything else showing up here is a column that needs converting back to numeric.)

## Missing values (columns with any NaN)
- none reported as NaN in train

## Duplicates
- within train (ignoring id): 67601 (38.6%)
- within test  (ignoring id): 26387 (32.0%)
- rows in BOTH train and test (feature+label match): 8421

## Categorical features
- proto: 133 levels in train, 131 in test; 2 only-in-train, 0 only-in-test
- service: 13 levels in train, 13 in test; 0 only-in-train, 0 only-in-test
    service levels: ['-', 'dhcp', 'dns', 'ftp', 'ftp-data', 'http', 'irc', 'pop3', 'radius', 'smtp', 'snmp', 'ssh', 'ssl']
- state: 9 levels in train, 7 in test; 4 only-in-train, 2 only-in-test
  (only-in-test levels are why the encoders need handle_unknown.)

## TTL leakage evidence (sttl by class)
- sttl | normal: most common values -> 31:39455, 254:11230, 0:2859, 62:2167
- sttl | attack: most common values -> 254:103513, 62:15514, 0:303, 255:11
- the single rule 'sttl >= 32 => attack' already gets 92.1% training accuracy, so everything gets run with and without the TTL columns and the without-TTL number is the one we report.

