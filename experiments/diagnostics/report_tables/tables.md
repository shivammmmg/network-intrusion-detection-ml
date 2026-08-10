generated_from: experiments/diagnostics/

## T1 — SHAP global importance

| model | rank | feature | mean_abs_shap_value | scale |
| --- | --- | --- | --- | --- |
| random_forest | 1 | ackdat | 0.0552 | probability |
| random_forest | 2 | sbytes | 0.0380 | probability |
| random_forest | 3 | tcprtt | 0.0316 | probability |
| random_forest | 4 | dbytes | 0.0298 | probability |
| random_forest | 5 | dload | 0.0294 | probability |
| random_forest | 6 | rate | 0.0291 | probability |
| random_forest | 7 | sloss | 0.0252 | probability |
| random_forest | 8 | smean | 0.0242 | probability |
| random_forest | 9 | state | 0.0233 | probability |
| random_forest | 10 | ct_dst_sport_ltm | 0.0225 | probability |
| xgboost | 1 | ackdat | 1.0111 | raw |
| xgboost | 2 | dload | 0.9709 | raw |
| xgboost | 3 | ct_dst_sport_ltm | 0.6708 | raw |
| xgboost | 4 | sbytes | 0.5233 | raw |
| xgboost | 5 | ct_srv_dst | 0.3749 | raw |
| xgboost | 6 | smean | 0.3747 | raw |
| xgboost | 7 | service | 0.3316 | raw |
| xgboost | 8 | ct_dst_src_ltm | 0.3107 | raw |
| xgboost | 9 | synack | 0.2957 | raw |
| xgboost | 10 | dbytes | 0.2735 | raw |

**Rankings are comparable across the two models; magnitudes are not.**

## T2 — Permutation importance

| model | rank | feature | importance_mean | importance_std |
| --- | --- | --- | --- | --- |
| logistic_regression | 1 | ct_dst_sport_ltm | 0.1256 | 0.0010 |
| logistic_regression | 2 | state_con | 0.0934 | 0.0005 |
| logistic_regression | 3 | dpkts | 0.0632 | 0.0005 |
| logistic_regression | 4 | proto_unas | 0.0612 | 0.0010 |
| logistic_regression | 5 | swin | 0.0504 | 0.0005 |
| logistic_regression | 6 | proto_infrequent_sklearn | 0.0423 | 0.0007 |
| logistic_regression | 7 | dbytes | 0.0416 | 0.0008 |
| logistic_regression | 8 | sloss | 0.0409 | 0.0013 |
| logistic_regression | 9 | spkts | 0.0369 | 0.0004 |
| logistic_regression | 10 | proto_udp | 0.0366 | 0.0003 |
| neural_network | 1 | swin | 0.2063 | 0.0005 |
| neural_network | 2 | sload | 0.1437 | 0.0007 |
| neural_network | 3 | ct_dst_sport_ltm | 0.1341 | 0.0017 |
| neural_network | 4 | dmean | 0.0778 | 0.0012 |
| neural_network | 5 | rate | 0.0752 | 0.0007 |
| neural_network | 6 | dwin | 0.0677 | 0.0002 |
| neural_network | 7 | ct_dst_src_ltm | 0.0581 | 0.0014 |
| neural_network | 8 | dbytes | 0.0507 | 0.0015 |
| neural_network | 9 | djit | 0.0321 | 0.0003 |
| neural_network | 10 | dload | 0.0319 | 0.0002 |
| random_forest | 1 | ackdat | 0.0378 | 0.0002 |
| random_forest | 2 | sbytes | 0.0271 | 0.0003 |
| random_forest | 3 | state | 0.0259 | 0.0002 |
| random_forest | 4 | ct_dst_sport_ltm | 0.0199 | 0.0001 |
| random_forest | 5 | dbytes | 0.0174 | 0.0002 |
| random_forest | 6 | tcprtt | 0.0161 | 0.0002 |
| random_forest | 7 | swin | 0.0144 | 0.0002 |
| random_forest | 8 | service | 0.0143 | 0.0001 |
| random_forest | 9 | sloss | 0.0134 | 0.0002 |
| random_forest | 10 | smean | 0.0132 | 0.0002 |
| xgboost | 1 | ackdat | 0.0219 | 0.0001 |
| xgboost | 2 | dload | 0.0139 | 0.0002 |
| xgboost | 3 | ct_dst_sport_ltm | 0.0062 | 0.0001 |
| xgboost | 4 | ct_dst_src_ltm | 0.0055 | 0.0000 |
| xgboost | 5 | smean | 0.0048 | 0.0001 |
| xgboost | 6 | ct_srv_src | 0.0047 | 0.0001 |
| xgboost | 7 | dbytes | 0.0040 | 0.0001 |
| xgboost | 8 | sbytes | 0.0040 | 0.0000 |
| xgboost | 9 | service | 0.0029 | 0.0001 |
| xgboost | 10 | ct_srv_dst | 0.0024 | 0.0001 |

## T3 — Error summary

| model | TP | TN | FP | FN | false_positive_rate | false_negative_rate |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 43213 | 22293 | 14707 | 2119 | 0.3975 | 0.0467 |
| neural_network | 43936 | 23567 | 13433 | 1396 | 0.3631 | 0.0308 |
| random_forest | 44646 | 26747 | 10253 | 686 | 0.2771 | 0.0151 |
| xgboost | 44449 | 26753 | 10247 | 883 | 0.2769 | 0.0195 |

## T4 — Calibration summary

| model | brier_score | expected_calibration_error | worst_bin | worst_bin_mean_predicted | worst_bin_gap | worst_bin_count |
| --- | --- | --- | --- | --- | --- | --- |
| random_forest | 0.0843 | 0.0872 | 2 | 0.5337 | -0.3260 | 8233 |
| xgboost | 0.0888 | 0.0942 | 5 | 0.8586 | -0.4675 | 8233 |
| neural_network | 0.0976 | 0.0845 | 5 | 0.7474 | -0.3381 | 8233 |
| logistic_regression | 0.1406 | 0.0929 | 6 | 0.8444 | -0.3151 | 8234 |

## T5 — Drift summary

| section | rank | feature | psi | ks_statistic | binning_strategy | psi_low_resolution |
| --- | --- | --- | --- | --- | --- | --- |
| top_psi | 1 | ct_dst_sport_ltm | 0.8332 | 0.2811 | value | false |
| top_psi | 2 | dmean | 0.5394 | 0.3188 | quantile | false |
| top_psi | 3 | dpkts | 0.5233 | 0.3070 | quantile | false |
| top_psi | 4 | state | 0.5102 | 0.3064 | value | false |
| top_psi | 5 | dbytes | 0.5031 | 0.3200 | quantile | false |
| top_psi | 6 | dload | 0.4497 | 0.3151 | quantile | false |
| top_psi | 7 | dinpkt | 0.4362 | 0.3070 | quantile | false |
| top_psi | 8 | rate | 0.4029 | 0.2884 | quantile | false |
| top_psi | 9 | sbytes | 0.3992 | 0.2701 | quantile | false |
| top_psi | 10 | dur | 0.3854 | 0.2974 | quantile | false |
| top_ks | 1 | dbytes | 0.5031 | 0.3200 | quantile | false |
| top_ks | 2 | dmean | 0.5394 | 0.3188 | quantile | false |
| top_ks | 3 | dload | 0.4497 | 0.3151 | quantile | false |
| top_ks | 4 | dpkts | 0.5233 | 0.3070 | quantile | false |
| top_ks | 5 | dinpkt | 0.4362 | 0.3070 | quantile | false |
| top_ks | 6 | state | 0.5102 | 0.3064 | value | false |
| top_ks | 7 | dur | 0.3854 | 0.2974 | quantile | false |
| top_ks | 8 | rate | 0.4029 | 0.2884 | quantile | false |
| top_ks | 9 | sload | 0.3615 | 0.2867 | quantile | false |
| top_ks | 10 | djit | 0.2559 | 0.2856 | quantile | false |

Drift scalars: 39 total features; 29 (74.36%) with PSI > 0.2; 0 PSI-degenerate; 6 low-resolution.

## T6 — Importance × drift overlap

| model | feature | permutation_rank | drift_rank_by_psi | drift_rank_by_ks | psi | ks_statistic | psi_degenerate | psi_low_resolution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | ct_dst_sport_ltm | 1 | 1 | 11 | 0.8332 | 0.2811 | false | false |
| logistic_regression | dpkts | 3 | 3 | 4 | 0.5233 | 0.3070 | false | false |
| logistic_regression | dbytes | 7 | 5 | 1 | 0.5031 | 0.3200 | false | false |
| neural_network | sload | 2 | 12 | 9 | 0.3615 | 0.2867 | false | false |
| neural_network | ct_dst_sport_ltm | 3 | 1 | 11 | 0.8332 | 0.2811 | false | false |
| neural_network | dmean | 4 | 2 | 2 | 0.5394 | 0.3188 | false | false |
| neural_network | rate | 5 | 8 | 8 | 0.4029 | 0.2884 | false | false |
| neural_network | dbytes | 8 | 5 | 1 | 0.5031 | 0.3200 | false | false |
| neural_network | djit | 9 | 20 | 10 | 0.2559 | 0.2856 | false | false |
| neural_network | dload | 10 | 6 | 3 | 0.4497 | 0.3151 | false | false |
| random_forest | sbytes | 2 | 9 | 12 | 0.3992 | 0.2701 | false | false |
| random_forest | state | 3 | 4 | 6 | 0.5102 | 0.3064 | false | false |
| random_forest | ct_dst_sport_ltm | 4 | 1 | 11 | 0.8332 | 0.2811 | false | false |
| random_forest | dbytes | 5 | 5 | 1 | 0.5031 | 0.3200 | false | false |
| xgboost | dload | 2 | 6 | 3 | 0.4497 | 0.3151 | false | false |
| xgboost | ct_dst_sport_ltm | 3 | 1 | 11 | 0.8332 | 0.2811 | false | false |
| xgboost | dbytes | 7 | 5 | 1 | 0.5031 | 0.3200 | false | false |
| xgboost | sbytes | 8 | 9 | 12 | 0.3992 | 0.2701 | false | false |

Overlapping features per model: logistic_regression: 3, neural_network: 7, random_forest: 4, xgboost: 4.

## T7 — TTL comparison

| metric | no_ttl_frozen | with_ttl_refit | delta |
| --- | --- | --- | --- |
| pr_auc | 0.9859 | 0.9870 | +0.0011 |
| roc_auc | 0.9809 | 0.9824 | +0.0015 |
| f1_locked_threshold | 0.8887 | 0.8910 | +0.0022 |
| precision_locked_threshold | 0.8127 | 0.8146 | +0.0020 |
| recall_locked_threshold | 0.9805 | 0.9831 | +0.0026 |
| accuracy_locked_threshold | 0.8648 | 0.8675 | +0.0027 |

locked_threshold: 0.46306103
threshold_note: Validation-selected no-TTL XGBoost threshold, applied unchanged to both arms for like-for-like secondary operating-point metrics.

## T8 — TTL rank shift

| feature | no_ttl_rank | no_ttl_importance_mean | with_ttl_rank | with_ttl_importance_mean |
| --- | --- | --- | --- | --- |
| ackdat | 1 | 0.0219 | 33 | 0.0000 |
| dload | 2 | 0.0139 | 13 | 0.0010 |
| ct_dst_sport_ltm | 3 | 0.0062 | 2 | 0.0076 |
| ct_dst_src_ltm | 4 | 0.0055 | 3 | 0.0057 |
| smean | 5 | 0.0048 | 4 | 0.0055 |
| ct_srv_src | 6 | 0.0047 | 5 | 0.0041 |
| dbytes | 7 | 0.0040 | 7 | 0.0030 |
| sbytes | 8 | 0.0040 | 6 | 0.0033 |
| service | 9 | 0.0029 | 10 | 0.0024 |
| ct_srv_dst | 10 | 0.0024 | 9 | 0.0024 |

TTL shortcut-feature evidence (with_ttl_refit):

| feature | rank | importance_mean |
| --- | --- | --- |
| sttl | 1 | 0.0817 |
| ct_state_ttl | 11 | 0.0019 |
| dttl | 28 | 0.0002 |

## T9 — Trust-question evidence map

| question | headline_evidence | source_file |
| --- | --- | --- |
| Why is the model making these predictions? | Top SHAP: random_forest ackdat=0.0552 (probability); xgboost ackdat=1.0111 (raw); top permutation: logistic_regression ct_dst_sport_ltm=0.1256; neural_network swin=0.2063; random_forest ackdat=0.0378; xgboost ackdat=0.0219. | shap/shap_global_importance_{random_forest,xgboost}.csv; importance/permutation_importance_{models}.csv |
| Can its probabilities be trusted? | Best Brier: random_forest=0.0843; ECE=0.0872; worst-bin gap=-0.3260 (bin 2). | calibration/calibration_summary.json |
| Is it relying on unstable or dataset-specific features? | PSI>0.2: 29/39 (74.36%); overlap counts: logistic_regression=3, neural_network=7, random_forest=4, xgboost=4; with-TTL: sttl rank 1 (0.0817), ct_state_ttl rank 11 (0.0019), dttl rank 28 (0.0002). | drift/drift_psi_ks.csv; drift/drift_importance_overlap.csv; ttl_ablation/ttl_rank_shift.csv; ttl_ablation/ttl_permutation_importance.csv |
| Why does it fail on certain samples? | Lowest FPR: xgboost=0.2769; lowest FNR: random_forest=0.0151; 8 selected local-SHAP cases. | errors/error_summary.json; shap/representative_cases.csv |

