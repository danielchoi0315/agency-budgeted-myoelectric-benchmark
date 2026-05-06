# J2 Validation Summary

## Submission Status
- Status: provisionally ready

## Critical Checks
- PASS [critical] cemhsey_failed_trial_excluded: Matched exclusions: ['S4_Day1_Session1_Task1_Trial1.mat']
- PASS [critical] db10_multimodal_direction: assist_only macro-F1 exceeds user_only on every DB10 split family
- PASS [critical] db10_absolute_performance_floor: Best DB10 agency macro-F1 observed: 0.5722
- PASS [critical] db10_active_grasp_floor: Best DB10 agency active-grasp macro-F1 observed: 0.6938
- PASS [supporting] external_agency_margin_direction: best agency-margin policy improves macro-F1 over user_only on executed external robustness datasets
- PASS [supporting] canonical_model_manifest_alignment: Benchmark manifests match config model targets: [{'dataset_id': 'db10', 'user_model': 'torch_mlp', 'assist_model': 'torch_mlp', 'matches_config': True}, {'dataset_id': 'hyser', 'user_model': 'torch_mlp', 'assist_model': 'extra_trees', 'matches_config': True}, {'dataset_id': 'cemhsey', 'user_model': 'extra_trees', 'assist_model': 'extra_trees', 'matches_config': True}, {'dataset_id': 'grabmyo', 'user_model': 'extra_trees', 'assist_model': 'torch_mlp', 'matches_config': True}]
- PASS [critical] db10_primary_pairwise_significance: DB10 taus with beneficial Holm-significant active-F1 and active-risk results across all DB10 split families: ['0.02', '0.05', '0.10']

## Dataset Audits
| dataset_id   | status   |   file_count |   total_gb |
|:-------------|:---------|-------------:|-----------:|
| db10         | PASS     |          244 |     223.6  |
| hyser        | PASS     |        57501 |     142.76 |
| cemhsey      | PASS     |           20 |     307.78 |
| grabmyo      | PASS     |        30705 |       9.42 |

## Prepared Dataset Summary
| dataset_id   |   rows |   subjects |   sessions |   days |   labels |   user_dim |   assist_dim | assist_context_modes   |
|:-------------|-------:|-----------:|-----------:|-------:|---------:|-----------:|-------------:|:-----------------------|
| db10         | 189000 |         45 |          1 |      0 |       11 |         96 |          190 | annotated_object_proxy |
| hyser        |  40340 |         20 |          2 |      2 |       34 |       1280 |         2048 |                        |
| cemhsey      |  24020 |         13 |          1 |     11 |        7 |       1600 |         2560 |                        |
| grabmyo      |  76755 |         43 |          3 |      3 |       17 |        160 |          256 |                        |

## Benchmark Manifests
| dataset_id   | user_model   | assist_model   | config_user_model   | config_assist_model   | matches_config   | sequence_roles   |
|:-------------|:-------------|:---------------|:--------------------|:----------------------|:-----------------|:-----------------|
| db10         | torch_mlp    | torch_mlp      | torch_mlp           | torch_mlp             | True             | assist|user      |
| hyser        | torch_mlp    | extra_trees    | torch_mlp           | extra_trees           | True             |                  |
| cemhsey      | extra_trees  | extra_trees    | extra_trees         | extra_trees           | True             |                  |
| grabmyo      | extra_trees  | torch_mlp      | extra_trees         | torch_mlp             | True             |                  |

## Benchmark Coverage
| dataset_id   | split_family                    |   n_splits |   n_policies |
|:-------------|:--------------------------------|-----------:|-------------:|
| db10         | db10_able_to_amputee            |         15 |           16 |
| db10         | db10_amputee_loso               |         15 |           16 |
| db10         | db10_mixed_to_amputee           |         15 |           16 |
| hyser        | hyser_subject_logo              |         20 |           16 |
| hyser        | hyser_within_subject_dayshift   |          1 |           16 |
| cemhsey      | cemhsey_forward_day_logo        |         10 |           16 |
| grabmyo      | grabmyo_subject_logo            |         43 |           16 |
| grabmyo      | grabmyo_within_subject_dayshift |          2 |           16 |

## Benchmark Observations
| dataset_id   | split_family                    | best_overall_policy         |   best_overall_macro_f1 |   best_overall_active_macro_f1 |   best_overall_active_risk_coverage_auc |   best_overall_mean_ali | best_agency_policy     |   best_agency_macro_f1 |   best_agency_active_macro_f1 |   best_agency_active_risk_coverage_auc |   best_agency_mean_ali |   user_only_macro_f1 |   user_only_active_macro_f1 |   user_only_active_risk_coverage_auc |   user_only_risk_coverage_auc |   assist_only_macro_f1 |
|:-------------|:--------------------------------|:----------------------------|------------------------:|-------------------------------:|----------------------------------------:|------------------------:|:-----------------------|-----------------------:|------------------------------:|---------------------------------------:|-----------------------:|---------------------:|----------------------------:|-------------------------------------:|------------------------------:|-----------------------:|
| db10         | db10_able_to_amputee            | confidence_blend_alpha_0.75 |                0.556824 |                       0.690195 |                                0.222972 |               0.0947213 | agency_margin_tau_0.20 |               0.47651  |                      0.591193 |                               0.251611 |              0.0445796 |            0.102835  |                   0.0814183 |                             0.880204 |                      0.815534 |               0.556606 |
| db10         | db10_amputee_loso               | assist_only                 |                0.570061 |                       0.680558 |                                0.248024 |               0.0573554 | agency_margin_tau_0.20 |               0.552622 |                      0.657734 |                               0.25405  |              0.0525128 |            0.0950532 |                   0.0806035 |                             0.889884 |                      0.839451 |               0.570061 |
| db10         | db10_mixed_to_amputee           | confidence_blend_alpha_0.50 |                0.585322 |                       0.710159 |                                0.186447 |               0.049727  | agency_margin_tau_0.20 |               0.572159 |                      0.693784 |                               0.189462 |              0.0483577 |            0.112882  |                   0.0880473 |                             0.879156 |                      0.80549  |               0.585298 |
| hyser        | hyser_subject_logo              | confidence_blend_alpha_0.50 |                0.357913 |                       0.357913 |                                0.425108 |               0.0272995 | agency_margin_tau_0.20 |               0.34226  |                      0.34226  |                               0.450213 |              0.0123824 |            0.325055  |                   0.325055  |                             0.467254 |                      0.467254 |               0.287886 |
| hyser        | hyser_within_subject_dayshift   | confidence_blend_alpha_0.50 |                0.446087 |                       0.446087 |                                0.347029 |               0.0188878 | agency_margin_tau_0.20 |               0.432038 |                      0.432038 |                               0.366463 |              0.013429  |            0.385725  |                   0.385725  |                             0.404726 |                      0.404726 |               0.392863 |
| cemhsey      | cemhsey_forward_day_logo        | confidence_blend_alpha_0.75 |                0.670094 |                       0.670094 |                                0.222969 |               0.0346505 | agency_margin_tau_0.20 |               0.650254 |                      0.650254 |                               0.230487 |              0.0148142 |            0.57756   |                   0.57756   |                             0.258028 |                      0.258028 |               0.643597 |
| grabmyo      | grabmyo_subject_logo            | confidence_blend_alpha_0.50 |                0.656336 |                       0.656336 |                                0.145915 |               0.0442494 | agency_margin_tau_0.20 |               0.617028 |                      0.617028 |                               0.164167 |              0.0135661 |            0.558837  |                   0.558837  |                             0.234997 |                      0.234997 |               0.646403 |
| grabmyo      | grabmyo_within_subject_dayshift | confidence_blend_alpha_0.50 |                0.712698 |                       0.712698 |                                0.111723 |               0.0341144 | agency_margin_tau_0.20 |               0.680972 |                      0.680972 |                               0.130947 |              0.012844  |            0.626559  |                   0.626559  |                             0.190959 |                      0.190959 |               0.69286  |

## CEMHSEY Exclusions
- Configured failed trials: S4_Day1_Session1_Task1_Trial1.mat
- Matched failed trials: S4_Day1_Session1_Task1_Trial1.mat
- Excluded member paths: 1

## Interpretation
- The workflow now regenerates exclusion-aware prepared data, paired stats, and a manuscript-facing report from declared targets.
- DB10 currently uses a declared annotated object-context proxy in the assistive feature block, so the result is an offline context-assisted benchmark rather than an end-to-end vision claim.
- The current blocking infrastructure issues are cleared; the remaining risk is manuscript-level framing and reviewer-facing justification, not benchmark failure.

