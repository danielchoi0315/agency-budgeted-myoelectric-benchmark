# Benchmark Validation Summary

## Submission Status
- Status: not submission ready

## Critical Checks
- FAIL [critical] cemhsey_failed_trial_excluded: Matched exclusions: ['<none>']
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
_No data_

## Benchmark Manifests
| dataset_id   | user_model   | assist_model   | config_user_model   | config_assist_model   | matches_config   | sequence_roles   |
|:-------------|:-------------|:---------------|:--------------------|:----------------------|:-----------------|:-----------------|
| db10         | torch_mlp    | torch_mlp      | torch_mlp           | torch_mlp             | True             | assist|user      |
| hyser        | torch_mlp    | extra_trees    | torch_mlp           | extra_trees           | True             |                  |
| cemhsey      | extra_trees  | extra_trees    | extra_trees         | extra_trees           | True             |                  |
| grabmyo      | extra_trees  | torch_mlp      | extra_trees         | torch_mlp             | True             |                  |

## Benchmark Coverage
_No data_

## Benchmark Observations
_No data_

## Interpretation
- The workflow now regenerates exclusion-aware prepared data, paired stats, and a manuscript-facing report from declared targets.
- DB10 currently uses a declared annotated object-context proxy in the assistive feature block, so the result is an offline context-assisted benchmark rather than an end-to-end vision claim.
- External robustness direction is positive, but at least one DB10 primary matched-budget gate still fails.
- Until every critical DB10 gate passes, this is not a TNSRE-safe submission package.
