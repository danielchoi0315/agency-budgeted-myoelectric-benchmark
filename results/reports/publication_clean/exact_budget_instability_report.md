# Exact-Budget Instability Report

## Bottom Line
- Status: fixed_tau_unstable_under_exact_budget
- The exact-budget audit holds plain-confidence to a dense global threshold bank and reports residual budget gaps explicitly.
- If fixed taus still fail here, the instability story is scientific rather than a comparator-support artifact.

## Dataset Overview
| dataset_id   |   required_split_family_count |   n_universal_exact_budget_taus | fixed_tau_supported   |   n_unique_best_taus |   unique_best_taus |   worst_confidence_gap |   lowest_confidence_exact_match_fraction |   worst_safety_harmful_sig_metric_count | instability_story   |
|:-------------|------------------------------:|--------------------------------:|:----------------------|---------------------:|-------------------:|-----------------------:|-----------------------------------------:|----------------------------------------:|:--------------------|
| db10         |                             3 |                               0 | False                 |                    1 |                0.2 |            0.00447619  |                                 0        |                                       2 | fixed_tau_unstable  |
| hyser        |                             2 |                               0 | False                 |                    1 |                0.2 |            0.000968616 |                                 0.3      |                                       0 | fixed_tau_unstable  |
| cemhsey      |                             1 |                               0 | False                 |                    1 |                0.2 |            0.00155678  |                                 0.746154 |                                       1 | fixed_tau_unstable  |

## Required-Scope Tau Summary
| dataset_id   | split_family                  |   tau | confidence_all_primary_metrics_sig_beneficial   |   safety_harmful_sig_metric_count |   confidence_mean_abs_intervention_rate_gap |   confidence_exact_match_fraction |
|:-------------|:------------------------------|------:|:------------------------------------------------|----------------------------------:|--------------------------------------------:|----------------------------------:|
| db10         | db10_able_to_amputee          |  0.02 | False                                           |                                 1 |                                 0.00196825  |                         0.0666667 |
| db10         | db10_able_to_amputee          |  0.05 | False                                           |                                 1 |                                 0.00107937  |                         0.133333  |
| db10         | db10_able_to_amputee          |  0.1  | False                                           |                                 1 |                                 0.000984127 |                         0.133333  |
| db10         | db10_able_to_amputee          |  0.15 | False                                           |                                 2 |                                 0.00101587  |                         0.133333  |
| db10         | db10_able_to_amputee          |  0.2  | False                                           |                                 2 |                                 0.00115873  |                         0.0666667 |
| db10         | db10_amputee_loso             |  0.02 | False                                           |                                 1 |                                 0.00138095  |                         0.133333  |
| db10         | db10_amputee_loso             |  0.05 | False                                           |                                 1 |                                 0.00136508  |                         0         |
| db10         | db10_amputee_loso             |  0.1  | False                                           |                                 2 |                                 0.00314286  |                         0         |
| db10         | db10_amputee_loso             |  0.15 | False                                           |                                 2 |                                 0.001       |                         0.2       |
| db10         | db10_amputee_loso             |  0.2  | False                                           |                                 1 |                                 0.00125397  |                         0.266667  |
| db10         | db10_mixed_to_amputee         |  0.02 | False                                           |                                 1 |                                 0.00125397  |                         0.133333  |
| db10         | db10_mixed_to_amputee         |  0.05 | False                                           |                                 1 |                                 0.00150794  |                         0.0666667 |
| db10         | db10_mixed_to_amputee         |  0.1  | False                                           |                                 2 |                                 0.00447619  |                         0         |
| db10         | db10_mixed_to_amputee         |  0.15 | False                                           |                                 1 |                                 0.00277778  |                         0.133333  |
| db10         | db10_mixed_to_amputee         |  0.2  | False                                           |                                 2 |                                 0.00173016  |                         0.266667  |
| hyser        | hyser_subject_logo            |  0.02 | False                                           |                                 0 |                                 0.000644802 |                         0.475     |
| hyser        | hyser_subject_logo            |  0.05 | False                                           |                                 0 |                                 0.000820225 |                         0.4       |
| hyser        | hyser_subject_logo            |  0.1  | False                                           |                                 0 |                                 0.000869734 |                         0.4       |
| hyser        | hyser_subject_logo            |  0.15 | False                                           |                                 0 |                                 0.000968616 |                         0.3       |
| hyser        | hyser_subject_logo            |  0.2  | False                                           |                                 0 |                                 0.000964857 |                         0.35      |
| hyser        | hyser_within_subject_dayshift |  0.02 | False                                           |                                 0 |                                 0.0007417   |                         0.35      |
| hyser        | hyser_within_subject_dayshift |  0.05 | False                                           |                                 0 |                                 0.000892237 |                         0.35      |
| hyser        | hyser_within_subject_dayshift |  0.1  | False                                           |                                 0 |                                 0.000691916 |                         0.45      |
| hyser        | hyser_within_subject_dayshift |  0.15 | False                                           |                                 0 |                                 0.000893947 |                         0.3       |
| hyser        | hyser_within_subject_dayshift |  0.2  | True                                            |                                 0 |                                 0.000891404 |                         0.35      |
| cemhsey      | cemhsey_forward_day_logo      |  0.02 | False                                           |                                 1 |                                 0.000595238 |                         0.9       |
| cemhsey      | cemhsey_forward_day_logo      |  0.05 | False                                           |                                 1 |                                 0.00114469  |                         0.815385  |
| cemhsey      | cemhsey_forward_day_logo      |  0.1  | False                                           |                                 1 |                                 0.00100733  |                         0.830769  |
| cemhsey      | cemhsey_forward_day_logo      |  0.15 | False                                           |                                 0 |                                 0.00155678  |                         0.746154  |
| cemhsey      | cemhsey_forward_day_logo      |  0.2  | True                                            |                                 0 |                                 0.00141941  |                         0.761538  |

## Interpretation
- `fixed_tau_supported = True` means at least one tau clears every required split family on both primary metrics with no harmful significant safety row under the exact-budget audit.
- `n_unique_best_taus > 1` or non-empty `metric_disagreement_families` indicates operating-point instability even when a universal tau exists somewhere else.
- Small residual budget gaps with no universal tau mean the fixed-policy story is failing on behavior, not on comparator mismatch.
