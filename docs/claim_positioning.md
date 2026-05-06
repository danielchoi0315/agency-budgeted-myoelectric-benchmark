# Claim Positioning

## Primary Manuscript Claim

This is a reproducible offline benchmark for agency-budgeted intervention policies in myoelectric prosthetic grasping under distribution shift. The strongest supported contribution is the benchmark definition and the performance-agency frontier it exposes, not a claim that one fixed operating point is universally best.

## Supported Wording

- The benchmark evaluates calibrated user and assistive posterior traces through user-only, assist-only, confidence blend, SetACSA, matched confidence-threshold, and agency-margin policies.
- The primary DB10 evidence is split-specific: `db10_amputee_loso` and `db10_mixed_to_amputee` are claim-bearing; `db10_able_to_amputee` is a boundary transfer result.
- Exact-budget confidence thresholding is a deliberately strong audit. When it blocks a universal fixed-tau claim, that is a scientific result about operating-point instability.
- Hyser and CEMHSEY support external robustness framing for day/session/longitudinal shift, not direct clinical or online validation.

## Disallowed Wording

- SOTA
- state of the art
- best published
- best-in-class
- clinically validated
- real-world validated
- online shared-control validation
- universal winner
- improves trust
- improves embodiment
- reduces workload
- haptic effectiveness
- hardware validation

## Results Order

1. Benchmark contract: datasets, split hierarchy, policy layer, matched comparators, and claim gate.
2. DB10 primary performance and matched-budget SetACSA comparison.
3. Exact-budget confidence audit showing fixed operating-point limits.
4. Offline earliest-safe and stable-safe prefix diagnostics.
5. Hyser and CEMHSEY external robustness checks.
6. Literature comparability and benchmark-only claim boundary.

## Evidence Files

- `results/reports/publication_clean/db10_publication_aggregate_policy_metrics.csv`
- `results/reports/publication_clean/db10_confidence_gate_pairwise.csv`
- `results/reports/publication_clean/db10_confidence_gate_exact_budget_pairwise.csv`
- `results/reports/publication_clean/db10_earliest_safe_summary.csv`
- `results/reports/publication_clean/hyser_*`
- `results/reports/publication_clean/cemhsey_*`
- `results/reports/publication_clean/direct_comparability_matrix.*`
- `results/reports/publication_clean/exact_budget_instability_*`
