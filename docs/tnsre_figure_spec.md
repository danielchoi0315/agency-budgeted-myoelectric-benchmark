# TNSRE Figure Specification

This document records the final reviewer-facing figure set for the IEEE TNSRE manuscript package. The figures support an offline, open-data, agency-budgeted policy benchmark for myoelectric prosthetic grasping under distribution shift.

## Claim Boundary

Allowed figure wording:

- offline open-data benchmark
- reproducible policy-layer benchmark
- agency-budgeted intervention benchmark
- matched-budget policy comparison
- model-implied agency-loss accounting
- performance-agency trade-off
- external EMG stress-test check
- fixed operating-point instability

Disallowed figure wording:

- SOTA or state of the art
- best published or best-in-class
- clinically validated or real-world validated
- online shared-control validation
- haptic, workload, trust, embodiment, or hardware validation
- universal fixed-tau winner

## Final Figure Set

| ID | File stem | Role |
|---|---|---|
| Fig. 1 | `fig1_benchmark_flow` | Dataset contract, fixed decoders, policy layer, offline evaluation, and evidence boundary. |
| Fig. 2 | `fig2_db10_frontier` | DB10 primary performance-agency trade-off curves. |
| Fig. 3 | `fig3_setacsa_pairwise` | DB10 agency-margin versus matched SetACSA paired comparisons with nominal CIs and Holm-adjusted beneficial tests. |
| Fig. 4 | `fig4_exact_budget_audit` | Exact-budget confidence-threshold benefit and exact-match audit by split and budget. |
| Fig. 5 | `fig5_external_robustness` | Hyser and CEMHSEY external EMG stress-test trade-offs, not primary prosthetic clinical evidence. |
| Fig. 6 | `fig6_db10_earliest_safe` | DB10 offline prefix timing diagnostics. |
| Fig. S1-S5 | `figS_*` | Dataset-specific matched-budget and earliest-safe supplementary diagnostics. |

## Visual Standards

- Use split-family-aware panels and label primary, boundary, and external roles clearly.
- Preserve subject/session/day-level inference units in captions and legends.
- Label risk-coverage AUC as lower-is-better.
- State that earliest-safe and stable-safe timing diagnostics are offline prefix analyses.
- Use vector PDF for Overleaf and high-resolution PNG for review.
- Do not paste raw-data examples or local workstation paths into figure assets.
