# Claim Positioning

## Primary Manuscript Claim

The strongest claim supported by the current repository is:

- `J1-Frontier` is a reproducible open-data benchmark for causal-prefix
  intervention policies under matched intervention budgets, and the current real
  run shows that no fixed operating point universally dominates a matched plain
  confidence-threshold baseline across the amputee-relevant primary families.

This is a benchmark/frontier claim, not a method-win claim.

## Supported Wording

Abstract-safe wording:

- `J1-Frontier` evaluates causal-prefix intervention policies under matched
  intervention budgets on amputee-relevant open-data split families.
- In the current J1 release, no fixed `tau` achieved Holm-significant gains
  over the matched plain confidence-threshold comparator on both
  `active_macro_f1` and `active_risk_coverage_auc` across every required
  primary family.
- Boundary, leakage, and provenance gates pass, so the negative
  universal-winner result is scientific evidence rather than a pipeline failure.

Discussion-safe wording:

- The contribution is the benchmark definition and the frontier it reveals, not
  a claim that one frozen policy is universally best.
- Matched plain confidence thresholds are the primary comparator because weak
  comparators would hide the real tradeoff structure.
- The strongest in-dataset evidence is `j1_amputee_loso` and
  `j1_mixed_to_amputee`; `j1_able_to_amputee` is a boundary condition and
  `j1_subject_logo` is descriptive only.
- External datasets remain supporting robustness evidence, not part of the core
  J1 causal-prefix claim.

## Boundary Cases To State Explicitly

- `j1_amputee_loso` is the top primary family because it is the cleanest
  subject-disjoint amputee-to-amputee test.
- `j1_mixed_to_amputee` is also primary, but second to `j1_amputee_loso`
  because pooled able-bodied plus amputee training can hide target-population
  weakness.
- `j1_able_to_amputee` is boundary evidence only. It should stress-test
  transportability, not gate the main claim.
- `j1_subject_logo` should not be treated as independent primary corroboration
  because it overlaps heavily with the amputee holdout structure while being
  able-bodied dominated.
- `J1` itself is still single-source, single-session, and single-day in the
  current bundle, so claims about source/day/device shift must remain supported
  by external datasets until the J1 bundle becomes multi-source.

## Disallowed Wording

- `SOTA`
- `state of the art`
- `best published`
- `clinically validated`
- `universal winner`
- `revolutionary controller`
- `improves trust`
- `improves embodiment`
- `reduces workload`
- `translational readiness`

## Results Order

1. The benchmark definition: causal-prefix contract, split hierarchy, matched
   comparator design, and fail-closed provenance.
2. Primary result: no fixed `tau` universally clears the matched-budget gate on
   the amputee-relevant primary families.
3. Frontier package: Pareto/non-dominated structure, universal-`tau` regret,
   and family-conditioned tradeoffs.
4. Safety/timing and modality-time value analyses.
5. Boundary transfer result on `j1_able_to_amputee`.
6. External robustness support on Hyser, CEMHSEY, and GRABMyo.

## Evidence Files

- `results/reports/j1_open/j1_scorecard.md`
- `results/reports/j1_open/j1_frontier_readout.md`
- `results/reports/j1_open/stats/primary_pairwise.csv`
- `results/reports/j1_open/publication_clean/j1_confidence_gate_pairwise.csv`
- `results/reports/j1_open/publication_clean/j1_confidence_gate_match_summary.csv`
- `results/reports/j1_open/split_report.md`

