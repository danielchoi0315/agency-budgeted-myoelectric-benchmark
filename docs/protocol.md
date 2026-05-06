# Benchmark Protocol

## Research Question

Can an agency-budgeted shared-autonomy policy improve offline prosthetic grasp
control under distribution shift while maintaining a pre-declared agency-loss
budget?

## Claim Boundary

Allowed claims:

- offline intent inference under distribution shift
- offline probability calibration
- offline intervention timing
- ALI/performance tradeoffs
- risk-coverage and intervention-budget comparisons

Disallowed claims:

- haptic effectiveness
- embodiment improvement
- trust improvement
- workload reduction
- clinical efficacy
- hardware validation

## Datasets

1. DB10/MeganePro is the primary dataset and the only dataset used for the main
   multimodal and cross-population story.
2. Hyser is an external robustness dataset for two-day HD-sEMG shift.
3. CEMHSEY is an external robustness dataset for 11-day longitudinal drift.

No raw samples from different datasets will be pooled into a single classifier.
Dataset-specific decoders feed a shared policy layer.

## Primary Comparisons

- User-only
- Assistive-only
- Confidence blend
- Selective prediction
- SetACSA
- Agency-margin policy

Primary manuscript emphasis:

- `db10_amputee_loso`
- `db10_mixed_to_amputee`

Boundary result to report explicitly:

- `db10_able_to_amputee`

## Primary Metrics

- Primary endpoint: risk-coverage AUC at matched intervention budget on DB10.
- Secondary metrics: macro-F1, mean expected ALI, Brier, NLL, ECE, median decision time.
- Diagnostics: balanced accuracy, intervention rate, action-change rate, and calibration plots.
- DB10 publication extension diagnostics: earliest-safe episode rate, stable-safe
  episode rate, median earliest stable-safe time, and empirical matched-budget
  comparisons against a plain assist-confidence threshold policy.
- External publication extension diagnostics: Hyser and CEMHSEY matched-budget
  confidence-threshold ablations plus earliest-safe and stable-safe summaries.

## Literature Comparison Boundary

The primary manuscript endpoint remains matched-budget DB10 risk-coverage AUC.
The literature-comparison machinery may also track `active_macro_f1` where that
is the closest available reported external metric. That does not upgrade those
papers to direct comparators by itself. Direct-comparison status is controlled by
the target-specific matrix in `config/literature_comparability.yaml`, not by a
single metric match.

Local DB10 paper-style anchor reproductions are kept in the publication extension
lane and are reported as pipeline-validation evidence only. They do not convert
the main cross-population or amputee-LOSO policy targets into direct SOTA
comparisons.

Primary claim wording must stay split-specific. The repository currently supports
the strongest benchmark language on `db10_amputee_loso` and
`db10_mixed_to_amputee`. `db10_able_to_amputee` remains a boundary transfer
result and should not be used to justify a universal threshold-dominance claim.

Hyser and CEMHSEY should be described as external robustness checks for
matched-budget performance, risk-coverage, and safe-intervention behavior. They
do not currently support a uniform claim that mean expected ALI is lower than a
rate-matched confidence threshold on every external split family.

All statistical tests are subject/session/day-level. Window-level hypothesis
tests are prohibited.

