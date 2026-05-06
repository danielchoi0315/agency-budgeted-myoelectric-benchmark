# TNSRE Figure Specification for J2

Last updated: 2026-05-02

This document specifies the reviewer-facing final figure set for the J2 IEEE
TNSRE manuscript. It is a figure specification only: do not treat it as a plot
generation request, and do not submit the existing `publication_clean` PNGs as
final journal composites without rebuilding them into split-family-aware panels.

## Scope

The final figure package supports this manuscript claim:

> J2 is a reproducible offline open-data benchmark for agency-budgeted
> intervention policies in myoelectric grasp classification under distribution
> shift.

The figure package must not support or imply direct-SOTA, online control,
clinical, hardware, haptic, trust, workload, embodiment, or end-to-end vision
claims.

Authoritative source hierarchy:

1. J2-specific manuscript and claim files:
   - `docs/manuscript_work/j2_tnsre_full_manuscript_draft.md`
   - `docs/manuscript_work/j2_claim_positioning.md`
   - `docs/manuscript_work/subagent_inputs/09_figures_tables_supplement.md`
2. J2 handoff files:
   - `docs/j2_publication_handoff/publication_asset_map.md`
   - `docs/j2_publication_handoff/execution_ledger.md`
   - `docs/j2_publication_handoff/remaining_work_and_submission_checklist.md`
3. Publication-clean outputs:
   - `results/reports/publication_clean/*`
   - `results/reports/stats/*`
4. Protocol, configuration, audits, and manifests:
   - `docs/protocol.md`
   - `config/config.yaml`
   - `config/literature_anchors.yaml`
   - `config/literature_comparability.yaml`
   - `results/audits/*_audit.json`
   - `results/real/*/benchmark_manifest.json`

Do not use `results/reports/j1_open/*` as J2 evidence. If an exact-budget source
contains J1 rows, filter the final J2 figure to DB10, Hyser, and CEMHSEY.

## Evidence Hierarchy

Primary DB10 evidence:

- `db10_amputee_loso`
- `db10_mixed_to_amputee`

Boundary DB10 evidence:

- `db10_able_to_amputee`

Supporting external robustness:

- `hyser_subject_logo`
- `hyser_within_subject_dayshift`
- `cemhsey_forward_day_logo`

Supplementary only:

- GRABMyo, unless a full publication-clean extension is generated.

Current claim gate:

- Source: `results/reports/publication_clean/sota_status.json`
- Status: `claim_safe_benchmark_only`
- Direct-SOTA eligibility: false for every registered target.

## Final Figure Set

| ID | Placement | Export stem | Role |
| --- | --- | --- | --- |
| Fig. 1 | Main | `tnsre_j2_fig01_benchmark_architecture_split_hierarchy` | Benchmark contract, split hierarchy, policy layer, and claim gate |
| Fig. 2 | Main | `tnsre_j2_fig02_db10_matched_budget_performance` | DB10 primary performance and matched-budget comparisons |
| Fig. 3 | Main | `tnsre_j2_fig03_db10_timing_diagnostics` | DB10 earliest-safe and stable-safe offline timing diagnostics |
| Fig. 4 | Main | `tnsre_j2_fig04_external_robustness_hyser_cemhsey` | Hyser and CEMHSEY external robustness composite |
| Fig. S1 | Supplement | `tnsre_j2_figs01_hyser_robustness_detail` | Hyser detailed matched-budget and timing diagnostics |
| Fig. S2 | Supplement | `tnsre_j2_figs02_cemhsey_robustness_detail` | CEMHSEY detailed matched-budget and timing diagnostics |
| Fig. S3 | Supplement | `tnsre_j2_figs03_claim_boundary_exact_budget` | Direct comparability, claim gate, and exact-budget instability |
| Fig. S4 | Supplement | `tnsre_j2_figs04_db10_anchor_reproduction_context` | DB10 anchor reproduction context and non-SOTA comparability |

Required export formats for every final figure:

- `figures/main/<export_stem>.pdf` or `figures/supplement/<export_stem>.pdf`
- `figures/main/<export_stem>.svg` or `figures/supplement/<export_stem>.svg`
- `figures/main/<export_stem>.tiff` or `figures/supplement/<export_stem>.tiff`
- `figures/main/<export_stem>.png` or `figures/supplement/<export_stem>.png`
- `figures/provenance/<export_stem>_sources.json`

The source manifest for each figure must list source file paths, file checksums,
plotting script or notebook path, software versions, and any filters applied.

## Main Figures

### Figure 1. Benchmark Architecture and Split Hierarchy

Export stem: `tnsre_j2_fig01_benchmark_architecture_split_hierarchy`

Status: final rendered asset is missing. Build a schematic, not a data plot.

Required sources:

- `docs/protocol.md`
- `config/config.yaml`
- `config/literature_anchors.yaml`
- `config/literature_comparability.yaml`
- `results/reports/anchor_report.md`
- `results/reports/publication_clean/anchor_report.md`
- `results/audits/db10_audit.json`
- `results/audits/hyser_audit.json`
- `results/audits/cemhsey_audit.json`
- `results/audits/grabmyo_audit.json`
- `results/real/db10/benchmark_manifest.json`
- `results/real/hyser/benchmark_manifest.json`
- `results/real/cemhsey/benchmark_manifest.json`
- `results/real/grabmyo/benchmark_manifest.json`

Required panels:

- A. Public-data contract: DB10 primary, Hyser/CEMHSEY external robustness,
  GRABMyo supplementary unless publication-clean outputs are added.
- B. Prepared-bundle contract: datasets are prepared independently; raw samples
  and label vocabularies are not pooled across datasets.
- C. Split hierarchy: DB10 primary families, DB10 boundary transfer family, and
  external subject/day or longitudinal shifts.
- D. Policy-layer evaluation: calibrated user and assistive decoder traces feed
  user-only, assist-only, confidence blend, selective prediction, SetACSA,
  agency-margin, and matched confidence-threshold policies.
- E. Claim gate: inference units are subject/session/day units, literature
  anchors are partial/contextual/boundary, and direct-SOTA claims are blocked.

Reviewer-facing claim boundary:

- State that DB10 uses `annotated_object_proxy` context.
- State that the result is offline annotated context-assisted benchmarking, not
  end-to-end vision or deployed perception.
- Do not imply online shared-control validation.

### Figure 2. DB10 Primary Performance and Matched-Budget Comparison

Export stem: `tnsre_j2_fig02_db10_matched_budget_performance`

Status: source data and check figure exist; final split-family panels are
missing. The check figure is:

- `results/reports/publication_clean/db10_matched_budget_ablation.png`

Required sources:

- `results/reports/publication_clean/db10_publication_aggregate_policy_metrics.csv`
- `results/reports/publication_clean/db10_publication_metrics_by_policy_unit.csv`
- `results/reports/publication_clean/db10_confidence_gate_match_summary.csv`
- `results/reports/publication_clean/db10_confidence_gate_pairwise.csv`
- `results/reports/publication_clean/db10_confidence_gate_exact_budget_pairwise.csv`
- `results/reports/publication_clean/db10_confidence_gate_iso_budget_pairwise.csv`
- `results/reports/publication_clean/db10_confidence_gate_iso_budget_unit_deltas.csv`
- `results/reports/publication_clean/db10_plain_conf_exact_budget_selection_by_unit.csv`
- `results/reports/stats/primary_pairwise.csv`
- `results/reports/stats/secondary_pairwise.csv`

Required panels:

- A. Active macro-F1 by DB10 split family. Mark `db10_amputee_loso` and
  `db10_mixed_to_amputee` as primary; mark `db10_able_to_amputee` as boundary.
- B. Active risk-coverage AUC by DB10 split family. Axis label must state
  "lower is better."
- C. Matched-budget active macro-F1 across tau for agency-margin, SetACSA, and
  matched plain confidence.
- D. Paired agency-margin versus SetACSA deltas for tau 0.02, 0.05, and 0.10 in
  the two primary DB10 families, using unit-level confidence intervals.
- E. Exact-budget plain-confidence audit inset or boundary panel showing that
  exact-budget confidence thresholding blocks a universal fixed-tau win.

Reviewer-facing claim boundary:

- Supported: agency-margin improves over matched SetACSA on active macro-F1 and
  active risk-coverage AUC at low-to-middle tau values in DB10 primary families.
- Required caveat: exact-budget confidence thresholding is a strong comparator
  and prevents a universal agency-margin dominance claim.
- `assist_only` and confidence blend are context/upper-envelope anchors, not
  agency-preserving deployment policies.

### Figure 3. DB10 Earliest-Safe and Stable-Safe Timing

Export stem: `tnsre_j2_fig03_db10_timing_diagnostics`

Status: source data and check figure exist; final split-family panels are
missing. The check figure is:

- `results/reports/publication_clean/db10_earliest_safe.png`

Required sources:

- `results/reports/publication_clean/db10_earliest_safe_summary.csv`
- `results/reports/publication_clean/db10_earliest_safe_pairwise.csv`
- `results/reports/publication_clean/db10_earliest_safe_by_unit.csv`
- `results/reports/publication_clean/db10_earliest_safe_by_episode.csv`
- `results/reports/publication_clean/db10_earliest_safe_exact_budget_pairwise.csv`
- `results/reports/publication_clean/db10_earliest_safe_iso_budget_pairwise.csv`
- `results/reports/publication_clean/db10_earliest_safe_iso_budget_unit_deltas.csv`

Required panels:

- A. Stable-safe episode rate by tau and policy family for
  `db10_amputee_loso`.
- B. Stable-safe episode rate by tau and policy family for
  `db10_mixed_to_amputee`.
- C. Median earliest stable-safe time by tau in the two primary DB10 families.
  Axis label must state that lower time is earlier and that timing is evaluated
  only among episodes with a stable-safe event.
- D. Paired safe-episode, stable-safe, and final-correct deltas versus SetACSA.
- E. Exact-budget plain-confidence safety audit showing that safe-episode gains
  can coexist with stable-safe or final-correct tradeoffs.

Reviewer-facing claim boundary:

- These are offline prefix diagnostics computed from causal prefixes.
- Do not call them online latency, real-time safety, clinical safety, haptic
  effectiveness, or user-experience outcomes.

### Figure 4. External Robustness Composite: Hyser and CEMHSEY

Export stem: `tnsre_j2_fig04_external_robustness_hyser_cemhsey`

Status: current dataset-level PNGs exist, but a final combined figure is
missing. Check figures:

- `results/reports/publication_clean/hyser_matched_budget_ablation.png`
- `results/reports/publication_clean/hyser_earliest_safe.png`
- `results/reports/publication_clean/cemhsey_matched_budget_ablation.png`
- `results/reports/publication_clean/cemhsey_earliest_safe.png`

Required Hyser sources:

- `results/reports/publication_clean/hyser_publication_aggregate_policy_metrics.csv`
- `results/reports/publication_clean/hyser_publication_metrics_by_policy_unit.csv`
- `results/reports/publication_clean/hyser_confidence_gate_match_summary.csv`
- `results/reports/publication_clean/hyser_confidence_gate_pairwise.csv`
- `results/reports/publication_clean/hyser_confidence_gate_exact_budget_pairwise.csv`
- `results/reports/publication_clean/hyser_confidence_gate_iso_budget_pairwise.csv`
- `results/reports/publication_clean/hyser_earliest_safe_summary.csv`
- `results/reports/publication_clean/hyser_earliest_safe_pairwise.csv`
- `results/reports/publication_clean/hyser_earliest_safe_exact_budget_pairwise.csv`

Required CEMHSEY sources:

- `results/reports/publication_clean/cemhsey_publication_aggregate_policy_metrics.csv`
- `results/reports/publication_clean/cemhsey_publication_metrics_by_policy_unit.csv`
- `results/reports/publication_clean/cemhsey_confidence_gate_match_summary.csv`
- `results/reports/publication_clean/cemhsey_confidence_gate_pairwise.csv`
- `results/reports/publication_clean/cemhsey_confidence_gate_exact_budget_pairwise.csv`
- `results/reports/publication_clean/cemhsey_confidence_gate_iso_budget_pairwise.csv`
- `results/reports/publication_clean/cemhsey_earliest_safe_summary.csv`
- `results/reports/publication_clean/cemhsey_earliest_safe_pairwise.csv`
- `results/reports/publication_clean/cemhsey_earliest_safe_exact_budget_pairwise.csv`

Required panels:

- A. Hyser active macro-F1 across matched intervention budgets for
  `hyser_subject_logo` and `hyser_within_subject_dayshift`.
- B. Hyser active risk-coverage AUC and/or stable-safe episode rate across tau.
- C. CEMHSEY active macro-F1 across matched intervention budgets for
  `cemhsey_forward_day_logo`.
- D. CEMHSEY stable-safe episode rate and median earliest stable-safe time.
- E. External exact-budget audit inset.

Reviewer-facing claim boundary:

- Hyser and CEMHSEY are supporting external robustness checks.
- They do not replace the DB10 primary claim hierarchy.
- They do not establish universal agency-margin superiority, direct DB10
  replication, or direct comparability to published Hyser/CEMHSEY results.
- GRABMyo must not be added as a claim-bearing robustness panel unless
  publication-clean paired, exact-budget, and earliest-safe outputs are
  generated.

## Supplement Figures

### Figure S1. Hyser Robustness Detail

Export stem: `tnsre_j2_figs01_hyser_robustness_detail`

Required sources:

- `results/reports/publication_clean/hyser_matched_budget_ablation.png`
- `results/reports/publication_clean/hyser_earliest_safe.png`
- `results/reports/publication_clean/hyser_publication_aggregate_policy_metrics.csv`
- `results/reports/publication_clean/hyser_confidence_gate_pairwise.csv`
- `results/reports/publication_clean/hyser_earliest_safe_summary.csv`
- `results/reports/publication_clean/hyser_earliest_safe_pairwise.csv`

Required panels:

- A. Matched-budget active macro-F1 detail.
- B. Matched-budget active risk-coverage AUC detail.
- C. Stable-safe episode rate by policy family.
- D. Median earliest stable-safe time by policy family.

Claim boundary: detailed support for external robustness only.

### Figure S2. CEMHSEY Robustness Detail

Export stem: `tnsre_j2_figs02_cemhsey_robustness_detail`

Required sources:

- `results/reports/publication_clean/cemhsey_matched_budget_ablation.png`
- `results/reports/publication_clean/cemhsey_earliest_safe.png`
- `results/reports/publication_clean/cemhsey_publication_aggregate_policy_metrics.csv`
- `results/reports/publication_clean/cemhsey_confidence_gate_pairwise.csv`
- `results/reports/publication_clean/cemhsey_earliest_safe_summary.csv`
- `results/reports/publication_clean/cemhsey_earliest_safe_pairwise.csv`

Required panels:

- A. Matched-budget active macro-F1 detail.
- B. Matched-budget active risk-coverage AUC detail.
- C. Stable-safe episode rate by policy family.
- D. Median earliest stable-safe time by policy family.

Claim boundary: detailed support for external robustness only.

### Figure S3. Claim Boundary, Direct Comparability, and Exact-Budget Instability

Export stem: `tnsre_j2_figs03_claim_boundary_exact_budget`

Required sources:

- `results/reports/publication_clean/direct_comparability_matrix.csv`
- `results/reports/publication_clean/direct_comparability_matrix.json`
- `results/reports/publication_clean/direct_comparability_matrix.md`
- `results/reports/publication_clean/sota_status.json`
- `results/reports/publication_clean/sota_report.md`
- `results/reports/publication_clean/exact_budget_instability_report.md`
- `results/reports/publication_clean/exact_budget_instability_summary.csv`
- `results/reports/publication_clean/exact_budget_instability_by_tau.csv`

Required panels:

- A. Direct-comparability counts by target; every target has zero direct
  anchors and `direct_sota_eligible = False`.
- B. Exact-budget instability by dataset; DB10, Hyser, and CEMHSEY have
  `fixed_tau_supported = False`.
- C. Required-scope tau heatmap showing confidence primary-metric support,
  harmful safety rows, and exact-match fraction by split family and tau.
- D. Claim wording gate: benchmark-only terms allowed; direct-SOTA and
  clinical/online validation terms blocked.

Claim boundary: this figure is a guardrail, not an attempt to rank against
published papers.

### Figure S4. DB10 Anchor Reproduction Context

Export stem: `tnsre_j2_figs04_db10_anchor_reproduction_context`

Required sources:

- `results/reports/publication_clean/db10_anchor_reproductions.md`
- `results/reports/publication_clean/db10_anchor_reproductions.json`
- `results/reports/publication_clean/direct_comparability_matrix.csv`
- `results/reports/publication_clean/sota_status.json`

Required panels:

- A. Wang-style local reproduction context.
- B. Cognolato-style local reproduction context.
- C. Comparability status showing why these are qualitative pipeline-validation
  anchors rather than direct published-performance claims.

Claim boundary: anchor reproductions are context and validation checks only.

## Visual Standards

General layout:

- Use split-family-aware composites. Do not average over split families in final
  claim-bearing panels unless the panel explicitly labels itself as a descriptive
  dataset-level summary.
- Use panel labels `A`, `B`, `C`, etc. in the upper-left of each panel.
- Keep all labels readable at IEEE two-column print size.
- Prefer vector-native text and lines; rasterize only dense heatmaps or images
  when necessary.
- Use one consistent policy legend across all figures.

Typography and export:

- Use Arial, Helvetica, or an IEEE-compatible sans-serif.
- Minimum final label size: 7 pt; preferred axis and legend labels: 8-9 pt.
- Export vector PDF/SVG plus 300 dpi or higher TIFF/PNG at final print size.
- Use white or transparent backgrounds; no decorative gradients.

Color and shape:

- Use a colorblind-safe palette.
- Reserve one color consistently for each policy family:
  - user-only baseline
  - assist-only/context anchor
  - confidence blend
  - SetACSA
  - agency-margin
  - matched plain-confidence threshold
- Use line style or marker shape in addition to color for policy families.
- Use visual badges or facet subtitles for evidence role: primary, boundary, or
  supporting external.

Statistical display:

- Pairwise deltas must use subject/session/day-level units as reported in the
  source CSV `unit_cols`; do not display window-level significance.
- Confidence intervals must come from the paired output CSVs, not from visual
  smoothing or ad hoc recalculation.
- When showing Holm results, distinguish estimate size from gate status.
- Do not use stars without a legend explaining the paired unit and Holm gate.

Metric directions:

- Active macro-F1: higher is better.
- Macro-F1: higher is better.
- Risk-coverage AUC and active risk-coverage AUC: lower is better.
- Mean ALI: lower is lower model-implied agency loss.
- Intervention rate: lower is fewer interventions, but not automatically better
  unless interpreted with performance and risk.
- Median earliest stable-safe time: lower is earlier among stable-safe episodes.

Captions:

- Captions must say "offline" for all policy and timing figures.
- Captions for DB10 figures must mention `annotated_object_proxy` when context
  assistance is relevant.
- Captions for timing figures must say that prefix timing diagnostics do not
  establish online latency or clinical safety.
- Captions for external robustness figures must say that Hyser and CEMHSEY are
  supporting robustness checks.
- Captions for comparability figures must say that direct-SOTA wording is
  blocked.

## Source Data Standards

Core aggregate metric columns:

- `split_family`
- `policy`
- `accuracy`
- `macro_f1`
- `active_macro_f1`
- `balanced_accuracy`
- `mean_ali`
- `intervention_rate`
- `risk_coverage_auc`
- `active_risk_coverage_auc`
- `ece`
- `median_decision_time_ms`
- `n`
- `n_active`

Core paired comparison columns:

- `split_family`
- `comparison_family`
- `metric`
- `policy_a`
- `policy_b`
- `unit_cols`
- `n_units`
- `estimate`
- `ci_low`
- `ci_high`
- `p_value`
- `beneficial`
- `holm_reject`
- `beneficial_and_holm_significant`

Core timing summary columns:

- `dataset_id`
- `split_family`
- `policy`
- `safe_episode_rate`
- `stable_safe_episode_rate`
- `final_correct_rate`
- `median_earliest_intervene_s`
- `median_earliest_correct_s`
- `median_earliest_safe_s`
- `median_earliest_stable_safe_s`

Exact-budget audit fields to preserve:

- `required_split_families`
- `universal_exact_budget_taus`
- `n_universal_exact_budget_taus`
- `fixed_tau_supported`
- `worst_confidence_gap`
- `lowest_confidence_exact_match_fraction`
- `worst_safety_harmful_sig_metric_count`
- `instability_story`

## Claim Boundaries

Allowed figure-level wording:

- offline open benchmark
- reproducible policy-layer benchmark
- agency-budgeted intervention benchmark
- matched-budget intervention comparison
- model-implied agency-loss accounting
- performance-agency frontier
- external EMG robustness check
- fixed operating-point instability

Disallowed figure-level wording:

- SOTA
- state of the art
- best published
- best-in-class
- clinically validated
- real-world validated
- online shared-control validation
- haptic effectiveness
- trust improvement
- workload reduction
- embodiment improvement
- hardware validation
- end-to-end vision
- universal fixed-tau winner

Reviewer-facing captions and callouts should expose the constraints rather than
hide them. The strongest supported result is not a universal method win; it is a
benchmark/frontier result showing that matched-budget policy evaluation changes
the interpretation of active performance, risk, intervention rate, and
model-implied agency loss under declared distribution shifts.

## Final QA Checklist

- [ ] Every final figure has a source manifest.
- [ ] Every final figure has PDF, SVG, TIFF, and PNG exports.
- [ ] Current `publication_clean` PNGs are used only as checks or source panels,
      not pasted as final manuscript figures unless rebuilt and relabeled.
- [ ] All DB10 panels mark primary and boundary split families explicitly.
- [ ] All external panels mark Hyser/CEMHSEY as supporting robustness evidence.
- [ ] GRABMyo is supplementary only unless publication-clean extension outputs
      are generated.
- [ ] All risk-coverage axes state that lower is better.
- [ ] All timing captions state that the metric is an offline prefix diagnostic.
- [ ] Pairwise significance is subject/session/day-level, not window-level.
- [ ] No direct-SOTA or clinical/online validation language appears in titles,
      legends, captions, or callouts.

