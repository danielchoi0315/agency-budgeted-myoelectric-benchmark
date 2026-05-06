# Claims Ledger

| Claim | Status | Evidence Required | Reviewer Risk Control |
|---|---|---|---|
| The repository defines a reproducible offline policy-layer benchmark for agency-budgeted myoelectric prosthetic grasping under distribution shift. | Primary | Protocol, dataset audits, split manifests, benchmark manifests, committed aggregate outputs, figure QA. | Frame the contribution as a benchmark and frontier analysis, not as a universal controller win. |
| DB10/MeganePro is the claim-bearing dataset for amputee-relevant policy evaluation. | Primary | `db10_amputee_loso`, `db10_mixed_to_amputee`, and boundary `db10_able_to_amputee` results. | Keep split-specific wording; do not average away the primary/boundary hierarchy. |
| Matched SetACSA and matched confidence-threshold policies are necessary comparators for agency-margin policies. | Primary | DB10 pairwise tables, matched-budget summaries, exact-budget confidence audit. | Avoid weak-comparator claims; state when exact-budget confidence thresholding blocks universal dominance. |
| Unit-level inference is subject/session/day-level rather than window-level. | Primary | Pairwise CSV `unit_cols`, statistics tables, manuscript captions. | Do not describe window-level significance or inflate the effective sample size. |
| Hyser and CEMHSEY provide external robustness checks for shift behavior. | Secondary | External publication-clean metrics, pairwise outputs, earliest-safe summaries, figure panels. | Do not use external datasets to replace the DB10 primary claim hierarchy. |
| GRABMyo is supplementary unless a full publication-clean extension is generated. | Boundary | Audit and canonical benchmark manifests. | Do not promote GRABMyo into the main robustness claim without matched-budget and timing outputs. |
| Offline prefix timing diagnostics are not online latency, clinical safety, haptic effectiveness, trust, workload, embodiment, or user acceptance. | Boundary | Timing tables, figure captions, ethics statement, limitations. | Every timing claim must say offline/prefix and must not imply deployed prosthesis safety. |
| Literature anchors are contextual unless the target-specific comparability matrix says otherwise. | Boundary | Direct-comparability matrix and benchmark-only status report. | Block direct-SOTA, best-published, clinical-validation, and universal-operating-point language. |
