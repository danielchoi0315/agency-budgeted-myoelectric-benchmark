# Data Access and Raw-Data Policy

This repository does not redistribute raw participant data. All raw datasets must be obtained from their original public sources and stored outside git.

## Datasets

| Dataset | Manuscript role | Raw data source |
|---|---|---|
| DB10/MeganePro | Primary amputee-relevant multimodal grasp benchmark | Original DB10/MeganePro release and associated publications |
| Hyser | External two-day HD-sEMG robustness check | Original Hyser/PhysioNet release |
| CEMHSEY | External 11-day HD-sEMG robustness check | Original CEMHSEY release |
| GRABMyo | Supplementary HD-sEMG robustness check | Original GRABMyo release |

See `docs/sources.md` and `manuscript/tnsre_overleaf/references.bib` for formal source citations.

## Expected Local Layout

The code accepts explicit paths through command-line arguments. For reproducible local work, use a layout like:

```text
<DATA_ROOT>/raw/db10
<DATA_ROOT>/raw/hyser
<DATA_ROOT>/raw/cemhsey
<DATA_ROOT>/raw/grabmyo
<WORK_ROOT>/processed/db10
<WORK_ROOT>/processed/hyser
<WORK_ROOT>/processed/cemhsey
<WORK_ROOT>/processed/grabmyo
```

Do not commit raw archives, extracted WFDB files, MATLAB files, video, raw EMG arrays, or model binaries. The repository safety check is:

```bash
python scripts/check_no_raw_data.py --root .
```

## Committed Derived Outputs

The repository includes frozen manuscript-facing derived artifacts:

- aggregate policy metrics,
- unit-level policy metrics,
- pairwise comparison tables,
- exact-budget confidence audit summaries,
- earliest-safe summaries,
- direct-comparability and claim-boundary reports,
- publication figures and figure QA reports.

The largest excluded derived files are per-episode timing dumps named `*_earliest_safe_by_episode.csv`. They are not needed to regenerate the submitted figures/tables from the committed aggregate outputs.

