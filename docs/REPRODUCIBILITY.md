# Reproducibility Guide

This guide separates three reproducibility levels.

## Level 1: Verify the Release Repository

Install the package and run tests:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,deep]"
python scripts/check_no_raw_data.py --root .
python scripts/check_release_integrity.py --root .
pytest
```

Expected result: tests pass and no likely raw-data files are present.

## Level 2: Regenerate Submitted Figures From Committed Outputs

Install R packages used by `scripts/build_tnsre_figures.R` if they are not already available:

```r
install.packages(c("tidyverse", "ggplot2", "ggprism", "patchwork", "scales", "svglite", "ragg"))
```

Run:

```bash
Rscript scripts/build_tnsre_figures.R --repo-root .
python scripts/check_tnsre_figures.py --out-dir results/reports/tnsre_figures --recursive --min-pdfs 11 --min-width 1800 --min-height 1800 --min-dpi 300
```

Expected result: 11 PDF and 11 PNG figure assets pass QA in `results/reports/tnsre_figures/`.

## Level 3: Rebuild From Raw Public Datasets

Raw datasets must be downloaded from the original public repositories and placed outside git. The examples below use placeholders; replace them with actual local paths.

```bash
python scripts/audit_dataset.py --dataset db10 --root <DATA_ROOT>/raw/db10 --out results/audits/db10_audit.json
python scripts/audit_dataset.py --dataset hyser --root <DATA_ROOT>/raw/hyser --out results/audits/hyser_audit.json
python scripts/audit_dataset.py --dataset cemhsey --root <DATA_ROOT>/raw/cemhsey --out results/audits/cemhsey_audit.json
python scripts/audit_dataset.py --dataset grabmyo --root <DATA_ROOT>/raw/grabmyo --out results/audits/grabmyo_audit.json
```

Prepare canonical datasets:

```bash
python scripts/prepare_real_dataset.py --dataset db10 --root <DATA_ROOT>/raw/db10 --out <WORK_ROOT>/processed/db10 --db10-context-mode annotated_object_proxy
python scripts/prepare_real_dataset.py --dataset hyser --root <DATA_ROOT>/raw/hyser --out <WORK_ROOT>/processed/hyser
python scripts/prepare_cemhsey_resumable.py --root <DATA_ROOT>/raw/cemhsey --parts-dir <WORK_ROOT>/interim/cemhsey_parts --out <WORK_ROOT>/processed/cemhsey
python scripts/prepare_real_dataset.py --dataset grabmyo --root <DATA_ROOT>/raw/grabmyo --out <WORK_ROOT>/processed/grabmyo
```

Run canonical model lanes:

```bash
python scripts/run_real_baseline.py --dataset db10 --prepared-root <WORK_ROOT>/processed/db10 --out results/real/db10 --user-model torch_mlp --assist-model torch_mlp
python scripts/run_real_baseline.py --dataset hyser --prepared-root <WORK_ROOT>/processed/hyser --out results/real/hyser --user-model torch_mlp --assist-model extra_trees
python scripts/run_real_baseline.py --dataset cemhsey --prepared-root <WORK_ROOT>/processed/cemhsey --out results/real/cemhsey --user-model extra_trees --assist-model extra_trees
python scripts/run_real_baseline.py --dataset grabmyo --prepared-root <WORK_ROOT>/processed/grabmyo --out results/real/grabmyo --user-model extra_trees --assist-model torch_mlp
```

Regenerate reports:

```bash
python scripts/postprocess_real_stats.py --real-root results/real --out-root results/reports/stats
python scripts/build_anchor_report.py --config config/config.yaml --audits-root results/audits --real-root results/real --stats-root results/reports/stats --out-root results/reports
python scripts/run_db10_publication_extension.py --prepared-root <WORK_ROOT>/processed/db10 --out-root results/reports/publication_clean --user-model torch_mlp --assist-model torch_mlp
python scripts/run_publication_extension.py --dataset hyser --prepared-root <WORK_ROOT>/processed/hyser --out-root results/reports/publication_clean --user-model torch_mlp --assist-model extra_trees
python scripts/run_publication_extension.py --dataset cemhsey --prepared-root <WORK_ROOT>/processed/cemhsey --out-root results/reports/publication_clean --user-model extra_trees --assist-model extra_trees
python scripts/build_comparability_matrix.py --targets config/literature_anchors.yaml --comparability config/literature_comparability.yaml --out-root results/reports/publication_clean
python scripts/build_sota_report.py --config config/config.yaml --anchors config/literature_anchors.yaml --comparability config/literature_comparability.yaml --anchor-status results/reports/anchor_status.json --out-root results/reports/publication_clean
python scripts/build_exact_budget_instability_report.py --out-root results/reports/publication_clean
```

Finally regenerate figures as in Level 2.

## Claim Boundary

The exact-budget confidence audit is intentionally a strong diagnostic comparator. If thresholds are selected from evaluated traces, interpret the result as an oracle-style fairness stress test, not as a prospective deployment protocol.

