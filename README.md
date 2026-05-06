# Agency-Budgeted Myoelectric Policy Benchmark

Publication repository for the TNSRE manuscript:

**Agency-Budgeted Policy Evaluation for Myoelectric Prosthetic Grasping Under Distribution Shift: An Offline Open-Data Benchmark**

This repository contains the code, configuration, manuscript-facing derived outputs, audit manifests, and figure-generation scripts used for an offline policy-layer benchmark of agency-budgeted myoelectric prosthetic grasping under distribution shift. The repository is designed for reproducibility review: it does not redistribute raw participant data, but it includes the instructions and derived artifacts needed to inspect the benchmark claims and regenerate manuscript figures from the committed aggregate outputs.

## Scope

The benchmark evaluates shared-autonomy intervention policies from calibrated posterior traces under declared distribution shifts. The primary claim-bearing dataset is DB10/MeganePro. Hyser and CEMHSEY are external high-density EMG robustness checks, and GRABMyo is supplementary.

Supported claims:

- Offline policy-layer evaluation for myoelectric prosthetic grasping.
- Unit-level inference over subject/session/day units rather than windows.
- Matched SetACSA-style, confidence-gate, confidence-blend, user-only, assist-only, and agency-margin policy comparisons.
- Decoder-relative agency-loss index (ALI), risk-coverage, intervention-rate, exact-budget confidence, and timing diagnostics.

Unsupported claims:

- Online shared-control validation.
- Clinical efficacy or device safety.
- Haptic effectiveness, workload, trust, embodiment, perceived agency, or user acceptance.
- End-to-end deployable perception or prosthetic hardware validation.
- Universal fixed operating point or state-of-the-art superiority.

## Repository Layout

```text
config/                         Dataset, split, policy, and literature-comparability configuration
docs/                           Reproducibility, data-access, claim-boundary, and source documentation
src/myoagency/                   Reusable benchmark package
scripts/                        Command-line entry points for audits, preparation, benchmarking, reports, and figures
tests/                          Synthetic-data unit and integration tests
workflow/                       Snakemake workflow rules
results/reports/publication_clean/
                                Frozen manuscript-facing CSV/JSON/Markdown outputs
results/reports/tnsre_figures/  Final PDF/PNG figure assets and figure QA report
results/reports/stats*/         Inferential statistics tables
results/audits/                 Dataset audit manifests, no raw data
results/real/*/benchmark_manifest.json
                                Canonical model-lane manifests
manuscript/tnsre_overleaf/      Overleaf-ready manuscript source package
```

## Quick Start

Create a Python 3.11 environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,deep]"
```

Run the release safety checks and tests:

```bash
python scripts/check_no_raw_data.py --root .
python scripts/check_release_integrity.py --root .
python scripts/check_tnsre_figures.py --out-dir results/reports/tnsre_figures --recursive --min-pdfs 11 --min-width 1800 --min-height 1800 --min-dpi 300
pytest
```

Regenerate the TNSRE figure package from committed aggregate outputs:

```bash
Rscript scripts/build_tnsre_figures.R --repo-root .
python scripts/check_tnsre_figures.py --out-dir results/reports/tnsre_figures --recursive --min-pdfs 11 --min-width 1800 --min-height 1800 --min-dpi 300
```

The figure command uses `results/reports/publication_clean/` as input and writes PDF/PNG assets plus a manifest to `results/reports/tnsre_figures/`.

The validation record for this release is in [docs/RELEASE_QA.md](docs/RELEASE_QA.md).

## Full Benchmark Reproduction

Raw datasets are public but are not redistributed in this repository. To rebuild from raw data, obtain the original datasets described in [docs/DATA.md](docs/DATA.md), place them outside git, then run the preparation and benchmark commands described in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

The committed CSV/JSON outputs in `results/reports/publication_clean/` are the frozen manuscript-facing artifacts. Large per-episode timing dumps are intentionally excluded from git; summary, unit-level, pairwise, exact-budget, and figure-generation inputs are included.

## Manuscript Package

The Overleaf-ready source package is in [manuscript/tnsre_overleaf](manuscript/tnsre_overleaf). It contains the LaTeX source, references, tables, figures, and submission notes used for the TNSRE package.

## Data and Code Availability Statement

Suggested manuscript text:

> Code, configuration files, split and benchmark manifests, frozen aggregate outputs, figure-generation scripts, and manuscript source are available at `https://github.com/danielchoi0315/agency-budgeted-myoelectric-benchmark`. Raw public datasets are not redistributed and should be obtained from their original repositories.

## Citation

If you use this repository, cite the associated manuscript and the original public datasets. A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

