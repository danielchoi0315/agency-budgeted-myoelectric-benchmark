# Publication Repository Checklist

- [ ] `python scripts/check_no_raw_data.py --root .` passes.
- [ ] `python scripts/check_release_integrity.py --root .` passes.
- [ ] `pytest` passes in a fresh Python 3.11 environment.
- [ ] `Rscript scripts/build_tnsre_figures.R --repo-root .` regenerates all TNSRE figure assets.
- [ ] `python scripts/check_tnsre_figures.py --out-dir results/reports/tnsre_figures --recursive --min-pdfs 11 --min-width 1800 --min-height 1800 --min-dpi 300` passes.
- [ ] No raw dataset files, raw archives, model binaries, or local workstation paths are committed.
- [ ] The manuscript data/code availability statement points to `https://github.com/danielchoi0315/agency-budgeted-myoelectric-benchmark`.
- [ ] The final release is archived with a DOI after acceptance.

