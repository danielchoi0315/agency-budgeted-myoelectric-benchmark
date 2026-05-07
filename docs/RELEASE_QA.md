# Release QA Record

Validation run on 2026-05-06 from the curated publication repository and a fresh extraction of the Overleaf package.

## Repository Safety

- `python scripts/check_no_raw_data.py --root .`: passed.
- `python scripts/check_release_integrity.py --root .`: passed.
- Sensitive-path scan for workstation home directories, local data-drive roots, legacy workspace names, placeholder authors, and editorial-worker notes: passed.
- Raw-data extension scan for MATLAB, HDF5, NumPy, WFDB, archive, model-binary, and media artifacts: passed.

## Code Tests

- `python -m pytest`: 93 passed, 9 warnings.
- Warnings were PyTorch nested-tensor informational warnings in model smoke tests, not assertion failures.

## Figure QA

- `Rscript scripts/build_tnsre_figures.R --repo-root .`: regenerated 22 assets.
- `python scripts/check_tnsre_figures.py --out-dir results/reports/tnsre_figures --recursive --min-pdfs 11 --min-width 1800 --min-height 900 --min-dpi 300 --report results/reports/tnsre_figures/tnsre_figure_qa.json`: passed.
- `python scripts/check_figure_fonts.py --pdf-dir results/reports/tnsre_figures/pdf`: passed.
- Final committed figure set: 11 PDF assets and 11 PNG assets.
- PNG DPI: approximately 600 dpi for all assets.

## Manuscript Compile

- The final Overleaf ZIP was extracted into a clean verification directory.
- Package manifest check: passed.
- Package figure check: 11 PDF assets and 11 PNG assets; all PNGs at least 1800 px wide, at least 900 px high, and at least 300 dpi.
- `main.tex`: compiled with pdfLaTeX + BibTeX + two pdfLaTeX passes.
- `supplement.tex`: compiled with two pdfLaTeX passes.
- Final local compile check: main manuscript 9 pages; supplement 5 pages; no undefined citations or references in final logs.
- Rendered page check: redesigned Fig. 1 appears centered at full text width as a compact A-E evidence-boundary matrix with no boundary escape, overlap, or caption collision.
- Main-text figure check: Fig. 5 appears in the main manuscript as the external EMG stress-test frontier; the DB10 earliest-safe timing diagnostic remains supplementary.

## Remaining Human Checks

- Confirm all author affiliations and ordering.
- Confirm funding, conflicts of interest, and acknowledgments.
- Confirm MIT license is acceptable to all authors and the institution before journal submission.
