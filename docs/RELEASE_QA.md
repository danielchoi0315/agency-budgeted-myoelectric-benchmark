# Release QA Record

Validation run on 2026-05-05 from the curated publication repository.

## Repository Safety

- `python scripts/check_no_raw_data.py --root .`: passed.
- `python scripts/check_release_integrity.py --root .`: passed.
- Sensitive-path scan for workstation home directories, local data-drive roots, legacy workspace names, placeholder authors, and editorial-worker notes: passed.
- Raw-data extension scan for MATLAB, HDF5, NumPy, WFDB, archive, model-binary, and media artifacts: passed.

## Code Tests

- `python -m pytest`: 151 passed, 9 warnings.
- Warnings were PyTorch nested-tensor informational warnings in model smoke tests, not assertion failures.

## Figure QA

- `Rscript scripts/build_tnsre_figures.R --repo-root .`: regenerated 22 assets.
- `python scripts/check_tnsre_figures.py --out-dir results/reports/tnsre_figures --recursive --min-pdfs 11 --min-width 1800 --min-height 1800 --min-dpi 300`: passed.
- Final committed figure set: 11 PDF assets and 11 PNG assets.
- PNG DPI: approximately 600 dpi for all assets.

## Manuscript Compile

- `main.tex`: compiled with pdfLaTeX + BibTeX + two pdfLaTeX passes.
- `supplement.tex`: compiled with pdfLaTeX.
- Final local compile check: main manuscript 10 pages; supplement 5 pages; no undefined citations or references in final logs.

## Remaining Human Checks

- Confirm all author affiliations and ordering.
- Confirm funding, conflicts of interest, and acknowledgments.
- Confirm MIT license is acceptable to all authors and the institution before journal submission.
