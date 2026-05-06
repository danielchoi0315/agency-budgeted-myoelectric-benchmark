
# TNSRE Overleaf-Ready Manuscript Package

Main document: `main.tex`
Supplement: `supplement.tex`
Bibliography: `references.bib`
Figures: `figs/`
Tables: `tables/`

## Import to Overleaf
1. Upload this zip as a new Overleaf project.
2. Set `main.tex` as the main file.
3. Compile with pdfLaTeX + BibTeX. Overleaf's IEEE journal template environment should include `IEEEtran.cls`.
4. Confirm author affiliations, funding/conflict metadata, and repository licensing before submission.

## TNSRE-targeting choices
- Uses `\documentclass[journal]{IEEEtran}`.
- Keeps the regular-paper abstract under 250 words.
- Intersperses figures and tables in the manuscript.
- Uses numbered IEEE references through BibTeX.
- Includes an ethics / secondary-data statement.
- Keeps the exact-budget confidence-threshold audit in the main Results.
- Avoids clinical, hardware, haptic, online-control, and subjective-agency claims.

## Local verification status
The manuscript source was locally checked with MiKTeX `pdflatex` and BibTeX after the final editorial pass. `main.tex` compiled to a 10-page PDF after the standard LaTeX/BibTeX/LaTeX/LaTeX sequence, and `supplement.tex` compiled to a 5-page standalone supplement. A fresh extraction of the final zip was also compiled successfully with no blocking LaTeX log hits.

Please still compile once in Overleaf and inspect page count, float placement, author metadata, and figure legibility before submission.

