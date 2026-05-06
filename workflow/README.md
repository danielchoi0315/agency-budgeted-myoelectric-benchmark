# Workflow

The top-level `Snakefile` provides a lightweight publication workflow for regenerating reviewer-facing figures from committed aggregate outputs and running release checks.

Full raw-data reproduction remains a staged process because raw datasets are not redistributed in this repository:

1. Audit original public dataset roots.
2. Prepare dataset-specific feature bundles outside git.
3. Train dataset-specific user and assistive decoders.
4. Calibrate posterior traces.
5. Run shared policy-layer comparisons.
6. Aggregate subject/session/day-level metrics.
7. Regenerate publication figures and tables.
8. Run release integrity, raw-data, figure, and test checks.
