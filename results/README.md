# Results Directory

This directory contains manuscript-facing derived outputs only. It intentionally excludes raw data, downloaded archives, full prepared arrays, trained model binaries, and large per-episode timing dumps.

Included:

- `reports/publication_clean/`: frozen CSV/JSON/Markdown outputs used by manuscript tables, claim-boundary audits, and figure generation.
- `reports/tnsre_figures/`: final PDF/PNG assets, figure manifest, and figure QA report.
- `reports/stats/` and `reports/stats_clean/`: inferential statistics tables.
- `audits/`: dataset audit manifests.
- `real/*/benchmark_manifest.json`: canonical model-lane manifests for each dataset.

Excluded:

- raw EMG or gaze/video files,
- downloaded archives,
- `.npy`, `.npz`, `.parquet`, `.pkl`, `.joblib`,
- per-episode timing dumps larger than GitHub's normal file limits.

