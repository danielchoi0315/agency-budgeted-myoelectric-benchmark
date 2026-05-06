# Workflow

The top-level `Snakefile` currently runs the synthetic smoke pipeline. Real
dataset rules should be added after raw datasets are downloaded and audited.

The rule order for real data should remain:

1. audit raw dataset roots
2. create manifests and modality maps
3. preprocess windows/features
4. train dataset-specific decoders
5. calibrate prediction traces
6. run shared policy grid
7. aggregate subject/session/day metrics
8. generate figures and tables


