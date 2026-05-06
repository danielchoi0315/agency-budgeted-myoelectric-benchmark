
# Draft Cover Letter for IEEE TNSRE

Dear Editor-in-Chief and Associate Editor,

We submit the manuscript "Agency-Budgeted Policy Evaluation for Myoelectric Prosthetic Grasping Under Distribution Shift: An Offline Open-Data Benchmark" for consideration as a regular full-length paper in IEEE Transactions on Neural Systems and Rehabilitation Engineering.

The manuscript targets TNSRE's rehabilitative and neural engineering scope by addressing a software-control problem for upper-limb myoelectric prostheses: when should an assistive policy be allowed to alter the command implied by a user's EMG decoder? The contribution is an offline open-data policy-layer benchmark, not a clinical or hardware validation study. We separate dataset-specific decoders from a shared intervention-policy layer and evaluate how agency-budgeted assistance trades active-grasp performance against decoder-relative agency loss under declared distribution shifts.

The primary claim-bearing dataset is DB10/MeganePro, with amputee leave-one-subject-out and mixed-to-amputee split families as the primary settings. Hyser and CEMHSEY are included as external high-density EMG robustness checks under session/day shift, not as substitutes for DB10 or as clinical validation datasets. The DB10 assistive branch uses an annotated object-context proxy; we describe it as an offline context source for policy auditing, not as an end-to-end perception or hardware-control system.

We include matched SetACSA comparisons and a deliberately strong exact-budget confidence-threshold audit in the main Results. The exact-budget audit shows that no universal fixed operating point is supported and that confidence thresholding can outperform agency-margin on DB10 active performance, while agency-margin reduces decoder-relative agency loss. We frame this as the central benchmark finding: the benchmark exposes a performance--agency frontier rather than supporting a universal method-superiority claim.

The manuscript makes no claim of clinical efficacy, online shared-control validation, haptic effectiveness, hardware validation, or measured perceived agency. The agency-loss index is explicitly defined as model-implied and decoder-relative. DB10 assistive context is described as an offline annotated object-context proxy rather than as a deployable perception system.

All datasets are public resources, and no new participant data were collected. During peer review we will provide a public reproducibility repository containing the benchmark code, split definitions, prepared-bundle schemas or reconstruction scripts, policy grids, metric scripts, figure-generation scripts, and frozen result manifests. Raw public datasets will be referenced rather than redistributed.

Sincerely,

[Corresponding Author]

