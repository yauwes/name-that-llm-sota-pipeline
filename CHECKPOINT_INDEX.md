# Name That LLM — verified research checkpoint

Official best balanced accuracy: **0.665**. Target0.99/rank1 remains unachieved and cannot be guaranteed. No new competition submission from this research session.

- [Current Indonesian handoff, EDA, methods, failures and next actions](readme.md)
- [Executed research report — 30 code cells](ntl_session_20261103.ipynb). This is a report, **not** a competition inference notebook.
- [517-file source snapshot](ntl_research_checkpoint_20261103.zip) and [SHA-256 manifest](checkpoint_manifest_20261103.json).
- [E37 complete private version2: all44 remote members verified](e37_v2_private_verified.json).
- [Five final adapters reproduce all2699 heldout predictions](e37_final_inference_verified.json), with [source/library/terminal-handle contract](e37_final_contract_addendum.json).
- [Isolated E38/E39 restore, cached replays and20actualhead inferences](score_heads_restore_verified.json).
- [E40 external-origin metadata audit](e40_external_origin_audit.json).
- [Measured championV1 archive, official0.665](ntl_champion_v1_0665_ARCHIVED.ipynb). Contract repairV3 remains UNVALIDATED for full CUDA serving/quality.

| Experiment | Result | Decision |
|---|---|---|
| E37 local MiniLM top-two-layer training |5/5folds,81.01min; train0.28152,inner-val0.17407,standalone-test0.18705; fixedfusion0.58784 versus baseline0.59709, deltaCI95%[-0.01557,-0.00296]|Rejected|
| E38 nonlinear score heads |Inner-selected fixedfusion0.59562 versus0.59709; deltaCI crosses0|No promotion|
| E39 soft provider×model heads |Inner-selected fixedfusion0.59451 versus0.59709; deltaCI crosses0|No promotion|
| E40 metadata-only audit |49syntactic dataset prefixes recovered for69530rows;44617rows have unrecovered detailed origin|Diagnostic only; no classifier or new score|

These are historical research comparisons, not championCV or hidden-test estimates. E37 training/inference have terminated successfully; do not restart them. All inner selections exhausted the8epoch budget, which is not proof of convergence. Another agent owns the TPU job; this session does not modify or stop it.

Private restoration dependencies:

- [Base numeric version4](https://www.kaggle.com/datasets/ajikajadeh/ntl-reader-e19-e22-checkpoint/versions/4) plus [E33–E35 delta version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e33-e35-research-checkpoint/versions/1) for earlier research.
- [E37 complete version2](https://www.kaggle.com/datasets/ajikajadeh/ntl-e37-minilm-checkpoints/versions/2):502.64MiB/44files, all remote bytes verified. [Version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e37-minilm-checkpoints/versions/1) retains the earlier resumable snapshot used for actual CPU/MPS resume proofs.
- [E38 version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e38-score-head-checkpoint/versions/1):21files with heads, four score arrays and two original stage1adapters, all bytes verified; then [E39 delta version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e39-hierarchical-checkpoint/versions/1):33files, all bytes verified.

Restore source first, overlay required private artifacts, attach official train.csv and pinned backbones only where needed. Public E37 progress is aggregate-only and does not prove local weights are present. Source-cache provenance limits remain. No raw passages, credentials or numeric model weights are uploaded publicly. Main is preserved. The index is named CHECKPOINT_INDEX.md to avoid README.md/readme.md collisions on Mac.
