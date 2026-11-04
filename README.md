# Name That LLM — verified research checkpoint

Official best balanced accuracy: **0.665**. Target0.99/rank1 is unachieved and cannot be guaranteed. No new competition submission from this research session.

- [Indonesian handoff, validation caveats and full experiment history](readme.md)
- [Executed EDA/research notebook — 29 code cells](ntl_session_20261103.ipynb)
- [478-file source snapshot](ntl_research_checkpoint_20261103.zip) and [SHA-256 manifest](checkpoint_manifest_20261103.json)
- [E37 private snapshot: all10 remote members verified](e37_private_verified.json), [actual CPU inference parity](e37_cpu_snapshot_parity.json) and [actual MPS epoch-resume parity](e37_resume_parity.json)
- [E38: all21 remote members verified](e38_private_verified.json) and [ten remote heads reproduce predictions](e38_remote_head_inference_verified.json)
- [E39: all33 remote members verified](e39_private_verified.json) and [ten remote heads reproduce predictions](e39_remote_head_inference_verified.json)
- [Prior E33–E35 isolated restore/replay proof](context_identity_restore_verification.json)

E38 nonlinear score heads: inner-selected fixed fusion0.59562 versus baseline0.59709; E39 soft provider×model heads:0.59451 versus0.59709. Neither improves the baseline reliably; neither is promoted. These are head-only experiments on historical caches, not championCV or hidden-test predictions. Source-cache provenance limitations remain.

**E37 local MiniLM training is still active**, with3/5folds complete at this snapshot; do not duplicate its process or alter signature-bound source. Eight-epoch budget is not proof of convergence. The first private snapshot contains inner0epoch8/refit0epoch3, not the newer complete folds. Public progress_summary is aggregate-only and must not become a resumable fold cache. Another agent owns the TPU job; this session does not control it.

Restore source first, then the private checkpoints needed by the experiment:

- [Base numeric version4](https://www.kaggle.com/datasets/ajikajadeh/ntl-reader-e19-e22-checkpoint/versions/4) and [E33–E35 delta version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e33-e35-research-checkpoint/versions/1).
- [E37 partial training snapshot version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e37-minilm-checkpoints/versions/1), plus pinned official MiniLM backbone.
- [E38 heads, four score arrays and two original stage1adapters version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e38-score-head-checkpoint/versions/1), then [E39 small delta version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e39-hierarchical-checkpoint/versions/1).

Attach official train.csv. Pretrained upstream backbones and raw passages are not redistributed. Q25-v5 repair remains **UNVALIDATED** for full CUDA serving/quality. OriginalV3 scored0.055; championV1 remains0.665. Public uploads exclude credentials, raw passages and numeric model weights. This branch preserves main.
