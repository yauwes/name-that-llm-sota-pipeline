# Name That LLM — verified research checkpoint

Verified official best balanced accuracy: **0.665**. Target0.99/rank1 remains unachieved. No new competition submission from E15–E35.

- [Current Indonesian handoff and full experiment history](readme.md)
- [Executed EDA and research notebook — 25 code cells](ntl_session_20261103.ipynb)
- [411-file source snapshot](ntl_research_checkpoint_20261103.zip) and [SHA-256 manifest](checkpoint_manifest_20261103.json)
- [E33–E35 isolated restore and three cached pipeline replays](context_identity_restore_verification.json)
- [Five restored identity heads reproduce held-out predictions](identity_checkpoint_inference_verification.json)
- [All43 new remote checkpoint members downloaded and SHA-256 verified](private_delta_roundtrip_verified.json)
- [Clean restore and six cached CPU replays](restore_verification_20261104.json)
- [Ten restored checkpoints reproduce held-out predictions](checkpoint_inference_verification_20261104.json)

E34 tests a same-source pair verifier with prompt-disjoint inner selection and fixed fusion weight. It reduces mean test0.59709→0.58153 and is rejected. E33/E35 test3530–4040row grouping contexts with frozen thresholds: primary target predictions stay unchanged; these fixtures do not establish a leaderboard gain. Historical gold has been inspected repeatedly. None of these scores is a fresh hidden-test estimate or a replacement for the0.665champion.

Private storage: [base numeric checkpoint version4](https://www.kaggle.com/datasets/ajikajadeh/ntl-reader-e19-e22-checkpoint/versions/4) plus [E33–E35 delta version1](https://www.kaggle.com/datasets/ajikajadeh/ntl-e33-e35-research-checkpoint/versions/1). Restore latest public source, overlay basev4 then deltav1, attach official train.csv. The13.29MiB delta has43members including5head files and a hash-only context index, so audit replay does not require raw external paragraphs. The prior base retains355entries/50joblibfiles. Public fold JSON is aggregate-only.

The Q25-v5 contract repair remains **UNVALIDATED** for full CUDA serving and quality. OriginalV3 scored0.055; championV1 remains0.665. GPU quota was0h and TPU20h at the4NovemberAPIcheck. E33–E35 ran locally and all own jobs terminated.

This branch preserves main. Inherited score names are historical. Public uploads exclude credentials, raw competition passages and numeric model weights. Full numeric state is stored privately on Kaggle.
