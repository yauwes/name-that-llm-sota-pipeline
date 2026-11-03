# Name That LLM — verified research checkpoint

Official best verified public balanced accuracy: **0.665**. Target0.99 and rank1 remain unachieved. No new competition submission from E15–E32.

- [Current handoff and measured experiments](readme.md)
- [Executed EDA and experiment notebook — 22 code cells](ntl_session_20261103.ipynb)
- [Source archive](ntl_research_checkpoint_20261103.zip) and [SHA-256 manifest](checkpoint_manifest_20261103.json)
- [Clean restore and six cached CPU replays](restore_verification_20261104.json)
- [Ten restored checkpoints reproduce held-out predictions](checkpoint_inference_verification_20261104.json)

E27 style/logit ablation and E28 probability summaries did not establish a replacement for the champion. E29 quantifies the gap between oracle context and inferred groups. E30 improves group recovery on a small Gaussian proxy; E31/E32 show a gain with dense18-response context but uncertain transfer to sparse2-response cases. These are historical local experiments, not measured leaderboard gains.

The private checkpoint contains full fold JSON as well as derived arrays and trained heads. Restore public source first, overlay private data, then attach the official train.csv. Public fold JSON is aggregate-only. Read the handoff for version-specific paths and verification scope.

The Q25-v5 contract repair remains **UNVALIDATED**, pending real CUDA full-branch and quality checks. Original V3 scored0.055; champion V1 remains0.665.

This branch preserves main. Inherited score names are historical. Public uploads exclude credentials, raw competition passages and numeric model weights. Full numeric state is stored privately on Kaggle.
