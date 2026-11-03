# Name That LLM — verified research checkpoint

Official best verified public balanced accuracy: **0.665**. Target0.99 remains unachieved. No new competition submission from E15–E26.

- [Current handoff and measured experiments](readme.md)
- [Executed EDA and experiment notebook — 17 code cells](ntl_session_20261103.ipynb)
- [Source archive](ntl_research_checkpoint_20261103.zip)
- [SHA-256 manifest](checkpoint_manifest_20261103.json)
- [Q25-v5 contract repair — UNVALIDATED, DO NOT SUBMIT](ntl_apex_v3_contract_fixed_UNVALIDATED.ipynb)

E23 found a concrete V3 loader defect: Q25-v5 adapters target32layers, while V3 instantiated27 and used the wrong readout indices. The author's reference establishes the corrected contract. A CPU-only probe in the exact V3 Docker resolves all six adapters after correction. Full CUDA inference, resource limits and quality remain unverified.

E24 paired ranking did not improve ordinary heads. E25 StyleDistance fusion improved a weaker local branch from0.57352 to0.58751; E26 on stronger LoRA was0.59709 to0.59857 with a confidence interval crossing zero. These are historical local experiments, not measured improvements over the0.665 champion.

Original Apex V3 submission57930534 scored0.055. Its fatal-exception-to-constant-fallback defect was reproduced. Both repairs remain ineligible for submission until real runtime and quality gates pass.

This branch preserves main. Inherited pipeline files and aspirational score names are historical. Numeric features/heads are in private Kaggle storage; this public snapshot excludes credentials and raw competition passages.
