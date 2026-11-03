# Name That LLM — verified research checkpoint

Official best verified public balanced accuracy: **0.665**. Target0.99 remains unachieved. No new submission from E15–E22.

- [Read the current handoff and measured experiments](readme.md)
- [Executed EDA and experiment notebook](ntl_session_20261103.ipynb)
- [Source archive](ntl_research_checkpoint_20261103.zip)
- [SHA-256 manifest](checkpoint_manifest_20261103.json)

E19/E20 fusion gains did not exclude zero; E21 did not improve the stronger LoRA baseline. Agent500 data replacement was rejected in validation on all five rotations. All thirteen report code cells executed successfully.

This branch preserves main and carries the current audit. Existing pipeline files inherited from main are historical; their names and aspirational scores are not measured leaderboard results. The current handoff records limitations and rejected methods.

Latest official result: previous Apex V3 submission57930534 scored **0.055**. Its fatal-exception-to-constant-prediction defect was reproduced on synthetic IDs. A guarded repair is archived as UNVALIDATED, pending real GPU dryrun; champion V1 remains0.665.
