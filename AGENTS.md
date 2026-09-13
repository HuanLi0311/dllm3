# Experiment workspace

- Active training/evaluation code lives in `reproduction/`; use its two
  aggregate launchers documented in `README.md`.
- Organize runs by paper figure/table, exporting all required metrics per run.
  Preserve settings, metric definitions, and layouts when changing model scope.
- Keep all historical results, checkpoints, logs, protocols, and evidence bundles.
  Do not overwrite existing runs or reinterpret old records as new runs.
- Legacy implementations are archived in
  `release_evidence/legacy_code_20260914.tar.gz`; do not restore runtime
  dependencies on them. `third_party/SMDM` remains a required upstream backbone.
- Cleanup here does not authorize changes to manuscripts in `../assets/`.
- Prefer the smallest implementation using existing dependencies. Preserve
  runnable checks for non-trivial changes.
- After runner changes, run `python -m reproduction.suite_common`,
  `python -m reproduction.trace self-check`, and both launcher dry runs using
  the interpreter paths in `README.md`; do not start training just to check cleanup.
