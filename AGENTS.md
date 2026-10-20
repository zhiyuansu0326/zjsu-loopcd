# Workspace constraints

- Run GPU evaluations only on Slurm partition `scavenger-h200`, using account
  `scavenger-h200`. The user explicitly prohibited `h200-hp`; do not switch
  partitions to shorten queues. Only an explicit new user instruction can
  change this restriction.
- Use the existing `venv/` environment. Full checkpoints belong in allocated
  GPU jobs, not in the interactive editing process.
- Preserve completed results and their source snapshots. Resume partial runs
  only after checking their manifests and saved candidate records.
