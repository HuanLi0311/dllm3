# CAGD anchor-budget sensitivity protocol

Status: frozen before execution on 2026-09-09.

## Question

How does continual retention change as the number of condition anchors grows
under an otherwise matched CAGD training protocol? This is a sensitivity test
of the finite-anchor prediction, not a numerical verification of the coverage
bound.

## Fixed design

- Model and stream: the SMDM-219M four-task factual benchmark used in the main
  experiment, in both forward and exact-reverse order.
- Method: CAGD only. The existing anchor-64 result is reused.
- Anchor budgets per previous task: `4, 8, 16, 32, 64, 120`.
- Selection: source order, exactly balanced over the four facts in each task.
  The supports are nested and budget 120 contains every training prompt.
- Seed: 3407 for optimization, generation, and evaluation.
- All other settings match the main experiment: full-parameter adaptation,
  1,000 updates per task, batch size 4, learning rate `5e-5`, gradient clipping
  at 1, replay weight 1, 32 reverse-generation steps, and 32 evaluation mask
  draws.
- Endpoints: final average held-out loss, past-task forgetting, and final-task
  loss. Forward and reverse orders are reported separately; there are no error
  bars because this sensitivity study uses one seed.

## Validity checks

A run is excluded if it is incomplete or non-finite, its source or dependency
hash differs from the reused anchor-64 run, any fixed metadata field differs,
the replay count is incorrect, or replay is not exactly balanced over facts.
Exploratory or partial outputs are never read by the paper figure builder.
