# Qwen3-0.6B exact-reverse order extension

Status: frozen on 2026-09-12 before retained reverse-order runs.

## Question

Does the autoregressive CAGD result survive moving the same factual task from
the end of a continual stream to its beginning?

## Matrix

- Model: official Qwen3-0.6B base checkpoint.
- Methods: Sequential and CAGD (`gd` in the original runner).
- Seeds: 3407, 3408, and 3409.
- Existing forward stream: `d2p_8-11 -> p2d_12-15 -> d2p_16-19 -> p2d_20-23`.
- New reverse stream: its exact reversal.
- Fixed task: `p2d_20-23`, evaluated after the complete stream in both orders.

The six new cells use the existing factual Qwen configuration: final
Transformer block, 1,000 AdamW updates per task, batch size 4, learning rate
`5e-5`, gradient-norm clipping at 1, zero weight decay, 64 balanced prompt
anchors per old task, greedy teacher generation, and token-level teacher KL
with weight and temperature 1.  The existing archived forward cells are not
rerun.

## Endpoints

The primary endpoint is fixed-task final loss on `p2d_20-23`.  Supporting
endpoints are final average loss, past-task forgetting, last-task loss, and
the fixed task's reverse-order loss immediately after learning.  The latter
gives within-run fixed-task forgetting; reverse-minus-forward fixed-task loss
measures position sensitivity.

## Audit

A run is invalid if its method, seed, task sequence, model, trainable scope,
or locked hyperparameters differ; an endpoint is absent or non-finite; the
source or this protocol changes while a run is active; or a task's encoded
train/evaluation hash differs from the archived forward cell with the same
method and seed.  Results are summarized only after all six reverse cells
complete.

- Reverse wrapper SHA-256: `8408131a0dcf8a07a3875a4a5c349c495876c1f89407c697ea8ec769c67c773a`.
- Reused base runner SHA-256: `a8a48d15340d01b2261f0eba8551e42fd138fba10a73359794eba251c57947c0`.
