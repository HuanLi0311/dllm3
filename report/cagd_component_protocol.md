# Condition-anchored distillation component protocol

Status: frozen before all non-smoke component runs on 2026-09-07.

## Question

Does soft teacher-distribution matching improve continual retention beyond
hard replay on generated completions, and how does it compare with replaying
stored real prompt--answer pairs?

## Methods

- `cagd`: retain balanced old prompts, greedily generate completions with the
  method's frozen previous-stage model, and minimize teacher--student KL on
  independently masked completion positions.
- `hard_replay`: retain the same balanced prompts and use the same generation
  procedure, but train on generated completions with the native one-hot DLLM
  loss.
- `real_replay`: retain balanced real prompt--answer rows and train on them
  with the same native one-hot DLLM loss.

All replay losses have weight 1. No Fisher or other parameter-space penalty is
used. Sequential and joint-training references are reused from the audited R16
matrix rather than recomputed.

## Locked studies

### Two-task component isolation

- Stream: `d2p_4-5 -> p2d_6-7`.
- Purpose: after Task 1, all three methods start Task 2 from the same model;
  `cagd` and `hard_replay` therefore use identical prompt anchors, teacher
  snapshot, generated completions, masks, and minibatch indices. Their only
  intentional difference is soft KL versus one-hot replay loss.
- Methods: `cagd`, `hard_replay`, `real_replay`.
- Seeds: 3407, 3408, 3409.

### Four-task deployed comparison

- Forward stream: `d2p_8-11 -> p2d_12-15 -> d2p_16-19 -> p2d_20-23`.
- Reverse stream: its exact reversal.
- Purpose: compare the methods as deployed continual learners, allowing each
  method's previous-stage model and generated completions to evolve naturally.
- Methods: `cagd`, `hard_replay`, `real_replay`.
- Seeds: 3407, 3408, 3409.

### Fresh-fact sensitivity

- Forward stream: `d2p_24-25 -> p2d_26-27 -> d2p_28-29`.
- Reverse stream: its exact reversal.
- Methods and seeds match the four-task comparison.
- This tests sensitivity to the remaining previously unused fact groups; it is
  not a task-count-matched replication of the four-task study.

## Shared training and evaluation

- Model: SMDM checkpoint with 219,050,496 trainable parameters.
- Training: all parameters, 1,000 AdamW steps per task, batch size 4, learning
  rate `5e-5`, gradient-norm clip 1, and zero weight decay.
- DLLM objective: answer-only independent Bernoulli masking with
  `t ~ Uniform(1e-3, 1)`; empty masks contribute zero.
- Replay: 64 rows per previous task, balanced exactly over facts; 32 reverse
  generation steps and greedy decoding for generated replay.
- Evaluation: 32 paired mask draws per test template plus greedy generation.
- Statistical unit: one optimization seed. Report paired differences, means,
  SEM, and all individual seed values.

## Primary comparisons

1. `cagd - hard_replay` final average held-out loss.
2. `cagd - hard_replay` past-task forgetting.
3. Final-task loss for the same contrast, to expose stability--plasticity
   trade-offs.

The two-task contrast is the clean component test. The longer streams measure
end-to-end behavior, not a fixed-replay causal contrast after methods diverge.
`real_replay` is a storage-rich reference rather than a privacy or compute
matched baseline.

## Audit and invalidation

A run is invalid if its source or protocol changes while active, the task
sequence or any locked hyperparameter differs, replay is not balanced, a
required result is missing or non-finite, or the two-task `cagd` and
`hard_replay` generated-row hashes differ within a seed. Smoke outputs under
`runs/cagd_smoke` are development artifacts and are never summarized as
evidence.
