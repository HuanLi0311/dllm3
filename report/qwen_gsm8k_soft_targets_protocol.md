# Qwen GSM8K soft-target intervention

Status: frozen

## Question

Do CAGD soft targets preserve generated GSM8K behavior better than hard
generated replay when prompts, teacher completions, initialization, and update
budget are identical?

## Frozen design

- Model: Qwen3-0.6B, full-parameter fine-tuning.
- Seeds: 3407, 3408, and 3409.
- Stream: GSM8K followed by the existing prompt-disjoint Dolly summarization
  task. This is deliberately a two-task intervention: adding the later
  creative-writing boundary would make the preceding teacher method-dependent
  and would no longer hold generated support fixed.
- Stage 0: 1,000 AdamW updates on all 5,250 GSM8K training rows, followed by
  greedy evaluation on all 1,319 official test questions.
- Shared artifacts: each seed writes one immutable stage-0 checkpoint and one
  set of greedy completions from the first 64 GSM8K training prompts. Both
  branches load those exact artifacts.
- Branches:
  - `hard_replay`: native one-hot causal loss on the saved completion tokens.
  - `cagd`: token-level teacher KL on the same saved completion trajectories,
    using the shared stage-0 checkpoint as the frozen teacher.
- Branch training: 1,000 AdamW updates on the same 120 Dolly summarization
  rows, batch size 2, learning rate 5e-5, no weight decay, gradient clipping
  at 1, replay weight 1, and temperature 1. Current and replay minibatch RNGs
  are paired by seed.
- Generation: greedy, at most 128 new tokens for replay and 256 for GSM8K
  evaluation. Maximum training sequence length is 576.

The stage-0 JSON, checkpoint digest, anchor manifest, replay-row digest, data
digests, model inventory, and runtime settings must match within each paired
seed. Outputs are immutable; an existing output is never overwritten.

## Endpoints

Primary:

1. Final GSM8K exact match.
2. Conditional retention among questions answered correctly at stage 0:
   `count(correct_stage0 and correct_final) / count(correct_stage0)`.
3. Final teacher-forced Dolly summarization loss.

Secondary:

- Dolly answer-token accuracy.
- Final GSM8K `####` delimiter rate.
- Adaptation-only and complete branch wall time; the shared stage-0 cost is
  reported separately.

All reported contrasts are paired as `cagd - hard_replay` by seed. Higher is
better for exact match and conditional retention; lower is better for Dolly
loss and wall time.
