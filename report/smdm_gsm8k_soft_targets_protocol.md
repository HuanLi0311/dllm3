# SMDM GSM8K soft-target intervention

Status: frozen

## Question

On matched masked-diffusion language models, do CAGD soft targets preserve
generated GSM8K behavior better than hard generated replay when prompts,
teacher completions, initialization, and update budget are identical?

## Frozen design

- Models: SMDM-219M and SMDM-1.14B, full-parameter fine-tuning.
- Seeds: 3407, 3408, and 3409.
- Stream: GSM8K followed by the existing prompt-disjoint Dolly summarization
  task.  The two-task stream keeps the teacher and generated support identical
  between the two branches.
- Stage 0: 1,000 AdamW updates on all 5,250 GSM8K training rows, followed by
  masked-diffusion evaluation on all 1,319 official test questions.
- Shared artifacts: each model/seed writes one immutable stage-0 checkpoint
  and one set of completions generated from the first 64 GSM8K training
  prompts.  Both branches load those exact artifacts.
- `hard_replay`: native one-hot masked-diffusion loss on the saved completion
  tokens.
- `cagd`: token-level teacher KL on the same saved completion trajectories,
  using the shared stage-0 checkpoint as the frozen teacher.
- Branch training: 1,000 updates on the same 120 Dolly summarization rows,
  batch size 2, learning rate 5e-5, no weight decay, gradient clipping at 1,
  replay weight 1, and temperature 1.  Current and replay RNGs are paired by
  seed.
- Replay generation uses 32 diffusion steps, CFG 0.8, and at most 128 new
  tokens.  GSM8K evaluation uses 256 steps, CFG 0.1, and at most 256 new
  tokens.  Final summarization loss uses 16 fixed Monte Carlo samples.

The stage-0 JSON, checkpoint digest, anchor manifest, replay-row digest, data
digests, tokenizer digest, and runtime settings must match within each paired
seed.  Existing outputs are never overwritten.

## Endpoints

Primary endpoints are final GSM8K exact match, retention conditional on being
correct at stage 0, and final Dolly summarization loss.  Dolly answer-token
accuracy, delimiter rate, and wall time are secondary.  All contrasts are
paired as `cagd - hard_replay` within seed.
