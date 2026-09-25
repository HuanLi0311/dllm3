# SMDM GSM8K soft-target intervention

Status: frozen

## Question

For an SMDM-1.14B checkpoint with non-floor GSM8K behavior, does CAGD soft
distillation preserve generated behavior better than hard generated replay
when prompts, teacher completions, initialization, and update budget match?

## Frozen design

- Model: SMDM-1.14B, full-parameter fine-tuning.
- Shared stage 0: the official GSM8K-SFT checkpoint
  `mdm-1028M-3300e18-rsl-gsm8k.safetensors`, SHA-256
  `1e968c26419d5b041adf3b1825e6d2b10887c45cdab76e60ea2d8341df31618f`.
  It must achieve at least 10% exact match on the full 1,319-example GSM8K
  test set under the experiment decoder before any branch may start.
- Replay support: one immutable set of completions generated from the first
  64 standard GSM8K training prompts. All six branches load this exact replay
  artifact and the exact same stage-0 checkpoint.
- Branch seeds: 3407, 3408, and 3409 for each of `hard_replay` and `cagd`.
- Stream after stage 0: the existing prompt-disjoint Dolly summarization task.
- `hard_replay`: native one-hot masked-diffusion loss on the saved completion
  tokens.
- `cagd`: token-level teacher KL on the same saved completion trajectories,
  using the shared stage-0 checkpoint as the frozen teacher.
- Branch training: 1,000 updates on the same 120 Dolly summarization rows,
  batch size 2, learning rate 5e-5, no weight decay, gradient clipping at 1,
  replay weight 1, and temperature 1. Current and replay RNGs are paired by
  seed.
- Replay generation uses 32 diffusion steps, CFG 0.8, and at most 128 new
  tokens. GSM8K evaluation uses 256 steps, CFG 0.1, and at most 256 new
  tokens. Final summarization loss uses 16 fixed Monte Carlo samples.

The stage-0 JSON, checkpoint digest, anchor manifest, replay-row digest, data
digests, tokenizer digest, and runtime settings must be identical across all
six branches. Existing outputs are never overwritten.

## Endpoints

Primary endpoints are final GSM8K exact match, retention conditional on being
correct at stage 0, and final Dolly summarization loss. Dolly answer-token
accuracy, delimiter rate, and wall time are secondary. All contrasts are
paired as `cagd - hard_replay` within branch seed.
