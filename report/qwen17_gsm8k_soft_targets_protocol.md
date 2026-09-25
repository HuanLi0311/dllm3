# Qwen GSM8K soft-target intervention

Status: frozen

## Scale extension

This is the exact Qwen3-0.6B soft-target intervention repeated with
Qwen3-1.7B.  The only intentional model-level change is the frozen local
Qwen3-1.7B snapshot; all data, seeds, full-parameter training settings,
generated-support controls, endpoints, and paired audits remain unchanged.

- Model: Qwen3-1.7B, full-parameter fine-tuning.
- Seeds: 3407, 3408, and 3409.
- Stream: GSM8K followed by Dolly summarization.
- Each seed has one shared 1,000-update GSM8K checkpoint and one immutable set
  of greedy completions from the first 64 GSM8K training prompts.
- `hard_replay` applies one-hot causal loss to those saved completions.
- `cagd` applies token-level teacher KL to the same completion trajectories,
  using the shared checkpoint as the frozen teacher.
- Each branch receives 1,000 summarization updates with batch size 2, learning
  rate 5e-5, replay weight 1, temperature 1, and gradient clipping at 1.
- Evaluation covers all 1,319 GSM8K test questions and the 40 summarization
  test rows.

Primary endpoints are final GSM8K exact match, retention conditional on being
correct at stage 0, and final summarization loss.  Wall time is secondary.
Contrasts are paired as `cagd - hard_replay` within seed.
