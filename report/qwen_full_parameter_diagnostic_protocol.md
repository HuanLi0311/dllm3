# Qwen3-0.6B full-parameter diagnostic

Status: frozen before launch on 2026-09-10.

## Question

Does the direction of the existing Qwen result survive when all parameters,
rather than only the final Transformer block, are optimized?

## Paired diagnostic

- Official Qwen3-0.6B base checkpoint, seed 3407.
- Existing four-task forward factual stream:
  `d2p_8-11 -> p2d_12-15 -> d2p_16-19 -> p2d_20-23`.
- Sequential versus GD (the AR specialization of CAGD).
- All model parameters trainable.
- Otherwise identical to the existing scale experiment: 1,000 AdamW steps per
  task, batch size 4, learning rate `5e-5`, zero weight decay, clipping at 1,
  answer-only causal cross-entropy, and the same held-out prompts.
- GD retains 64 balanced prompts per previous task, greedily generates at most
  32 tokens with the frozen pre-task teacher, and matches teacher distributions
  on the generated answer tokens with temperature and weight 1.

The primary paired endpoints are final average held-out answer loss and
past-task forgetting. Final average answer-token accuracy and final-task loss
check whether retention was purchased by blocking new-task acquisition.

## Decision rule and scope

The full-parameter result supports the existing qualitative conclusion if GD
beats Sequential on both primary endpoints without materially degrading the
final task. This is a one-seed sensitivity diagnostic, not a replacement for
the three-seed last-block result and not evidence about full-parameter 4B
training. No manuscript file is changed by this diagnostic.
