# Slice fidelity and forgetting protocol

Status: frozen before launch on 2026-09-26.

## Question

Within the same model, task pair, replay cache, optimization seed, and
regularization scale, does the held-out Fisher approximation that is more
faithful on a parameter slice also produce less final forgetting when the
regularizer is restricted to that slice?

## Design

- Model: SMDM-219M; all 219,050,496 parameters remain trainable.
- Tasks: `d2p_8-11 -> p2d_12-15`, using the R23 data, answer-only loss,
  1,000 AdamW steps per task, batch size 4, learning rate `5e-5`, and clip 1.
- Replay: the R23 soft generative replay protocol with 64 Task-A prompts,
  balanced 16/16/16/16 across facts. Within a seed/slice cell, Rank-1+GD and
  Diagonal+GD share the exact post-Task-A state, frozen teacher, generated
  replay rows, Task-B minibatch stream, and mask stream.
- Seeds: 3407, 3408, and 3409.
- Slices: attention (including `norm_1`) and MLP (including `norm_2`) in
  layers 0, 8, and 17, for six slices and 18 paired cells.
- Fidelity: fit mean-gradient rank-1 and empirical-Fisher diagonal matrices
  on the 40 designated Task-A training rows; score both against 40 disjoint
  Task-A test prompts with independent native mask draws. The score is
  `log(error_diagonal / error_rank1)`, positive when rank-1 is more faithful.
- Intervention: apply either matrix only on its measured slice while all
  parameters train. Set the diagonal multiplier to 1,000 and choose the
  rank-1 multiplier per cell so both represented matrices have equal weighted
  trace. This controls total slice stiffness, not directional stiffness.

The primary behavioral contrast is
`forgetting_diagonal - forgetting_rank1`, positive when rank-1 forgets less.
The primary association is Spearman correlation between this contrast and the
held-out fidelity score over the 18 paired cells. A fixed 100,000-draw
permutation test shuffles behavioral contrasts among slices within each seed.
Sign agreement, Pearson correlation, per-seed correlations, final-average-loss
contrasts, and realized-update directional errors are secondary diagnostics.

## Interpretation boundary

This is a controlled causal slice intervention, but it remains one model and
one synthetic two-task stream with three seeds. Slices within a seed share a
Task-A distribution and are not independent model replications. A null result
rules out a strong local fidelity-forgetting link under this setup; it does not
show that Fisher fidelity is never useful. A positive result is exploratory
evidence of alignment, not a universal causal law.

Every cell must reject an existing output, verify disjoint calibration/test
prompts, exact replay balance, equal weighted trace, finite results, and
source/input hashes. The complete 18-cell grid is summarized regardless of
result sign.
