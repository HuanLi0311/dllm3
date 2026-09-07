# CAGD natural-task protocol

Status: frozen on 2026-09-07 after implementation smoke tests and before any
formal natural-task run.

## Question and scope

Does condition-anchored generative distillation mitigate forgetting when the
conditions and outputs are natural instructions rather than synthetic factual
associations?  The study tests the same fixed stream with a masked diffusion
language model and an autoregressive language model.  The forward order alone
is predeclared because exact order reversal is already tested in the factual
study; this extension isolates a content and task-format shift.

## Data

- Source: `databricks/databricks-dolly-15k`, revision
  `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, CC BY-SA 3.0.
- Builder: `experiments/build_dolly_stream.py`, SHA-256
  `c5da8fecb05759610d203b95986f2bc52129dc95e0d19866b2664a9bb1f93657`.
- Selection seed: 20260907.  Rows are ordered by a SHA-256 key before taking
  the split; no outcome from a model is used in selection.
- Stream: `closed_qa -> summarization -> creative_writing`.
- Each task has 120 training rows and 40 held-out rows.  All 480 prompts and
  source rows are distinct.
- A row must fit both tokenizers within 256 total tokens and 160 prompt tokens.
  Eligible counts before selection are 473, 237, and 461, respectively.
- Selected-row semantic digest:
  `5083b050bdc3255b2e77c4d51af9fff012d536cdf101a391104a8ab25a3adc88`.
- Materialized JSONL SHA-256:
  `a3847b527b517a6a778d85c17ad6a597e66c0f27f4137dde25da496092e219a0`.

## Methods and matrix

Methods are Sequential, CAGD, and hard generated replay.  CAGD and hard replay
retain the first 64 rows in the already hash-randomized training split of each
old task and greedily generate one completion for every retained prompt.
CAGD matches the frozen teacher distribution on those generated trajectories;
hard replay applies the backend's native one-hot loss to the same type of
trajectory.  Both replay losses have weight 1.  No Fisher or other
parameter-space penalty is used.

Seeds are 3407, 3408, and 3409.  The formal matrix contains
2 backends x 3 methods x 3 seeds = 18 cells.  CAGD and hard replay must have
identical generated-row digests at the first replay transition within each
backend and seed.  Later digests may differ because the learners have diverged.

## Masked diffusion backend

- SMDM checkpoint with 219,050,496 trainable parameters; update all parameters.
- 1,000 AdamW steps per task, batch size 2, learning rate `5e-5`, zero weight
  decay, gradient-norm clip 1.
- Answer-only independent Bernoulli masks with
  `t ~ Uniform(1e-3, 1)`; empty masks contribute zero.
- CAGD uses teacher KL at independently masked generated-answer positions,
  temperature 1.
- Replay uses 32 deterministic reverse steps, context length 256, CFG 0.8,
  and temperature 0.
- Evaluation uses 16 fixed Monte Carlo mask draws per held-out row and batch
  size 2.
- Runner SHA-256:
  `eb2e88ceb8de3aec433187c367b3a73fe534260a8f8504be3b483e87610994cc`.

## Autoregressive backend

- Official Qwen3-0.6B base snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`; update only the final
  Transformer block (15,730,944 parameters).
- 1,000 AdamW steps per task, batch size 2, learning rate `5e-5`, zero weight
  decay, gradient-norm clip 1.
- Answer-only next-token cross-entropy for new data; CAGD uses token-level
  teacher KL with temperature 1 on greedy teacher completions.
- Generate at most 96 new tokens in batches of 4; maximum full sequence length
  is 256.  Evaluation batch size is 2.
- Runner SHA-256:
  `a45839488e8f848069bdea0da5ecddb975305cd1e909ae268cce891377693dd3`.

## Endpoints and interpretation

Primary endpoints are final average held-out answer loss, past-task forgetting
(final loss minus loss when learned, averaged over the first two tasks), and
final-task loss.  Mean, SEM, every seed, and paired differences are reported;
lower is better.  Answer-token accuracy is secondary.

The primary contrasts are CAGD minus hard generated replay and CAGD minus
Sequential within backend and seed.  A retention gain is interpreted only
alongside final-task loss.  Cross-backend agreement concerns direction, not
effect-size equality, because the objectives and trainable parameter fractions
differ.

## Audit and invalidation

Smoke artifacts under `runs/cagd_natural/smoke` are development checks and are
excluded.  A formal cell is invalid if the source, protocol, materialized data,
manifest, model snapshot, task order, seed, method, or a locked hyperparameter
changes during execution; a required endpoint is missing or non-finite; the
matrix is incomplete or contains an extra cell; or the first-transition CAGD
and hard-replay generated digests do not match within backend and seed.
