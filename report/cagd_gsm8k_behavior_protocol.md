# GSM8K behavioral-retention protocol

Status: locked before formal execution.

## Question

Does condition-anchored generative distillation preserve an externally scored
language capability, rather than only lowering teacher-forced loss, after a
model learns later language-generation tasks?

## Stream and model

- Model: the frozen local Qwen3-0.6B base snapshot used by the existing natural
  experiment; only its final Transformer block is trainable.
- Task order: GSM8K, Dolly summarization, Dolly creative writing.
- GSM8K uses all 5,250 vendored training examples and all 1,319 official test
  examples.  The prompt is `Question: ...\nAnswer:` and the supervised response
  is the original rationale ending in `#### <answer>`.
- The two Dolly tasks reuse the exact 120 training and 40 held-out rows from the
  frozen natural-task stream.

## Comparison

- Methods: Sequential and CAGD.
- Paired seeds: 3407, 3408, 3409.
- Each task receives 1,000 AdamW updates, batch size 2, learning rate 5e-5,
  gradient clipping at 1, and no weight decay.
- CAGD retains 64 prompt-only anchors per old task, uses greedy teacher
  completions of at most 128 new tokens, temperature 1, and weight beta=1.
- Both methods share initialization, task data, update counts, and seed.

## Endpoints

The primary endpoint is greedy GSM8K final-answer exact match after the last
task, scored on all 1,319 test questions by the final number after `####`, or
the last generated number when the delimiter is absent.  Generation permits at
most 256 new tokens.  We also record exact match immediately after GSM8K,
retention change, answer-format rate, every decoded output, and final-task loss
and answer-token accuracy.

We report the paired seed differences and a two-way seed-by-example bootstrap
interval.  With only three training seeds, results are described as a large and
consistent behavioral improvement only if all paired seeds favor CAGD; formal
significance is claimed only if the predeclared interval excludes zero.

## Qualitative examples

Examples are not selected manually.  Among test questions answered correctly
immediately after GSM8K by the shared seed-3407 model, we sort by official test
index and show the first three for which final Sequential is wrong and final
CAGD is correct.  We report the total number of such cases and the reverse
count (Sequential correct, CAGD wrong).  If fewer than three exist, all are
shown and the criterion is declared unmet.

Smoke runs use separate output paths and cannot enter the formal summary.
