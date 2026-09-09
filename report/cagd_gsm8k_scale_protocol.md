# Five-model GSM8K behavioral-retention protocol

Status: frozen on 2026-09-08 after feasibility smoke; formal launch remains
subject to the requested pre-launch audit.

## Question and matrix

Does prompt-conditioned generative distillation retain usable GSM8K behavior
across the two available SMDM checkpoints and three Qwen3 checkpoints?  The
formal matrix has Sequential and CAGD at seeds 3407, 3408, and 3409.  The six
completed Qwen3-0.6B runs under `cagd_gsm8k_behavior_v1` are reused byte for
byte and must not be rerun.  Missing cells are SMDM-219M, SMDM-1.14B,
Qwen3-1.7B, and Qwen3-4B.

## Shared task stream and endpoints

- Stream: GSM8K, Dolly summarization, Dolly creative writing.
- GSM8K uses all 5,250 vendored training rows and all 1,319 official test
  rows.  Dolly uses the frozen 120-train/40-test rows per task.
- Each task receives 1,000 updates, batch size 2, AdamW with zero weight
  decay, learning rate `5e-5`, gradient clipping at 1, and seed-paired
  minibatches.
- CAGD stores the first 64 prompt-only anchors per prior task, generates at
  most 128 completion tokens from the frozen teacher, and applies unit-weight,
  temperature-1 distribution distillation.  Sequential receives no old-task
  signal.
- Five reported endpoints are post-GSM8K exact match, final exact match,
  final-minus-post-GSM8K retention change, final `####` answer-format rate,
  and final creative-writing task loss.  Exact match uses the same canonical
  numeric parser for both backends: parse the final number following the last
  `####`, or the completion's last number when the delimiter is absent.
- Benchmark decoding returns at most 256 new tokens, stops scoring at the
  first EOS, parses only tokens after the prompt, and stores every decoded
  completion and parsed result.

## Backend-native implementations

- Qwen3-0.6B/1.7B/4B use causal completion loss, greedy decoding, and the last
  Transformer block as the trainable subspace.  The 1.7B and 4B cells delegate
  to `qwen_cagd_gsm8k_behavior.py` with its frozen optimization and evaluation
  settings, including generation batch size 16.
- SMDM uses configuration `Diff_LLaMA_170M` (219,050,496 measured parameters)
  and `Diff_LLaMA_1028M` (1,142,367,744 measured parameters), with all
  parameters trainable.  Training corrupts answer tokens with
  `t ~ Uniform(1e-3,1)` and applies the native importance-weighted denoising
  objective.  Replay uses 32 deterministic denoising steps and CFG 0.8.
  GSM8K evaluation uses 256 deterministic denoising steps and CFG 0.1.
- SMDM generation batch size is 8.  This batching choice changes throughput,
  not the independently computed per-example transfer schedule.
- SMDM examples are grouped only when their tokenized prompt lengths are
  exactly equal.  Each sequence has `prompt_length + token_cap` positions,
  transfer counts are computed independently per example, and only the
  per-example completion slice is decoded.  This removes the padding,
  cross-example transfer, and token-cap ambiguity of a mixed-length batch.
- Final SMDM creative-writing loss is averaged over 16 fixed Monte Carlo mask
  draws; Qwen final loss is exact teacher-forced causal cross-entropy.  Losses
  are therefore interpretable within a checkpoint, not across backends.

## Fixed artifacts

- GSM8K train SHA-256:
  `52ebf7c73927f7434abbb2f7b705a82fb3dbdd4695438b7654de78b701c23b36`.
- GSM8K test SHA-256:
  `8530a3775b96385370842171f226d83de7c5b27d779be54ef0411d255a939818`.
- Dolly stream SHA-256:
  `a3847b527b517a6a778d85c17ad6a597e66c0f27f4137dde25da496092e219a0`.
- SMDM tokenizer tree SHA-256:
  `f2bb6b928f472c0a69142cf12ab7b6d74b7e43050f561eded292ac33a2edf559`.
- SMDM checkpoint SHA-256 values (219M, 1.14B):
  `2d8c9b9a730715f2c772d5bc740e12951fc160e5e8511a16835f3537401ea9bb`,
  `ce96ce67a051613b6d7feb419c99c0b4db5bfcfaaa0833ed7f7ecbc6632841d6`.
- Qwen snapshot commits (1.7B, 4B):
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Qwen snapshot inventory SHA-256 values (1.7B, 4B):
  `ec3da139e35f8bdd716b8f3858ebb9ba5042629b0e5bbd122a283ddfcea48ade`,
  `62d9e88fca7b194225dbc4d7caafc7d64b96608c33fa68103a1cf39e98b187e8`.

## Feasibility gate and reporting

Smoke outputs live only under `runs/cagd_gsm8k_scale/smoke/`.  Before this
document is frozen, smoke tests must show correct prompt/completion slicing,
EOS truncation, parser parity, finite training and generation, and an
SMDM-1.14B CAGD peak below 40 GiB.  Formal outputs live under
`runs/cagd_gsm8k_scale/formal/<model>/<method>/s<seed>.json`; temporary or
failed outputs cannot enter the summary.  The summarizer must fail closed on
missing cells, provenance mismatches, non-identical paired post-GSM8K outputs,
incomplete 1,319-example records, or recomputation errors.

The final feasibility gates preserved the batch sizes and token caps that
determine peak generation shapes while shortening work-count knobs such as
training steps, denoising steps, benchmark cardinality, and (for Qwen3-4B)
the number of replay anchors.  Qwen3-4B used batch 16 with a 256-new-token
benchmark cap and a 128-token replay cap;
its peak was 17,769,640,448 allocated bytes, 27,581,743,104 reserved bytes,
and 26,829 MiB in one-second external sampling.  SMDM-1.14B used exact-length
batches up to 8 with the same caps, all student parameters, and a resident
teacher; its peak was 13,733,395,968 allocated bytes, 18,505,269,248 reserved
bytes, and 18,173 MiB externally.  Both are below the 40-GiB A100 limit.
The audited logs and sampling traces are
`runs/cagd_gsm8k_scale/logs/smoke_qwen3_4b_cagd_formal_shape_s3407_v2*`
and
`runs/cagd_gsm8k_scale/logs/smoke_smdm_1.14b_cagd_formal_shape_s3407_v3*`.
Smoke metrics are used only for correctness and resource feasibility; their
accuracy values neither select hyperparameters nor enter the formal summary.

Formal execution is exactly:

```bash
cd iclr_3
nohup experiments/launch_gsm8k_scale_formal.sh \
  > runs/cagd_gsm8k_scale/logs/formal_launcher.log 2>&1 &
```

The launcher preflights both Python environments, the frozen protocol, eight
idle GPUs, and every output, lock, log, PID, exit, and GPU-sampling path; it
refuses the entire launch if any check fails.  To reduce wall time, all six
SMDM-1.14B cells occupy GPUs 0--5 concurrently, Qwen3-4B cells run in two
three-cell queues on GPUs 6--7, and the shorter Qwen3-1.7B and SMDM-219M cells
follow on GPUs 0--5.  Every GPU runs at most one cell at a time, all cards are
the same A100-PCIE-40GB model, and the summarizer still requires exact paired
post-GSM8K record equality.  The master launcher explicitly waits for all
eight worker PIDs and exits nonzero if any queue fails.  The runner also uses
an exclusive output lock and refuses to overwrite JSON.  A failed or
interrupted cell is inspected and relaunched only under a new, explicit
attempt path; existing artifacts are never deleted automatically.

Frozen implementation SHA-256 values are:

- runner `bed5dc7cf6bcad10d0b333244e494bffd9c5f2b20f5b83d7b3cb5f7e400bd38c`;
- summarizer `8424eded4db62e01c0c575ab3b9e13e065a84df265e8f62817f1a083a3797bb5`;
- launcher `20442a262bcc27645ccec721a1e2a87ba4e185fc1d5bac4d0adbad507ef899e6`.

SMDM runs use Python 3.9.25, PyTorch 2.4.1+cu121, Transformers 4.31.0,
and safetensors 0.4.5.  Qwen runs use Python 3.11.15, PyTorch 2.7.0+cu128,
and Transformers 4.57.6.  All formal computation runs on `air-node-03` with
NVIDIA A100-PCIE-40GB GPUs.  The final summary reports mean and SEM over the
three paired seeds for all five endpoints; it does not compare raw loss
magnitudes across backends.
