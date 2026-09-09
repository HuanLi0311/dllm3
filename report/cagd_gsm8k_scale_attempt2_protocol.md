# Five-model GSM8K behavioral retention: attempt-2 protocol

Status: frozen

## Reason for attempt 2

Attempt 1 assumed that independently seeded SMDM stage-0 runs would be
bitwise identical.  Its three SMDM-1.14B pairs violated that assumption, so
the attempt was stopped with zero result JSONs.  Attempt 2 makes the common
prefix an explicit immutable dependency.

## Matrix and reuse

The endpoint matrix remains two methods (`seq`, `cagd`) by three paired seeds
(3407, 3408, 3409) for SMDM-219M (configuration name 170M; measured
219,050,496 parameters), SMDM-1.14B (configuration name 1028M; measured
1,142,367,744 parameters), Qwen3-1.7B, and Qwen3-4B.  The six completed
Qwen3-0.6B v1 results are reused and are never rerun.

Qwen endpoints retain the frozen v1 AR runner and settings.  Only SMDM uses
the paired-prefix runner introduced here.

## Canonical SMDM stage 0

For each `(SMDM checkpoint, seed)`, one `stage0` process trains GSM8K and
evaluates all 1,319 GSM8K test examples.  It atomically writes a safetensors
model-state checkpoint, then atomically writes a JSON manifest last.  The
manifest records the checkpoint path, byte count and SHA-256; base checkpoint
path and SHA-256; runner, protocol and dependency hashes; data and tokenizer
hashes; actual settings; model identity and measured parameter count; seed;
complete training statistics; and every decoded benchmark record.

Both Sequential and CAGD branches independently recompute and validate every
provenance field and the canonical checkpoint bytes/SHA.  They load the same
canonical checkpoint and embed a verbatim copy of the same stage-0 record.
Neither branch trains or evaluates stage 0 again.  The summarizer must verify
that paired endpoints name the same JSON/checkpoint hashes and contain an
identical stage-0 object.

## Locked formal settings

The data, task order, parsing, metrics, optimizer, training counts, masking,
generation, and provenance are unchanged from
`cagd_gsm8k_scale_protocol.md`: GSM8K -> Dolly summarization -> Dolly creative
writing; 1,000 updates per task; batch 2; learning rate 5e-5; no weight decay;
gradient clip 1; 64 prompt anchors per prior task; replay cap 128 tokens with
32 denoising steps and CFG 0.8; GSM8K cap 256 tokens with 256 denoising steps,
CFG 0.1 and generation batch 8; all SMDM parameters trainable; 16 Monte Carlo
masks for final loss.  Completion parsing uses only tokens after the prompt,
truncates at the first EOS, and uses the final numeric value after `####` or,
when absent, the completion's final numeric value.

The five reported metrics remain GSM8K exact match after stage 0, final GSM8K
exact match, retention change, final `####` format rate, and final
creative-writing loss.  Smoke accuracy is not used for hyperparameter
selection and never enters formal summaries.

## Execution and preservation

All attempt-2 stage artifacts, endpoints, logs, PID files, exit files, GPU
samples, and summaries use `runs/cagd_gsm8k_scale/attempt2/`.  Existing v1,
smoke, reused Qwen3-0.6B, and invalid attempt-1 artifacts are read-only.
Existing outputs or locks cause a hard failure.

The formal launcher will use a two-phase schedule.  GPUs 0--5 first build the
six SMDM canonical prefixes in parallel, while GPUs 6--7 independently run
the Qwen3-4B queues.  SMDM branches start only after all six canonical jobs
succeed.  GPUs 0--5 then run the six SMDM-1.14B branches, followed by the six
Qwen3-1.7B endpoints and the six SMDM-219M branches.  Every GPU runs at most
one cell at a time, worker failures propagate to the master, and no dependent
branch launches after a canonical failure.

## Pre-formal gates

Required before freezing:

1. Both supported Python environments compile the paired runner and its
   runnable self-check passes.
2. A short SMDM-219M canonical stage and both branches complete; paired branch
   JSONs reference the same artifact hashes and have byte-equivalent stage-0
   objects.
3. An SMDM-1.14B CAGD branch gate exercises the formal batch and token-cap
   shapes with resident student and teacher below the 40 GiB device limit.
4. The summarizer rejects a tampered canonical record/hash, audits all 1,319
   formal benchmark rows against vendored GSM8K, and checks exact task order.
5. The complete command map, source hashes, external peak GPU samples, and
   launcher preflight are reviewed before `Status` changes to `frozen`.

The SMDM-219M smoke-v2 gate passed with one canonical prefix and both branches.
The independent audit recomputed the result fields and verified that
`seq.stages[0] == cagd.stages[0] == canonical.stage`; its canonical JSON SHA-256
is `a6295315fe8972f1debb8f6c856dfbbf14e83ddcde102549af37ea24d21ed4d7`
and checkpoint SHA-256 is
`9690660c7c72b1fcb0c0c5a1102d82abe69ed68bbe8eba484e6113b0dbeb4baa`.
These smoke files are not formal inputs.

The SMDM-1.14B CAGD formal-shape gate passed with generation batch 8,
benchmark/replay caps 256/128, and replay counts 64 then 128.  The observed
maximum generation increment was 8.  PyTorch peak allocation/reservation in
the branch was 13,737,754,112/14,627,635,200 bytes; the canonical process peak
reservation was 18,503,172,096 bytes.  All are below one 40 GiB A100.  The
gate shortened only work-count knobs and its accuracy is not a formal result.

## Frozen implementation manifest candidate

The launcher recomputes these hashes during preflight and refuses a mismatch:

```
05689a1b7d67433f2f474ed3371f148fcdadada9580339bb3c05012f6e6647da  experiments/smdm_cagd_gsm8k_paired.py
fb0d65e65b5ca2702363910c5fb50b9f0ffc8b073cfde9106942273c857a9aab  experiments/audit_gsm8k_scale_attempt2.py
3ea34aa42a9cc114859f5185764a3e628165f6069d8673ae375273e65b4a7f5d  experiments/summarize_gsm8k_scale_attempt2.py
a27e608682380e28c95328ef6efe6fd18ddaf50f2099244f13d60f0a5ac1d5a0  experiments/launch_gsm8k_scale_attempt2.sh
bed5dc7cf6bcad10d0b333244e494bffd9c5f2b20f5b83d7b3cb5f7e400bd38c  experiments/cagd_gsm8k_scale.py
b51aac8a6fd9eecb1727979a7d19cbcee1d66fd705422c0d7412bcef83178d16  report/cagd_gsm8k_scale_protocol.md
```

## Formal paths and commands

Six canonical manifests and checkpoints will be created only at
`runs/cagd_gsm8k_scale/attempt2/stage0/{smdm_1.14b,smdm_219m}/s{3407,3408,3409}/stage0.{json,safetensors}`.
The 24 new endpoints will be created only at
`runs/cagd_gsm8k_scale/attempt2/formal/{smdm_1.14b,smdm_219m,qwen3_1.7b,qwen3_4b}/{seq,cagd}/s{3407,3408,3409}.json`.
Each job also creates the corresponding `.pid`, `.log`, `.exit`, and
`.gpu.csv` stem under `runs/cagd_gsm8k_scale/attempt2/logs/`.  The audited
aggregate will be written to `runs/cagd_gsm8k_scale/attempt2/summary.json` and
`report/cagd_gsm8k_scale_attempt2_results.md`.  Existing paths fail closed.

Read-only review commands are:

```
cd /home/JJ_Group/lih2511/test/dllm/iclr_3
experiments/launch_gsm8k_scale_attempt2.sh --print-plan
ssh air-node-03 'cd /home/JJ_Group/lih2511/test/dllm/iclr_3 && experiments/launch_gsm8k_scale_attempt2.sh --preflight'
```

After final approval and only after the status is frozen, the single launch
command is:

```
ssh air-node-03 'cd /home/JJ_Group/lih2511/test/dllm/iclr_3 && mkdir -p runs/cagd_gsm8k_scale/attempt2/logs && test ! -e runs/cagd_gsm8k_scale/attempt2/logs/formal_launcher.log && nohup experiments/launch_gsm8k_scale_attempt2.sh > runs/cagd_gsm8k_scale/attempt2/logs/formal_launcher.log 2>&1 < /dev/null & echo $!'
```

After all jobs exit successfully, the one summary command is:

```
cd /home/JJ_Group/lih2511/test/dllm/iclr_3
PYTHONNOUSERSITE=1 /home/JJ_Group/lih2511/.conda/envs/VisualSim2Real/bin/python experiments/summarize_gsm8k_scale_attempt2.py
```
