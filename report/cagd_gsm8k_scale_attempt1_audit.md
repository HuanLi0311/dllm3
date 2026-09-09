# Five-model GSM8K scale study: attempt 1 audit

Status: invalidated and stopped; no result from this attempt is eligible for
the five-model table.

Attempt 1 launched eight first-wave cells on `air-node-03` under master PID
`1040968`.  It produced zero formal JSON results.  Every launched cell was
stopped and its launcher-recorded exit code is `143`:

| Cell | PID | Exit |
|---|---:|---:|
| SMDM-1.14B, Sequential, seed 3407 | 1041452 | 143 |
| SMDM-1.14B, CAGD, seed 3407 | 1041594 | 143 |
| SMDM-1.14B, Sequential, seed 3408 | 1041827 | 143 |
| SMDM-1.14B, CAGD, seed 3408 | 1042201 | 143 |
| SMDM-1.14B, Sequential, seed 3409 | 1042555 | 143 |
| SMDM-1.14B, CAGD, seed 3409 | 1042823 | 143 |
| Qwen3-4B, Sequential, seed 3407 | 1043301 | 143 |
| Qwen3-4B, CAGD, seed 3407 | 1043902 | 143 |

## Invalidation evidence

The frozen protocol required the two methods in each `(model, seed)` pair to
have an identical stage-0 prefix.  Independently training the SMDM prefix on
different GPUs did not meet that requirement.  The final logged stage-0 loss
for every SMDM-1.14B pair differed:

| Seed | Sequential log line | CAGD log line |
|---:|---|---|
| 3407 | `stage_step=1000/1000 current=2.00853` | `stage_step=1000/1000 current=2.07062` |
| 3408 | `stage_step=1000/1000 current=0.28049` | `stage_step=1000/1000 current=0.36377` |
| 3409 | `stage_step=1000/1000 current=0.30294` | `stage_step=1000/1000 current=0.38472` |

The complete lines, including the zero replay and penalty terms, remain in
`runs/cagd_gsm8k_scale/logs/smdm_1.14b_{seq,cagd}_s{3407,3408,3409}.log`.
The divergence is consistent with a non-bitwise-deterministic SMDM CUDA
training path (FlashAttention is available in this backend), so matching
seeds alone cannot establish a paired prefix.

## Stop and preservation record

`SIGINT` was sent first, but these non-interactive background jobs inherited
an ignored `SIGINT` disposition and continued running.  `SIGTERM` was then
sent to the eight exact PIDs after checking each `/proc/<pid>/cmdline`; the
launcher recorded exit `143` for all cells and returned all eight GPUs to
idle.  No `SIGKILL` was used.

All attempt-1 `.pid`, `.log`, `.exit`, `.gpu.csv`, and `.json.lock` artifacts
are intentionally retained.  The stale locks are evidence of termination and
must not be removed or reused.  No result JSON exists.

## Attempt-2 correction

Attempt 2 trains one canonical SMDM stage 0 per `(checkpoint, seed)`, writes
its model state and complete benchmark record as immutable, hashed artifacts,
and initializes both Sequential and CAGD branches from that same state.  Both
branches embed the canonical stage-0 record rather than recomputing it.  This
makes the paired prefix an enforced data dependency instead of an assumption
about CUDA kernel determinism.
