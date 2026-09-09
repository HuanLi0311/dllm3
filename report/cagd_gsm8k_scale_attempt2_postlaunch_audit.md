# Attempt-2 post-launch audit log

Formal training was launched with the frozen manifest in
`cagd_gsm8k_scale_attempt2_protocol.md`.  A read-only audit of the first
completed canonical artifact exposed a `NameError` in the offline summarizer:
`_audit_canonical` received `path` but referred to `canonical_path` when
constructing the expected checkpoint path.  This was corrected by changing
that one reference to `path`.

- Launch-time summarizer SHA-256:
  `3ea34aa42a9cc114859f5185764a3e628165f6069d8673ae375273e65b4a7f5d`
- Corrected summarizer SHA-256:
  `ab5220ee0b333cac70a2f76de7fec6edc58428dcc9ab7cbf82e6ec23b798b1ec`

The summarizer is not a dependency of either training runner and is never
loaded by a formal worker, so this correction changes no training,
generation, checkpoint, metric, or provenance record.  The corrected file
compiled, passed its self-check, and validated the first completed formal
SMDM-219M canonical artifact against all 1,319 vendored GSM8K records.

## Completion and recovery

All 24 planned new endpoints completed.  The six existing Qwen3-0.6B
endpoints were reused without rerunning them, giving 30 validated endpoints
in the final five-model matrix.  All six immutable SMDM stage-0 JSON and
checkpoint pairs completed and were independently rehashed.

The original SMDM-1.14B Sequential seed-3408 worker exited before importing
Torch with a transient, concurrently triggered Python `multiprocessing`
import error.  It created no result JSON and did not begin model loading or
training.  Its original log, PID record, GPU trace, nonzero exit record, and
failure traceback remain preserved.  The recovery launcher then created the
previously absent result path and completed that endpoint, Qwen3-1.7B
Sequential seed 3408, and SMDM-219M Sequential seed 3408 with exit status 0.
No prior source, result, or log was deleted or overwritten.

- Recovery launcher SHA-256:
  `6cb171b61c4ede7714b0d8597bfa1cfd46226c72d452a4053c84143448ee42ad`
- Final summary SHA-256:
  `c165182c330d6a7176316e0db593b205947d177d37af73ab0e268e1c41a1413d`
- Rendered results report SHA-256:
  `cc9d6db3e154faaa2b5e110405c16a3231a167fbdb299c0d8125a147a1c04ac7`

The final fail-closed summarizer validated all 30 endpoints, all six SMDM
canonical prefixes, exactly 1,319 benchmark records at each scored stage,
three 1,000-step task records per endpoint, the declared anchor and replay
counts, parser-derived metrics, data and tokenizer hashes, and paired stage-0
equality.  It reports `status: ok`, 24 new runs, six reused runs, and six
validated canonical prefixes.  All formal workers and both launchers had
exited and all eight GPUs were released after completion.  The main launcher
retains a nonzero aggregate status solely because it preserves the original
pre-training import failure; the recovery records supply the unique completed
endpoint at that formal output path.
