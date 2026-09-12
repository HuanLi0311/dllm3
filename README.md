# Experiment reproduction

## Environment

- CUDA GPUs and the two existing environments: `smdm-baseline` for SMDM and
  `opr` for Qwen/TRACE. Override them with `SMDM_PYTHON` and `AR_PYTHON`.
- SMDM stack: Python 3.9, PyTorch 2.4.1, Transformers 4.31.0,
  tokenizers 0.13.3, Lightning 2.1.2, xFormers 0.0.28.post1.
- Qwen/TRACE stack: Python 3.10, PyTorch 2.8.0+cu128, Transformers 4.57.6,
  TRL 0.24.0, vLLM 0.11.0, DeepSpeed 0.18.2, and Liger Kernel.
- Cached SMDM-219M/SMDM-1.14B and Qwen3-0.6B/1.7B/4B checkpoints at the
  paths declared in `reproduction/suite_common.py`.
- Run commands from `/home/JJ_Group/lih2511/test/dllm/iclr_3`.


## TRACE large experiment: training and evaluation

The launcher executes all eight stages and their evaluations for Sequential,
Vanilla Replay (`replay`), SDFT, OPR-RU (`opr`), OPR-SC (`opr_sc`), and CAGD,
then emits one comparison `summary.json` per seed and a three-seed aggregate.

```bash
nohup env TRACE_SEEDS="3407 3408 3409" \
  TRACE_GPUS="0,1,2,3,4,5,6,7" TRAINABLE_SCOPE=all \
  TRACE_METHODS="sequential replay sdft opr opr_sc cagd" \
  TRACE_RUN_ROOT=runs/reproduction/trace_full \
  bash reproduction/run_trace_experiment.sh \
  > runs/reproduction/trace_full.log 2>&1 &
```

Print the TRACE matrix without training:

```bash
TRACE_SEEDS="3407" \
  bash reproduction/run_trace_experiment.sh --dry-run
```
