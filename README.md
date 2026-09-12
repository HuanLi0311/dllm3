# Experiment reproduction

# Download Qwen3-4B
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
