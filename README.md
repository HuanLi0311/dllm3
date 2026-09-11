# Experiment reproduction

## Environment

- CUDA GPUs and the existing environment at `/home/JJ_Group/lih2511/.conda/envs/opr`.
- Python 3.10, PyTorch 2.8.0+cu128, Transformers 4.57.6, TRL 0.24.0,
  vLLM 0.11.0, DeepSpeed 0.18.2, Liger Kernel, xFormers, and safetensors.
- Cached SMDM-219M/SMDM-1.14B and Qwen3-0.6B/1.7B/4B checkpoints at the
  paths declared in `reproduction/suite_common.py`.
- Run commands from `/home/JJ_Group/lih2511/test/dllm/iclr_3`.

## All non-TRACE paper figures and tables

Each named experiment writes one `summary.json` containing every metric or
trajectory required by that figure/table. Existing output is never overwritten;
use a new `RUN_ID`, or `RESUME=1` only for an interrupted run.

```bash
nohup env RUN_ID=reported PAPER_GPUS="0 1 2 3 4 5 6 7" \
  TRAINABLE_SCOPE=reported \
  bash reproduction/run_all_paper_experiments.sh \
  > runs/reproduction/reported.log 2>&1 &
```

`TRAINABLE_SCOPE` accepts `reported`, `last_block`, or `all`. Model switches are
controlled by `SMDM_MODELS`, `QWEN_MODELS`, `PRIMARY_SMDM_MODEL`, and
`PRIMARY_QWEN_MODEL`. Select individual figure/table runners with
`PAPER_EXPERIMENTS`; valid names are:

```text
table_cagd_main table_cagd_components table_cagd_deployed_components
table_cagd_fresh figure_loss_dynamics figure_anchor_budget
table_cagd_natural figure_qwen_cagd table_gsm8k_behavior
tables_qualitative table_parameter_controls
```

Full-parameter SMDM and Qwen-0.6B/1.7B reproduction:

```bash
nohup env RUN_ID=fullparam_smdm_qwen06_qwen17 PAPER_SEEDS="3407 3408 3409" \
  PAPER_GPUS="0 1 2 3 4 5 6 7" TRAINABLE_SCOPE=all \
  SMDM_MODELS="smdm_219m smdm_1.14b" \
  QWEN_MODELS="qwen3_0.6b qwen3_1.7b" \
  PRIMARY_SMDM_MODEL=smdm_219m PRIMARY_QWEN_MODEL=qwen3_0.6b \
  NATURAL_SMDM_MODELS=smdm_219m NATURAL_QWEN_MODELS=qwen3_0.6b \
  bash reproduction/run_all_paper_experiments.sh \
  > runs/reproduction/fullparam_smdm_qwen06_qwen17.log 2>&1 &
```

Print the complete non-TRACE command matrix without training:

```bash
RUN_ID=check bash reproduction/run_all_paper_experiments.sh --dry-run
```

## TRACE large experiment: training and evaluation

The launcher executes all eight stages and their evaluations for Sequential,
Replay, SDFT, OPR, and CAGD, then emits one comparison `summary.json` per seed and
order.

```bash
nohup env TRACE_SEEDS="3407 3408 3409" TRACE_ORDERS="canonical reverse" \
  TRACE_GPUS="0,1,2,3,4,5,6,7" TRAINABLE_SCOPE=all \
  TRACE_METHODS="sequential replay sdft opr cagd" \
  TRACE_RUN_ROOT=runs/reproduction/trace_full \
  bash reproduction/run_trace_experiment.sh \
  > runs/reproduction/trace_full.log 2>&1 &
```

Print the TRACE matrix without training:

```bash
TRACE_SEEDS="3407" TRACE_ORDERS="canonical reverse" \
  bash reproduction/run_trace_experiment.sh --dry-run
```
