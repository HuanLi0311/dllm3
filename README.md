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
Vanilla Replay (`replay`), SDFT, OPR-RU (`opr`), OPR-SC (`opr_sc`), and CAGD,
then emits one comparison `summary.json` per seed and order.

```bash
nohup env TRACE_SEEDS="3407 3408 3409" TRACE_ORDERS="canonical reverse" \
  TRACE_GPUS="0,1,2,3,4,5,6,7" TRAINABLE_SCOPE=all \
  TRACE_METHODS="sequential replay sdft opr opr_sc cagd" \
  TRACE_RUN_ROOT=runs/reproduction/trace_full \
  bash reproduction/run_trace_experiment.sh \
  > runs/reproduction/trace_full.log 2>&1 &
```

Print the TRACE matrix without training:

```bash
TRACE_SEEDS="3407" TRACE_ORDERS="canonical reverse" \
  bash reproduction/run_trace_experiment.sh --dry-run
```
