# Experiment reproduction

## Environment

Run commands from this directory. SMDM uses Python 3.9.25 / PyTorch 2.4.1
(CUDA 12.1), with dependencies in `requirements.txt`. Qwen and TRACE use
Python 3.10.21 / PyTorch 2.8.0 (CUDA 12.8), with dependencies in
`requirements-qwen.txt`. Keep the two environments separate.

```bash
export PAPER_PYTHON=/home/JJ_Group/lih2511/.conda/envs/opr/bin/python
export AR_PYTHON="$PAPER_PYTHON"
export SMDM_PYTHON=/home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python
export PAPER_TORCHRUN=/home/JJ_Group/lih2511/.conda/envs/opr/bin/torchrun
mkdir -p runs/reproduction
```

Checkpoint profiles are in `reproduction/suite_common.py`; tokenizer and
locked datasets are retained in `tokenizer/`, `third_party/SMDM/data/`,
`runs/data/`, and `data/trace_opr/`. On another machine, set the interpreter
paths above and the checkpoint paths in the profiles. TRACE accepts
`TRACE_MODEL=/absolute/checkpoint/path` independently.

## Paper experiments, excluding TRACE

This launcher runs all 11 figure/table experiment groups, including training,
evaluation, and `summary.json` generation. Each group exports all its required
metrics and per-seed results, with means, SEMs, and paired differences where
multiple seeds are specified.

```bash
nohup env PAPER_SEEDS="3407 3408 3409" PAPER_GPUS="0 1 2 3 4 5 6 7" \
  TRAINABLE_SCOPE=all SMDM_MODELS="smdm_219m" \
  QWEN_MODELS="qwen3_0.6b qwen3_1.7b" \
  PAPER_RUN_ROOT=runs/reproduction/paper_full \
  bash reproduction/run_all_paper_experiments.sh \
  > runs/reproduction/paper_full.log 2>&1 &
```

`TRAINABLE_SCOPE=all` updates all parameters; `last_block` updates only the
last block; `reported` uses full SMDM and last-block Qwen adaptation.
`SMDM_MODELS` and `QWEN_MODELS` select scale-study checkpoints; defaults include
SMDM-219M/1.14B and Qwen3-0.6B/1.7B/4B. Set `PRIMARY_SMDM_MODEL` and
`PRIMARY_QWEN_MODEL` for single-checkpoint studies, and `NATURAL_SMDM_MODELS`
and `NATURAL_QWEN_MODELS` for the Dolly study. Anchor-budget and qualitative
groups use seed 3407 by default; `ANCHOR_SEEDS` and `QUALITATIVE_SEED` override
these. Set `RESUME=1` to reuse completed cells in the same run directory;
otherwise use a new directory to avoid overwriting results.

An individual figure/table can be launched with, for example:

```bash
"$PAPER_PYTHON" -m reproduction.table_cagd_main \
  --run-root runs/reproduction/main_only --seeds "3407 3408 3409" \
  --gpus "0 1 2 3 4 5 6 7" --trainable all --model smdm_219m
```

## TRACE large experiment: training and evaluation

The large experiment is split into SDFT and the other five methods. Both
launchers run seeds 3407, 3408, and 3409 sequentially over the eight canonical
stages. The selected GPUs jointly run each distributed training stage.

```bash
nohup env TRACE_GPUS="0,1,2,3,4,5,6,7" TRAINABLE_SCOPE=all \
  bash reproduction/run_trace_sdft.sh \
  > runs/reproduction/trace_sdft.log 2>&1 &

nohup env TRACE_GPUS="0,1,2,3,4,5,6,7" TRAINABLE_SCOPE=all \
  bash reproduction/run_trace_without_sdft.sh \
  > runs/reproduction/trace_without_sdft.log 2>&1 &
```

Each method exports eight final scores, eight scores when learned, ACC, and
BWT. Results and full logs are available immediately after each seed at
`<run-root>/seed<seed>/summary.json` and
`<run-root>/seed<seed>/orchestrator.log`; the cross-seed aggregate is written
to `<run-root>/summary.json` after all three seeds finish. Set `TRACE_SEEDS` or
`TRACE_RUN_ROOT` to override the defaults. Completed stages and evaluations
are reused. An incomplete stage directory stops the launcher for inspection
rather than overwriting it.

## Checks and retained utilities

These checks do not launch model training:

```bash
"$PAPER_PYTHON" -m reproduction.suite_common
"$PAPER_PYTHON" -m reproduction.trace self-check
bash reproduction/run_all_paper_experiments.sh --dry-run
TRACE_SEEDS="3407 3408 3409" bash reproduction/run_trace_experiment.sh --dry-run
bash reproduction/run_trace_sdft.sh --dry-run
bash reproduction/run_trace_without_sdft.sh --dry-run
```

`reproduction/prepare_dolly_stream.py` builds the locked natural-instruction
stream; `render_paper_figures.py` and `render_loss_dynamics.py` preserve the
original figure-generation tools and their archived-summary input formats.
Use their `--help` for paths; they are not needed to run training/evaluation.

Historical results, checkpoints, logs, frozen protocols, and evidence bundles
remain unchanged in `runs/`, `results/`, `report/`, `release_evidence/`, and
`release_extension_evidence/`. Historical commands in `EXPERIMENT.md`,
`runs/commands.md`, and `report/` describe their original implementations,
not the current entry points. The removed legacy code and the original OPR
checkout are recoverable from
[`release_evidence/legacy_code_20260914.tar.gz`](release_evidence/legacy_code_20260914.tar.gz).
Extract that archive into a separate empty directory, not over this checkout.
Current runners depend only on `reproduction/` and the retained SMDM backbone,
not on the archived code or an OPR checkout.
The exact removal inventory and verification results are recorded in
[`release_evidence/cleanup_20260914.md`](release_evidence/cleanup_20260914.md).
