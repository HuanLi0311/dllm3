# Experiment reproduction

## Environment

Run commands from this directory and keep the SMDM and Qwen environments
separate.

### TRACE environment

The reference environment is the one used for Table 5: Linux, Python 3.10.21,
PyTorch 2.8.0 with CUDA 12.8, Transformers 4.57.6, vLLM 0.11.0, and eight
NVIDIA A100 PCIe 40GB GPUs. `requirements-qwen.txt` pins the tested Python
packages. A newer GPU may be used, but it is not the Table 5 reference
hardware; its NVIDIA driver must support the CUDA 12.8 wheels.

Create a clean environment and point both TRACE launchers to it:

```bash
conda create -n cagd-trace python=3.10.21 -y
conda activate cagd-trace
python -m pip install -r requirements-qwen.txt

export PYTHONNOUSERSITE=1
export PAPER_PYTHON="$CONDA_PREFIX/bin/python"
export AR_PYTHON="$PAPER_PYTHON"
export PAPER_TORCHRUN="$CONDA_PREFIX/bin/torchrun"
export TRACE_MODEL=/absolute/path/to/Qwen3-4B-Instruct-2507
mkdir -p runs/reproduction
```

## TRACE large experiment: training and evaluation

The large experiment is split into SDFT and the other five methods. Both
launchers run seeds 3407, 3408, and 3409 sequentially over the eight canonical
stages. The selected GPUs jointly run each distributed training stage. On one
eight-GPU host, run these commands one after the other; use separate hosts if
they are launched simultaneously.

```bash
nohup env TRACE_SEEDS="3407 3408 3409" \
  TRACE_GPUS="0,1,2,3,4,5,6,7" TRAINABLE_SCOPE=all \
  TRACE_RUN_ROOT=runs/reproduction/trace_sdft \
  bash reproduction/run_trace_sdft.sh \
  > runs/reproduction/trace_sdft.log 2>&1 &

nohup env TRACE_SEEDS="3407 3408 3409" \
  TRACE_GPUS="0,1,2,3,4,5,6,7" TRAINABLE_SCOPE=all \
  TRACE_RUN_ROOT=runs/reproduction/trace_without_sdft \
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
