# Condition-Anchored Distillation for Continual Language Models

This directory contains the experiments for *Stabilizing Language Models under
Continual Learning via Condition-Anchored Distillation*.  CAGD keeps a small
set of old prompts, lets the frozen previous model reconstruct completions and
generation states, and matches the teacher's predictive distribution while
learning the next language task.

The formulation is presented for continual language-model adaptation and
covers the two tested objectives: autoregressive generation and masked
diffusion language modeling.  The diffusion backend in this repository is
specifically SMDM; the autoregressive backend is Qwen3.

## Evidence and scope

The paper separates three questions:

- Does CAGD improve retention without preventing acquisition of the current
  task?
- With identical generated completions, does soft distribution matching add
  anything beyond one-hot hard replay?
- Does the behavior persist across task content, task order, and AR versus
  masked-diffusion objectives?

The controlled factual study uses full-parameter SMDM-219M adaptation.  The
natural Dolly stream contains closed QA, summarization, and creative writing
with disjoint train and held-out prompts.  Its SMDM run updates all parameters;
the Qwen3-0.6B run updates only the final Transformer block.  Cross-backend
agreement is therefore directional evidence, not an effect-size comparison.

The exact frozen protocols are:

- [component protocol](report/cagd_component_protocol.md)
- [natural-task protocol](report/cagd_natural_protocol.md)

Validated aggregates are written to:

- `runs/cagd_component/two_task_summary.json`
- `runs/cagd_component/main_summary.json`
- `runs/cagd_component/fresh_summary.json`
- `runs/cagd_natural/summary.json`

Earlier audited Sequential, Joint, and three-scale Qwen references reused by
the paper remain under `runs/r16_native_mask/` and
`runs/qwen_continual_scale/`.  Fisher studies retained for historical or
appendix-only comparison are not part of the CAGD mechanism claim.

## Environments

Run SMDM experiments with:

```bash
PYTHONNOUSERSITE=1 /home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python
```

Run Qwen experiments with:

```bash
PYTHONNOUSERSITE=1 /home/JJ_Group/lih2511/.conda/envs/VisualSim2Real/bin/python
```

The reported jobs use NVIDIA A100 40GB GPUs.  The two environments are kept
separate because their pinned PyTorch and Transformers stacks differ.

## Data

The factual streams use the vendored reversal-association data.  The shared
natural stream is materialized once from a fixed Databricks Dolly 15K revision:

```bash
PYTHONNOUSERSITE=1 /home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python \
  experiments/build_dolly_stream.py
```

The resulting files are `runs/data/dolly_natural_stream.jsonl` and
`runs/data/dolly_natural_stream_manifest.json`.  Both backends consume these
same rows and splits.

## Lightweight checks

From this directory:

```bash
python experiments/dllm_rank1_multitask.py --self-check
python experiments/smdm_cagd_natural.py --self-check
python experiments/qwen_cagd_natural.py --self-check
python experiments/build_dolly_stream.py --self-check
python experiments/summarize_cagd_components.py --self-check
python experiments/summarize_cagd_natural.py --self-check
python experiments/make_cagd_figures.py --self-check
```

## Representative formal runs

Masked-diffusion factual component run:

```bash
python experiments/dllm_rank1_multitask.py \
  --method cagd --cagd-two-task-protocol \
  --eval-mc-samples 32 --seed 3407 --generation-seed 3407 \
  --output runs/cagd_component/two_task/s3407/cagd.json
```

Natural masked-diffusion and AR runs:

```bash
python experiments/smdm_cagd_natural.py \
  --formal --method cagd --seed 3407 \
  --output runs/cagd_natural/formal/smdm/s3407/cagd.json

python experiments/qwen_cagd_natural.py \
  --formal --method cagd --seed 3407 \
  --output runs/cagd_natural/formal/qwen/s3407/cagd.json
```

The strict aggregators reject incomplete matrices, unexpected cells,
non-finite endpoints, protocol drift, and mismatched first-transition
generations between CAGD and hard replay:

```bash
python experiments/summarize_cagd_components.py \
  --run-root runs/cagd_component/main --protocol main \
  --output runs/cagd_component/main_summary.json

python experiments/summarize_cagd_components.py \
  --run-root runs/cagd_component/fresh --protocol fresh \
  --output runs/cagd_component/fresh_summary.json

python experiments/summarize_cagd_natural.py \
  --run-root runs/cagd_natural/formal \
  --output runs/cagd_natural/summary.json
```

## Paper

The LaTeX source is in `../assets/iclr_3`.  For live rebuilding:

```bash
cd ../assets/iclr_3
latexmk -pdf -pvc -interaction=nonstopmode main.tex
```

The generated submission file is `../assets/iclr_3/main.pdf`.  See
[NOTICE.md](../NOTICE.md) for the paper narrative, section structure, upstream
attribution, and claim boundaries.
