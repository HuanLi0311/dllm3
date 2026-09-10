# TRACE 4B Full-Parameter Experiment

## Environment dependencies

- Linux compute node with 8 NVIDIA A100 40GB GPUs
- At least 130GB free storage per seed for the shared, OPR, and CAGD checkpoints
- Conda environment: `/home/JJ_Group/lih2511/.conda/envs/opr`
- Python 3.10.21 and CUDA 12.8
- `torch==2.8.0`
- `transformers==4.57.6`
- `accelerate==1.15.0`
- `deepspeed==0.18.2`
- `vllm==0.11.0`
- `liger-kernel==0.8.2`
- `evaluate==0.4.6`
- `rouge-score==0.1.2`
- `fuzzywuzzy==0.18.0`
- `sacrebleu==2.5.1`
- `nltk==3.10.3`

The launcher expects the Qwen3-4B-Instruct-2507 snapshot, TRACE data, and OPR
scorers at the paths already fixed in `experiments/trace_opr_cagd.py`.

## Launch

From `/home/JJ_Group/lih2511/test/dllm/iclr_3`:

```bash
TRACE_SEED=3407
mkdir -p "runs/trace_opr_cagd/seed${TRACE_SEED}"
nohup bash experiments/launch_trace_opr_cagd_seed.sh "${TRACE_SEED}" \
  > "runs/trace_opr_cagd/seed${TRACE_SEED}/orchestrator.log" 2>&1 &
```
