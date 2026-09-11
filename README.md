# TRACE 4B Full-Parameter Experiment

## Environment dependencies

- Linux compute node with 8 homogeneous CUDA GPUs visible to the job; the
  launcher supports the current 8-GPU A100 40GB, H200, or H20 nodes
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

Start a fresh reverse-order run of both OPR-RU and CAGD. The launcher trains
all eight stages, evaluates each task when it is acquired, evaluates all eight
tasks after the final stage, and writes a summary containing the per-task
scores, ACC, and BWT for each method. Run from the `iclr_3` directory:

```bash
cd /home/JJ_Group/lih2511/test/dllm/iclr_3
TRACE_SEED=3407
TRACE_RUN="runs/trace_opr_cagd/seed${TRACE_SEED}_reverse"
test ! -e "$TRACE_RUN"
mkdir -p "$TRACE_RUN"
nohup env TRACE_METHODS="opr cagd" \
  bash experiments/launch_trace_opr_cagd_seed.sh "$TRACE_SEED" reverse \
  > "$TRACE_RUN/orchestrator.log" 2>&1 &
```

The reverse task order is `20Minuten`, `NumGLUE-ds`, `NumGLUE-cm`,
`ScienceQA`, `Py150`, `MeetingBank`, `FOMC`, and `C-STANCE`. OPR-RU and CAGD
share stage 0 and are then run sequentially so their results use the same
initial checkpoint. The fresh-run guard above prevents an earlier result
directory from being reused.
