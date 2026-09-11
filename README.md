# TRACE 4B Full-Parameter Experiment

## Environment

- 8 CUDA GPUs; the launcher uses devices `0,1,2,3,4,5,6,7`.
- Python: `/home/JJ_Group/lih2511/.conda/envs/opr/bin/python`
- Installed stack: Python 3.10, PyTorch 2.8.0+cu128, Transformers 4.57.6,
  TRL 0.24.0, vLLM 0.11.0, DeepSpeed 0.18.2, and Liger Kernel.
- Model: cached `Qwen3-4B-Instruct-2507` snapshot referenced by the launcher.
- Data: `data/trace_opr/{C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}`.

## Launch one seed

The default method list is `sequential replay sdft opr cagd`. Completed stages
are reused; an incomplete stage directory stops the run for inspection.

```bash
cd /home/JJ_Group/lih2511/test/dllm/iclr_3
TRACE_SEED=3407
TRACE_RUN="runs/trace_opr_cagd/seed${TRACE_SEED}"
mkdir -p "$TRACE_RUN"
nohup env TRACE_METHODS="sequential replay sdft opr cagd" \
  bash experiments/launch_trace_opr_cagd_seed.sh "$TRACE_SEED" canonical \
  > "$TRACE_RUN/full.log" 2>&1 &
```

## Launch all three seeds serially on one 8-GPU machine

```bash
cd /home/JJ_Group/lih2511/test/dllm/iclr_3
mkdir -p runs/trace_opr_cagd
nohup bash -c '
for seed in 3407 3408 3409; do
  run="runs/trace_opr_cagd/seed${seed}"
  mkdir -p "$run"
  TRACE_METHODS="sequential replay sdft opr cagd" \
    bash experiments/launch_trace_opr_cagd_seed.sh "$seed" canonical \
    > "$run/full.log" 2>&1 || exit
done
' > runs/trace_opr_cagd/all_seeds.log 2>&1 &
```

For exact reverse order, replace `canonical` with `reverse`; outputs use the
suffix `_reverse`. To run a subset, set `TRACE_METHODS`, for example
`TRACE_METHODS="sequential opr cagd"`.
