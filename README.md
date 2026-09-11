# TRACE 4B Full-Parameter Experiment

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
