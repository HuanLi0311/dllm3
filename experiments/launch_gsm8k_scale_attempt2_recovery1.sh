#!/usr/bin/env bash
# Recover the GPU-2 queue after its first process failed before importing torch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT/runs/cagd_gsm8k_scale/attempt2"
SMDM_RUNNER="$ROOT/experiments/smdm_cagd_gsm8k_paired.py"
QWEN_RUNNER="$ROOT/experiments/cagd_gsm8k_scale.py"
QWEN_PY="${QWEN_PY:-/home/JJ_Group/lih2511/.conda/envs/VisualSim2Real/bin/python}"
SMDM_PY="${SMDM_PY:-/home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python}"
GPU=2

[[ "$(hostname)" == air-node-03 ]] || { echo "recovery must run on air-node-03" >&2; exit 2; }
IFS=, read -r memory utilization < <(
  nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
)
memory=${memory// /}
utilization=${utilization// /}
[[ "$memory" -lt 100 && "$utilization" -lt 5 ]] || {
  echo "GPU $GPU is not idle: ${memory}MiB, ${utilization}%" >&2
  exit 4
}

run_monitored() {
  local stem=$1 pid monitor exit_code
  shift
  env CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 "$@" > "$stem.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$stem.pid"
  (
    while kill -0 "$pid" 2>/dev/null; do
      printf '%s,' "$(date -u +%FT%TZ)"
      nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 1
    done
  ) > "$stem.gpu.csv" 2>&1 &
  monitor=$!
  set +e
  wait "$pid"
  exit_code=$?
  set -e
  wait "$monitor" || true
  printf '%s\n' "$exit_code" > "$stem.exit"
  [[ "$exit_code" -eq 0 ]]
}

run_cell() {
  local model=$1 method=$2 seed=$3 output stem canonical checkpoint
  output="$RUN_ROOT/formal/$model/$method/s$seed.json"
  stem="$RUN_ROOT/logs/recovery1_${model}_${method}_s${seed}"
  mkdir -p "$(dirname "$output")" "$(dirname "$stem")"
  for target in "$output" "$output.lock" "$stem.log" "$stem.pid" "$stem.exit" "$stem.gpu.csv"; do
    [[ ! -e "$target" ]] || { echo "refusing existing recovery artifact: $target" >&2; return 3; }
  done
  if [[ "$model" == smdm_* ]]; then
    canonical="$RUN_ROOT/stage0/$model/s$seed/stage0.json"
    checkpoint="$RUN_ROOT/stage0/$model/s$seed/stage0.safetensors"
    [[ -f "$canonical" && -f "$checkpoint" ]] || { echo "missing canonical prefix" >&2; return 4; }
    run_monitored "$stem" "$SMDM_PY" "$SMDM_RUNNER" \
      --mode branch --model-id "$model" --method "$method" --seed "$seed" --formal \
      --output "$output" --stage0-json "$canonical" --stage0-checkpoint "$checkpoint"
  else
    run_monitored "$stem" "$QWEN_PY" "$QWEN_RUNNER" \
      --model-id "$model" --method "$method" --seed "$seed" --formal --output "$output"
  fi
}

# ponytail: this fixed three-cell queue is the entire recovery scope; no generic scheduler needed.
run_cell smdm_1.14b seq 3408
run_cell qwen3_1.7b seq 3408
run_cell smdm_219m seq 3408
