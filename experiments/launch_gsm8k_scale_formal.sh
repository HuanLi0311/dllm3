#!/usr/bin/env bash
# Launch the 24 missing formal cells; Qwen3-0.6B is intentionally reused.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/experiments/cagd_gsm8k_scale.py"
RUN_ROOT="$ROOT/runs/cagd_gsm8k_scale"
QWEN_PY=/home/JJ_Group/lih2511/.conda/envs/VisualSim2Real/bin/python
SMDM_PY=/home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python

[[ "$(hostname)" == air-node-03 ]] || { echo "formal compute must run on air-node-03" >&2; exit 2; }

PLAN=(
  "0 smdm_1.14b seq 3407"  "1 smdm_1.14b cagd 3407"
  "2 smdm_1.14b seq 3408"  "3 smdm_1.14b cagd 3408"
  "4 smdm_1.14b seq 3409"  "5 smdm_1.14b cagd 3409"
  "6 qwen3_4b seq 3407"    "6 qwen3_4b seq 3408"    "6 qwen3_4b seq 3409"
  "7 qwen3_4b cagd 3407"   "7 qwen3_4b cagd 3408"   "7 qwen3_4b cagd 3409"
  "0 qwen3_1.7b seq 3407"  "1 qwen3_1.7b cagd 3407"
  "2 qwen3_1.7b seq 3408"  "3 qwen3_1.7b cagd 3408"
  "4 qwen3_1.7b seq 3409"  "5 qwen3_1.7b cagd 3409"
  "0 smdm_219m seq 3407"   "1 smdm_219m cagd 3407"
  "2 smdm_219m seq 3408"   "3 smdm_219m cagd 3408"
  "4 smdm_219m seq 3409"   "5 smdm_219m cagd 3409"
)

print_plan() {
  printf '%-4s %-13s %-6s %-6s\n' GPU MODEL SEED METHOD
  for item in "${PLAN[@]}"; do
    read -r gpu model method seed <<< "$item"
    printf '%-4s %-13s %-6s %-6s\n' "$gpu" "$model" "$seed" "$method"
  done
}

if [[ "${1:-}" == --print-plan ]]; then
  print_plan
  exit 0
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--print-plan]" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/formal" "$RUN_ROOT/logs"
[[ -x "$QWEN_PY" && -x "$SMDM_PY" ]] || { echo "missing Python environment" >&2; exit 4; }
grep -q '^Status: frozen' "$ROOT/report/cagd_gsm8k_scale_protocol.md" \
  || { echo "protocol is not frozen" >&2; exit 4; }
PYTHONNOUSERSITE=1 "$QWEN_PY" -c 'import torch, transformers; assert torch.cuda.is_available()'
PYTHONNOUSERSITE=1 "$SMDM_PY" -c 'import asyncio, torch, transformers; assert torch.cuda.is_available()'
mapfile -t GPU_STATE < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
[[ "${#GPU_STATE[@]}" -eq 8 ]] || { echo "expected exactly eight GPUs" >&2; exit 4; }
for state in "${GPU_STATE[@]}"; do
  IFS=, read -r index memory utilization <<< "$state"
  memory=${memory// /}; utilization=${utilization// /}
  [[ "$memory" -lt 100 && "$utilization" -lt 5 ]] \
    || { echo "GPU $index is not idle: ${memory}MiB, ${utilization}%" >&2; exit 4; }
done
for item in "${PLAN[@]}"; do
  read -r _ model method seed <<< "$item"
  output="$RUN_ROOT/formal/$model/$method/s$seed.json"
  stem="$RUN_ROOT/logs/${model}_${method}_s${seed}"
  for target in "$output" "$output.lock" "$stem.log" "$stem.pid" "$stem.exit" "$stem.gpu.csv"; do
    [[ ! -e "$target" ]] || { echo "refusing existing cell artifact: $target" >&2; exit 3; }
  done
done

run_cell() {
  local gpu=$1 model=$2 method=$3 seed=$4 python output stem pid monitor exit_code
  python=$QWEN_PY
  [[ "$model" == smdm_* ]] && python=$SMDM_PY
  output="$RUN_ROOT/formal/$model/$method/s$seed.json"
  stem="$RUN_ROOT/logs/${model}_${method}_s${seed}"
  mkdir -p "$(dirname "$output")"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 "$python" "$RUNNER" \
    --model-id "$model" --method "$method" --seed "$seed" --formal --output "$output" \
    > "$stem.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$stem.pid"
  (
    while kill -0 "$pid" 2>/dev/null; do
      printf '%s,' "$(date -u +%FT%TZ)"
      nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu \
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
  [[ "$exit_code" -eq 0 ]] || return "$exit_code"
}

launch_queue() {
  local gpu=$1
  shift
  local cell model method seed
  for cell in "$@"; do
    IFS=: read -r model method seed <<< "$cell"
    run_cell "$gpu" "$model" "$method" "$seed"
  done
}

workers=()
launch_queue 0 smdm_1.14b:seq:3407 qwen3_1.7b:seq:3407 smdm_219m:seq:3407 & workers+=("$!")
sleep 8
launch_queue 1 smdm_1.14b:cagd:3407 qwen3_1.7b:cagd:3407 smdm_219m:cagd:3407 & workers+=("$!")
sleep 8
launch_queue 2 smdm_1.14b:seq:3408 qwen3_1.7b:seq:3408 smdm_219m:seq:3408 & workers+=("$!")
sleep 8
launch_queue 3 smdm_1.14b:cagd:3408 qwen3_1.7b:cagd:3408 smdm_219m:cagd:3408 & workers+=("$!")
sleep 8
launch_queue 4 smdm_1.14b:seq:3409 qwen3_1.7b:seq:3409 smdm_219m:seq:3409 & workers+=("$!")
sleep 8
launch_queue 5 smdm_1.14b:cagd:3409 qwen3_1.7b:cagd:3409 smdm_219m:cagd:3409 & workers+=("$!")
sleep 8
launch_queue 6 qwen3_4b:seq:3407 qwen3_4b:seq:3408 qwen3_4b:seq:3409 & workers+=("$!")
sleep 8
launch_queue 7 qwen3_4b:cagd:3407 qwen3_4b:cagd:3408 qwen3_4b:cagd:3409 & workers+=("$!")
status=0
for worker in "${workers[@]}"; do
  wait "$worker" || status=1
done
exit "$status"
