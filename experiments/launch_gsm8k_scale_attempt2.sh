#!/usr/bin/env bash
# Launch six immutable SMDM prefixes and the 24 missing formal endpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT/runs/cagd_gsm8k_scale/attempt2"
SMDM_RUNNER="$ROOT/experiments/smdm_cagd_gsm8k_paired.py"
QWEN_RUNNER="$ROOT/experiments/cagd_gsm8k_scale.py"
PROTOCOL="$ROOT/report/cagd_gsm8k_scale_attempt2_protocol.md"
QWEN_PY="${QWEN_PY:-/home/JJ_Group/lih2511/.conda/envs/VisualSim2Real/bin/python}"
SMDM_PY="${SMDM_PY:-/home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python}"

CANONICAL_PLAN=(
  "0 smdm_1.14b 3407" "1 smdm_1.14b 3408" "2 smdm_1.14b 3409"
  "3 smdm_219m 3407"  "4 smdm_219m 3408"  "5 smdm_219m 3409"
)
ENDPOINT_PLAN=(
  "0 smdm_1.14b seq 3407"  "1 smdm_1.14b cagd 3407"
  "2 smdm_1.14b seq 3408"  "3 smdm_1.14b cagd 3408"
  "4 smdm_1.14b seq 3409"  "5 smdm_1.14b cagd 3409"
  "0 qwen3_1.7b seq 3407"  "1 qwen3_1.7b cagd 3407"
  "2 qwen3_1.7b seq 3408"  "3 qwen3_1.7b cagd 3408"
  "4 qwen3_1.7b seq 3409"  "5 qwen3_1.7b cagd 3409"
  "0 smdm_219m seq 3407"   "1 smdm_219m cagd 3407"
  "2 smdm_219m seq 3408"   "3 smdm_219m cagd 3408"
  "4 smdm_219m seq 3409"   "5 smdm_219m cagd 3409"
  "6 qwen3_4b seq 3407"    "6 qwen3_4b seq 3408"    "6 qwen3_4b seq 3409"
  "7 qwen3_4b cagd 3407"   "7 qwen3_4b cagd 3408"   "7 qwen3_4b cagd 3409"
)

print_plan() {
  printf 'CANONICAL PREFIXES\n%-4s %-13s %-6s\n' GPU MODEL SEED
  for item in "${CANONICAL_PLAN[@]}"; do
    read -r gpu model seed <<< "$item"
    printf '%-4s %-13s %-6s\n' "$gpu" "$model" "$seed"
  done
  printf '\nFORMAL ENDPOINTS\n%-4s %-13s %-6s %-6s\n' GPU MODEL METHOD SEED
  for item in "${ENDPOINT_PLAN[@]}"; do
    read -r gpu model method seed <<< "$item"
    printf '%-4s %-13s %-6s %-6s\n' "$gpu" "$model" "$method" "$seed"
  done
}

canonical_json() {
  printf '%s/stage0/%s/s%s/stage0.json' "$RUN_ROOT" "$1" "$2"
}

canonical_checkpoint() {
  printf '%s/stage0/%s/s%s/stage0.safetensors' "$RUN_ROOT" "$1" "$2"
}

check_absent() {
  local target
  for target in "$@"; do
    [[ ! -e "$target" ]] || {
      echo "refusing existing attempt-2 artifact: $target" >&2
      return 3
    }
  done
}

preflight() {
  local require_frozen=$1 item gpu model method seed output stem status_line relative digest
  [[ "$(hostname)" == air-node-03 ]] || {
    echo "formal compute must run on air-node-03" >&2
    return 2
  }
  [[ -x "$QWEN_PY" && -x "$SMDM_PY" ]] || {
    echo "missing Python environment" >&2
    return 4
  }
  PYTHONNOUSERSITE=1 "$QWEN_PY" -c 'import torch, transformers; assert torch.cuda.is_available()'
  PYTHONNOUSERSITE=1 "$SMDM_PY" -c \
    'import asyncio, safetensors, torch, transformers; assert torch.cuda.is_available()'
  status_line="$(grep -m1 '^Status:' "$PROTOCOL")"
  if [[ "$require_frozen" -eq 1 && "$status_line" != 'Status: frozen' ]]; then
    echo "attempt-2 protocol is not frozen: $status_line" >&2
    return 4
  fi
  for relative in \
    experiments/smdm_cagd_gsm8k_paired.py \
    experiments/audit_gsm8k_scale_attempt2.py \
    experiments/summarize_gsm8k_scale_attempt2.py \
    experiments/launch_gsm8k_scale_attempt2.sh \
    experiments/cagd_gsm8k_scale.py \
    report/cagd_gsm8k_scale_protocol.md; do
    digest="$(sha256sum "$ROOT/$relative" | cut -d' ' -f1)"
    grep -Fqx "$digest  $relative" "$PROTOCOL" || {
      echo "protocol hash manifest differs: $relative" >&2
      return 4
    }
  done
  mapfile -t gpu_state < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  [[ "${#gpu_state[@]}" -eq 8 ]] || { echo "expected eight GPUs" >&2; return 4; }
  for item in "${gpu_state[@]}"; do
    IFS=, read -r gpu memory utilization <<< "$item"
    memory=${memory// /}; utilization=${utilization// /}
    [[ "$memory" -lt 100 && "$utilization" -lt 5 ]] || {
      echo "GPU $gpu is not idle: ${memory}MiB, ${utilization}%" >&2
      return 4
    }
  done
  for item in "${CANONICAL_PLAN[@]}"; do
    read -r _ model seed <<< "$item"
    output="$(canonical_json "$model" "$seed")"
    checkpoint="$(canonical_checkpoint "$model" "$seed")"
    stem="$RUN_ROOT/logs/stage0_${model}_s${seed}"
    check_absent "$output" "$output.lock" "$checkpoint" "$checkpoint.lock" \
      "$stem.log" "$stem.pid" "$stem.exit" "$stem.gpu.csv"
  done
  for item in "${ENDPOINT_PLAN[@]}"; do
    read -r _ model method seed <<< "$item"
    output="$RUN_ROOT/formal/$model/$method/s$seed.json"
    stem="$RUN_ROOT/logs/${model}_${method}_s${seed}"
    check_absent "$output" "$output.lock" \
      "$stem.log" "$stem.pid" "$stem.exit" "$stem.gpu.csv"
  done
  check_absent "$RUN_ROOT/summary.json" "$ROOT/report/cagd_gsm8k_scale_attempt2_results.md"
  printf 'preflight=ok protocol="%s" qwen_python=%s smdm_python=%s\n' \
    "$status_line" "$QWEN_PY" "$SMDM_PY"
}

run_monitored() {
  local gpu=$1 stem=$2 pid monitor exit_code
  shift 2
  mkdir -p "$(dirname "$stem")"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 "$@" > "$stem.log" 2>&1 &
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
  [[ "$exit_code" -eq 0 ]]
}

run_canonical() {
  local gpu=$1 model=$2 seed=$3 output checkpoint stem
  output="$(canonical_json "$model" "$seed")"
  checkpoint="$(canonical_checkpoint "$model" "$seed")"
  stem="$RUN_ROOT/logs/stage0_${model}_s${seed}"
  mkdir -p "$(dirname "$output")"
  run_monitored "$gpu" "$stem" "$SMDM_PY" "$SMDM_RUNNER" \
    --mode stage0 --model-id "$model" --method seq --seed "$seed" --formal \
    --output "$output" --stage0-checkpoint "$checkpoint"
}

run_endpoint() {
  local gpu=$1 model=$2 method=$3 seed=$4 output stem canonical checkpoint
  output="$RUN_ROOT/formal/$model/$method/s$seed.json"
  stem="$RUN_ROOT/logs/${model}_${method}_s${seed}"
  mkdir -p "$(dirname "$output")"
  if [[ "$model" == smdm_* ]]; then
    canonical="$(canonical_json "$model" "$seed")"
    checkpoint="$(canonical_checkpoint "$model" "$seed")"
    run_monitored "$gpu" "$stem" "$SMDM_PY" "$SMDM_RUNNER" \
      --mode branch --model-id "$model" --method "$method" --seed "$seed" --formal \
      --output "$output" --stage0-json "$canonical" --stage0-checkpoint "$checkpoint"
  else
    run_monitored "$gpu" "$stem" "$QWEN_PY" "$QWEN_RUNNER" \
      --model-id "$model" --method "$method" --seed "$seed" --formal --output "$output"
  fi
}

run_endpoint_queue() {
  local gpu=$1
  shift
  local cell model method seed
  for cell in "$@"; do
    IFS=: read -r model method seed <<< "$cell"
    run_endpoint "$gpu" "$model" "$method" "$seed"
  done
}

case "${1:-}" in
  --print-plan) print_plan; exit 0 ;;
  --preflight) preflight 0; exit 0 ;;
  '') ;;
  *) echo "usage: $0 [--print-plan|--preflight]" >&2; exit 2 ;;
esac

preflight 1

# Qwen3-4B runs independently while the six canonical SMDM prefixes are built.
qwen_workers=()
run_endpoint_queue 6 qwen3_4b:seq:3407 qwen3_4b:seq:3408 qwen3_4b:seq:3409 &
qwen_workers+=("$!")
sleep 8
run_endpoint_queue 7 qwen3_4b:cagd:3407 qwen3_4b:cagd:3408 qwen3_4b:cagd:3409 &
qwen_workers+=("$!")
sleep 8

canonical_workers=()
for item in "${CANONICAL_PLAN[@]}"; do
  read -r gpu model seed <<< "$item"
  run_canonical "$gpu" "$model" "$seed" &
  canonical_workers+=("$!")
  sleep 8
done
canonical_status=0
for worker in "${canonical_workers[@]}"; do
  wait "$worker" || canonical_status=1
done

endpoint_workers=()
if [[ "$canonical_status" -eq 0 ]]; then
  run_endpoint_queue 0 smdm_1.14b:seq:3407 qwen3_1.7b:seq:3407 smdm_219m:seq:3407 &
  endpoint_workers+=("$!")
  run_endpoint_queue 1 smdm_1.14b:cagd:3407 qwen3_1.7b:cagd:3407 smdm_219m:cagd:3407 &
  endpoint_workers+=("$!")
  run_endpoint_queue 2 smdm_1.14b:seq:3408 qwen3_1.7b:seq:3408 smdm_219m:seq:3408 &
  endpoint_workers+=("$!")
  run_endpoint_queue 3 smdm_1.14b:cagd:3408 qwen3_1.7b:cagd:3408 smdm_219m:cagd:3408 &
  endpoint_workers+=("$!")
  run_endpoint_queue 4 smdm_1.14b:seq:3409 qwen3_1.7b:seq:3409 smdm_219m:seq:3409 &
  endpoint_workers+=("$!")
  run_endpoint_queue 5 smdm_1.14b:cagd:3409 qwen3_1.7b:cagd:3409 smdm_219m:cagd:3409 &
  endpoint_workers+=("$!")
fi

status=$canonical_status
for worker in "${endpoint_workers[@]}" "${qwen_workers[@]}"; do
  wait "$worker" || status=1
done
exit "$status"
