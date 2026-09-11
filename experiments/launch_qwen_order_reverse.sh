#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=/home/JJ_Group/lih2511/.conda/envs/VisualSim2Real/bin/python
runner="$root/experiments/qwen_order_reverse.py"
summarizer="$root/experiments/summarize_qwen_order_reverse.py"
run_root="$root/runs/qwen_order_reverse"

"$python" "$runner" --self-check
mkdir -p "$run_root/reverse/seq" "$run_root/reverse/gd"

pids=()
gpu=0
for method in seq gd; do
  for seed in 3407 3408 3409; do
    output="$run_root/reverse/$method/s${seed}.json"
    log="$run_root/reverse/$method/s${seed}.log"
    test ! -e "$output"
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 \
      "$python" "$runner" --method "$method" --seed "$seed" --output "$output" \
      > "$log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
test "$status" -eq 0
"$python" "$summarizer"
