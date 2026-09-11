#!/usr/bin/env bash
set -uo pipefail

cd /home/JJ_Group/lih2511/test/dllm/iclr_3 || exit 72
root="runs/cagd_loss_dynamics/table7"
test ! -e "$root" || { printf 'refusing existing result directory: %s\n' "$root" >&2; exit 73; }
mkdir -p "$root/logs"
python=/home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python
pids=()

run_cell() {
  local order=$1 method=$2 seed=$3 gpu=$4
  local label=$method
  test "$method" != gd || label=cagd
  local stem="${order}_${label}_s${seed}"
  local output="$root/$order/s$seed/$label.json"
  mkdir -p "$(dirname "$output")"
  (
    export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$gpu"
    printf 'launch_utc=%s host=%s physical_gpu=%s order=%s method=%s seed=%s\n' \
      "$(date -u +%FT%TZ)" "$(hostname)" "$gpu" "$order" "$method" "$seed"
    sha256sum experiments/dllm_rank1_multitask.py
    /usr/bin/time -v timeout --signal=TERM --kill-after=60s 10800 \
      "$python" experiments/dllm_rank1_multitask.py \
      --final-protocol --method "$method" --order "$order" \
      --eval-mc-samples 32 --seed "$seed" --generation-seed "$seed" \
      --record-step-loss --record-eval-loss-every 100 --output "$output"
    code=$?
    printf '%s\n' "$code" >"$root/logs/$stem.exit"
    exit "$code"
  ) >"$root/logs/$stem.log" 2>&1 &
  pids+=("$!")
}

wait_batch() {
  local failed=0 pid
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  pids=()
  return "$failed"
}

run_cell forward seq 3407 0
run_cell forward gd 3407 1
run_cell forward seq 3408 2
run_cell forward gd 3408 3
run_cell forward seq 3409 4
run_cell forward gd 3409 5
run_cell reverse seq 3407 6
run_cell reverse gd 3407 7
wait_batch || exit 1

run_cell reverse seq 3408 0
run_cell reverse gd 3408 1
run_cell reverse seq 3409 2
run_cell reverse gd 3409 3
wait_batch || exit 1

"$python" experiments/plot_table7_training_loss.py \
  --root "$root" \
  --output ../assets/iclr_3/figures/situ_glu_redraw/table7_loss_dynamics
