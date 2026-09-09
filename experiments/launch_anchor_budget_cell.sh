#!/usr/bin/env bash
set -u

if (( $# != 3 )); then
  printf 'usage: %s BUDGET ORDER PHYSICAL_GPU\n' "$0" >&2
  exit 64
fi
budget=$1
order=$2
gpu=$3
case "$budget" in 4|8|16|32|120) ;; *) printf 'invalid or already available budget\n' >&2; exit 64 ;; esac
case "$order" in forward|reverse) ;; *) printf 'invalid order\n' >&2; exit 64 ;; esac
case "$gpu" in 0|1|2|3|4|5|6|7) ;; *) printf 'invalid GPU\n' >&2; exit 64 ;; esac

cd /home/JJ_Group/lih2511/test/dllm/iclr_3 || exit 72
stem="${order}_m${budget}_s3407"
output="runs/cagd_anchor_budget/${order}/m${budget}_s3407.json"
log="runs/cagd_anchor_budget/logs/${stem}.log"
exit_file="runs/cagd_anchor_budget/logs/${stem}.exit"
mkdir -p "$(dirname "$output")" "$(dirname "$log")"
for target in "$output" "$log" "$exit_file"; do
  test ! -e "$target" || { printf 'refusing existing target: %s\n' "$target" >&2; exit 73; }
done

export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$gpu"
{
  printf 'launch_utc=%s host=%s physical_gpu=%s budget=%s order=%s seed=3407\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$gpu" "$budget" "$order"
  sha256sum experiments/dllm_rank1_multitask.py report/cagd_anchor_budget_protocol.md
  /usr/bin/time -v timeout --signal=TERM --kill-after=60s 7200 \
    /home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python \
    experiments/dllm_rank1_multitask.py \
    --method cagd --order "$order" --replay-per-task "$budget" \
    --eval-mc-samples 32 --seed 3407 --generation-seed 3407 \
    --output "$output"
} >"$log" 2>&1
code=$?
printf '%s\n' "$code" >"$exit_file"
exit "$code"
