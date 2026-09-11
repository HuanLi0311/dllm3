#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
"${PAPER_PYTHON:-/home/JJ_Group/lih2511/.conda/envs/opr/bin/python}" -m reproduction.suite_common

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${PAPER_RUN_ROOT:-$root/runs/reproduction/$run_id}"
seeds="${PAPER_SEEDS:-3407 3408 3409}"
gpus="${PAPER_GPUS:-0 1 2 3 4 5 6 7}"
trainable="${TRAINABLE_SCOPE:-reported}"
experiments="${PAPER_EXPERIMENTS:-table_cagd_main table_cagd_components table_cagd_deployed_components table_cagd_fresh figure_loss_dynamics figure_anchor_budget table_cagd_natural figure_qwen_cagd table_gsm8k_behavior tables_qualitative table_parameter_controls}"
smdm_models="${SMDM_MODELS:-smdm_219m smdm_1.14b}"
qwen_models="${QWEN_MODELS:-qwen3_0.6b qwen3_1.7b qwen3_4b}"
primary_smdm="${PRIMARY_SMDM_MODEL:-smdm_219m}"
primary_qwen="${PRIMARY_QWEN_MODEL:-qwen3_0.6b}"
natural_smdm_models="${NATURAL_SMDM_MODELS:-$primary_smdm}"
natural_qwen_models="${NATURAL_QWEN_MODELS:-$primary_qwen}"
extra=()
[[ "${RESUME:-0}" == 1 ]] && extra+=(--resume)
[[ "${1:-}" == --dry-run ]] && extra+=(--dry-run)
[[ $# -eq 0 || "${1:-}" == --dry-run ]] || { echo "usage: $0 [--dry-run]" >&2; exit 2; }

selected() {
    [[ " $experiments " == *" $1 "* ]]
}

run_table() {
    local module=$1
    shift
    selected "$module" || return 0
    "${PAPER_PYTHON:-/home/JJ_Group/lih2511/.conda/envs/opr/bin/python}" -m "reproduction.$module" \
        --run-root "$run_root/$module" --seeds "$seeds" --gpus "$gpus" \
        --trainable "$trainable" "${extra[@]}" "$@"
}

run_table table_cagd_main --model "$primary_smdm"
run_table table_cagd_components --model "$primary_smdm"
run_table table_cagd_deployed_components --model "$primary_smdm"
run_table table_cagd_fresh --model "$primary_smdm"
run_table figure_loss_dynamics --model "$primary_smdm"
run_table figure_anchor_budget --model "$primary_smdm" --seeds "${ANCHOR_SEEDS:-3407}"
run_table table_cagd_natural --smdm-models "$natural_smdm_models" --qwen-models "$natural_qwen_models"
run_table figure_qwen_cagd --models "$qwen_models"
run_table table_gsm8k_behavior --smdm-models "$smdm_models" --qwen-models "$qwen_models"
run_table tables_qualitative --model "$primary_qwen" --seeds "${QUALITATIVE_SEED:-3407}"
run_table table_parameter_controls --models "$smdm_models"

printf 'completed_run_root=%s\n' "$run_root"
