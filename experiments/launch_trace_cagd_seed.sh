#!/usr/bin/env bash
set -euo pipefail

seed=${1:-3407}
order=${2:-canonical}
case "$order" in
    canonical) order_suffix= ;;
    reverse) order_suffix=_reverse ;;
    *) echo "task order must be canonical or reverse" >&2; exit 2 ;;
esac
repo=/home/JJ_Group/lih2511/test/dllm
python=/home/JJ_Group/lih2511/.conda/envs/opr/bin/python
torchrun=/home/JJ_Group/lih2511/.conda/envs/opr/bin/torchrun
runner=$repo/iclr_3/experiments/trace_opr_cagd.py
model=/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
run=$repo/iclr_3/runs/trace_opr_cagd/seed${seed}${order_suffix}
runner_args=(--task-order "$order")

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/tmp/trace_cagd_triton_lih2511
export VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p "$TRITON_CACHE_DIR"
cd "$repo"

checkpoint() {
    "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1/stage_result.json"
}

train_stage() {
    stage=$1
    source=$2
    support=$3
    output=$4
    if [[ -f "$output/stage_result.json" ]]; then
        return
    fi
    if [[ -e "$output" ]]; then
        echo "incomplete output requires inspection: $output" >&2
        exit 1
    fi
    "$torchrun" --standalone --nproc_per_node=8 "$runner" "${runner_args[@]}" train-cagd \
        --checkpoint "$source" --stage "$stage" --seed "$seed" --support "$support" --output "$output"
}

stage_inference() {
    stage=$1
    source=$2
    evaluation=$3
    support=${4:-}
    if [[ -f "$evaluation" && ( -z "$support" || -f "$support" ) ]]; then
        return
    fi
    args=(stage-inference --method cagd --checkpoint "$source" --stage "$stage" --seed "$seed")
    [[ -f "$evaluation" ]] || args+=(--evaluation "$evaluation")
    [[ -z "$support" || -f "$support" ]] || args+=(--next-support "$support")
    "$python" "$runner" "${runner_args[@]}" "${args[@]}"
}

shared=$run/shared/stage0
if [[ ! -f "$shared/stage_result.json" ]]; then
    if [[ -e "$shared" ]]; then
        echo "incomplete output requires inspection: $shared" >&2
        exit 1
    fi
    # Stage 0 is shared SFT; CAGD begins at the first task transition.
    "$torchrun" --standalone --nproc_per_node=8 "$runner" "${runner_args[@]}" train-opr \
        --checkpoint "$model" --stage 0 --seed "$seed" --output "$shared"
fi
shared_checkpoint=$(checkpoint "$shared")
cagd_support=$shared/support_cagd_stage1.jsonl
stage_inference 0 "$shared_checkpoint" "$shared/evaluation.json" "$cagd_support"

source=$shared_checkpoint
support=$cagd_support
for stage in 1 2 3 4 5 6 7; do
    output=$run/cagd/stage$stage
    train_stage "$stage" "$source" "$support" "$output"
    source=$(checkpoint "$output")
    if [[ "$stage" -lt 7 ]]; then
        support=$output/support_cagd_stage$((stage + 1)).jsonl
        stage_inference "$stage" "$source" "$output/evaluation.json" "$support"
    else
        stage_inference "$stage" "$source" "$output/evaluation.json"
    fi
done
[[ -f "$run/cagd/summary.json" ]] || \
    "$python" "$runner" "${runner_args[@]}" summarize --run "$run" --method cagd
