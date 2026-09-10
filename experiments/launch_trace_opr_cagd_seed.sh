#!/usr/bin/env bash
set -euo pipefail

seed=${1:-3407}
repo=/home/JJ_Group/lih2511/test/dllm
python=/home/JJ_Group/lih2511/.conda/envs/opr/bin/python
torchrun=/home/JJ_Group/lih2511/.conda/envs/opr/bin/torchrun
runner=$repo/iclr_3/experiments/trace_opr_cagd.py
model=/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
run=$repo/iclr_3/runs/trace_opr_cagd/seed$seed

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/tmp/trace_opr_triton_lih2511
export VLLM_USE_V1=0
mkdir -p "$TRITON_CACHE_DIR"

checkpoint() {
    "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1/stage_result.json"
}

train_stage() {
    method=$1
    stage=$2
    source=$3
    support=$4
    output=$5
    if [[ -f "$output/stage_result.json" ]]; then
        return
    fi
    if [[ -e "$output" ]]; then
        echo "incomplete output requires inspection: $output" >&2
        exit 1
    fi
    "$torchrun" --standalone --nproc_per_node=8 "$runner" "train-$method" \
        --checkpoint "$source" --stage "$stage" --seed "$seed" --support "$support" --output "$output"
}

stage_inference() {
    method=$1
    stage=$2
    source=$3
    evaluation=$4
    support=${5:-}
    if [[ -f "$evaluation" && ( -z "$support" || -f "$support" ) ]]; then
        return
    fi
    args=(stage-inference --method "$method" --checkpoint "$source" --stage "$stage" --seed "$seed")
    [[ -f "$evaluation" ]] || args+=(--evaluation "$evaluation")
    [[ -z "$support" || -f "$support" ]] || args+=(--next-support "$support")
    "$python" "$runner" "${args[@]}"
}

shared=$run/shared/stage0
if [[ ! -f "$shared/stage_result.json" ]]; then
    if [[ -e "$shared" ]]; then
        echo "incomplete output requires inspection: $shared" >&2
        exit 1
    fi
    "$torchrun" --standalone --nproc_per_node=8 "$runner" train-opr \
        --checkpoint "$model" --stage 0 --seed "$seed" --output "$shared"
fi
shared_checkpoint=$(checkpoint "$shared")
opr_support=$shared/support_opr_stage1.jsonl
cagd_support=$shared/support_cagd_stage1.jsonl
stage_inference opr 0 "$shared_checkpoint" "$shared/evaluation.json" "$opr_support"
if [[ ! -f "$cagd_support" ]]; then
    "$python" "$runner" stage-inference --method cagd --checkpoint "$shared_checkpoint" \
        --stage 0 --seed "$seed" --next-support "$cagd_support"
fi

for method in opr cagd; do
    source=$shared_checkpoint
    support=$shared/support_${method}_stage1.jsonl
    for stage in 1 2 3 4 5 6 7; do
        output=$run/$method/stage$stage
        train_stage "$method" "$stage" "$source" "$support" "$output"
        source=$(checkpoint "$output")
        if [[ "$stage" -lt 7 ]]; then
            support=$output/support_${method}_stage$((stage + 1)).jsonl
            stage_inference "$method" "$stage" "$source" "$output/evaluation.json" "$support"
        else
            stage_inference "$method" "$stage" "$source" "$output/evaluation.json"
        fi
    done
    [[ -f "$run/$method/summary.json" ]] || \
        "$python" "$runner" summarize --run "$run" --method "$method"
done
