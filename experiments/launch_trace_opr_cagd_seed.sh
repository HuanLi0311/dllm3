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
read -r -a methods <<< "${TRACE_METHODS:-sequential replay sdft opr cagd}"

for method in "${methods[@]}"; do
    case "$method" in
        sequential|replay|sdft|opr|cagd) ;;
        *) echo "unknown TRACE method: $method" >&2; exit 2 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/tmp/trace_opr_cagd_triton_lih2511
export VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p "$TRITON_CACHE_DIR" "$run"
cd "$repo"
"$python" "$runner" "${runner_args[@]}" self-check

checkpoint() {
    "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1/stage_result.json"
}

selected() {
    local wanted=$1 method
    for method in "${methods[@]}"; do
        [[ "$method" == "$wanted" ]] && return 0
    done
    return 1
}

train_stage() {
    local method=$1 stage=$2 source=$3 output=$4 support=${5:-}
    if [[ -f "$output/stage_result.json" ]]; then
        return
    fi
    if [[ -e "$output" ]]; then
        echo "incomplete output requires inspection: $output" >&2
        exit 1
    fi
    local args=("train-$method" --checkpoint "$source" --stage "$stage" --seed "$seed" --output "$output")
    [[ -z "$support" ]] || args+=(--support "$support")
    "$torchrun" --standalone --nproc_per_node=8 "$runner" "${runner_args[@]}" "${args[@]}"
}

stage_inference() {
    local method=$1 stage=$2 source=$3 evaluation=${4:-} support=${5:-}
    if [[ ( -z "$evaluation" || -f "$evaluation" ) && ( -z "$support" || -f "$support" ) ]]; then
        return
    fi
    local args=(stage-inference --method "$method" --checkpoint "$source" --stage "$stage" --seed "$seed")
    [[ -z "$evaluation" || -f "$evaluation" ]] || args+=(--evaluation "$evaluation")
    [[ -z "$support" || -f "$support" ]] || args+=(--next-support "$support")
    "$python" "$runner" "${runner_args[@]}" "${args[@]}"
}

shared=$run/shared/stage0
if [[ " ${methods[*]} " != " sdft " ]]; then
    train_stage opr 0 "$model" "$shared"
    shared_checkpoint=$(checkpoint "$shared")
    stage_inference sequential 0 "$shared_checkpoint" "$shared/evaluation.json"
    selected replay && stage_inference replay 0 "$shared_checkpoint" "" "$shared/support_replay_stage1.jsonl"
    selected opr && stage_inference opr 0 "$shared_checkpoint" "" "$shared/support_opr_stage1.jsonl"
    selected cagd && stage_inference cagd 0 "$shared_checkpoint" "" "$shared/support_cagd_stage1.jsonl"
fi

run_shared_sft_method() {
    local method=$1 source support= next_support= stage output
    source=$(checkpoint "$shared")
    case "$method" in
        replay|opr|cagd) support=$shared/support_${method}_stage1.jsonl ;;
    esac
    for stage in 1 2 3 4 5 6 7; do
        output=$run/$method/stage$stage
        train_stage "$method" "$stage" "$source" "$output" "$support"
        source=$(checkpoint "$output")
        next_support=
        if [[ "$stage" -lt 7 ]]; then
            case "$method" in
                replay|opr|cagd) next_support=$output/support_${method}_stage$((stage + 1)).jsonl ;;
            esac
        fi
        stage_inference "$method" "$stage" "$source" "$output/evaluation.json" "$next_support"
        support=$next_support
    done
    [[ -f "$run/$method/summary.json" ]] || \
        "$python" "$runner" "${runner_args[@]}" summarize --run "$run" --method "$method"
}

run_sdft() {
    local source=$model stage output
    for stage in 0 1 2 3 4 5 6 7; do
        output=$run/sdft/stage$stage
        train_stage sdft "$stage" "$source" "$output"
        source=$(checkpoint "$output")
        stage_inference sdft "$stage" "$source" "$output/evaluation.json"
    done
    [[ -f "$run/sdft/summary.json" ]] || \
        "$python" "$runner" "${runner_args[@]}" summarize --run "$run" --method sdft
}

for method in "${methods[@]}"; do
    if [[ "$method" == sdft ]]; then
        run_sdft
    else
        run_shared_sft_method "$method"
    fi
done
