#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --dry-run ]]; then
    export TRACE_DRY_RUN=1
    shift
fi

if (( $# == 0 )); then
    read -r -a trace_seeds <<< "${TRACE_SEEDS:-3407}"
    read -r -a trace_orders <<< "${TRACE_ORDERS:-canonical}"
    for trace_order in "${trace_orders[@]}"; do
        for trace_seed in "${trace_seeds[@]}"; do
            "$0" "$trace_seed" "$trace_order"
        done
    done
    exit 0
fi

seed=${1:-3407}
order=${2:-canonical}
case "$order" in
    canonical) order_suffix= ;;
    reverse) order_suffix=_reverse ;;
    *) echo "task order must be canonical or reverse" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python=${PAPER_PYTHON:-/home/JJ_Group/lih2511/.conda/envs/opr/bin/python}
torchrun=${PAPER_TORCHRUN:-/home/JJ_Group/lih2511/.conda/envs/opr/bin/torchrun}
runner=$root/reproduction/trace.py
model=${TRACE_MODEL:-/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
run_base=${TRACE_RUN_ROOT:-$root/runs/reproduction/trace}
run=$run_base/seed${seed}${order_suffix}
trainable=${TRAINABLE_SCOPE:-all}
case "$trainable" in all|last_block) ;; *) echo "TRAINABLE_SCOPE must be all or last_block" >&2; exit 2 ;; esac
runner_args=(--task-order "$order")
read -r -a methods <<< "${TRACE_METHODS:-sequential replay sdft opr cagd}"

for method in "${methods[@]}"; do
    case "$method" in
        sequential|replay|sdft|opr|cagd) ;;
        *) echo "unknown TRACE method: $method" >&2; exit 2 ;;
    esac
done

if [[ "${TRACE_DRY_RUN:-0}" == 1 ]]; then
    printf 'seed=%s order=%s methods=%s trainable=%s gpus=%s output=%s\n' \
        "$seed" "$order" "${methods[*]}" "$trainable" "${TRACE_GPUS:-0,1,2,3,4,5,6,7}" "$run"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=${TRACE_GPUS:-0,1,2,3,4,5,6,7}
nproc=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/tmp/trace_opr_cagd_triton_lih2511
export VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p "$TRITON_CACHE_DIR" "$run"
cd "$root"
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
    local args=("train-$method" --checkpoint "$source" --stage "$stage" --seed "$seed" --trainable "$trainable" --output "$output")
    [[ -z "$support" ]] || args+=(--support "$support")
    "$torchrun" --standalone --nproc_per_node="$nproc" "$runner" "${runner_args[@]}" "${args[@]}"
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

[[ -f "$run/summary.json" ]] || \
    "$python" "$runner" "${runner_args[@]}" summarize-comparison --run "$run" --methods "${methods[@]}"
