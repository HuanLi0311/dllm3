#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --dry-run ]]; then
    export TRACE_DRY_RUN=1
    shift
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python=${PAPER_PYTHON:-/home/JJ_Group/lih2511/.conda/envs/opr/bin/python}
torchrun=${PAPER_TORCHRUN:-/home/JJ_Group/lih2511/.conda/envs/opr/bin/torchrun}
runner=$root/reproduction/trace.py
model=${TRACE_MODEL:-/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
run_base=${TRACE_RUN_ROOT:-$root/runs/reproduction/trace}
trainable=${TRAINABLE_SCOPE:-all}
parallel_seeds=${TRACE_PARALLEL_SEEDS:-0}
case "$trainable" in all|last_block) ;; *) echo "TRAINABLE_SCOPE must be all or last_block" >&2; exit 2 ;; esac
case "$parallel_seeds" in 0|1) ;; *) echo "TRACE_PARALLEL_SEEDS must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$parallel_seeds" == 1 ]]; then
    export TRACE_VLLM_GPU_MEMORY_UTILIZATION=${TRACE_VLLM_GPU_MEMORY_UTILIZATION:-0.32}
fi
read -r -a methods <<< "${TRACE_METHODS:-sequential replay sdft opr opr_sc cagd}"
for method in "${methods[@]}"; do
    case "$method" in
        sequential|replay|sdft|opr|opr_sc|cagd) ;;
        *) echo "unknown TRACE method: $method" >&2; exit 2 ;;
    esac
done

run_seed() {
    local trace_seed=$1 seed_log
    if [[ "${TRACE_DRY_RUN:-0}" == 1 ]]; then
        "$0" "$trace_seed"
        return
    fi
    seed_log=$run_base/seed${trace_seed}/orchestrator.log
    mkdir -p "$(dirname "$seed_log")"
    {
        printf 'seed=%s started=%s\n' "$trace_seed" "$(date --iso-8601=seconds)"
        "$0" "$trace_seed" || return $?
        printf 'seed=%s result=%s\n' "$trace_seed" "$run_base/seed${trace_seed}/summary.json"
        "$python" -m json.tool "$run_base/seed${trace_seed}/summary.json"
    } 2>&1 | tee -a "$seed_log"
}

if (( $# == 0 )); then
    read -r -a trace_seeds <<< "${TRACE_SEEDS:-3407}"
    if [[ "${TRACE_DRY_RUN:-0}" == 1 || "$parallel_seeds" == 0 ]]; then
        for trace_seed in "${trace_seeds[@]}"; do
            run_seed "$trace_seed"
        done
    else
        if (( ${#trace_seeds[@]} > 3 )); then
            echo "TRACE parallel mode supports at most three full-GPU seeds" >&2
            exit 2
        fi
        pids=()
        trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
        for trace_seed in "${trace_seeds[@]}"; do
            run_seed "$trace_seed" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do
            wait "$pid" || failed=1
        done
        (( failed == 0 )) || exit 1
    fi
    if [[ "${TRACE_DRY_RUN:-0}" != 1 ]]; then
        "$python" "$runner" summarize-suite --run-root "$run_base" \
            --seeds "${trace_seeds[@]}" --orders canonical
    fi
    exit 0
fi

seed=${1:-3407}
run=$run_base/seed${seed}
runner_args=(--task-order canonical)

if [[ "${TRACE_DRY_RUN:-0}" == 1 ]]; then
    printf 'seed=%s order=canonical methods=%s trainable=%s gpus=%s output=%s\n' \
        "$seed" "${methods[*]}" "$trainable" "${TRACE_GPUS:-0,1,2,3,4,5,6,7}" "$run"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=${TRACE_GPUS:-0,1,2,3,4,5,6,7}
nproc=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
cache_root=${TRACE_CACHE_ROOT:-/tmp/cagd_trace/$(basename "$run_base")}/seed${seed}
export TRITON_CACHE_DIR=$cache_root/triton
export TORCH_EXTENSIONS_DIR=$cache_root/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$cache_root/torchinductor
export VLLM_CACHE_ROOT=$cache_root/vllm
export VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$run"
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
    local command="train-$method"
    [[ "$method" != opr_sc ]] || command=train-opr-sc
    local args=("$command" --checkpoint "$source" --stage "$stage" --seed "$seed" --trainable "$trainable" --output "$output")
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
    selected opr_sc && stage_inference opr_sc 0 "$shared_checkpoint" "" "$shared/support_opr_sc_stage1.jsonl"
    selected cagd && stage_inference cagd 0 "$shared_checkpoint" "" "$shared/support_cagd_stage1.jsonl"
fi

run_shared_sft_method() {
    local method=$1 source support= next_support= stage output
    source=$(checkpoint "$shared")
    case "$method" in
        replay|opr|opr_sc|cagd) support=$shared/support_${method}_stage1.jsonl ;;
    esac
    for stage in 1 2 3 4 5 6 7; do
        output=$run/$method/stage$stage
        train_stage "$method" "$stage" "$source" "$output" "$support"
        source=$(checkpoint "$output")
        next_support=
        if [[ "$stage" -lt 7 ]]; then
            case "$method" in
                replay|opr|opr_sc|cagd) next_support=$output/support_${method}_stage$((stage + 1)).jsonl ;;
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
