#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRACE_SEEDS=${TRACE_SEEDS:-"3407 3408 3409"}
export TRACE_PARALLEL_SEEDS=${TRACE_PARALLEL_SEEDS:-1}
export TRACE_METHODS="sequential replay opr opr_sc cagd"
export TRACE_RUN_ROOT=${TRACE_RUN_ROOT:-$root/runs/reproduction/trace_without_sdft}
exec "$root/reproduction/run_trace_experiment.sh" "$@"
