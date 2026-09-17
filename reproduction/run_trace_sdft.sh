#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRACE_SEEDS=${TRACE_SEEDS:-"3407 3408 3409"}
export TRACE_METHODS=sdft
export TRACE_RUN_ROOT=${TRACE_RUN_ROOT:-$root/runs/reproduction/trace_sdft}
exec "$root/reproduction/run_trace_experiment.sh" "$@"
