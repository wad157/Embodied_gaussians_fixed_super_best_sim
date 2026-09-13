#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/embodied_gaussians_sim_unified_three_repeats_v1}"
GPU_ID="${SIM_GPU_ID:-0}"
EG_PY="${EG_ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"

run_one() {
    local dataset_key="$1"
    local repeat="$2"
    local seed="$3"
    local output="$OUTPUT_ROOT/$repeat/$dataset_key"
    if [[ -f "$output/status.txt" ]] && rg -q '^status=complete$' "$output/status.txt"; then
        echo "[EG batch] reuse completed $dataset_key $repeat seed=$seed"
        return
    fi
    if [[ -e "$output" ]]; then
        echo "ERROR: incomplete/colliding output exists: $output" >&2
        exit 1
    fi
    echo "[EG batch] start $dataset_key $repeat seed=$seed"
    SIM_GPU_ID="$GPU_ID" EG_ACTUATION=psm-fk-collision-only \
        bash "$ROOT_DIR/scripts/run_embodied_gaussians_sim_baseline_once.sh" \
        "$dataset_key" "$repeat" "$seed" "$output"
}

mkdir -p "$OUTPUT_ROOT"
run_one sim01 repeat_01 0
run_one sim01 repeat_02 1
run_one sim02 repeat_01 0
run_one sim02 repeat_02 1
run_one sim02 repeat_03 2
run_one sim03 repeat_01 0
run_one sim03 repeat_02 1
run_one sim03 repeat_03 2

"$EG_PY" "$ROOT_DIR/scripts/summarize_embodied_gaussians_sim_baseline.py" \
    --root "$OUTPUT_ROOT"
echo "[EG batch] complete $OUTPUT_ROOT"
