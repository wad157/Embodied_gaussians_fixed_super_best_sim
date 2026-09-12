#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_ROOT="${1:-$ROOT_DIR/outputs/endogaussian_sim_unified_three_repeats_v1}"
NEW_ROOT="${2:-$ROOT_DIR/outputs/endogaussian_sim_unified_three_repeats_v2}"

if [[ $# -gt 2 ]]; then
    echo "用法：bash scripts/run_endogaussian_sim_baseline_v2_two_gpus.sh [旧结果根目录] [新结果根目录]" >&2
    exit 2
fi
if [[ ! -d "$OLD_ROOT/runs/repeat_02/sim03/model/point_cloud/iteration_3000" ]]; then
    echo "ERROR: 旧根目录没有六个已完成 checkpoint：$OLD_ROOT" >&2
    exit 1
fi
if [[ -e "$NEW_ROOT/comparison_mean_std.json" ]]; then
    echo "ERROR: v2 已汇总，拒绝覆盖：$NEW_ROOT" >&2
    exit 1
fi
mkdir -p "$NEW_ROOT/logs"

run_reexport() {
    local dataset_key="$1" repeat_id="$2" seed="$3" gpu="$4"
    local old_run="$OLD_ROOT/runs/$repeat_id/$dataset_key"
    local new_run="$NEW_ROOT/runs/$repeat_id/$dataset_key"
    if [[ -f "$new_run/status.txt" ]] && grep -q '^status=complete$' "$new_run/status.txt"; then
        return
    fi
    if [[ -e "$new_run" ]]; then
        echo "ERROR: v2 中存在不完整 re-export：$new_run" >&2
        return 1
    fi
    SIM_GPU_ID="$gpu" bash "$ROOT_DIR/scripts/reexport_endogaussian_sim_baseline_once.sh" \
        "$dataset_key" "$repeat_id" "$seed" "$old_run" "$new_run" \
        'v1_absolute_xyz_decoder_rejected_after_visual_audit'
}

run_fresh() {
    local dataset_key="$1" repeat_id="$2" seed="$3" gpu="$4"
    local run="$NEW_ROOT/runs/$repeat_id/$dataset_key"
    if [[ -f "$run/status.txt" ]] && grep -q '^status=complete$' "$run/status.txt"; then
        return
    fi
    if [[ -e "$run" ]]; then
        echo "ERROR: v2 中存在不完整 fresh run：$run" >&2
        return 1
    fi
    SIM_GPU_ID="$gpu" bash "$ROOT_DIR/scripts/run_endogaussian_sim_baseline_once.sh" \
        "$dataset_key" "$repeat_id" "$seed" "$run"
}

# Re-export the six valid frozen checkpoints. Only the rejected trajectory
# decoder changes; RGB/alpha are hard-linked and their metrics are recomputed.
run_reexport sim01 repeat_01 0 0 & p0=$!
run_reexport sim02 repeat_01 0 1 & p1=$!
wait "$p0"
wait "$p1"

run_reexport sim03 repeat_01 0 0 & p0=$!
run_reexport sim01 repeat_02 1 1 & p1=$!
wait "$p0"
wait "$p1"

run_reexport sim02 repeat_02 1 0 & p0=$!
run_reexport sim03 repeat_02 1 1 & p1=$!
wait "$p0"
wait "$p1"

# Seed 2 has not been trained and therefore runs entirely with decoder v2.
run_fresh sim03 repeat_03 2 0 & p0=$!
run_fresh sim01 repeat_03 2 1 & p1=$!
wait "$p0"
wait "$p1"
run_fresh sim02 repeat_03 2 1

# All nine runs now exist, so the ordinary fail-closed summarizer only validates,
# aggregates, and writes the mean/sample-standard-deviation tables.
SIM_GPU_ID=0 bash "$ROOT_DIR/scripts/run_endogaussian_sim_baseline_three_repeats.sh" \
    "$NEW_ROOT"
