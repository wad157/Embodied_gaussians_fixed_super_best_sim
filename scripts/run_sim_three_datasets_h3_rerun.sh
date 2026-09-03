#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/sim_three_datasets_h3_rerun_v1}"

SIM01="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"
SIM02="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"
SIM03="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"

SIM01_ASSETS="$ROOT_DIR/outputs/sim01_planar_x_cotracker_foundation_v1/flow_depth_assets"
SIM02_ASSETS="$ROOT_DIR/outputs/sim02_planar_y_cotracker_foundation_v1/flow_depth_assets"
SIM03_ASSETS="$ROOT_DIR/outputs/sim_cotracker3_foundation_depth_v1/flow_depth_assets"

SIM01_DEPTH="$SIM01/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM02_DEPTH="$SIM02/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM03_DEPTH="$SIM03/estimated_depth/foundation_stereo_rgb_v1"

SIM01_BASELINE="$ROOT_DIR/outputs/sim_three_datasets_previous_c_complete_v2/sim01"
SIM02_BASELINE="$ROOT_DIR/outputs/sim_three_datasets_previous_c_complete_v2/sim02"
SIM03_BASELINE="$ROOT_DIR/outputs/sim_foundation_depth_complete_v1"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_three_datasets_h3_rerun.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT/logs"
printf 'status=running\nalgorithm=h3_h5_adam_h1_safe_projection\n' \
    >"$RUN_ROOT/status.txt"

run_dataset() {
    local dataset_id="$1"
    local dataset="$2"
    local assets="$3"
    local depth="$4"
    local baseline="$5"
    local output="$RUN_ROOT/$dataset_id"

    SIM_DATASET="$dataset" \
    SIM_ESTIMATED_DEPTH_ROOT="$depth" \
    FLOW_DEPTH_ASSET_ROOT="$assets" \
    SIM_FOUNDATION_BASELINE_ROOT="$baseline" \
        bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_hierarchical_complete.sh" \
        "$output" >"$RUN_ROOT/logs/${dataset_id}.log" 2>&1
}

# Each dataset uses GPU0 for 7:1 reconstruction and GPU1 for 80/20 prediction.
# Datasets run sequentially to keep GPU memory and Warp execution isolated.
run_dataset sim01 "$SIM01" "$SIM01_ASSETS" "$SIM01_DEPTH" "$SIM01_BASELINE"
run_dataset sim02 "$SIM02" "$SIM02_ASSETS" "$SIM02_DEPTH" "$SIM02_BASELINE"
run_dataset sim03 "$SIM03" "$SIM03_ASSETS" "$SIM03_DEPTH" "$SIM03_BASELINE"

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_h1_safe_three_datasets.py" \
    --sim01 "$RUN_ROOT/sim01/comparison_complete.json" \
    --sim02 "$RUN_ROOT/sim02/comparison_complete.json" \
    --sim03 "$RUN_ROOT/sim03/comparison_complete.json" \
    --output "$RUN_ROOT/comparison_all.md"

printf 'status=complete\nalgorithm=h3_h5_adam_h1_safe_projection\n' \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "三套H3测评完成：$RUN_ROOT/comparison_all.md"

