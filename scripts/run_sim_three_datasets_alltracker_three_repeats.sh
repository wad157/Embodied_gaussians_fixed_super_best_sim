#!/usr/bin/env bash
set -euo pipefail

# 将已有 AllTracker 完整测评作为第1次，复用其观测资产，
# 在 GPU0 上连续、串行执行第2和第3次，再对三次结果统一汇总。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/sim_three_datasets_alltracker_medium_h3w4_three_repeats_v1}"
BASELINE_ROOT="${SIM_ALLTRACKER_BASELINE_ROOT:-$ROOT_DIR/outputs/sim_three_datasets_alltracker_medium_h3w4_complete_v1}"
ASSET_ROOT="${SIM_ALLTRACKER_ASSET_ROOT:-$BASELINE_ROOT/assets}"
REPEAT_COUNT=3
NEW_REPEAT_COUNT=2
GPU_ID="${SIM_GPU_ID:-0}"

SIM01="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"
SIM02="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"
SIM03="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"

SIM01_DEPTH="$SIM01/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM02_DEPTH="$SIM02/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM03_DEPTH="$SIM03/estimated_depth/foundation_stereo_rgb_v1"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_three_datasets_alltracker_three_repeats.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$RUN_ROOT" >&2
    exit 1
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SIM_GPU_ID 必须是非负整数：$GPU_ID" >&2
    exit 2
fi

for required in \
    "$PYTHON" \
    "$BASELINE_ROOT/sim01/comparison_complete.json" \
    "$BASELINE_ROOT/sim02/comparison_complete.json" \
    "$BASELINE_ROOT/sim03/comparison_complete.json" \
    "$SIM01/episode.json" "$SIM02/episode.json" "$SIM03/episode.json" \
    "$SIM01_DEPTH/depth_generation_summary.json" \
    "$SIM02_DEPTH/depth_generation_summary.json" \
    "$SIM03_DEPTH/depth_generation_summary.json" \
    "$ASSET_ROOT/sim01/flow_depth_assets/bindings.npz" \
    "$ASSET_ROOT/sim01/flow_depth_assets/observations.npz" \
    "$ASSET_ROOT/sim01/flow_depth_assets/report.json" \
    "$ASSET_ROOT/sim02/flow_depth_assets/bindings.npz" \
    "$ASSET_ROOT/sim02/flow_depth_assets/observations.npz" \
    "$ASSET_ROOT/sim02/flow_depth_assets/report.json" \
    "$ASSET_ROOT/sim03/flow_depth_assets/bindings.npz" \
    "$ASSET_ROOT/sim03/flow_depth_assets/observations.npz" \
    "$ASSET_ROOT/sim03/flow_depth_assets/report.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少 AllTracker 正式测评输入：$required" >&2
        exit 1
    fi
done

mkdir -p "$RUN_ROOT/logs"
printf '%s\n' \
    'status=running' \
    "pid=$$" \
    'tracker=AllTracker' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count=3' \
    'existing_repeat_count=1' \
    'new_repeat_count=2' \
    "repeat_01_source=$BASELINE_ROOT" \
    'execution=continuous_serial' \
    "gpu=$GPU_ID" \
    'initial_distance_stiffness=0.20' \
    'initial_shape_stiffness=0.004' \
    'stiffness_update_mode=differentiable_global' \
    'global_horizon_weights=1.5,2.0,4.0' \
    >"$RUN_ROOT/status.txt"

run_dataset() {
    local repeat_id="$1"
    local dataset_id="$2"
    local dataset="$3"
    local depth_root="$4"
    local flow_depth_assets="$ASSET_ROOT/$dataset_id/flow_depth_assets"
    local output="$RUN_ROOT/$repeat_id/$dataset_id"
    local log="$RUN_ROOT/logs/${repeat_id}_${dataset_id}.log"

    printf '%s\n' "start=$repeat_id/$dataset_id" >>"$RUN_ROOT/logs/progress.log"
    SIM_DATASET="$dataset" \
    SIM_ESTIMATED_DEPTH_ROOT="$depth_root" \
    FLOW_DEPTH_ASSET_ROOT="$flow_depth_assets" \
    SIM_METHOD_B="pbd_alltracker_foundation_depth" \
    SIM_B_LABEL="B：PBD + AllTracker轨迹校正 + FoundationStereo RGB深度" \
    SIM_TRACKER_NAME="AllTracker" \
    SIM_METHOD_C="pbd_alltracker_foundation_depth_global_only_h3w4" \
    SIM_C_LABEL="C：B + 全局刚度/阻尼更新（H1:H2:H3=1.5:2:4）" \
    SIM_STIFFNESS_UPDATE_MODE="differentiable_global" \
    SIM_STIFFNESS_AUTOGRAD_UNROLL_STEPS=4 \
    SIM_STIFFNESS_AUTOGRAD_REGION_COUNT=12 \
    SIM_INITIAL_PAPER_DISTANCE_STIFFNESS=0.20 \
    SIM_INITIAL_PAPER_SHAPE_STIFFNESS=0.004 \
    SIM_STIFFNESS_DISTANCE_MINIMUM=0.01 \
    SIM_STIFFNESS_DISTANCE_MAXIMUM=2.00 \
    SIM_STIFFNESS_MAXIMUM_LOG_OFFSET=2.302585093 \
    SIM_STIFFNESS_H1_WEIGHT=1.5 \
    SIM_STIFFNESS_H2_WEIGHT=2.0 \
    SIM_STIFFNESS_H3_WEIGHT=4.0 \
    SIM_RUN_CAPABILITIES_SERIAL=1 \
    SIM_GPU_ID="$GPU_ID" \
        bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_dataset_complete.sh" \
        "$output" >"$log" 2>&1
    printf '%s\n' "complete=$repeat_id/$dataset_id" >>"$RUN_ROOT/logs/progress.log"
}

for repeat_number in $(seq 2 "$REPEAT_COUNT"); do
    repeat_id="$(printf 'repeat_%02d' "$repeat_number")"
    run_dataset "$repeat_id" sim01 "$SIM01" "$SIM01_DEPTH"
    run_dataset "$repeat_id" sim02 "$SIM02" "$SIM02_DEPTH"
    run_dataset "$repeat_id" sim03 "$SIM03" "$SIM03_DEPTH"
done

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_three_datasets_alltracker_three_repeats.py" \
    --root "$RUN_ROOT" \
    --baseline-root "$BASELINE_ROOT" \
    --asset-root "$ASSET_ROOT" \
    --repeat-count "$REPEAT_COUNT"

printf '%s\n' \
    'status=complete' \
    'tracker=AllTracker' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count=3' \
    'existing_repeat_count=1' \
    "new_repeat_count=$NEW_REPEAT_COUNT" \
    "repeat_01_source=$BASELINE_ROOT" \
    'execution=continuous_serial' \
    "gpu=$GPU_ID" \
    'initial_distance_stiffness=0.20' \
    'initial_shape_stiffness=0.004' \
    'stiffness_update_mode=differentiable_global' \
    'global_horizon_weights=1.5,2.0,4.0' \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "完成：$RUN_ROOT/comparison_mean_std.md"
