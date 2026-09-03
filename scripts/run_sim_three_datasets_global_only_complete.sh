#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/sim_three_datasets_global_only_foundation_complete_v1}"

SIM01="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"
SIM02="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"
SIM03="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"

SIM01_ASSETS="$ROOT_DIR/outputs/sim01_planar_x_cotracker_foundation_v1/flow_depth_assets"
SIM02_ASSETS="$ROOT_DIR/outputs/sim02_planar_y_cotracker_foundation_v1/flow_depth_assets"
SIM03_ASSETS="$ROOT_DIR/outputs/sim_cotracker3_foundation_depth_v1/flow_depth_assets"

SIM01_DEPTH="$SIM01/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM02_DEPTH="$SIM02/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM03_DEPTH="$SIM03/estimated_depth/foundation_stereo_rgb_v1"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_three_datasets_global_only_complete.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$RUN_ROOT" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT/logs"
printf 'status=running\nalgorithm=differentiable_global_distance_and_damping\ndepth=foundation_stereo_per_dataset\n' \
    >"$RUN_ROOT/status.txt"

run_dataset() {
    local dataset_id="$1"
    local dataset="$2"
    local assets="$3"
    local depth="$4"
    local output="$RUN_ROOT/$dataset_id"

    SIM_DATASET="$dataset" \
    SIM_ESTIMATED_DEPTH_ROOT="$depth" \
    FLOW_DEPTH_ASSET_ROOT="$assets" \
    SIM_METHOD_C="pbd_cotracker_foundation_depth_global_only" \
    SIM_STIFFNESS_UPDATE_MODE="differentiable_global" \
    SIM_STIFFNESS_AUTOGRAD_UNROLL_STEPS=4 \
    SIM_STIFFNESS_AUTOGRAD_REGION_COUNT=12 \
    SIM_C_LABEL="C：B + 全局distance刚度与全局阻尼更新（无区域/逐粒子刚度）" \
        bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_dataset_complete.sh" \
        "$output" >"$RUN_ROOT/logs/${dataset_id}.log" 2>&1
}

# 数据集之间顺序运行；单个数据集内部GPU0执行7:1重建，GPU1执行80/20预测。
# A/B/C均重新运行，不复用旧测评数字。
run_dataset sim01 "$SIM01" "$SIM01_ASSETS" "$SIM01_DEPTH"
run_dataset sim02 "$SIM02" "$SIM02_ASSETS" "$SIM02_DEPTH"
run_dataset sim03 "$SIM03" "$SIM03_ASSETS" "$SIM03_DEPTH"

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_h1_safe_three_datasets.py" \
    --sim01 "$RUN_ROOT/sim01/comparison_complete.json" \
    --sim02 "$RUN_ROOT/sim02/comparison_complete.json" \
    --sim03 "$RUN_ROOT/sim03/comparison_complete.json" \
    --method-c "pbd_cotracker_foundation_depth_global_only" \
    --output "$RUN_ROOT/comparison_all.md"

printf 'status=complete\nalgorithm=differentiable_global_distance_and_damping\ndepth=foundation_stereo_per_dataset\n' \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "三套全局刚度完整测评完成：$RUN_ROOT/comparison_all.md"
