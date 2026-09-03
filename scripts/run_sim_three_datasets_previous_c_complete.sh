#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/sim_three_datasets_previous_c_complete_v1}"

SIM01="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"
SIM02="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"
SIM03="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"

SIM01_ASSETS="$ROOT_DIR/outputs/sim01_planar_x_cotracker_foundation_v1"
SIM02_ASSETS="$ROOT_DIR/outputs/sim02_planar_y_cotracker_foundation_v1"
SIM03_ASSETS="$ROOT_DIR/outputs/sim_cotracker3_foundation_depth_v1"

SIM01_DEPTH="$SIM01/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM02_DEPTH="$SIM02/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
SIM03_DEPTH="$SIM03/estimated_depth/foundation_stereo_rgb_v1"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_three_datasets_previous_c_complete.sh [新输出目录]" >&2
    exit 2
fi
if [[ -f "$RUN_ROOT/status.txt" ]] \
    && grep -q '^status=complete$' "$RUN_ROOT/status.txt"; then
    echo "ERROR: 测评已经完成，拒绝覆盖：$RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT/logs"
printf 'status=running\npid=%s\n' "$$" >"$RUN_ROOT/status.txt"

# SIM-01上次FoundationStereo在280/300帧处中断。生成器会跳过已有帧，
# 这里只续做缺失帧；轨迹、深度和三角面绑定始终来自SIM-01自身。
SIM_DATASET="$SIM01" \
SIM_ESTIMATED_DEPTH_ROOT="$SIM01_DEPTH" \
SIM_GPU_ID=0 \
SIM_RECTIFICATION_ALPHA=0.85 \
SIM_DEPTH_SOURCE_LABEL="sim01_foundation_stereo_rgb_rig_aware_v1" \
    bash "$ROOT_DIR/scripts/prepare_sim_numbered_dataset_assets.sh" \
    "$SIM01_ASSETS" >"$RUN_ROOT/logs/sim01_prepare.log" 2>&1

run_dataset() {
    local dataset_id="$1"
    local dataset="$2"
    local asset_root="$3"
    local depth_root="$4"
    local output="$RUN_ROOT/$dataset_id"

    if [[ -f "$output/comparison_complete.json" ]]; then
        echo "跳过已完成数据集：$dataset_id"
        return
    fi
    if [[ -e "$output" ]]; then
        echo "ERROR: 数据集存在不完整输出，拒绝覆盖：$output" >&2
        exit 1
    fi

    SIM_DATASET="$dataset" \
    SIM_ESTIMATED_DEPTH_ROOT="$depth_root" \
    FLOW_DEPTH_ASSET_ROOT="$asset_root/flow_depth_assets" \
        bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_dataset_complete.sh" \
        "$output" >"$RUN_ROOT/logs/${dataset_id}_complete.log" 2>&1
}

# 每个数据集内部：GPU0运行7:1重建，GPU1运行80/20未来预测；
# 每项均依次重新运行A纯PBD、B轨迹校正、C上次H1-safe刚度算法。
run_dataset sim01 "$SIM01" "$SIM01_ASSETS" "$SIM01_DEPTH"
run_dataset sim02 "$SIM02" "$SIM02_ASSETS" "$SIM02_DEPTH"
run_dataset sim03 "$SIM03" "$SIM03_ASSETS" "$SIM03_DEPTH"

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_h1_safe_three_datasets.py" \
    --sim01 "$RUN_ROOT/sim01/comparison_complete.json" \
    --sim02 "$RUN_ROOT/sim02/comparison_complete.json" \
    --sim03 "$RUN_ROOT/sim03/comparison_complete.json" \
    --output "$RUN_ROOT/comparison_all.md"

printf 'status=complete\nalgorithm=previous_c_h3_h5_adam_h1_safe_projection\n' \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "三套完整测评完成：$RUN_ROOT/comparison_all.md"
