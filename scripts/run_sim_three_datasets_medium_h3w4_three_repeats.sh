#!/usr/bin/env bash
set -euo pipefail

# 三个数据集分别进行3次完整A/B/C测评，并汇总均值与样本标准差。
# 每个数据集内部由通用脚本并行使用GPU0（重建）和GPU1（未来预测）。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
RUN_ROOT="${1:-$ROOT_DIR/outputs/sim_three_datasets_medium_h3w4_three_repeats_v1}"
REPEAT_COUNT=3
WAIT_FOR_PIDS="${SIM_WAIT_FOR_PIDS:-}"

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
    echo "用法：bash scripts/run_sim_three_datasets_medium_h3w4_three_repeats.sh [新输出目录]" >&2
    exit 2
fi
if [[ -f "$RUN_ROOT/status.txt" ]] && grep -q '^status=complete$' "$RUN_ROOT/status.txt"; then
    echo "ERROR: 三次复测已经完成，拒绝覆盖：$RUN_ROOT" >&2
    exit 1
fi
for required in \
    "$PYTHON" \
    "$SIM01/episode.json" "$SIM02/episode.json" "$SIM03/episode.json" \
    "$SIM01_ASSETS/bindings.npz" "$SIM01_ASSETS/observations.npz" "$SIM01_ASSETS/report.json" \
    "$SIM02_ASSETS/bindings.npz" "$SIM02_ASSETS/observations.npz" "$SIM02_ASSETS/report.json" \
    "$SIM03_ASSETS/bindings.npz" "$SIM03_ASSETS/observations.npz" "$SIM03_ASSETS/report.json" \
    "$SIM01_DEPTH/depth_generation_summary.json" \
    "$SIM02_DEPTH/depth_generation_summary.json" \
    "$SIM03_DEPTH/depth_generation_summary.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少正式测评输入：$required" >&2
        exit 1
    fi
done

mkdir -p "$RUN_ROOT/logs"
if [[ -n "$WAIT_FOR_PIDS" ]]; then
    printf '%s\n' \
        'status=queued' \
        "pid=$$" \
        "wait_for_pids=$WAIT_FOR_PIDS" \
        'datasets=sim01,sim02,sim03' \
        'repeat_count=3' \
        >"$RUN_ROOT/status.txt"
    for wait_pid in $WAIT_FOR_PIDS; do
        if [[ ! "$wait_pid" =~ ^[0-9]+$ ]]; then
            echo "ERROR: 非法等待PID：$wait_pid" >&2
            exit 2
        fi
        if kill -0 "$wait_pid" 2>/dev/null; then
            printf '%s\n' "waiting_for_pid=$wait_pid" >>"$RUN_ROOT/logs/progress.log"
            tail --pid="$wait_pid" -f /dev/null
        fi
    done
fi
printf '%s\n' \
    'status=running' \
    "pid=$$" \
    'datasets=sim01,sim02,sim03' \
    'repeat_count=3' \
    'initial_distance_stiffness=0.20' \
    'initial_shape_stiffness=0.004' \
    'stiffness_update_mode=differentiable_global' \
    'global_horizon_weights=1.5,2.0,4.0' \
    'distance_bounds=0.01,2.00' \
    'maximum_log_offset=2.302585093' \
    >"$RUN_ROOT/status.txt"

run_dataset() {
    local repeat_id="$1"
    local dataset_id="$2"
    local dataset="$3"
    local asset_root="$4"
    local depth_root="$5"
    local output="$RUN_ROOT/$repeat_id/$dataset_id"
    local log="$RUN_ROOT/logs/${repeat_id}_${dataset_id}.log"

    if [[ -f "$output/comparison_complete.json" ]]; then
        echo "跳过已完成：$repeat_id/$dataset_id" | tee -a "$RUN_ROOT/logs/progress.log"
        return
    fi
    if [[ -e "$output" ]]; then
        echo "ERROR: 存在不完整输出，拒绝覆盖：$output" >&2
        exit 1
    fi

    printf '%s\n' "start=$repeat_id/$dataset_id" >>"$RUN_ROOT/logs/progress.log"
    SIM_DATASET="$dataset" \
    SIM_ESTIMATED_DEPTH_ROOT="$depth_root" \
    FLOW_DEPTH_ASSET_ROOT="$asset_root" \
    SIM_METHOD_C="pbd_cotracker_foundation_depth_global_only_h3w4" \
    SIM_STIFFNESS_UPDATE_MODE="differentiable_global" \
    SIM_STIFFNESS_AUTOGRAD_UNROLL_STEPS=4 \
    SIM_STIFFNESS_AUTOGRAD_REGION_COUNT=12 \
    SIM_C_LABEL="C：B + 全局刚度/阻尼更新（H1:H2:H3=1.5:2:4）" \
    SIM_INITIAL_PAPER_DISTANCE_STIFFNESS=0.20 \
    SIM_INITIAL_PAPER_SHAPE_STIFFNESS=0.004 \
    SIM_STIFFNESS_DISTANCE_MINIMUM=0.01 \
    SIM_STIFFNESS_DISTANCE_MAXIMUM=2.00 \
    SIM_STIFFNESS_MAXIMUM_LOG_OFFSET=2.302585093 \
    SIM_STIFFNESS_H1_WEIGHT=1.5 \
    SIM_STIFFNESS_H2_WEIGHT=2.0 \
    SIM_STIFFNESS_H3_WEIGHT=4.0 \
        bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_dataset_complete.sh" \
        "$output" >"$log" 2>&1
    printf '%s\n' "complete=$repeat_id/$dataset_id" >>"$RUN_ROOT/logs/progress.log"
}

for repeat_number in $(seq 1 "$REPEAT_COUNT"); do
    repeat_id="$(printf 'repeat_%02d' "$repeat_number")"
    # 数据集之间顺序执行，确保每套只读取自己的深度与轨迹缓存。
    run_dataset "$repeat_id" sim01 "$SIM01" "$SIM01_ASSETS" "$SIM01_DEPTH"
    run_dataset "$repeat_id" sim02 "$SIM02" "$SIM02_ASSETS" "$SIM02_DEPTH"
    run_dataset "$repeat_id" sim03 "$SIM03" "$SIM03_ASSETS" "$SIM03_DEPTH"
done

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_three_datasets_three_repeats.py" \
    --root "$RUN_ROOT" \
    --repeat-count "$REPEAT_COUNT"

printf '%s\n' \
    'status=complete' \
    'datasets=sim01,sim02,sim03' \
    'repeat_count=3' \
    'initial_distance_stiffness=0.20' \
    'initial_shape_stiffness=0.004' \
    'stiffness_update_mode=differentiable_global' \
    'global_horizon_weights=1.5,2.0,4.0' \
    >"$RUN_ROOT/status.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$RUN_ROOT/SHA256SUMS"
echo "完成：$RUN_ROOT/comparison_mean_std.md"
