#!/usr/bin/env bash
set -euo pipefail

# SIM-03 / AllTracker 新联合协议：每个方法只进行一次完整因果 rollout。
# 前80%按7:1留出评估重建；后20%冻结视觉与刚度更新并评估未来预测。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"
ASSET_ROOT="${SIM_ALLTRACKER_ASSET_ROOT:-$ROOT_DIR/outputs/sim_three_datasets_alltracker_medium_h3w4_complete_v1/assets/sim03/flow_depth_assets}"
DEPTH_ROOT="$DATASET/estimated_depth/foundation_stereo_rgb_v1"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim03_alltracker_unified_80_20_7to1_once_v1}"
NODE_MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
GPU_ID="${SIM_GPU_ID:-0}"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FUTURE_START=$((FRAME_COUNT * 4 / 5))

METHOD_B="pbd_alltracker_foundation_depth"
METHOD_C="pbd_alltracker_foundation_depth_global_only_h3w4"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim03_alltracker_unified_protocol_once.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SIM_GPU_ID 必须是非负整数：$GPU_ID" >&2
    exit 2
fi
for required in \
    "$PYTHON" \
    "$DATASET/episode.json" \
    "$ASSET_ROOT/bindings.npz" \
    "$ASSET_ROOT/observations.npz" \
    "$ASSET_ROOT/report.json" \
    "$DEPTH_ROOT/depth_generation_summary.json" \
    "$NODE_MANIFEST" \
    "$BOUNDARY"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少联合协议输入：$required" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_ROOT/logs"
printf '%s\n' \
    'status=running' \
    "pid=$$" \
    'dataset=SIM-03' \
    'tracker=AllTracker' \
    'methods=B,C' \
    'runs_per_method=1' \
    "frame_count=$FRAME_COUNT" \
    "assimilation_end_exclusive=$FUTURE_START" \
    'reconstruction_protocol=first_80_percent_7_to_1_holdout' \
    'future_protocol=last_20_percent_open_loop' \
    'single_rollout_per_method=true' \
    "gpu=$GPU_ID" \
    >"$OUTPUT_ROOT/status.txt"

run_method() {
    local method="$1"
    local group="$OUTPUT_ROOT/methods/$method"
    local log="$OUTPUT_ROOT/logs/${method}.log"
    local stiffness_args=(--no-online-stiffness-update)
    if [[ "$method" == "$METHOD_C" ]]; then
        stiffness_args=(
            --online-stiffness-update
            --stiffness-update-mode differentiable_global
            --stiffness-log-learning-rate 0.03
            --stiffness-maximum-log-step 0.02
            --stiffness-signal-ema-decay 0.90
            --stiffness-spatial-smoothing-iterations 3
            --stiffness-spatial-smoothing-blend 0.35
            --stiffness-strain-signal-weight 0.20
            --stiffness-autograd-unroll-steps 4
            --stiffness-autograd-region-count 12
            --stiffness-distance-minimum 0.01
            --stiffness-distance-maximum 2.00
            --stiffness-autograd-maximum-log-offset 2.302585093
            --stiffness-global-horizon-weights 1.5 2.0 4.0
        )
    fi

    mkdir -p "$group"
    printf '%s\n' "start=$method" >>"$OUTPUT_ROOT/logs/progress.log"
    SIM_DATASET="$DATASET" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    TORCH_EXTENSIONS_DIR="$ROOT_DIR/.tmp/torch_extensions_sim03_unified_gpu${GPU_ID}" \
        bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
            --evaluation-start-frame 0 \
            --evaluation-frame-count 0 \
            --evaluation-physics-steps-per-frame 2 \
            --evaluation-label "sim03_unified_${method}" \
            --benchmark-output "$group/artifacts" \
            --evaluation-render-images \
            --evaluation-holdout-stride 8 \
            --evaluation-holdout-offset 7 \
            --evaluation-open-loop-start-frame "$FUTURE_START" \
            --evaluation-render-frame-mode holdout_future \
            --visual-feedback-mode trajectory \
            --flow-depth-bindings "$ASSET_ROOT/bindings.npz" \
            --flow-depth-observations "$ASSET_ROOT/observations.npz" \
            --flow-depth-position-gain 0.70 \
            --flow-depth-velocity-gain 0.20 \
            --flow-depth-absolute-position-weight 0.85 \
            --flow-depth-solver-regularization 0.01 \
            --flow-depth-solver-iterations 24 \
            --flow-depth-robust-residual-mm 5.0 \
            --flow-depth-maximum-position-correction-mm 2.0 \
            --flow-depth-maximum-velocity-correction-m-s 0.06 \
            --sim-grasp-boundary-mode known_grasp_region \
            "${stiffness_args[@]}" \
            --initial-paper-distance-stiffness 0.20 \
            --initial-paper-shape-stiffness 0.004 \
            >"$log" 2>&1

    for capability in reconstruction_7to1 future_80to20; do
        local metrics="$OUTPUT_ROOT/metrics/$capability/$method"
        local selection_args=()
        mkdir -p "$metrics"
        if [[ "$capability" == "reconstruction_7to1" ]]; then
            selection_args=(
                --frame-start 0
                --frame-end-exclusive "$FUTURE_START"
                --frame-stride 8
                --frame-offset 7
            )
        else
            selection_args=(--frame-start "$FUTURE_START")
        fi
        "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
            --reference-dataset "$DATASET" \
            --prediction "$group/artifacts/predicted_trajectories.npz" \
            --controlled-boundary "$BOUNDARY" \
            --evaluation-node-manifest "$NODE_MANIFEST" \
            "${selection_args[@]}" \
            --output "$metrics/trajectory_metrics.json" \
            >>"$log" 2>&1
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
            "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
            --reference-dataset "$DATASET" \
            --prediction-dir "$group/artifacts/renders" \
            "${selection_args[@]}" \
            --device cuda:0 \
            --output "$metrics/render_metrics.json" \
            >>"$log" 2>&1
    done
    printf '%s\n' "complete=$method" >>"$OUTPUT_ROOT/logs/progress.log"
}

run_method "$METHOD_B"
run_method "$METHOD_C"

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim03_unified_protocol.py" \
    --root "$OUTPUT_ROOT" \
    --dataset "$DATASET" \
    --observation-report "$ASSET_ROOT/report.json" \
    --future-start "$FUTURE_START"

printf '%s\n' \
    'status=complete' \
    'dataset=SIM-03' \
    'tracker=AllTracker' \
    'methods=B,C' \
    'runs_per_method=1' \
    "frame_count=$FRAME_COUNT" \
    "assimilation_end_exclusive=$FUTURE_START" \
    'reconstruction_protocol=first_80_percent_7_to_1_holdout' \
    'future_protocol=last_20_percent_open_loop' \
    'single_rollout_per_method=true' \
    "gpu=$GPU_ID" \
    >"$OUTPUT_ROOT/status.txt"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "完成：$OUTPUT_ROOT/comparison_unified.md"
