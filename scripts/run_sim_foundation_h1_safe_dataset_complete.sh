#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1}"
ASSET_ROOT="${FLOW_DEPTH_ASSET_ROOT:-$ROOT_DIR/outputs/sim_inplane_cotracker3_foundation_depth_v1/flow_depth_assets}"
DEPTH_ROOT="${SIM_ESTIMATED_DEPTH_ROOT:-$DATASET/estimated_depth/foundation_stereo_rgb_v1}"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_inplane_foundation_h1_safe_complete_v1}"
NODE_MANIFEST="${SIM_EVALUATION_NODE_MANIFEST:-$DATASET/evaluation/evaluation_points_30_non_grasp.json}"
BOUNDARY="${SIM_GRASP_BOUNDARY:-$DATASET/task_inputs/known_grasp_region_boundary.npz}"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FUTURE_START=$((FRAME_COUNT * 4 / 5))
METHOD_A="pbd"
METHOD_B="pbd_cotracker_foundation_depth"
METHOD_C="${SIM_METHOD_C:-pbd_cotracker_foundation_depth_hierarchical_stiffness_h1_safe}"
STIFFNESS_UPDATE_MODE="${SIM_STIFFNESS_UPDATE_MODE:-differentiable_hierarchical_relative}"
STIFFNESS_AUTOGRAD_UNROLL_STEPS="${SIM_STIFFNESS_AUTOGRAD_UNROLL_STEPS:-5}"
STIFFNESS_AUTOGRAD_REGION_COUNT="${SIM_STIFFNESS_AUTOGRAD_REGION_COUNT:-6}"
C_LABEL="${SIM_C_LABEL:-C-H1：B + H1短期约束的H3/H5全局/区域刚度更新}"

if [[ $# -gt 1 ]]; then
    echo "用法：SIM_DATASET=<数据集> FLOW_DEPTH_ASSET_ROOT=<资产> bash scripts/run_sim_foundation_h1_safe_dataset_complete.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
for required in \
    "$PYTHON" \
    "$ASSET_ROOT/bindings.npz" \
    "$ASSET_ROOT/observations.npz" \
    "$ASSET_ROOT/report.json" \
    "$DEPTH_ROOT/depth_generation_summary.json" \
    "$NODE_MANIFEST" \
    "$BOUNDARY"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少正式测评输入：$required" >&2
        exit 1
    fi
done
mkdir -p "$OUTPUT_ROOT/logs"
export SIM_DATASET="$DATASET"

run_method() {
    local gpu_id="$1"
    local capability="$2"
    local method="$3"
    local group="$OUTPUT_ROOT/$capability/$method"
    local feedback="off"
    local observation_args=()
    local stiffness_args=(--no-online-stiffness-update)
    local schedule_args=()
    local metric_args=()
    if [[ "$capability" == "reconstruction_7to1" ]]; then
        schedule_args=(
            --evaluation-holdout-stride 8
            --evaluation-holdout-offset 7
            --evaluation-render-frame-mode holdout
        )
        metric_args=(--frame-stride 8 --frame-offset 7)
    else
        schedule_args=(
            --evaluation-open-loop-start-frame "$FUTURE_START"
            --evaluation-render-frame-mode future
        )
        metric_args=(--frame-start "$FUTURE_START")
    fi
    if [[ "$method" != "$METHOD_A" ]]; then
        feedback="trajectory"
        observation_args=(
            --flow-depth-bindings "$ASSET_ROOT/bindings.npz"
            --flow-depth-observations "$ASSET_ROOT/observations.npz"
            --flow-depth-position-gain 0.70
            --flow-depth-velocity-gain 0.20
            --flow-depth-absolute-position-weight 0.85
            --flow-depth-solver-regularization 0.01
            --flow-depth-solver-iterations 24
            --flow-depth-robust-residual-mm 5.0
            --flow-depth-maximum-position-correction-mm 2.0
            --flow-depth-maximum-velocity-correction-m-s 0.06
        )
    fi
    if [[ "$method" == "$METHOD_C" ]]; then
        stiffness_args=(
            --online-stiffness-update
            --stiffness-update-mode "$STIFFNESS_UPDATE_MODE"
            --stiffness-log-learning-rate 0.03
            --stiffness-maximum-log-step 0.02
            --stiffness-signal-ema-decay 0.90
            --stiffness-spatial-smoothing-iterations 3
            --stiffness-spatial-smoothing-blend 0.35
            --stiffness-strain-signal-weight 0.20
            --stiffness-autograd-unroll-steps "$STIFFNESS_AUTOGRAD_UNROLL_STEPS"
            --stiffness-autograd-region-count "$STIFFNESS_AUTOGRAD_REGION_COUNT"
        )
    fi

    mkdir -p "$group"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        TORCH_EXTENSIONS_DIR="$ROOT_DIR/.tmp/torch_extensions_h1_transfer_gpu${gpu_id}" \
        bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
            --evaluation-start-frame 0 \
            --evaluation-frame-count 0 \
            --evaluation-physics-steps-per-frame 2 \
            --evaluation-label "${capability}_${method}" \
            --benchmark-output "$group/artifacts" \
            --evaluation-render-images \
            "${schedule_args[@]}" \
            --visual-feedback-mode "$feedback" \
            "${observation_args[@]}" \
            --sim-grasp-boundary-mode known_grasp_region \
            "${stiffness_args[@]}" \
            --initial-paper-distance-stiffness 0.20 \
            --initial-paper-shape-stiffness 0.004

    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction "$group/artifacts/predicted_trajectories.npz" \
        --controlled-boundary "$BOUNDARY" \
        --evaluation-node-manifest "$NODE_MANIFEST" \
        "${metric_args[@]}" \
        --output "$group/trajectory_metrics.json"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" \
        "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction-dir "$group/artifacts/renders" \
        "${metric_args[@]}" \
        --device cuda:0 \
        --output "$group/render_metrics.json"
}

run_capability() {
    local gpu_id="$1"
    local capability="$2"
    run_method "$gpu_id" "$capability" "$METHOD_A"
    run_method "$gpu_id" "$capability" "$METHOD_B"
    run_method "$gpu_id" "$capability" "$METHOD_C"
}

set +e
run_capability 0 reconstruction_7to1 >"$OUTPUT_ROOT/logs/reconstruction_gpu0.log" 2>&1 &
reconstruction_pid=$!
run_capability 1 future_80to20 >"$OUTPUT_ROOT/logs/future_gpu1.log" 2>&1 &
future_pid=$!
wait "$reconstruction_pid"
reconstruction_status=$?
wait "$future_pid"
future_status=$?
set -e
printf 'reconstruction_status=%s\nfuture_status=%s\n' \
    "$reconstruction_status" "$future_status" >"$OUTPUT_ROOT/status.txt"
if [[ "$reconstruction_status" -ne 0 || "$future_status" -ne 0 ]]; then
    exit 3
fi

if [[ "$STIFFNESS_UPDATE_MODE" == "differentiable_hierarchical_relative" ]]; then
    "$PYTHON" "$ROOT_DIR/scripts/summarize_sim_hierarchical_stiffness_diagnostics.py" \
        --evaluation-root "$OUTPUT_ROOT" \
        --method "$METHOD_C"
fi
"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_foundation_depth_evaluation.py" \
    --evaluation-root "$OUTPUT_ROOT" \
    --depth-summary "$DEPTH_ROOT/depth_generation_summary.json" \
    --observation-report "$ASSET_ROOT/report.json" \
    --method-b "$METHOD_B" \
    --label-b "B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度" \
    --method-c "$METHOD_C" \
    --label-c "$C_LABEL"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "完成：$OUTPUT_ROOT/comparison_complete.md"
