#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm}"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_known_grasp_full_$(date +%Y%m%d_%H%M%S)}"
NODE_MANIFEST="${SIM_EVALUATION_NODE_MANIFEST:-$DATASET/evaluation/evaluation_points_10_non_grasp.json}"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FUTURE_START=$((FRAME_COUNT * 4 / 5))
INITIAL_DISTANCE=0.20
INITIAL_SHAPE=0.004
ANALYSIS_FRAMES="0,123,126,132,160,200,240,287,359"
STIFFNESS_UPDATE_MODE="${STIFFNESS_UPDATE_MODE:-heuristic}"
STIFFNESS_LOG_LR="${STIFFNESS_LOG_LR:-0.30}"
STIFFNESS_LOG_CAP="${STIFFNESS_LOG_CAP:-0.10}"
STIFFNESS_EMA_DECAY="${STIFFNESS_EMA_DECAY:-0.60}"
STIFFNESS_SMOOTHING_ITERATIONS="${STIFFNESS_SMOOTHING_ITERATIONS:-2}"
STIFFNESS_SMOOTHING_BLEND="${STIFFNESS_SMOOTHING_BLEND:-0.30}"
STIFFNESS_STRAIN_WEIGHT="${STIFFNESS_STRAIN_WEIGHT:-0.80}"
STIFFNESS_AUTOGRAD_STEPS="${STIFFNESS_AUTOGRAD_STEPS:-4}"
STIFFNESS_AUTOGRAD_REGIONS="${STIFFNESS_AUTOGRAD_REGIONS:-12}"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_known_grasp_full_evaluation.sh [新输出目录]" >&2
    exit 2
fi
if [[ "$FRAME_COUNT" -ne 360 || "$FUTURE_START" -ne 288 ]]; then
    echo "ERROR: 正式协议要求360帧且80/20分界为288；实际为$FRAME_COUNT/$FUTURE_START" >&2
    exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
for required in \
    "$DATASET/episode.json" \
    "$DATASET/ground_truth/trajectories_3d.npz" \
    "$DATASET/ground_truth/trajectories_2d/stereo_left.npz" \
    "$DATASET/ground_truth/trajectories_2d/stereo_right.npz" \
    "$DATASET/ground_truth/tissue_state.npz" \
    "$DATASET/gui_assets/tissue_fixedsuperbest.npz" \
    "$BOUNDARY" \
    "$NODE_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少正式评估输入：$required" >&2
        exit 1
    fi
done

export SIM_DATASET="$DATASET"
mkdir -p "$OUTPUT_ROOT/reconstruction_7to1" "$OUTPUT_ROOT/future_80to20"

run_method() {
    local capability="$1"
    local key="$2"
    local feedback="$3"
    local stiffness="$4"
    local group="$OUTPUT_ROOT/$capability/$key"
    local stiffness_flag="--no-online-stiffness-update"
    local schedule_args=()
    local metric_args=()

    if [[ "$stiffness" == "on" ]]; then
        stiffness_flag="--online-stiffness-update"
    fi
    if [[ "$capability" == "reconstruction_7to1" ]]; then
        schedule_args=(
            --evaluation-holdout-stride 8
            --evaluation-holdout-offset 7
            --evaluation-render-frame-mode holdout
        )
        metric_args=(--frame-stride 8 --frame-offset 7)
    elif [[ "$capability" == "future_80to20" ]]; then
        schedule_args=(
            --evaluation-open-loop-start-frame "$FUTURE_START"
            --evaluation-render-frame-mode future
        )
        metric_args=(--frame-start "$FUTURE_START")
    else
        echo "ERROR: 未知能力：$capability" >&2
        exit 1
    fi

    mkdir -p "$group"
    echo "[统一夹持六实验] 开始 capability=$capability method=$key visual=$feedback stiffness=$stiffness"
    bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 0 \
        --evaluation-physics-steps-per-frame 2 \
        --evaluation-label "known_grasp_${capability}_${key}" \
        --benchmark-output "$group/artifacts" \
        --evaluation-render-images \
        "${schedule_args[@]}" \
        --visual-feedback-mode "$feedback" \
        "$stiffness_flag" \
        --sim-grasp-boundary-mode known_grasp_region \
        --stiffness-update-mode "$STIFFNESS_UPDATE_MODE" \
        --stiffness-log-learning-rate "$STIFFNESS_LOG_LR" \
        --stiffness-maximum-log-step "$STIFFNESS_LOG_CAP" \
        --stiffness-signal-ema-decay "$STIFFNESS_EMA_DECAY" \
        --stiffness-spatial-smoothing-iterations "$STIFFNESS_SMOOTHING_ITERATIONS" \
        --stiffness-spatial-smoothing-blend "$STIFFNESS_SMOOTHING_BLEND" \
        --stiffness-strain-signal-weight "$STIFFNESS_STRAIN_WEIGHT" \
        --stiffness-autograd-unroll-steps "$STIFFNESS_AUTOGRAD_STEPS" \
        --stiffness-autograd-region-count "$STIFFNESS_AUTOGRAD_REGIONS" \
        --initial-paper-distance-stiffness "$INITIAL_DISTANCE" \
        --initial-paper-shape-stiffness "$INITIAL_SHAPE"

    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction "$group/artifacts/predicted_trajectories.npz" \
        --controlled-boundary "$BOUNDARY" \
        --evaluation-node-manifest "$NODE_MANIFEST" \
        "${metric_args[@]}" \
        --output "$group/trajectory_metrics.json"

    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction-dir "$group/artifacts/renders" \
        "${metric_args[@]}" \
        --device cuda \
        --output "$group/render_metrics.json"

    "$PYTHON" "$ROOT_DIR/scripts/compare_sim_learned_stiffness_to_gt.py" \
        --dataset "$DATASET" \
        --material-diagnostics "$group/artifacts/material_diagnostics.npz" \
        --analysis-frames "$ANALYSIS_FRAMES" \
        --output-dir "$group/stiffness_gt_comparison"
}

for capability in reconstruction_7to1 future_80to20; do
    run_method "$capability" "pbd" "off" "off"
    run_method "$capability" "pbd_visual_residual" "residual" "off"
    run_method "$capability" "pbd_visual_residual_stiffness" "residual" "on"
done

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_two_capability_evaluation.py" \
    --evaluation-root "$OUTPUT_ROOT"
"$PYTHON" "$ROOT_DIR/scripts/summarize_known_grasp_full_diagnostics.py" \
    --dataset "$DATASET" \
    --evaluation-root "$OUTPUT_ROOT"

find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    > "$OUTPUT_ROOT/SHA256SUMS"
echo "[统一夹持六实验] 完成：$OUTPUT_ROOT/comprehensive_analysis.md"
