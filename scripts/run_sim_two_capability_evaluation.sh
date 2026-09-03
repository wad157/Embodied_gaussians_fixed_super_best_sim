#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm}"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_two_capability_$(date +%Y%m%d_%H%M%S)}"
NODE_MANIFEST="${SIM_EVALUATION_NODE_MANIFEST:-$DATASET/evaluation/evaluation_points_10_non_grasp.json}"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FUTURE_START=$((FRAME_COUNT * 4 / 5))
INITIAL_DISTANCE=0.20
INITIAL_SHAPE=0.004

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_two_capability_evaluation.sh [新输出目录]" >&2
    exit 2
fi
if [[ "$FRAME_COUNT" -ne 360 || "$FUTURE_START" -ne 288 ]]; then
    echo "ERROR: 当前正式协议要求 360 帧且 80/20 分界为 288；实际为 $FRAME_COUNT/$FUTURE_START" >&2
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
    "$DATASET/gui_assets/tissue_fixedsuperbest.npz" \
    "$NODE_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少正式六实验输入：$required" >&2
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
    echo "[六实验] 开始 capability=$capability method=$key visual=$feedback stiffness=$stiffness"
    bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 0 \
        --evaluation-physics-steps-per-frame 3 \
        --evaluation-label "${capability}_${key}" \
        --benchmark-output "$group/artifacts" \
        --evaluation-render-images \
        "${schedule_args[@]}" \
        --visual-feedback-mode "$feedback" \
        "$stiffness_flag" \
        --stiffness-log-learning-rate 0.30 \
        --stiffness-maximum-log-step 0.10 \
        --stiffness-signal-ema-decay 0.60 \
        --stiffness-spatial-smoothing-iterations 2 \
        --stiffness-spatial-smoothing-blend 0.30 \
        --initial-paper-distance-stiffness "$INITIAL_DISTANCE" \
        --initial-paper-shape-stiffness "$INITIAL_SHAPE"

    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction "$group/artifacts/predicted_trajectories.npz" \
        --evaluation-node-manifest "$NODE_MANIFEST" \
        "${metric_args[@]}" \
        --output "$group/trajectory_metrics.json"

    "$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction-dir "$group/artifacts/renders" \
        "${metric_args[@]}" \
        --device cuda \
        --output "$group/render_metrics.json"
}

for capability in reconstruction_7to1 future_80to20; do
    run_method "$capability" "pbd" "off" "off"
    run_method "$capability" "pbd_visual_residual" "residual" "off"
    run_method "$capability" "pbd_visual_residual_stiffness" "residual" "on"
done

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_two_capability_evaluation.py" \
    --evaluation-root "$OUTPUT_ROOT"

echo "[六实验] 完成：$OUTPUT_ROOT/comparison.md"
