#!/usr/bin/env bash
set -euo pipefail

# 对指定数据集/方法执行一次连续 rollout：
# 前80%同化，其中每8帧留出第8帧作7:1重建评估；后20%完全开环预测。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
ASSET_BASE="${SIM_ALLTRACKER_ASSET_BASE:-$ROOT_DIR/outputs/sim_three_datasets_alltracker_medium_h3w4_complete_v1/assets}"
GPU_ID="${SIM_GPU_ID:-0}"
REPEAT_ID="${SIM_REPEAT_ID:-unspecified}"

if [[ $# -ne 3 ]]; then
    echo "用法：bash scripts/run_sim_alltracker_unified_method_once.sh <sim01|sim02|sim03> <A|B|C> <评测根目录>" >&2
    exit 2
fi

DATASET_ID="$1"
METHOD_ID="$2"
OUTPUT_ROOT="$3"

case "$DATASET_ID" in
    sim01)
        DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_front_v2"
        DEPTH_ROOT="$DATASET/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
        ;;
    sim02)
        DATASET="$ROOT_DIR/data/sim/tissue_retraction_free_support_side_v2"
        DEPTH_ROOT="$DATASET/estimated_depth/foundation_stereo_rgb_rig_aware_v1"
        ;;
    sim03)
        DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"
        DEPTH_ROOT="$DATASET/estimated_depth/foundation_stereo_rgb_v1"
        ;;
    *)
        echo "ERROR: 未知数据集：$DATASET_ID" >&2
        exit 2
        ;;
esac

METHOD_A="pbd"
METHOD_B="pbd_alltracker_foundation_depth"
METHOD_C="pbd_alltracker_foundation_depth_global_only_h3w4"
case "$METHOD_ID" in
    A) METHOD="$METHOD_A" ;;
    B) METHOD="$METHOD_B" ;;
    C) METHOD="$METHOD_C" ;;
    *)
        echo "ERROR: 方法必须为 A、B 或 C：$METHOD_ID" >&2
        exit 2
        ;;
esac

ASSET_ROOT="$ASSET_BASE/$DATASET_ID/flow_depth_assets"
NODE_MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FUTURE_START=$((FRAME_COUNT * 4 / 5))
GROUP="$OUTPUT_ROOT/methods/$METHOD"
LOG="$OUTPUT_ROOT/logs/${METHOD}.log"
METHOD_STATUS="$OUTPUT_ROOT/status_${METHOD}.txt"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SIM_GPU_ID 必须是非负整数：$GPU_ID" >&2
    exit 2
fi
if [[ -e "$GROUP" ]]; then
    echo "ERROR: 拒绝覆盖已有方法输出：$GROUP" >&2
    exit 1
fi
required=(
    "$PYTHON"
    "$DATASET/episode.json"
    "$DEPTH_ROOT/depth_generation_summary.json"
    "$NODE_MANIFEST"
    "$BOUNDARY"
)
if [[ "$METHOD_ID" != "A" ]]; then
    required+=(
        "$ASSET_ROOT/bindings.npz"
        "$ASSET_ROOT/observations.npz"
        "$ASSET_ROOT/report.json"
    )
fi
for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: 缺少联合协议输入：$path" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_ROOT/logs" "$GROUP"
printf '%s\n' \
    'status=running' \
    "pid=$$" \
    "repeat=$REPEAT_ID" \
    "dataset=$DATASET_ID" \
    "method=$METHOD_ID" \
    "method_name=$METHOD" \
    "frame_count=$FRAME_COUNT" \
    "assimilation_end_exclusive=$FUTURE_START" \
    'protocol=single_rollout_first80_7to1_holdout_last20_open_loop' \
    "gpu=$GPU_ID" >"$METHOD_STATUS"
feedback="off"
observation_args=()
stiffness_args=(--no-online-stiffness-update)
if [[ "$METHOD_ID" != "A" ]]; then
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
if [[ "$METHOD_ID" == "C" ]]; then
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

SIM_DATASET="$DATASET" \
SIM_ESTIMATED_DEPTH_ROOT="$DEPTH_ROOT" \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_EXTENSIONS_DIR="$ROOT_DIR/.tmp/torch_extensions_unified_${DATASET_ID}_gpu${GPU_ID}" \
    bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
        --evaluation-start-frame 0 \
        --evaluation-frame-count 0 \
        --evaluation-physics-steps-per-frame 2 \
        --evaluation-label "${DATASET_ID}_${REPEAT_ID}_unified_${METHOD}" \
        --benchmark-output "$GROUP/artifacts" \
        --evaluation-render-images \
        --evaluation-holdout-stride 8 \
        --evaluation-holdout-offset 7 \
        --evaluation-open-loop-start-frame "$FUTURE_START" \
        --evaluation-render-frame-mode holdout_future \
        --visual-feedback-mode "$feedback" \
        "${observation_args[@]}" \
        --sim-grasp-boundary-mode known_grasp_region \
        "${stiffness_args[@]}" \
        --initial-paper-distance-stiffness 0.20 \
        --initial-paper-shape-stiffness 0.004 \
        >"$LOG" 2>&1

for capability in reconstruction_7to1 future_80to20; do
    METRICS="$OUTPUT_ROOT/metrics/$capability/$METHOD"
    selection_args=()
    mkdir -p "$METRICS"
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
        --prediction "$GROUP/artifacts/predicted_trajectories.npz" \
        --controlled-boundary "$BOUNDARY" \
        --evaluation-node-manifest "$NODE_MANIFEST" \
        "${selection_args[@]}" \
        --output "$METRICS/trajectory_metrics.json" \
        >>"$LOG" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
        "$ROOT_DIR/scripts/evaluate_sim_rendering_metrics.py" \
        --reference-dataset "$DATASET" \
        --prediction-dir "$GROUP/artifacts/renders" \
        "${selection_args[@]}" \
        --device cuda:0 \
        --output "$METRICS/render_metrics.json" \
        >>"$LOG" 2>&1
done

printf '%s\n' \
    'status=complete' \
    "repeat=$REPEAT_ID" \
    "dataset=$DATASET_ID" \
    "method=$METHOD_ID" \
    "method_name=$METHOD" \
    "frame_count=$FRAME_COUNT" \
    "assimilation_end_exclusive=$FUTURE_START" \
    'protocol=single_rollout_first80_7to1_holdout_last20_open_loop' \
    "gpu=$GPU_ID" >"$METHOD_STATUS"
