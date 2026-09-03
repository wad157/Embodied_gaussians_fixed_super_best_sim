#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm}"
ENV_ROOT="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
PYTHON="$ENV_ROOT/bin/python"
ASSET_ROOT="${FLOW_DEPTH_ASSET_ROOT:-$ROOT_DIR/outputs/sim_cotracker3_gt_depth_v1/flow_depth_assets}"
REFERENCE_ROOT="${SIM_REFERENCE_ROOT:-$ROOT_DIR/outputs/sim_cotracker_gt_depth_eval30_v2}"
PURE_PBD_ROOT="${SIM_PURE_PBD_ROOT:-$ROOT_DIR/outputs/sim_known_grasp_causal_full_pbd30hz_v1}"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_global_trackmean_warpfd_cos095_complete_v6}"
NODE_MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FUTURE_START=$((FRAME_COUNT * 4 / 5))
METHOD="pbd_cotracker_gt_depth_global_distribution"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_global_distribution_complete.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
if [[ "$FRAME_COUNT" -ne 360 || "$FUTURE_START" -ne 288 ]]; then
    echo "ERROR: 正式协议要求360帧且80/20分界为288" >&2
    exit 1
fi
for required in \
    "$PYTHON" \
    "$ASSET_ROOT/bindings.npz" \
    "$ASSET_ROOT/observations.npz" \
    "$NODE_MANIFEST" \
    "$BOUNDARY" \
    "$REFERENCE_ROOT/reconstruction_7to1/pbd_cotracker_gt_depth/trajectory_metrics.json" \
    "$REFERENCE_ROOT/future_80to20/pbd_cotracker_gt_depth/trajectory_metrics.json" \
    "$PURE_PBD_ROOT/reconstruction_7to1/pbd/render_metrics.json" \
    "$PURE_PBD_ROOT/future_80to20/pbd/render_metrics.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少完整测评输入：$required" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_ROOT/logs"
export SIM_DATASET="$DATASET"

run_case() {
    local gpu_id="$1"
    local capability="$2"
    local group="$OUTPUT_ROOT/$capability/$METHOD"
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
    mkdir -p "$group"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        TORCH_EXTENSIONS_DIR="/Media_HDD/jwshan/tmp/torch_extensions_global_distribution_gpu${gpu_id}" \
        bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
            --evaluation-start-frame 0 \
            --evaluation-frame-count 0 \
            --evaluation-physics-steps-per-frame 2 \
            --evaluation-label "${capability}_${METHOD}" \
            --benchmark-output "$group/artifacts" \
            --evaluation-render-images \
            "${schedule_args[@]}" \
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
            --online-stiffness-update \
            --sim-grasp-boundary-mode known_grasp_region \
            --stiffness-update-mode differentiable_global \
            --stiffness-log-learning-rate 0.03 \
            --stiffness-maximum-log-step 0.02 \
            --stiffness-signal-ema-decay 0.90 \
            --stiffness-spatial-smoothing-iterations 3 \
            --stiffness-spatial-smoothing-blend 0.35 \
            --stiffness-strain-signal-weight 0.20 \
            --stiffness-autograd-unroll-steps 4 \
            --stiffness-autograd-region-count 12 \
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

set +e
run_case 0 reconstruction_7to1 >"$OUTPUT_ROOT/logs/reconstruction_gpu0.log" 2>&1 &
reconstruction_pid=$!
run_case 1 future_80to20 >"$OUTPUT_ROOT/logs/future_gpu1.log" 2>&1 &
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

"$PYTHON" "$ROOT_DIR/scripts/summarize_sim_global_distribution_evaluation.py" \
    --evaluation-root "$OUTPUT_ROOT" \
    --reference-root "$REFERENCE_ROOT" \
    --pure-pbd-root "$PURE_PBD_ROOT"
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$OUTPUT_ROOT/SHA256SUMS"
echo "完成：$OUTPUT_ROOT/comparison_complete.md"
