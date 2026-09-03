#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"
ASSET_ROOT="$ROOT_DIR/outputs/sim_cotracker3_foundation_depth_v1/flow_depth_assets"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
NODE_MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim03_c_h1_retention_retest_v1}"

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
for required in \
    "$PYTHON" \
    "$ASSET_ROOT/bindings.npz" \
    "$ASSET_ROOT/observations.npz" \
    "$BOUNDARY" \
    "$NODE_MANIFEST"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少复测输入：$required" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_ROOT"
export SIM_DATASET="$DATASET"
CUDA_VISIBLE_DEVICES=0 \
TORCH_EXTENSIONS_DIR="$ROOT_DIR/.tmp/torch_extensions_sim03_double_gradient" \
bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
    --evaluation-start-frame 0 \
    --evaluation-frame-count 0 \
    --evaluation-physics-steps-per-frame 2 \
    --evaluation-label reconstruction_c_h1_retention_full_step \
    --benchmark-output "$OUTPUT_ROOT/artifacts" \
    --evaluation-holdout-stride 8 \
    --evaluation-holdout-offset 7 \
    --evaluation-render-frame-mode holdout \
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
    --online-stiffness-update \
    --stiffness-update-mode differentiable_hierarchical_relative \
    --stiffness-log-learning-rate 0.03 \
    --stiffness-maximum-log-step 0.02 \
    --stiffness-signal-ema-decay 0.90 \
    --stiffness-spatial-smoothing-iterations 3 \
    --stiffness-spatial-smoothing-blend 0.35 \
    --stiffness-strain-signal-weight 0.20 \
    --stiffness-autograd-unroll-steps 5 \
    --stiffness-autograd-region-count 6 \
    --initial-paper-distance-stiffness 0.20 \
    --initial-paper-shape-stiffness 0.004 \
    >"$OUTPUT_ROOT/run.log" 2>&1

"$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
    --reference-dataset "$DATASET" \
    --prediction "$OUTPUT_ROOT/artifacts/predicted_trajectories.npz" \
    --controlled-boundary "$BOUNDARY" \
    --evaluation-node-manifest "$NODE_MANIFEST" \
    --frame-stride 8 \
    --frame-offset 7 \
    --output "$OUTPUT_ROOT/trajectory_metrics.json" \
    >>"$OUTPUT_ROOT/run.log" 2>&1

echo "complete" >"$OUTPUT_ROOT/status.txt"
