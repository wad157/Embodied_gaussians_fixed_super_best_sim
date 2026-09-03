#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm}"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_stiffness_gt_diagnostic_$(date +%Y%m%d_%H%M%S)}"
NODE_MANIFEST="${SIM_EVALUATION_NODE_MANIFEST:-$DATASET/evaluation/evaluation_points_10_non_grasp.json}"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FIRST_GRASP_FRAME="$($PYTHON -c 'import json,sys; rows=json.load(open(sys.argv[1])); print(next(int(row["frame"]) for row in rows if row["grasped"]))' "$DATASET/task_inputs/phases.json")"
ANALYSIS_FRAMES="${SIM_STIFFNESS_ANALYSIS_FRAMES:-$FIRST_GRASP_FRAME,$((FIRST_GRASP_FRAME + 3)),$((FIRST_GRASP_FRAME + 9)),$((FRAME_COUNT * 4 / 9)),$((FRAME_COUNT * 5 / 9)),$((FRAME_COUNT * 2 / 3)),$((FRAME_COUNT * 4 / 5 - 1)),$((FRAME_COUNT - 1))}"
METHOD="${SIM_STIFFNESS_DIAGNOSTIC_METHOD:-pbd_visual_residual_stiffness}"

case "$METHOD" in
    pbd)
        VISUAL_FEEDBACK=off
        STIFFNESS_FLAG=--no-online-stiffness-update
        ;;
    pbd_visual_residual)
        VISUAL_FEEDBACK=residual
        STIFFNESS_FLAG=--no-online-stiffness-update
        ;;
    pbd_visual_residual_stiffness)
        VISUAL_FEEDBACK=residual
        STIFFNESS_FLAG=--online-stiffness-update
        ;;
    *)
        echo "ERROR: 未知诊断方法：$METHOD" >&2
        exit 2
        ;;
esac

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_sim_stiffness_gt_diagnostic.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有诊断：$OUTPUT_ROOT" >&2
    exit 1
fi
for required in \
    "$DATASET/episode.json" \
    "$DATASET/ground_truth/tissue_state.npz" \
    "$DATASET/ground_truth/trajectories_3d.npz" \
    "$DATASET/gui_assets/tissue_fixedsuperbest.npz" \
    "$NODE_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少诊断输入：$required" >&2
        exit 1
    fi
done

export SIM_DATASET="$DATASET"
mkdir -p "$OUTPUT_ROOT"

# 与正式六实验的7:1刚度重建组完全相同；只关闭PNG渲染并增加逐粒子材料导出。
bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
    --evaluation-start-frame 0 \
    --evaluation-frame-count 0 \
    --evaluation-physics-steps-per-frame 3 \
    --evaluation-label "reconstruction_7to1_${METHOD}_material_gt_diagnostic" \
    --benchmark-output "$OUTPUT_ROOT/artifacts" \
    --no-evaluation-render-images \
    --evaluation-holdout-stride 8 \
    --evaluation-holdout-offset 7 \
    --evaluation-render-frame-mode holdout \
    --visual-feedback-mode "$VISUAL_FEEDBACK" \
    "$STIFFNESS_FLAG" \
    --stiffness-log-learning-rate 0.30 \
    --stiffness-maximum-log-step 0.10 \
    --stiffness-signal-ema-decay 0.60 \
    --stiffness-spatial-smoothing-iterations 2 \
    --stiffness-spatial-smoothing-blend 0.30 \
    --initial-paper-distance-stiffness 0.20 \
    --initial-paper-shape-stiffness 0.004

"$PYTHON" "$ROOT_DIR/scripts/evaluate_sim_trajectory_metrics.py" \
    --reference-dataset "$DATASET" \
    --prediction "$OUTPUT_ROOT/artifacts/predicted_trajectories.npz" \
    --evaluation-node-manifest "$NODE_MANIFEST" \
    --frame-stride 8 \
    --frame-offset 7 \
    --output "$OUTPUT_ROOT/trajectory_metrics.json"

"$PYTHON" "$ROOT_DIR/scripts/compare_sim_learned_stiffness_to_gt.py" \
    --dataset "$DATASET" \
    --material-diagnostics "$OUTPUT_ROOT/artifacts/material_diagnostics.npz" \
    --analysis-frames "$ANALYSIS_FRAMES" \
    --output-dir "$OUTPUT_ROOT/stiffness_gt_comparison"

sha256sum \
    "$OUTPUT_ROOT/artifacts/material_diagnostics.npz" \
    "$OUTPUT_ROOT/trajectory_metrics.json" \
    "$OUTPUT_ROOT/stiffness_gt_comparison/comparison.json" \
    > "$OUTPUT_ROOT/SHA256SUMS"

echo "[刚度真值诊断] 完成：$OUTPUT_ROOT"
