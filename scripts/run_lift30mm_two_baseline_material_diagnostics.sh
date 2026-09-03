#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python"
DATASET="$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm"
MANIFEST="$DATASET/evaluation/evaluation_points_10_non_grasp.json"
STIFFNESS_RUN="$ROOT_DIR/outputs/sim_stiffness_gt_diagnostic_lift30mm_v1"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_stiffness_gt_three_method_lift30mm_$(date +%Y%m%d_%H%M%S)}"
ANALYSIS_FRAMES="123,126,132,160,200,240,287,359"

if [[ $# -gt 1 ]]; then
    echo "用法：bash scripts/run_lift30mm_two_baseline_material_diagnostics.sh [新输出目录]" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有三方法材料诊断：$OUTPUT_ROOT" >&2
    exit 1
fi
for required in \
    "$DATASET/episode.json" \
    "$DATASET/ground_truth/tissue_state.npz" \
    "$DATASET/gui_assets/tissue_fixedsuperbest.npz" \
    "$MANIFEST" \
    "$STIFFNESS_RUN/artifacts/material_diagnostics.npz" \
    "$STIFFNESS_RUN/trajectory_metrics.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: 缺少三方法材料诊断输入：$required" >&2
        exit 1
    fi
done
mkdir -p "$OUTPUT_ROOT"

for method in pbd pbd_visual_residual; do
    SIM_DATASET="$DATASET" \
    SIM_EVALUATION_NODE_MANIFEST="$MANIFEST" \
    SIM_STIFFNESS_DIAGNOSTIC_METHOD="$method" \
    SIM_STIFFNESS_ANALYSIS_FRAMES="$ANALYSIS_FRAMES" \
        bash "$ROOT_DIR/scripts/run_sim_stiffness_gt_diagnostic.sh" \
        "$OUTPUT_ROOT/$method"
done

"$PYTHON" "$ROOT_DIR/scripts/compare_sim_learned_stiffness_to_gt.py" \
    --dataset "$DATASET" \
    --material-diagnostics "$STIFFNESS_RUN/artifacts/material_diagnostics.npz" \
    --analysis-frames "$ANALYSIS_FRAMES" \
    --output-dir "$OUTPUT_ROOT/pbd_visual_residual_stiffness_reanalysis"

"$PYTHON" "$ROOT_DIR/scripts/summarize_three_method_stiffness_gt.py" \
    --method pbd \
        "$OUTPUT_ROOT/pbd" \
        "$OUTPUT_ROOT/pbd/stiffness_gt_comparison" \
    --method pbd_visual_residual \
        "$OUTPUT_ROOT/pbd_visual_residual" \
        "$OUTPUT_ROOT/pbd_visual_residual/stiffness_gt_comparison" \
    --method pbd_visual_residual_stiffness \
        "$STIFFNESS_RUN" \
        "$OUTPUT_ROOT/pbd_visual_residual_stiffness_reanalysis" \
    --output-dir "$OUTPUT_ROOT/three_method_comparison"

sha256sum \
    "$OUTPUT_ROOT/pbd/artifacts/material_diagnostics.npz" \
    "$OUTPUT_ROOT/pbd_visual_residual/artifacts/material_diagnostics.npz" \
    "$OUTPUT_ROOT/three_method_comparison/comparison.json" \
    > "$OUTPUT_ROOT/SHA256SUMS"

echo "[lift30mm三方法材料诊断] 完成：$OUTPUT_ROOT"
