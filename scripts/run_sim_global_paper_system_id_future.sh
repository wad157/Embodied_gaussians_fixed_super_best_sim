#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${SIM_DATASET:-$ROOT_DIR/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm}"
ENV_ROOT="${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}"
PYTHON="$ENV_ROOT/bin/python"
ASSET_ROOT="${FLOW_DEPTH_ASSET_ROOT:-$ROOT_DIR/outputs/sim_cotracker3_gt_depth_v1/flow_depth_assets}"
OUTPUT_ROOT="${1:-$ROOT_DIR/outputs/sim_global_paper_system_id_future_v1}"
NODE_MANIFEST="$DATASET/evaluation/evaluation_points_30_non_grasp.json"
BOUNDARY="$DATASET/task_inputs/known_grasp_region_boundary.npz"
FRAME_COUNT="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["frames"]))' "$DATASET/episode.json")"
FUTURE_START=$((FRAME_COUNT * 4 / 5))

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: 拒绝覆盖已有输出：$OUTPUT_ROOT" >&2
    exit 1
fi
for required in \
    "$ASSET_ROOT/bindings.npz" \
    "$ASSET_ROOT/observations.npz" \
    "$NODE_MANIFEST" \
    "$BOUNDARY"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: 缺少输入：$required" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_ROOT"
export SIM_DATASET="$DATASET"
export SIM_BENCHMARK_SKIP_RGB_ALIGNMENT=1

bash "$ROOT_DIR/scripts/run_sim_reconstruction_headless.sh" \
    --evaluation-start-frame 0 \
    --evaluation-frame-count 0 \
    --evaluation-physics-steps-per-frame 2 \
    --evaluation-label future_global_paper_system_id \
    --benchmark-output "$OUTPUT_ROOT/artifacts" \
    --no-evaluation-render-images \
    --evaluation-open-loop-start-frame "$FUTURE_START" \
    --evaluation-render-frame-mode future \
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
    --prediction "$OUTPUT_ROOT/artifacts/predicted_trajectories.npz" \
    --controlled-boundary "$BOUNDARY" \
    --evaluation-node-manifest "$NODE_MANIFEST" \
    --frame-start "$FUTURE_START" \
    --output "$OUTPUT_ROOT/trajectory_metrics.json"

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = json.loads((root / "trajectory_metrics.json").read_text())
summary = {
    "protocol": "future_80to20",
    "method": "paper_PBD+CoTracker_RGB+dataset_depth+global_distance_damping_coupling",
    "target_previous_3d_mean_mm": 1.2017,
    "trajectory_metrics": metrics,
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
