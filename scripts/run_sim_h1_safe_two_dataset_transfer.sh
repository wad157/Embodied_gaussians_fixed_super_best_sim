#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPLANE_DATASET="$ROOT_DIR/data/sim/tissue_retraction_closeup_inplane_full_v1"
INPLANE_ASSETS="$ROOT_DIR/outputs/sim_inplane_cotracker3_foundation_depth_v1"
INPLANE_DEPTH="$INPLANE_DATASET/estimated_depth/foundation_stereo_rgb_v2_rig_aware"
INPLANE_EVALUATION="$ROOT_DIR/outputs/sim_inplane_foundation_h1_safe_complete_v1"
LIFT_EVALUATION="$ROOT_DIR/outputs/sim_foundation_h1_safe_hierarchical_complete_v1"

SIM_DATASET="$INPLANE_DATASET" \
SIM_ESTIMATED_DEPTH_ROOT="$INPLANE_DEPTH" \
SIM_DEPTH_SOURCE_LABEL="foundation_stereo_rgb_rig_aware_v2" \
SIM_RECTIFICATION_ALPHA="0.85" \
    bash "$ROOT_DIR/scripts/prepare_sim_foundation_dataset_assets.sh" \
    "$INPLANE_ASSETS"
SIM_DATASET="$INPLANE_DATASET" \
SIM_ESTIMATED_DEPTH_ROOT="$INPLANE_DEPTH" \
FLOW_DEPTH_ASSET_ROOT="$INPLANE_ASSETS/flow_depth_assets" \
    bash "$ROOT_DIR/scripts/run_sim_foundation_h1_safe_dataset_complete.sh" \
    "$INPLANE_EVALUATION"

if [[ ! -f "$LIFT_EVALUATION/comparison_complete.json" ]]; then
    echo "ERROR: 夹起拉升正式结果不存在：$LIFT_EVALUATION" >&2
    exit 1
fi
"${ENV_PREFIX:-/Media_HDD/jwshan/conda_envs/eg_codex}/bin/python" \
    "$ROOT_DIR/scripts/summarize_sim_h1_safe_two_datasets.py" \
    --inplane "$INPLANE_EVALUATION/comparison_complete.json" \
    --lift "$LIFT_EVALUATION/comparison_complete.json" \
    --output "$ROOT_DIR/outputs/sim_h1_safe_two_dataset_summary_v1.md"
echo "全部完成：$ROOT_DIR/outputs/sim_h1_safe_two_dataset_summary_v1.md"
